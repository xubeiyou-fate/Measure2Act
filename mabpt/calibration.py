"""E11 model-independent event calibration and continuous mixture likelihood."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.edfa_ascent.relation import pack_scenes
from experiments.energy_predict_optimize.evaluation import compute_batch_metrics
from experiments.ascent_recomparison.common import fold_subsets, load_dataset, loader
from experiments.tpmo_ascent.protocol import load_protocol as load_legacy_data_protocol

from .evaluate import ROOT, _load_models
from .events import PROTOCOL, atomic_json, event_labels, event_membership, sha256
from .freeze import verify as verify_legacy_freeze
from .operator import (
    DEFAULT_ADE_SCALE,
    energy_kl_projection,
    exact_gibbs_transport,
    pairwise_trajectory_distance,
    support_cost,
)


def event_probabilities(
    mode_probabilities: torch.Tensor, mode_events: torch.Tensor
) -> torch.Tensor:
    if mode_probabilities.shape != mode_events.shape:
        raise ValueError("mode probabilities and events must share shape [B,K]")
    events = mode_probabilities.new_zeros((mode_probabilities.shape[0], 9))
    events.scatter_add_(1, mode_events, mode_probabilities)
    return events


def kernel_event_probabilities(
    mode_probabilities: torch.Tensor, membership: torch.Tensor
) -> torch.Tensor:
    if membership.shape != (*mode_probabilities.shape, 9):
        raise ValueError("membership must have shape [B,K,9]")
    probabilities = torch.einsum(
        "bk,bke->be", mode_probabilities.to(torch.float64), membership.to(torch.float64)
    )
    return probabilities / probabilities.sum(dim=1, keepdim=True)


def continuous_mixture_nll(
    predictions: torch.Tensor,
    probabilities: torch.Tensor,
    target: torch.Tensor,
    variance: torch.Tensor,
) -> torch.Tensor:
    predictions = predictions.to(torch.float64)
    probabilities = probabilities.to(torch.float64)
    target = target.to(torch.float64)
    variance = variance.to(device=predictions.device, dtype=torch.float64)
    if variance.shape != predictions.shape[-2:]:
        raise ValueError("variance must have shape [H,D]")
    log_component = -0.5 * (
        (predictions - target[:, None]).square() / variance[None, None]
        + torch.log(2.0 * math.pi * variance)[None, None]
    ).sum(dim=(-1, -2))
    tiny = torch.finfo(torch.float64).tiny
    return -torch.logsumexp(probabilities.clamp_min(tiny).log() + log_component, dim=1)


class CalibrationAccumulator:
    def __init__(self) -> None:
        self.probabilities: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []
        self.sums = defaultdict(float)
        self.count = 0
        self.date_counts = defaultdict(int)
        self.date_sums = defaultdict(lambda: defaultdict(float))

    def update(
        self,
        *,
        event_probability: torch.Tensor,
        event_label: torch.Tensor,
        mixture_nll: torch.Tensor,
        trajectory_probability: torch.Tensor,
        metrics: dict[str, torch.Tensor],
        pairwise: torch.Tensor,
        actor_dates: np.ndarray,
        duplicate_threshold: float,
        hard_zero_event: torch.Tensor,
    ) -> None:
        event_probability = event_probability.to(torch.float64)
        label_one_hot = F.one_hot(event_label, num_classes=9).to(torch.float64)
        tiny = torch.finfo(torch.float64).tiny
        rows = torch.arange(event_label.shape[0], device=event_label.device)
        event_nll = -event_probability[rows, event_label].clamp_min(tiny).log()
        event_brier = (event_probability - label_one_hot).square().sum(dim=1)
        trajectory_probability = trajectory_probability.to(torch.float64)
        entropy = -(
            trajectory_probability
            * trajectory_probability.clamp_min(tiny).log()
        ).sum(dim=1)
        event_entropy = -(
            event_probability * event_probability.clamp_min(tiny).log()
        ).sum(dim=1)
        modes = pairwise.shape[1]
        off_diagonal = ~torch.eye(
            modes, dtype=torch.bool, device=pairwise.device
        )[None]
        duplicate_rate = (
            (pairwise < duplicate_threshold) & off_diagonal
        ).sum(dim=(1, 2)).to(torch.float64) / (modes * (modes - 1))
        arrays = {
            "event_nll": event_nll.detach().cpu().numpy(),
            "event_brier": event_brier.detach().cpu().numpy(),
            "mixture_nll": mixture_nll.detach().cpu().numpy(),
            "energy_score": metrics["energy_score"].detach().cpu().numpy(),
            "top1_ade": metrics["top1_ade"].detach().cpu().numpy(),
            "top1_fde": metrics["top1_fde"].detach().cpu().numpy(),
            "effective_modes": entropy.exp().detach().cpu().numpy(),
            "effective_events": event_entropy.exp().detach().cpu().numpy(),
            "duplicate_pair_rate": duplicate_rate.detach().cpu().numpy(),
            "hard_support_zero_rate": hard_zero_event.to(torch.float64).detach().cpu().numpy(),
        }
        count = int(event_label.numel())
        self.count += count
        for name, values in arrays.items():
            self.sums[name] += float(values.sum())
        self.probabilities.append(event_probability.detach().cpu().numpy())
        self.labels.append(event_label.detach().cpu().numpy())
        dates = np.asarray(actor_dates, dtype=object)
        for date in sorted(set(dates.tolist())):
            mask = dates == date
            date_count = int(mask.sum())
            self.date_counts[str(date)] += date_count
            for name in ("event_nll", "event_brier", "mixture_nll", "energy_score"):
                self.date_sums[str(date)][name] += float(arrays[name][mask].sum())

    @staticmethod
    def _reliability(
        probabilities: np.ndarray, labels: np.ndarray, bins: int
    ) -> dict[str, object]:
        confidence = probabilities.max(axis=1)
        prediction = probabilities.argmax(axis=1)
        correct = prediction == labels
        edges = np.linspace(0.0, 1.0, bins + 1)
        result = []
        ece = 0.0
        for index in range(bins):
            upper = confidence <= edges[index + 1] if index == bins - 1 else confidence < edges[index + 1]
            mask = (confidence >= edges[index]) & upper
            count = int(mask.sum())
            mean_confidence = float(confidence[mask].mean()) if count else None
            accuracy = float(correct[mask].mean()) if count else None
            if count:
                ece += (count / len(labels)) * abs(accuracy - mean_confidence)
            result.append(
                {
                    "lower": float(edges[index]),
                    "upper": float(edges[index + 1]),
                    "count": count,
                    "mean_confidence": mean_confidence,
                    "accuracy": accuracy,
                }
            )
        return {"ece": ece, "bins": result}

    @staticmethod
    def _brier_decomposition(
        probabilities: np.ndarray, labels: np.ndarray, bins: int
    ) -> dict[str, float]:
        edges = np.linspace(0.0, 1.0, bins + 1)
        reliability = 0.0
        resolution = 0.0
        uncertainty = 0.0
        sample_count = len(labels)
        for event in range(9):
            observed = (labels == event).astype(np.float64)
            base_rate = observed.mean()
            uncertainty += base_rate * (1.0 - base_rate)
            for index in range(bins):
                upper = probabilities[:, event] <= edges[index + 1] if index == bins - 1 else probabilities[:, event] < edges[index + 1]
                mask = (probabilities[:, event] >= edges[index]) & upper
                count = int(mask.sum())
                if not count:
                    continue
                forecast = probabilities[mask, event].mean()
                frequency = observed[mask].mean()
                reliability += (count / sample_count) * (forecast - frequency) ** 2
                resolution += (count / sample_count) * (frequency - base_rate) ** 2
        return {
            "reliability": float(reliability),
            "resolution": float(resolution),
            "uncertainty": float(uncertainty),
            "decomposed_brier": float(reliability - resolution + uncertainty),
        }

    def summary(self, *, calibration_bins: int, reliability_bins: int) -> dict[str, object]:
        probabilities = np.concatenate(self.probabilities)
        labels = np.concatenate(self.labels)
        reliability = self._reliability(probabilities, labels, calibration_bins)
        return {
            "actors": self.count,
            **{name: value / self.count for name, value in self.sums.items()},
            "event_ece": reliability["ece"],
            "reliability_diagram": reliability["bins"],
            "brier_decomposition": self._brier_decomposition(
                probabilities, labels, reliability_bins
            ),
            "event_prevalence": np.bincount(labels, minlength=9).astype(float).tolist(),
            "date_metrics": {
                date: {
                    "actors": self.date_counts[date],
                    **{
                        name: self.date_sums[date][name] / self.date_counts[date]
                        for name in ("event_nll", "event_brier", "mixture_nll", "energy_score")
                    },
                }
                for date in sorted(self.date_counts)
            },
        }


@torch.inference_mode()
def run(
    *,
    fold: int,
    definition_path: Path,
    device: torch.device,
    workers: int,
    batch_size: int,
    max_validation_scenes: int | None,
) -> dict[str, object]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    if fold not in protocol["folds"] or definition["fold"] != fold:
        raise RuntimeError("E11 fold/definition mismatch")
    if definition["protocol_sha256"] != sha256(PROTOCOL):
        raise RuntimeError("E11 definition protocol hash mismatch")
    if not verify_legacy_freeze()["ok"]:
        raise RuntimeError("legacy freeze verification failed")
    legacy = load_legacy_data_protocol()
    legacy.assert_boundaries()
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(True)
    started = time.time()
    dataset = load_dataset(legacy)
    _, validation_data, validation_dates = fold_subsets(
        legacy,
        dataset,
        fold,
        max_validation_scenes=max_validation_scenes,
    )
    validation_loader = loader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        workers=workers,
        prefetch=4,
    )
    baseline, target_model, baseline_path, target_path = _load_models(fold, device)
    states = {"ascent": CalibrationAccumulator(), "mabpt": CalibrationAccumulator()}
    variance = torch.tensor(
        definition["constant_velocity_residual_variance_km2"],
        dtype=torch.float64,
        device=device,
    )
    turn_threshold = float(definition["turn_threshold_radians"])
    altitude_threshold = float(definition["altitude_threshold_km"])
    duplicate_threshold = float(protocol["duplicate_threshold_mean_distance_km"])
    scene_cursor = 0
    for data in validation_loader:
        data = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in data.items()
        }
        observations = data["obs_traj"].transpose(1, 0)
        target = data["pred_traj"].transpose(1, 0)
        source_support, source_logits, _ = baseline(data)
        source_probability = source_logits.softmax(dim=1)
        source_decision = source_logits.argmax(dim=1)
        source_metrics = compute_batch_metrics(
            source_support, source_probability, source_decision, target
        )
        target_support, _, target_decision, auxiliary = target_model(data)
        transported = exact_gibbs_transport(
            source_probability,
            support_cost(source_support, target_support),
            mass_weighted=True,
        )["transported"]
        mabpt_probability = energy_kl_projection(
            transported,
            auxiliary["centered_predicted_normalized_ade_risk"].to(torch.float64),
            pairwise_trajectory_distance(target_support) / DEFAULT_ADE_SCALE,
        )[0]
        mabpt_metrics = compute_batch_metrics(
            target_support.to(torch.float64),
            mabpt_probability,
            target_decision,
            target.to(torch.float64),
        )
        truth = event_labels(
            observations,
            target,
            turn_threshold=turn_threshold,
            altitude_threshold=altitude_threshold,
        )
        packed = pack_scenes(data["adj"])
        dates = validation_dates[scene_cursor : scene_cursor + packed.scene_count]
        actor_dates = np.asarray(dates, dtype=object)[
            packed.inverse.detach().cpu().numpy()
        ]
        for name, predictions, probabilities, metrics in (
            ("ascent", source_support, source_probability, source_metrics),
            ("mabpt", target_support, mabpt_probability, mabpt_metrics),
        ):
            mode_event = event_labels(
                observations,
                predictions,
                turn_threshold=turn_threshold,
                altitude_threshold=altitude_threshold,
            )
            hard_event_probability = event_probabilities(
                probabilities.to(torch.float64), mode_event
            )
            rows = torch.arange(truth.shape[0], device=truth.device)
            hard_zero_event = hard_event_probability[rows, truth] == 0
            membership = event_membership(
                observations,
                predictions,
                turn_threshold=turn_threshold,
                altitude_threshold=altitude_threshold,
            )
            event_probability = kernel_event_probabilities(probabilities, membership)
            pairwise = pairwise_trajectory_distance(predictions)
            states[name].update(
                event_probability=event_probability,
                event_label=truth,
                mixture_nll=continuous_mixture_nll(
                    predictions, probabilities, target, variance
                ),
                trajectory_probability=probabilities,
                metrics=metrics,
                pairwise=pairwise,
                actor_dates=actor_dates,
                duplicate_threshold=duplicate_threshold,
                hard_zero_event=hard_zero_event,
            )
        scene_cursor += packed.scene_count
    summaries = {
        name: state.summary(
            calibration_bins=int(protocol["calibration_bins"]),
            reliability_bins=int(protocol["reliability_bins"]),
        )
        for name, state in states.items()
    }
    summaries["relative_gain_mabpt_vs_ascent"] = {
        metric: (summaries["ascent"][metric] - summaries["mabpt"][metric])
        / summaries["ascent"][metric]
        for metric in ("event_nll", "event_brier", "mixture_nll", "energy_score")
    }
    return {
        "format_version": 1,
        "model": "MABPT",
        "experiment_id": "E11",
        "evidence_class": "legacy_development_only",
        "protocol_sha256": sha256(PROTOCOL),
        "definition": {
            "path": str(definition_path.relative_to(ROOT)),
            "sha256": sha256(definition_path),
            "turn_threshold_radians": turn_threshold,
            "altitude_threshold_km": altitude_threshold,
        },
        "fold": fold,
        "validation_scenes": len(validation_dates),
        "results": summaries,
        "integrity": {
            "event_definition_fit_on_training_only": True,
            "covariance_fit_on_training_only": True,
            "target_in_probability_forward": False,
            "post_hoc_calibration": False,
            "gate_or_residual_used": False,
        },
        "inputs": {
            "baseline_checkpoint": baseline_path,
            "target_checkpoint": target_path,
        },
        "runtime": {
            "device": str(device),
            "elapsed_seconds": time.time() - started,
            "peak_allocated_gpu_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=(1, 2))
    parser.add_argument("--definition", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-validation-scenes", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.definition is None:
        args.definition = ROOT / "artifacts/mabpt" / f"e11_definition_fold{args.fold}_v1.json"
    if args.smoke and args.max_validation_scenes is None:
        args.max_validation_scenes = 8
    result = run(
        fold=args.fold,
        definition_path=args.definition.resolve(),
        device=torch.device(args.device),
        workers=args.workers,
        batch_size=args.batch_size,
        max_validation_scenes=args.max_validation_scenes,
    )
    if args.output is None:
        suffix = "smoke" if args.smoke else "formal"
        args.output = ROOT / "artifacts/mabpt" / f"e11_fold{args.fold}_{suffix}_v1.json"
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "fold": result["fold"],
                "actors": result["results"]["mabpt"]["actors"],
                "relative_gain": result["results"]["relative_gain_mabpt_vs_ascent"],
                "elapsed_seconds": result["runtime"]["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
