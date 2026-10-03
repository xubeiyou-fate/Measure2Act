"""E12 training-threshold fit and traffic decision evaluation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from experiments.edfa_ascent.relation import pack_scenes
from experiments.ascent_recomparison.common import fold_subsets, load_dataset, loader
from experiments.tpmo_ascent.protocol import load_protocol as load_legacy_data_protocol

from .evaluate import ROOT, _load_models
from .events import atomic_json, sha256
from .freeze import verify as verify_legacy_freeze
from .operator import (
    DEFAULT_ADE_SCALE,
    energy_kl_projection,
    exact_gibbs_transport,
    pairwise_trajectory_distance,
    support_cost,
)


PROTOCOL = Path(__file__).with_name("e12_protocol.json")


def _model_outputs(baseline, target_model, data):
    source_support, source_logits, _ = baseline(data)
    source_probability = source_logits.softmax(dim=1)
    target_support, _, _, auxiliary = target_model(data)
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
    return {
        "ascent": (source_support, source_probability),
        "mabpt": (target_support, mabpt_probability),
    }


def pair_conflict_probabilities(
    predictions: torch.Tensor,
    probabilities: torch.Tensor,
    target: torch.Tensor,
    scene_index: torch.Tensor,
    *,
    horizontal_threshold: float,
    vertical_threshold: float,
    horizontal_scale: float,
    vertical_scale: float,
) -> dict[str, torch.Tensor]:
    packed = pack_scenes(scene_index)
    actors = packed.max_actors
    valid_pairs = (
        packed.valid[:, :, None]
        & packed.valid[:, None, :]
        & torch.triu(
            torch.ones(actors, actors, dtype=torch.bool, device=predictions.device),
            diagonal=1,
        )[None]
    )
    pair_index = torch.nonzero(valid_pairs, as_tuple=False)
    if not pair_index.numel():
        empty = probabilities.new_empty(0, dtype=torch.float64)
        return {
            "probability": empty,
            "hard_probability": empty,
            "label": empty.to(torch.bool),
            "first_true_step": empty.to(torch.long),
            "pair_scene": empty.to(torch.long),
        }
    pair_scene, pair_row, pair_column = pair_index.unbind(dim=1)
    row_actor = packed.global_index[pair_scene, pair_row]
    column_actor = packed.global_index[pair_scene, pair_column]
    row_prediction = predictions[row_actor].to(torch.float64)
    column_prediction = predictions[column_actor].to(torch.float64)
    relative = row_prediction[:, :, None] - column_prediction[:, None, :]
    horizontal = torch.linalg.vector_norm(relative[..., :2], dim=-1)
    vertical = relative[..., 2].abs()
    membership = torch.sigmoid(
        (horizontal_threshold - horizontal) / horizontal_scale
    ) * torch.sigmoid((vertical_threshold - vertical) / vertical_scale)
    mode_pair_event = 1.0 - torch.prod(1.0 - membership, dim=-1)
    hard_mode_pair_event = (
        (horizontal <= horizontal_threshold) & (vertical <= vertical_threshold)
    ).any(dim=-1)
    joint_mass = probabilities[row_actor].to(torch.float64)[:, :, None] * probabilities[
        column_actor
    ].to(torch.float64)[:, None, :]
    conflict_probability = (joint_mass * mode_pair_event).sum(dim=(1, 2))
    hard_probability = (
        joint_mass * hard_mode_pair_event.to(torch.float64)
    ).sum(dim=(1, 2))
    truth_relative = target[row_actor].to(torch.float64) - target[column_actor].to(
        torch.float64
    )
    truth_conflict_steps = (
        torch.linalg.vector_norm(truth_relative[..., :2], dim=-1)
        <= horizontal_threshold
    ) & (truth_relative[..., 2].abs() <= vertical_threshold)
    label = truth_conflict_steps.any(dim=1)
    first = truth_conflict_steps.to(torch.int64).argmax(dim=1)
    first = torch.where(label, first, torch.full_like(first, -1))
    return {
        "probability": conflict_probability,
        "hard_probability": hard_probability,
        "label": label,
        "first_true_step": first,
        "pair_scene": pair_scene,
    }


def _average_precision(probability: np.ndarray, label: np.ndarray) -> float:
    positives = int(label.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-probability, kind="stable")
    ordered = label[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered].sum() / positives)


def _ece(probability: np.ndarray, label: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        upper = probability <= edges[index + 1] if index == bins - 1 else probability < edges[index + 1]
        mask = (probability >= edges[index]) & upper
        if mask.any():
            result += mask.mean() * abs(label[mask].mean() - probability[mask].mean())
    return float(result)


class PairAccumulator:
    def __init__(self) -> None:
        self.probability: list[np.ndarray] = []
        self.hard_probability: list[np.ndarray] = []
        self.label: list[np.ndarray] = []
        self.first_step: list[np.ndarray] = []
        self.date_counts = defaultdict(int)
        self.date_sums = defaultdict(lambda: defaultdict(float))

    def update(self, result: dict[str, torch.Tensor], pair_dates: np.ndarray | None) -> None:
        probability = result["probability"].detach().cpu().numpy()
        hard_probability = result["hard_probability"].detach().cpu().numpy()
        label = result["label"].detach().cpu().numpy().astype(bool)
        first = result["first_true_step"].detach().cpu().numpy()
        self.probability.append(probability)
        self.hard_probability.append(hard_probability)
        self.label.append(label)
        self.first_step.append(first)
        if pair_dates is None:
            return
        tiny = np.finfo(np.float64).tiny
        nll = -(label * np.log(np.maximum(probability, tiny)) + (~label) * np.log(np.maximum(1.0 - probability, tiny)))
        brier = np.square(probability - label)
        for date in sorted(set(pair_dates.tolist())):
            mask = pair_dates == date
            count = int(mask.sum())
            self.date_counts[str(date)] += count
            self.date_sums[str(date)]["nll"] += float(nll[mask].sum())
            self.date_sums[str(date)]["brier"] += float(brier[mask].sum())

    def arrays(self):
        return (
            np.concatenate(self.probability),
            np.concatenate(self.hard_probability),
            np.concatenate(self.label),
            np.concatenate(self.first_step),
        )

    def summary(self, *, alert_threshold: float) -> dict[str, object]:
        probability, hard_probability, label, first = self.arrays()
        tiny = np.finfo(np.float64).tiny
        nll = -(label * np.log(np.maximum(probability, tiny)) + (~label) * np.log(np.maximum(1.0 - probability, tiny)))
        alert = probability >= alert_threshold
        positive = label.sum()
        negative = (~label).sum()
        true_positive = alert & label
        lead_time = (first[true_positive] + 1) * 5.0
        return {
            "pairs": int(len(label)),
            "positive_pairs": int(positive),
            "prevalence": float(label.mean()),
            "brier": float(np.square(probability - label).mean()),
            "nll": float(nll.mean()),
            "auprc": _average_precision(probability, label),
            "ece": _ece(probability, label),
            "alert_threshold": alert_threshold,
            "recall_at_fixed_fpr": float(true_positive.sum() / max(positive, 1)),
            "observed_fpr": float((alert & ~label).sum() / max(negative, 1)),
            "alerts": int(alert.sum()),
            "true_positive_alerts": int(true_positive.sum()),
            "mean_warning_lead_seconds": float(lead_time.mean()) if len(lead_time) else None,
            "median_warning_lead_seconds": float(np.median(lead_time)) if len(lead_time) else None,
            "hard_support_zero_rate_on_positive": float((hard_probability[label] == 0).mean()) if positive else None,
            "date_metrics": {
                date: {
                    "pairs": self.date_counts[date],
                    "nll": self.date_sums[date]["nll"] / self.date_counts[date],
                    "brier": self.date_sums[date]["brier"] / self.date_counts[date],
                }
                for date in sorted(self.date_counts)
            },
        }


@torch.inference_mode()
def _collect(
    *,
    fold: int,
    split: str,
    device: torch.device,
    workers: int,
    batch_size: int,
    max_scenes: int | None,
) -> tuple[dict[str, PairAccumulator], dict[str, object]]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    legacy = load_legacy_data_protocol()
    legacy.assert_boundaries()
    dataset = load_dataset(legacy)
    training_data, validation_data, validation_dates = fold_subsets(
        legacy,
        dataset,
        fold,
        max_train_scenes=max_scenes if split == "train" else None,
        max_validation_scenes=max_scenes if split == "validation" else None,
    )
    subset = training_data if split == "train" else validation_data
    dates = None if split == "train" else validation_dates
    data_loader = loader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        workers=workers,
        prefetch=4,
    )
    baseline, target_model, baseline_path, target_path = _load_models(fold, device)
    states = {"ascent": PairAccumulator(), "mabpt": PairAccumulator()}
    event = protocol["event"]
    scene_cursor = 0
    for data in data_loader:
        data = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in data.items()
        }
        outputs = _model_outputs(baseline, target_model, data)
        packed = pack_scenes(data["adj"])
        batch_dates = None if dates is None else dates[scene_cursor : scene_cursor + packed.scene_count]
        target = data["pred_traj"].transpose(1, 0)
        for model, (predictions, probabilities) in outputs.items():
            result = pair_conflict_probabilities(
                predictions,
                probabilities,
                target,
                data["adj"],
                horizontal_threshold=float(event["horizontal_threshold_km"]),
                vertical_threshold=float(event["vertical_threshold_km"]),
                horizontal_scale=float(event["horizontal_kernel_scale_km"]),
                vertical_scale=float(event["vertical_kernel_scale_km"]),
            )
            pair_dates = (
                None
                if batch_dates is None
                else np.asarray(batch_dates, dtype=object)[
                    result["pair_scene"].detach().cpu().numpy()
                ]
            )
            states[model].update(result, pair_dates)
        scene_cursor += packed.scene_count
    return states, {
        "baseline_checkpoint": baseline_path,
        "target_checkpoint": target_path,
        "scenes": len(subset),
    }


@torch.inference_mode()
def fit_thresholds(
    *, fold: int, device: torch.device, workers: int, batch_size: int, max_scenes: int | None
) -> dict[str, object]:
    started = time.time()
    states, inputs = _collect(
        fold=fold,
        split="train",
        device=device,
        workers=workers,
        batch_size=batch_size,
        max_scenes=max_scenes,
    )
    thresholds = {}
    diagnostics = {}
    for model, state in states.items():
        probability, _, label, _ = state.arrays()
        negatives = probability[~label]
        if not len(negatives):
            raise RuntimeError("E12 training split contains no negative pairs")
        threshold = float(np.quantile(negatives, 0.95, method="higher"))
        thresholds[model] = threshold
        diagnostics[model] = {
            "pairs": int(len(label)),
            "positive_pairs": int(label.sum()),
            "negative_pairs": int((~label).sum()),
            "training_fpr": float((negatives >= threshold).mean()),
        }
    return {
        "format_version": 1,
        "model": "MABPT",
        "experiment_id": "E12",
        "stage": "training_threshold_fit",
        "protocol_sha256": sha256(PROTOCOL),
        "fold": fold,
        "fit_split": "fold_training_only",
        "target_fpr": 0.05,
        "thresholds": thresholds,
        "diagnostics": diagnostics,
        "inputs": inputs,
        "validation_or_test_accessed": False,
        "elapsed_seconds": time.time() - started,
    }


@torch.inference_mode()
def evaluate(
    *,
    fold: int,
    threshold_path: Path,
    device: torch.device,
    workers: int,
    batch_size: int,
    max_scenes: int | None,
) -> dict[str, object]:
    started = time.time()
    thresholds = json.loads(threshold_path.read_text(encoding="utf-8"))
    if thresholds["fold"] != fold or thresholds["protocol_sha256"] != sha256(PROTOCOL):
        raise RuntimeError("E12 threshold artifact mismatch")
    states, inputs = _collect(
        fold=fold,
        split="validation",
        device=device,
        workers=workers,
        batch_size=batch_size,
        max_scenes=max_scenes,
    )
    results = {
        model: state.summary(alert_threshold=float(thresholds["thresholds"][model]))
        for model, state in states.items()
    }
    results["relative_gain_mabpt_vs_ascent"] = {
        "brier": (results["ascent"]["brier"] - results["mabpt"]["brier"])
        / results["ascent"]["brier"],
        "nll": (results["ascent"]["nll"] - results["mabpt"]["nll"])
        / results["ascent"]["nll"],
        "auprc": (results["mabpt"]["auprc"] - results["ascent"]["auprc"])
        / results["ascent"]["auprc"],
    }
    return {
        "format_version": 1,
        "model": "MABPT",
        "experiment_id": "E12",
        "evidence_class": "legacy_development_only",
        "protocol_sha256": sha256(PROTOCOL),
        "fold": fold,
        "threshold_artifact": {
            "path": str(threshold_path.relative_to(ROOT)),
            "sha256": sha256(threshold_path),
        },
        "results": results,
        "inputs": inputs,
        "integrity": {
            "alert_threshold_fit_on_training_only": True,
            "target_in_probability_forward": False,
            "regulatory_claim": False,
            "gate_or_residual_used": False,
        },
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("fit", "evaluate"), required=True)
    parser.add_argument("--fold", type=int, required=True, choices=(1, 2))
    parser.add_argument("--thresholds", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.smoke and args.max_scenes is None:
        args.max_scenes = 64
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(True)
    if not verify_legacy_freeze()["ok"]:
        raise RuntimeError("legacy freeze verification failed")
    if args.stage == "fit":
        result = fit_thresholds(
            fold=args.fold,
            device=device,
            workers=args.workers,
            batch_size=args.batch_size,
            max_scenes=args.max_scenes,
        )
    else:
        if args.thresholds is None:
            raise ValueError("--thresholds is required for E12 evaluation")
        result = evaluate(
            fold=args.fold,
            threshold_path=args.thresholds.resolve(),
            device=device,
            workers=args.workers,
            batch_size=args.batch_size,
            max_scenes=args.max_scenes,
        )
    result["runtime"] = {
        "device": str(device),
        "peak_allocated_gpu_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "stage": args.stage,
                "fold": args.fold,
                "thresholds": result.get("thresholds"),
                "results": result.get("results", {}).get("relative_gain_mabpt_vs_ascent"),
                "elapsed_seconds": result.get("elapsed_seconds"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
