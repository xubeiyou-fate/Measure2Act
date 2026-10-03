"""Evaluate frozen C127/C161 supports with the C162 TPMO operator."""

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
from experiments.metric_exact.evaluation import target_tail_threshold
from experiments.metric_exact.model import build_model as build_c127_model
from experiments.energy_predict_optimize.evaluation import compute_batch_metrics
from experiments.energy_predict_optimize.model import build_model as build_energy_model

from .operator import (
    ADE_SCALE,
    candidate_pairwise_distance,
    cross_support_cost,
    tpmo_probabilities,
    transported_prior,
)
from .protocol import atomic_json, load_protocol, sha256


ROOT = Path(__file__).resolve().parents[1]
CORE_METRICS = (
    "top1_ade",
    "top1_fde",
    "minade",
    "minfde",
    "energy_score",
    "nll",
    "brier",
)
GEOMETRY_METRICS = ("top1_ade", "top1_fde", "minade", "minfde")
PROBABILITY_ARMS = ("identity_prior", "hard_transport", "soft_transport", "tpmo")
ARMS = ("baseline_native", "c161_native", *PROBABILITY_ARMS)


def _gather(values: torch.Tensor, modes: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(values.shape[0], device=values.device)
    return values[rows, modes]


def _probability_metrics(
    predictions: torch.Tensor,
    probabilities: torch.Tensor,
    decision_mode: torch.Tensor,
    target: torch.Tensor,
    oracle_ade_mode: torch.Tensor,
    pairwise_distance: torch.Tensor,
    ade: torch.Tensor,
) -> dict[str, torch.Tensor]:
    one_hot = F.one_hot(oracle_ade_mode, num_classes=predictions.shape[1]).to(
        probabilities.dtype
    )
    energy = (probabilities * ade).sum(dim=1) - 0.5 * torch.einsum(
        "bi,bij,bj->b", probabilities, pairwise_distance, probabilities
    )
    tiny = torch.finfo(probabilities.dtype).tiny
    return {
        "energy_score": energy,
        "nll": -_gather(
            probabilities.clamp_min(tiny), oracle_ade_mode
        ).log(),
        "brier": torch.square(probabilities - one_hot).sum(dim=1),
        "confidence": _gather(probabilities, decision_mode),
    }


def _ece_sums(
    confidence: np.ndarray, correct: np.ndarray, bins: int = 15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    counts = np.zeros(bins, dtype=np.int64)
    confidence_sum = np.zeros(bins, dtype=np.float64)
    correct_sum = np.zeros(bins, dtype=np.float64)
    for index in range(bins):
        upper = (
            confidence <= edges[index + 1]
            if index == bins - 1
            else confidence < edges[index + 1]
        )
        mask = (confidence >= edges[index]) & upper
        counts[index] = int(mask.sum())
        confidence_sum[index] = float(confidence[mask].sum())
        correct_sum[index] = float(correct[mask].sum())
    return counts, confidence_sum, correct_sum


class StreamingAccumulator:
    """Actor sums plus calendar-date sums; dates are the inference units."""

    def __init__(self) -> None:
        self.count = 0
        self.sums = defaultdict(float)
        self.date_counts = defaultdict(int)
        self.date_sums = defaultdict(lambda: defaultdict(float))
        self.ece_counts = np.zeros(15, dtype=np.int64)
        self.ece_confidence = np.zeros(15, dtype=np.float64)
        self.ece_correct = np.zeros(15, dtype=np.float64)
        self.minfde_values: list[np.ndarray] = []
        self.tail_sum = 0.0
        self.tail_count = 0

    def update(
        self,
        metrics: dict[str, torch.Tensor],
        actor_dates: np.ndarray,
        tail: torch.Tensor,
    ) -> None:
        arrays = {
            key: value.detach().cpu().numpy()
            for key, value in metrics.items()
            if key in CORE_METRICS or key in {"confidence", "oracle_ade_rank1", "minfde"}
        }
        n = int(arrays["top1_ade"].shape[0])
        self.count += n
        for metric in CORE_METRICS:
            self.sums[metric] += float(arrays[metric].sum())
        self.minfde_values.append(arrays["minfde"].astype(np.float64, copy=False))
        tail_np = tail.detach().cpu().numpy().astype(bool)
        self.tail_sum += float(arrays["minfde"][tail_np].sum())
        self.tail_count += int(tail_np.sum())
        counts, conf_sum, correct_sum = _ece_sums(
            arrays["confidence"].astype(np.float64),
            arrays["oracle_ade_rank1"].astype(np.float64),
        )
        self.ece_counts += counts
        self.ece_confidence += conf_sum
        self.ece_correct += correct_sum
        dates = np.asarray(actor_dates, dtype=object)
        for date in sorted(set(dates.tolist())):
            mask = dates == date
            date_count = int(mask.sum())
            self.date_counts[str(date)] += date_count
            for metric in ("nll", "brier", "energy_score"):
                self.date_sums[str(date)][metric] += float(arrays[metric][mask].sum())

    def summary(self) -> dict[str, object]:
        if self.count == 0:
            raise RuntimeError("empty C162 accumulator")
        ece = 0.0
        for count, confidence, correct in zip(
            self.ece_counts, self.ece_confidence, self.ece_correct, strict=True
        ):
            if count:
                ece += (count / self.count) * abs(
                    correct / count - confidence / count
                )
        minfde = np.concatenate(self.minfde_values)
        p95 = float(np.quantile(minfde, 0.95))
        return {
            "agents": self.count,
            **{metric: self.sums[metric] / self.count for metric in CORE_METRICS},
            "ece": ece,
            "minfde_p95": p95,
            "tail_minfde": self.tail_sum / max(self.tail_count, 1),
            "tail_samples": self.tail_count,
            "date_metrics": {
                date: {
                    "actors": self.date_counts[date],
                    **{
                        metric: self.date_sums[date][metric] / self.date_counts[date]
                        for metric in ("nll", "brier", "energy_score")
                    },
                }
                for date in sorted(self.date_counts)
            },
        }


def _checkpoint_paths(protocol, fold: int) -> tuple[Path, Path]:
    spec = protocol.fold_inputs(fold)
    return ROOT / str(spec["baseline_path"]), ROOT / str(spec["candidate_path"])


def _load_models(protocol, fold: int, device: torch.device):
    baseline_path, candidate_path = _checkpoint_paths(protocol, fold)
    baseline = build_c127_model("B0_signed_coupled", batch_size=2048).to(device)
    baseline_checkpoint = torch.load(
        baseline_path, map_location=device, weights_only=False
    )
    baseline.load_state_dict(baseline_checkpoint["model_state_dict"])
    candidate = build_energy_model(batch_size=2048).to(device)
    candidate_checkpoint = torch.load(
        candidate_path, map_location=device, weights_only=False
    )
    expected_protocol = sha256(ROOT / "experiments/ascent_recomparison/protocol.json")
    if candidate_checkpoint.get("protocol_sha256") != expected_protocol:
        raise RuntimeError("C161 candidate checkpoint protocol hash mismatch")
    candidate.load_state_dict(candidate_checkpoint["model_state_dict"])
    return baseline.eval(), candidate.eval(), baseline_path, candidate_path


def _merge_candidate_metrics(
    native: dict[str, torch.Tensor], probability: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return {
        **{key: native[key] for key in native if key not in {"energy_score", "nll", "brier", "confidence"}},
        **probability,
    }


def _assert_geometry_equal(
    native: dict[str, torch.Tensor], other: dict[str, torch.Tensor]
) -> None:
    for metric in GEOMETRY_METRICS:
        if not torch.equal(native[metric], other[metric]):
            raise RuntimeError(f"C162 probability operator changed geometry metric {metric}")


@torch.inference_mode()
def run(
    *,
    fold: int,
    device: torch.device,
    workers: int,
    batch_size: int,
    smoke: bool,
    max_validation_scenes: int | None,
) -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_boundaries()
    if fold not in [int(value) for value in protocol.payload["evaluation"]["folds"]]:
        raise RuntimeError("C162 fold is outside frozen evaluation folds")
    if workers < 0 or batch_size < 1:
        raise ValueError("invalid loader configuration")
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    dataset = __import__("experiments.ascent_recomparison.common", fromlist=["load_dataset"]).load_dataset(protocol)
    _, validation_data, validation_dates = __import__(
        "experiments.ascent_recomparison.common", fromlist=["fold_subsets"]
    ).fold_subsets(protocol, dataset, fold, max_validation_scenes=max_validation_scenes)
    loader = __import__(
        "experiments.ascent_recomparison.common", fromlist=["loader"]
    ).loader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        workers=workers,
        prefetch=int(protocol.payload["evaluation"]["prefetch_factor"]),
    )
    baseline, candidate, baseline_path, candidate_path = _load_models(protocol, fold, device)
    states = {arm: StreamingAccumulator() for arm in ARMS}
    scene_cursor = 0
    support_cost_sum = 0.0
    support_cost_count = 0
    identity_cost_sum = 0.0
    hard_cost_sum = 0.0
    expected_cost_sum = 0.0
    assignment_entropy_sum = 0.0
    hard_winner_transfer = 0
    soft_winner_transfer = 0.0
    winner_count = 0
    solver_residual_sum = 0.0
    solver_objective_gain_sum = 0.0
    solver_min_probability = float("inf")
    physical_violations = 0
    exact_geometry_checks = 0

    for data in loader:
        data = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in data.items()
        }
        target = data["pred_traj"].transpose(1, 0)
        baseline_predictions, baseline_logits, baseline_aux = baseline(data)
        baseline_probabilities = baseline_logits.softmax(dim=1)
        baseline_decision = baseline_logits.argmax(dim=1)
        baseline_metrics = compute_batch_metrics(
            baseline_predictions, baseline_probabilities, baseline_decision, target
        )
        candidate_predictions, candidate_native_probabilities, candidate_decision, auxiliary = candidate(data)
        candidate_native_metrics = compute_batch_metrics(
            candidate_predictions,
            candidate_native_probabilities,
            candidate_decision,
            target,
        )
        pairwise = candidate_pairwise_distance(candidate_predictions)
        support_cost = cross_support_cost(baseline_predictions, candidate_predictions)
        transport = transported_prior(baseline_logits.double().softmax(dim=1), support_cost)
        predicted_distance = (
            auxiliary["centered_predicted_normalized_ade_risk"].double() * ADE_SCALE
        )
        tpmo, solver = tpmo_probabilities(
            transport["transported"], predicted_distance, pairwise
        )
        candidate_probability_arms = {
            "identity_prior": transport["identity"],
            "hard_transport": transport["hard"],
            "soft_transport": transport["transported"],
            "tpmo": tpmo,
        }
        for arm, probabilities in candidate_probability_arms.items():
            probability_metrics = _probability_metrics(
                candidate_predictions,
                probabilities,
                candidate_decision,
                target,
                candidate_native_metrics["oracle_ade_mode"],
                pairwise,
                torch.linalg.vector_norm(
                    candidate_predictions - target[:, None], dim=-1
                ).mean(dim=-1),
            )
            metrics = _merge_candidate_metrics(candidate_native_metrics, probability_metrics)
            _assert_geometry_equal(candidate_native_metrics, metrics)
        packed = pack_scenes(data["adj"])
        dates = validation_dates[scene_cursor : scene_cursor + packed.scene_count]
        if len(dates) != packed.scene_count:
            raise RuntimeError("C162 scene-date alignment failed")
        inverse = packed.inverse.detach().cpu().numpy()
        actor_dates = np.asarray(dates, dtype=object)[inverse]
        endpoint_travel = torch.linalg.vector_norm(
            target[:, -1] - data["obs_traj"][-1], dim=-1
        )
        tail = endpoint_travel >= target_tail_threshold(dataset)
        # Keep probability-arm updates aligned to packed actor dates and tail strata.
        for arm, probabilities in candidate_probability_arms.items():
            probability_metrics = _probability_metrics(
                candidate_predictions,
                probabilities,
                candidate_decision,
                target,
                candidate_native_metrics["oracle_ade_mode"],
                pairwise,
                torch.linalg.vector_norm(
                    candidate_predictions - target[:, None], dim=-1
                ).mean(dim=-1),
            )
            metrics = _merge_candidate_metrics(candidate_native_metrics, probability_metrics)
            states[arm].update(metrics, actor_dates, tail)
        states["baseline_native"].update(baseline_metrics, actor_dates, tail)
        states["c161_native"].update(candidate_native_metrics, actor_dates, tail)
        support_cost_sum += float(support_cost.sum().cpu())
        support_cost_count += int(support_cost.numel())
        identity_cost_sum += float(transport["identity_cost"].sum().cpu())
        hard_cost_sum += float(transport["hard_cost"].sum().cpu())
        expected_cost_sum += float(transport["expected_cost"].sum().cpu())
        assignment_entropy_sum += float(transport["assignment_entropy"].sum().cpu())
        baseline_oracle = baseline_metrics["oracle_ade_mode"]
        candidate_oracle = candidate_native_metrics["oracle_ade_mode"]
        hard_destination = transport["hard_permutation"].gather(
            1, baseline_oracle[:, None]
        ).squeeze(1)
        hard_winner_transfer += int((hard_destination == candidate_oracle).sum().cpu())
        permutation = transport["assignment_weights"].shape[1]
        all_perm = __import__("experiments.tpmo_ascent.operator", fromlist=["all_permutations"]).all_permutations(device=device)
        mapped_destination = all_perm[None, :, :].expand(
            baseline_oracle.shape[0], -1, -1
        ).gather(2, baseline_oracle[:, None, None].expand(-1, permutation, 1)).squeeze(-1)
        soft_winner_transfer += float(
            transport["assignment_weights"]
            .mul(mapped_destination.eq(candidate_oracle[:, None]))
            .sum()
            .cpu()
        )
        winner_count += int(baseline_oracle.numel())
        solver_residual_sum += float(solver["kkt_residual"].sum().cpu())
        solver_objective_gain_sum += float(solver["objective_gain"].sum().cpu())
        solver_min_probability = min(
            solver_min_probability, float(solver["minimum_probability"].min().cpu())
        )
        physical_violations += int((auxiliary["horizontal_control"] < 0).sum().cpu())
        exact_geometry_checks += 1
        scene_cursor += packed.scene_count

    if scene_cursor != len(validation_dates):
        raise RuntimeError("C162 did not consume exactly the validation scenes")
    summaries = {arm: states[arm].summary() for arm in ARMS}
    reference = json.loads(
        (ROOT / "artifacts/experiments/ascent_recomparison/final_comparison.json").read_text(
            encoding="utf-8"
        )
    )["fold_results"][str(fold)]
    replay_deltas = {}
    for arm, reference_name in (("baseline_native", "baseline"), ("c161_native", "candidate")):
        replay_deltas[arm] = {
            metric: summaries[arm][metric] - reference[reference_name]["overall"][metric]
            for metric in ("top1_ade", "top1_fde", "minade", "minfde", "energy_score", "nll", "brier")
        }
    result = {
        "format_version": 1,
        "cycle": "C162_TRANSPORTED_PRIOR_MEASURE_OPTIMIZATION",
        "protocol_sha256": sha256(protocol.path),
        "fold": fold,
        "seed": int(protocol.payload["evaluation"]["seed"]),
        "smoke": smoke,
        "validation_scenes": len(validation_dates),
        "arms": summaries,
        "replay_deltas_vs_C161_final": replay_deltas,
        "support_diagnostics": {
            "mean_cross_support_cost": support_cost_sum / support_cost_count,
            "mean_identity_cost": identity_cost_sum / winner_count,
            "mean_hard_assignment_cost": hard_cost_sum / winner_count,
            "mean_expected_assignment_cost": expected_cost_sum / winner_count,
            "mean_assignment_entropy": assignment_entropy_sum / winner_count,
            "hard_winner_transfer_rate": hard_winner_transfer / winner_count,
            "soft_winner_transfer_rate": soft_winner_transfer / winner_count,
        },
        "solver_diagnostics": {
            "mean_kkt_residual": solver_residual_sum / winner_count,
            "mean_objective_gain": solver_objective_gain_sum / winner_count,
            "minimum_probability": solver_min_probability,
            "max_iterations": int(protocol.payload["algorithm"]["solver_max_iterations"]),
            "backtracking_steps": int(protocol.payload["algorithm"]["solver_backtracking_steps"]),
        },
        "integrity": {
            "target_in_probability_forward": False,
            "physical_violation_count": physical_violations,
            "geometry_batch_identity_checks": exact_geometry_checks,
            "development_used": False,
            "locked_test_used": False,
            "trainable_parameters": 0,
        },
        "inputs": {
            "baseline_checkpoint": str(baseline_path.relative_to(ROOT)),
            "baseline_checkpoint_sha256": sha256(baseline_path),
            "candidate_checkpoint": str(candidate_path.relative_to(ROOT)),
            "candidate_checkpoint_sha256": sha256(candidate_path),
        },
        "runtime": {
            "python": __import__("sys").version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "elapsed_seconds": time.time() - started,
            "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(device),
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=(1, 2))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-validation-scenes", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.smoke and args.max_validation_scenes is None:
        args.max_validation_scenes = 8
    result = run(
        fold=args.fold,
        device=torch.device(args.device),
        workers=args.workers,
        batch_size=args.batch_size,
        smoke=args.smoke,
        max_validation_scenes=args.max_validation_scenes,
    )
    output = args.output
    if output is None:
        suffix = "smoke" if args.smoke else "formal"
        output = ROOT / "artifacts/tpmo_ascent" / f"fold{args.fold}_{suffix}.json"
    atomic_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
