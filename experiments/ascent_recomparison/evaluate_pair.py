"""Reload ASCENT and C161 checkpoints for paired calendar-date evaluation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.edfa_ascent.relation import pack_scenes
from experiments.metric_exact.evaluation import target_tail_threshold
from experiments.metric_exact.model import build_model as build_c127_model
from experiments.energy_predict_optimize.evaluation import (
    RankingMetricAccumulator,
    compute_batch_metrics,
)
from experiments.energy_predict_optimize.model import (
    VARIANT as ENERGY_VARIANT,
    build_model as build_energy_model,
)

from .common import atomic_json, fold_subsets, load_dataset, loader
from .protocol import load_protocol, sha256


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/ascent_recomparison"
RUN_ROOT = ROOT / "runs/ascent_recomparison"
METRICS = (
    "top1_ade",
    "top1_fde",
    "minade",
    "minfde",
    "energy_score",
    "nll",
    "brier",
)


class DateAccumulator:
    def __init__(self) -> None:
        self.counts: dict[str, int] = defaultdict(int)
        self.sums: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

    def update(
        self,
        scene_dates: list[str],
        scene_inverse: np.ndarray,
        metrics: dict[str, np.ndarray],
    ) -> None:
        actor_dates = np.asarray(scene_dates, dtype=object)[scene_inverse]
        for date in sorted(set(scene_dates)):
            mask = actor_dates == date
            count = int(mask.sum())
            self.counts[date] += count
            for metric in METRICS:
                self.sums[date][metric] += float(metrics[metric][mask].sum())

    def summarize(self) -> dict[str, dict[str, float]]:
        return {
            date: {
                "actors": self.counts[date],
                **{
                    metric: total / self.counts[date]
                    for metric, total in values.items()
                },
            }
            for date, values in sorted(self.sums.items())
        }


def paired_date_bootstrap(
    control: dict[str, dict[str, float]],
    candidate: dict[str, dict[str, float]],
    metric: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    dates = sorted(control)
    if dates != sorted(candidate):
        raise RuntimeError("C161 paired date sets differ")
    control_values = np.asarray([control[date][metric] for date in dates])
    candidate_values = np.asarray([candidate[date][metric] for date in dates])
    weights = np.asarray([control[date]["actors"] for date in dates], dtype=np.float64)
    candidate_weights = np.asarray(
        [candidate[date]["actors"] for date in dates], dtype=np.float64
    )
    if not np.array_equal(weights, candidate_weights):
        raise RuntimeError("C161 paired date actor counts differ")
    difference = control_values - candidate_values
    point = float(np.average(difference, weights=weights))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(dates), size=(replicates, len(dates)))
    sampled_weights = weights[draws]
    gains = (difference[draws] * sampled_weights).sum(axis=1) / sampled_weights.sum(
        axis=1
    )
    return {
        "metric": metric,
        "unit": "calendar_date",
        "dates": len(dates),
        "actors": int(weights.sum()),
        "replicates": replicates,
        "point_absolute_gain": point,
        "ci95": [float(value) for value in np.quantile(gains, [0.025, 0.975])],
    }


def _checkpoint_paths(protocol, fold: int, smoke: bool) -> tuple[Path, Path]:
    baseline = ROOT / str(
        protocol.payload["replication"]["baseline_checkpoints"][str(fold)]
    )
    suffix = "smoke" if smoke else "formal"
    candidate = RUN_ROOT / (
        f"{ENERGY_VARIANT}_fold{fold}_seed42_{suffix}/last.pt"
    )
    return baseline, candidate


def _load_models(protocol, fold: int, device: torch.device, smoke: bool):
    baseline_path, candidate_path = _checkpoint_paths(protocol, fold, smoke)
    if not baseline_path.is_file() or not candidate_path.is_file():
        raise FileNotFoundError(
            f"missing paired checkpoints: {baseline_path}, {candidate_path}"
        )
    baseline = build_c127_model("B0_signed_coupled", batch_size=2048).to(device)
    baseline_checkpoint = torch.load(
        baseline_path, map_location=device, weights_only=False
    )
    baseline.load_state_dict(baseline_checkpoint["model_state_dict"])
    candidate = build_energy_model(batch_size=2048).to(device)
    candidate_checkpoint = torch.load(
        candidate_path, map_location=device, weights_only=False
    )
    if candidate_checkpoint.get("protocol_sha256") != sha256(protocol.path):
        raise RuntimeError("C161 candidate checkpoint protocol mismatch")
    candidate.load_state_dict(candidate_checkpoint["model_state_dict"])
    return baseline.eval(), candidate.eval(), baseline_path, candidate_path


def _empty_state() -> dict[str, object]:
    return {
        "metrics": RankingMetricAccumulator(),
        "dates": DateAccumulator(),
        "tail_sum": 0.0,
        "tail_count": 0,
        "negative_controls": 0,
        "control_count": 0,
    }


def _update_state(
    state: dict[str, object],
    batch_metrics: dict[str, torch.Tensor],
    dates: list[str],
    inverse: np.ndarray,
    tail: torch.Tensor,
    controls: torch.Tensor,
) -> None:
    state["metrics"].update(batch_metrics)
    arrays = {
        name: value.detach().cpu().numpy() for name, value in batch_metrics.items()
    }
    state["dates"].update(dates, inverse, arrays)
    state["tail_sum"] += float(batch_metrics["minfde"][tail].sum().cpu())
    state["tail_count"] += int(tail.sum().cpu())
    state["negative_controls"] += int((controls < 0).sum().cpu())
    state["control_count"] += int(controls.numel())


def _summarize_state(state: dict[str, object]) -> dict[str, object]:
    overall = state["metrics"].summarize()
    overall["tail_samples"] = state["tail_count"]
    overall["tail_minfde"] = state["tail_sum"] / max(state["tail_count"], 1)
    overall["negative_horizontal_control_rate"] = state[
        "negative_controls"
    ] / max(state["control_count"], 1)
    return {"overall": overall, "date_metrics": state["dates"].summarize()}


@torch.no_grad()
def run(
    *,
    folds: list[int],
    device: torch.device,
    workers: int,
    eval_batch_size: int,
    smoke: bool,
    max_validation_scenes: int | None,
) -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_boundaries()
    allowed_folds = [int(value) for value in protocol.payload["replication"]["folds"]]
    if any(fold not in allowed_folds for fold in folds):
        raise RuntimeError("C161 evaluation fold is outside the frozen set")
    dataset = load_dataset(protocol)
    tail_threshold = target_tail_threshold(dataset)
    combined = {"baseline": _empty_state(), "candidate": _empty_state()}
    fold_results: dict[str, object] = {}
    checkpoint_receipts: dict[str, object] = {}
    torch.cuda.set_device(device)

    for fold in folds:
        _, validation_data, validation_dates = fold_subsets(
            protocol,
            dataset,
            fold,
            max_validation_scenes=max_validation_scenes,
        )
        validation_loader = loader(
            validation_data,
            batch_size=eval_batch_size,
            shuffle=False,
            workers=workers,
            prefetch=4,
        )
        baseline, candidate, baseline_path, candidate_path = _load_models(
            protocol, fold, device, smoke
        )
        fold_state = {"baseline": _empty_state(), "candidate": _empty_state()}
        scene_cursor = 0
        for data in validation_loader:
            data = {
                key: value.to(device, non_blocking=True)
                if torch.is_tensor(value)
                else value
                for key, value in data.items()
            }
            target = data["pred_traj"].transpose(1, 0)
            baseline_predictions, baseline_logits, baseline_aux = baseline(data)
            baseline_probabilities = baseline_logits.softmax(dim=1)
            baseline_decision = baseline_logits.argmax(dim=1)
            baseline_metrics = compute_batch_metrics(
                baseline_predictions,
                baseline_probabilities,
                baseline_decision,
                target,
            )
            candidate_predictions, candidate_probabilities, candidate_decision, candidate_aux = (
                candidate(data)
            )
            candidate_metrics = compute_batch_metrics(
                candidate_predictions,
                candidate_probabilities,
                candidate_decision,
                target,
            )
            packed = pack_scenes(data["adj"])
            dates = validation_dates[
                scene_cursor : scene_cursor + packed.scene_count
            ]
            if len(dates) != packed.scene_count:
                raise RuntimeError("C161 scene-date alignment failed")
            scene_cursor += packed.scene_count
            inverse = packed.inverse.detach().cpu().numpy()
            endpoint_travel = torch.linalg.vector_norm(
                target[:, -1] - data["obs_traj"][-1], dim=-1
            )
            tail = endpoint_travel >= tail_threshold
            for label, metrics, auxiliary in (
                ("baseline", baseline_metrics, baseline_aux),
                ("candidate", candidate_metrics, candidate_aux),
            ):
                for state in (fold_state[label], combined[label]):
                    _update_state(
                        state,
                        metrics,
                        dates,
                        inverse,
                        tail,
                        auxiliary["horizontal_control"],
                    )
        if scene_cursor != len(validation_dates):
            raise RuntimeError("C161 did not consume all validation scene dates")
        fold_summary = {
            label: _summarize_state(state) for label, state in fold_state.items()
        }
        fold_summary["relative_gains"] = {
            metric: (
                fold_summary["baseline"]["overall"][metric]
                - fold_summary["candidate"]["overall"][metric]
            )
            / fold_summary["baseline"]["overall"][metric]
            for metric in METRICS
        }
        fold_results[str(fold)] = fold_summary
        checkpoint_receipts[str(fold)] = {
            "baseline": {
                "path": baseline_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(baseline_path),
            },
            "candidate": {
                "path": candidate_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(candidate_path),
            },
        }
        del baseline, candidate
        torch.cuda.empty_cache()

    aggregate = {
        label: _summarize_state(state) for label, state in combined.items()
    }
    relative_gains = {
        metric: (
            aggregate["baseline"]["overall"][metric]
            - aggregate["candidate"]["overall"][metric]
        )
        / aggregate["baseline"]["overall"][metric]
        for metric in METRICS
    }
    analysis = protocol.payload["analysis"]
    bootstrap = {
        metric: paired_date_bootstrap(
            aggregate["baseline"]["date_metrics"],
            aggregate["candidate"]["date_metrics"],
            metric,
            replicates=int(analysis["bootstrap_replicates"]),
            seed=int(analysis["bootstrap_seed"]) + index,
        )
        for index, metric in enumerate(("top1_ade", "top1_fde"))
    }
    practical = float(analysis["minimum_practical_relative_gain"])
    gates = {
        "aggregate_top1_ade_relative_gain_at_least_3pct": relative_gains[
            "top1_ade"
        ]
        >= practical,
        "aggregate_top1_fde_relative_gain_at_least_3pct": relative_gains[
            "top1_fde"
        ]
        >= practical,
        "both_top1_metrics_improve_in_each_replication_fold": all(
            fold_results[str(fold)]["relative_gains"]["top1_ade"] > 0
            and fold_results[str(fold)]["relative_gains"]["top1_fde"] > 0
            for fold in folds
        ),
        "paired_date_bootstrap_ci_lower_positive_for_both_top1_metrics": all(
            bootstrap[metric]["ci95"][0] > 0
            for metric in ("top1_ade", "top1_fde")
        ),
        "aggregate_minade_not_worse": relative_gains["minade"] >= 0,
        "aggregate_minfde_not_worse": relative_gains["minfde"] >= 0,
        "aggregate_energy_not_worse": relative_gains["energy_score"] >= 0,
    }
    if not all(
        math.isfinite(float(value))
        for value in (*relative_gains.values(),)
    ):
        raise RuntimeError("C161 comparison produced non-finite gains")
    passed = all(gates.values())
    result = {
        "format_version": 1,
        "cycle": protocol.payload["cycle"],
        "protocol_sha256": sha256(protocol.path),
        "evaluation_kind": "smoke" if smoke else "formal_cross_fold_recomparison",
        "folds": folds,
        "seed": int(protocol.payload["replication"]["seed"]),
        "tail_threshold": tail_threshold,
        "fold_results": fold_results,
        "aggregate": aggregate,
        "relative_gains": relative_gains,
        "paired_date_bootstrap": bootstrap,
        "gates": gates,
        "passed": passed,
        "decision": "C161_ASCENT_IMPROVEMENT_REPLICATED" if passed else "C161_REPLICATION_FAILED",
        "checkpoint_receipts": checkpoint_receipts,
        "calibration_is_diagnostic_not_gating": True,
        "development_used": False,
        "locked_test_used": False,
        "claim_boundary": protocol.payload["claim_boundary"],
    }
    output = ARTIFACT_ROOT / (
        "smoke_comparison.json" if smoke else "final_comparison.json"
    )
    atomic_json(output, result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-validation-scenes", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        folds=args.folds,
        device=torch.device(args.device),
        workers=args.num_workers,
        eval_batch_size=args.eval_batch_size,
        smoke=args.smoke,
        max_validation_scenes=args.max_validation_scenes,
    )
