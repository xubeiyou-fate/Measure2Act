"""Aggregate frozen C162 folds and run the pre-registered gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .protocol import atomic_json, load_protocol, sha256


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("top1_ade", "top1_fde", "minade", "minfde", "energy_score", "nll", "brier")


def _weighted_average(values: list[float], weights: list[int]) -> float:
    return float(np.average(np.asarray(values, dtype=np.float64), weights=weights))


def _aggregate_arm(folds: dict[int, dict[str, object]], arm: str) -> dict[str, object]:
    summaries = [folds[fold]["arms"][arm] for fold in sorted(folds)]
    weights = [int(summary["agents"]) for summary in summaries]
    result: dict[str, object] = {
        "agents": int(sum(weights)),
        **{
            metric: _weighted_average([float(summary[metric]) for summary in summaries], weights)
            for metric in ("top1_ade", "top1_fde", "minade", "minfde", "energy_score", "nll", "brier", "ece", "tail_minfde")
        },
        "minfde_p95": _weighted_average([float(summary["minfde_p95"]) for summary in summaries], weights),
        "tail_samples": int(sum(int(summary["tail_samples"]) for summary in summaries)),
    }
    date_metrics: dict[str, dict[str, float]] = {}
    for summary in summaries:
        for date, metrics in summary["date_metrics"].items():
            if date in date_metrics:
                if int(date_metrics[date]["actors"]) != int(metrics["actors"]):
                    raise RuntimeError(f"date actor count mismatch for {date}")
                raise RuntimeError(f"validation date appears in both C162 folds: {date}")
            date_metrics[date] = metrics
    result["date_metrics"] = {date: date_metrics[date] for date in sorted(date_metrics)}
    return result


def _paired_date_bootstrap(
    control: dict[str, dict[str, float]],
    candidate: dict[str, dict[str, float]],
    metric: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    dates = sorted(control)
    if dates != sorted(candidate):
        raise RuntimeError("C162 paired date sets differ")
    control_values = np.asarray([control[date][metric] for date in dates], dtype=np.float64)
    candidate_values = np.asarray([candidate[date][metric] for date in dates], dtype=np.float64)
    weights = np.asarray([control[date]["actors"] for date in dates], dtype=np.float64)
    candidate_weights = np.asarray([candidate[date]["actors"] for date in dates], dtype=np.float64)
    if not np.array_equal(weights, candidate_weights):
        raise RuntimeError("C162 paired date actor counts differ")
    difference = control_values - candidate_values
    point = float(np.average(difference, weights=weights))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(dates), size=(replicates, len(dates)))
    sampled_weights = weights[draws]
    gains = (difference[draws] * sampled_weights).sum(axis=1) / sampled_weights.sum(axis=1)
    return {
        "metric": metric,
        "unit": "calendar_date",
        "dates": len(dates),
        "actors": int(weights.sum()),
        "replicates": replicates,
        "point_absolute_gain": point,
        "ci95": [float(value) for value in np.quantile(gains, [0.025, 0.975])],
    }


def aggregate(
    fold_paths: tuple[Path, Path],
    *,
    output: Path | None = None,
) -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_boundaries()
    fold_results = {}
    for path in fold_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("smoke"):
            raise RuntimeError("C162 aggregate refuses smoke artifacts")
        if payload.get("protocol_sha256") != sha256(protocol.path):
            raise RuntimeError(f"protocol hash mismatch in {path}")
        fold = int(payload["fold"])
        if fold in fold_results:
            raise RuntimeError(f"duplicate C162 fold {fold}")
        fold_results[fold] = payload
    expected_folds = {int(value) for value in protocol.payload["evaluation"]["folds"]}
    if set(fold_results) != expected_folds:
        raise RuntimeError(f"C162 aggregate requires folds {sorted(expected_folds)}")
    arms = {
        arm: _aggregate_arm(fold_results, arm)
        for arm in fold_results[min(fold_results)]["arms"]
    }
    tpmo = arms["tpmo"]
    c161 = arms["c161_native"]
    baseline = arms["baseline_native"]
    relative_gains = {
        metric: (float(c161[metric]) - float(tpmo[metric])) / float(c161[metric])
        for metric in METRICS
    }
    fold_relative = {
        str(fold): {
            metric: (
                float(payload["arms"]["c161_native"][metric])
                - float(payload["arms"]["tpmo"][metric])
            )
            / float(payload["arms"]["c161_native"][metric])
            for metric in METRICS
        }
        for fold, payload in sorted(fold_results.items())
    }
    bootstrap = {
        metric: _paired_date_bootstrap(
            c161["date_metrics"],
            tpmo["date_metrics"],
            metric,
            replicates=int(protocol.payload["evaluation"]["bootstrap_replicates"]),
            seed=int(protocol.payload["evaluation"]["bootstrap_seed"]) + index,
        )
        for index, metric in enumerate(("nll", "brier"))
    }
    gates = {
        "tpmo_nll_relative_gain_at_least_1pct": relative_gains["nll"] >= float(protocol.payload["gates"]["tpmo_nll_relative_gain_vs_C161_at_least"]),
        "tpmo_brier_relative_gain_at_least_1pct": relative_gains["brier"] >= float(protocol.payload["gates"]["tpmo_brier_relative_gain_vs_C161_at_least"]),
        "tpmo_nll_improves_in_each_fold": all(values["nll"] > 0 for values in fold_relative.values()),
        "tpmo_brier_improves_in_each_fold": all(values["brier"] > 0 for values in fold_relative.values()),
        "tpmo_nll_not_worse_than_native_B0": float(tpmo["nll"]) <= float(baseline["nll"]),
        "tpmo_brier_not_worse_than_native_B0": float(tpmo["brier"]) <= float(baseline["brier"]),
        "tpmo_energy_not_worse_than_C161_factor": float(tpmo["energy_score"]) <= float(c161["energy_score"]) * float(protocol.payload["gates"]["tpmo_energy_not_worse_than_C161_factor"]),
        "date_bootstrap_ci_lower_positive_for_nll_and_brier": all(
            bootstrap[metric]["ci95"][0] > 0 for metric in ("nll", "brier")
        ),
        "C161_geometry_exactly_unchanged": all(
            all(
                float(payload["arms"]["tpmo"][metric]) == float(payload["arms"]["c161_native"][metric])
                for metric in ("top1_ade", "top1_fde", "minade", "minfde")
            )
            for payload in fold_results.values()
        ),
        "probability_simplex_and_finite": all(
            payload["integrity"]["target_in_probability_forward"] is False
            and int(payload["integrity"]["physical_violation_count"]) == 0
            for payload in fold_results.values()
        ),
    }
    if not all(math.isfinite(value) for value in relative_gains.values()):
        raise RuntimeError("C162 aggregate produced non-finite relative gains")
    passed = bool(all(gates.values()))
    result = {
        "format_version": 1,
        "cycle": "C162_TRANSPORTED_PRIOR_MEASURE_OPTIMIZATION",
        "protocol_sha256": sha256(protocol.path),
        "folds": sorted(fold_results),
        "fold_results": {str(fold): payload for fold, payload in sorted(fold_results.items())},
        "aggregate": arms,
        "relative_gains_tpmo_vs_c161": relative_gains,
        "fold_relative_gains_tpmo_vs_c161": fold_relative,
        "paired_date_bootstrap": bootstrap,
        "gates": gates,
        "passed": passed,
        "decision": "C162_TPMO_PASSES_PRE_REGISTERED_SCREEN" if passed else "C162_TPMO_SCREEN_FAILED",
        "development_used": False,
        "locked_test_used": False,
        "claim_boundary": protocol.payload["claim_boundary"],
    }
    if output is not None:
        atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold1", type=Path, required=True)
    parser.add_argument("--fold2", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = aggregate((args.fold1, args.fold2), output=args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
