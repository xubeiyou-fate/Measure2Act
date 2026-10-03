"""Aggregate frozen C165 folds and apply the pre-registered gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .protocol import atomic_json, load_protocol, sha256


METRICS = ("top1_ade", "top1_fde", "minade", "minfde", "energy_score", "nll", "brier")


def _combine(folds: dict[int, dict[str, object]], arm: str) -> dict[str, object]:
    summaries = [folds[index]["arms"][arm] for index in sorted(folds)]
    weights = np.asarray([int(value["agents"]) for value in summaries], dtype=np.float64)
    result = {
        "agents": int(weights.sum()),
        **{
            metric: float(np.average([float(value[metric]) for value in summaries], weights=weights))
            for metric in (*METRICS, "ece", "tail_minfde", "minfde_p95")
        },
        "tail_samples": int(sum(int(value["tail_samples"]) for value in summaries)),
    }
    dates = {}
    for summary in summaries:
        for date, value in summary["date_metrics"].items():
            if date in dates:
                raise RuntimeError(f"date {date} appears in both C165 folds")
            dates[date] = value
    result["date_metrics"] = {date: dates[date] for date in sorted(dates)}
    return result


def _bootstrap(control, candidate, metric: str, *, replicates: int, seed: int):
    dates = sorted(control)
    if dates != sorted(candidate):
        raise RuntimeError("C165 control and candidate date sets differ")
    control_values = np.asarray([control[date][metric] for date in dates], dtype=np.float64)
    candidate_values = np.asarray([candidate[date][metric] for date in dates], dtype=np.float64)
    weights = np.asarray([control[date]["actors"] for date in dates], dtype=np.float64)
    candidate_weights = np.asarray([candidate[date]["actors"] for date in dates], dtype=np.float64)
    if not np.array_equal(weights, candidate_weights):
        raise RuntimeError("C165 date actor counts differ")
    difference = control_values - candidate_values
    point = float(np.average(difference, weights=weights))
    draws = np.random.default_rng(seed).integers(0, len(dates), size=(replicates, len(dates)))
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


def aggregate(paths: tuple[Path, Path], *, output: Path | None = None) -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_boundaries()
    folds = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("smoke"):
            raise RuntimeError("C165 aggregate refuses smoke artifacts")
        if payload.get("protocol_sha256") != sha256(protocol.path):
            raise RuntimeError(f"C165 protocol hash mismatch in {path}")
        fold = int(payload["fold"])
        if fold in folds:
            raise RuntimeError(f"duplicate C165 fold {fold}")
        folds[fold] = payload
    if set(folds) != {1, 2}:
        raise RuntimeError("C165 aggregate requires formal folds 1 and 2")

    arms = {
        arm: _combine(folds, arm)
        for arm in ("baseline_native", "c161_native", "tpmo", "mabpt")
    }
    baseline, c161, tpmo, mabpt = (arms[name] for name in ("baseline_native", "c161_native", "tpmo", "mabpt"))
    relative = {
        metric: (float(c161[metric]) - float(mabpt[metric])) / float(c161[metric])
        for metric in METRICS
    }
    versus_tpmo = {
        metric: (float(tpmo[metric]) - float(mabpt[metric])) / float(tpmo[metric])
        for metric in METRICS
    }
    versus_b0 = {
        metric: (float(mabpt[metric]) - float(baseline[metric])) / float(baseline[metric])
        for metric in METRICS
    }
    fold_relative = {
        str(fold): {
            metric: (
                float(payload["arms"]["c161_native"][metric])
                - float(payload["arms"]["mabpt"][metric])
            )
            / float(payload["arms"]["c161_native"][metric])
            for metric in METRICS
        }
        for fold, payload in sorted(folds.items())
    }
    bootstrap = {
        metric: _bootstrap(
            c161["date_metrics"],
            mabpt["date_metrics"],
            metric,
            replicates=int(protocol.payload["evaluation"]["bootstrap_replicates"]),
            seed=int(protocol.payload["evaluation"]["bootstrap_seed"]) + index,
        )
        for index, metric in enumerate(("nll", "brier"))
    }
    gates = {
        "mabpt_nll_relative_gain_at_least_1pct": relative["nll"] >= 0.01,
        "mabpt_brier_relative_gain_at_least_1pct": relative["brier"] >= 0.01,
        "mabpt_nll_improves_in_each_fold": all(value["nll"] > 0 for value in fold_relative.values()),
        "mabpt_brier_improves_in_each_fold": all(value["brier"] > 0 for value in fold_relative.values()),
        "mabpt_nll_not_worse_than_C162_TPMO": float(mabpt["nll"]) <= float(tpmo["nll"]),
        "mabpt_brier_not_worse_than_C162_TPMO": float(mabpt["brier"]) <= float(tpmo["brier"]),
        "mabpt_nll_not_worse_than_native_B0": float(mabpt["nll"]) <= float(baseline["nll"]),
        "mabpt_brier_not_worse_than_native_B0": float(mabpt["brier"]) <= float(baseline["brier"]),
        "mabpt_energy_not_worse_than_C161_factor": float(mabpt["energy_score"]) <= float(c161["energy_score"]) * 1.01,
        "date_bootstrap_ci_lower_positive_for_nll_and_brier": all(
            bootstrap[metric]["ci95"][0] > 0 for metric in ("nll", "brier")
        ),
        "C161_geometry_exactly_unchanged": all(
            all(
                float(payload["arms"]["mabpt"][metric])
                == float(payload["arms"]["c161_native"][metric])
                for metric in ("top1_ade", "top1_fde", "minade", "minfde")
            )
            for payload in folds.values()
        ),
        "mass_aware_target_free_integrity": all(
            payload["integrity"]["mass_aware_assignment_cost"] is True
            and payload["integrity"]["target_in_validation_probability_forward"] is False
            and payload["integrity"]["physical_violation_count"] == 0
            and payload["integrity"]["temperature_used"] is False
            and payload["integrity"]["gate_or_residual_used"] is False
            for payload in folds.values()
        ),
    }
    if not all(
        math.isfinite(float(value))
        for values in (relative, versus_tpmo, versus_b0)
        for value in values.values()
    ):
        raise RuntimeError("C165 relative metrics are non-finite")
    passed = bool(all(gates.values()))
    result = {
        "format_version": 1,
        "cycle": "C165_MASS_AWARE_BAYESIAN_PERMUTATION_TRANSPORT",
        "protocol_sha256": sha256(protocol.path),
        "folds": [1, 2],
        "fold_results": {str(fold): payload for fold, payload in sorted(folds.items())},
        "aggregate": arms,
        "relative_gains_mabpt_vs_c161": relative,
        "incremental_gains_mabpt_vs_tpmo": versus_tpmo,
        "relative_gap_mabpt_to_B0": versus_b0,
        "fold_relative_gains_mabpt_vs_c161": fold_relative,
        "paired_date_bootstrap": bootstrap,
        "gates": gates,
        "passed": passed,
        "decision": "C165_MABPT_PASSES_PRE_REGISTERED_SCREEN" if passed else "C165_MABPT_SCREEN_FAILED",
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
    print(json.dumps(aggregate((args.fold1, args.fold2), output=args.output), indent=2))


if __name__ == "__main__":
    main()
