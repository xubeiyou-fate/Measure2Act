"""Summarize frozen C127 P1 results and authorize at most one exact candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .model import VARIANTS
from .protocol import load_protocol, sha256


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/metric_exact"


def result_path(variant: str) -> Path:
    return RUN_ROOT / f"P1_{variant}_fold0_seed42_formal/training_summary.json"


def load_result(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("complete") is not True or payload.get("formal") is not True:
        raise RuntimeError(f"incomplete/non-formal C127 result: {path}")
    if payload.get("locked_test_used") is not False:
        raise RuntimeError(f"locked-test boundary violation: {path}")
    return payload


def relative_gain(control: float, candidate: float) -> float:
    return (control - candidate) / max(abs(control), 1e-12)


def paired_date_bootstrap(
    control: dict[str, object],
    candidate: dict[str, object],
    metric: str,
    *,
    replicates: int = 5000,
    seed: int = 127,
) -> dict[str, object]:
    dates = sorted(set(control) & set(candidate))
    if not dates or set(control) != set(candidate):
        raise RuntimeError("C127 date metrics are incomplete or mismatched")
    control_sum = np.asarray(
        [control[date][metric] * control[date]["actors"] for date in dates],
        dtype=np.float64,
    )
    candidate_sum = np.asarray(
        [candidate[date][metric] * candidate[date]["actors"] for date in dates],
        dtype=np.float64,
    )
    counts = np.asarray([control[date]["actors"] for date in dates], dtype=np.float64)
    rng = np.random.default_rng(seed)
    gains = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected = rng.integers(0, len(dates), len(dates))
        control_mean = control_sum[selected].sum() / counts[selected].sum()
        candidate_mean = candidate_sum[selected].sum() / counts[selected].sum()
        gains[index] = control_mean - candidate_mean
    return {
        "dates": dates,
        "replicates": replicates,
        "point_absolute_gain": float(
            control_sum.sum() / counts.sum() - candidate_sum.sum() / counts.sum()
        ),
        "ci95": [float(value) for value in np.quantile(gains, [0.025, 0.975])],
    }


def compare(control: dict[str, object], candidate: dict[str, object]) -> dict[str, object]:
    control_metrics = control["validation_metrics"]
    candidate_metrics = candidate["validation_metrics"]
    result: dict[str, object] = {}
    for metric in ("minade", "minfde"):
        control_value = float(control_metrics["overall"][metric])
        candidate_value = float(candidate_metrics["overall"][metric])
        result[metric] = {
            "control": control_value,
            "candidate": candidate_value,
            "relative_gain": relative_gain(control_value, candidate_value),
            "date_bootstrap": paired_date_bootstrap(
                control_metrics["date_metrics"],
                candidate_metrics["date_metrics"],
                metric,
            ),
        }
    dates = sorted(control_metrics["date_metrics"])
    result["same_direction_dates"] = sum(
        candidate_metrics["date_metrics"][date]["minade"]
        < control_metrics["date_metrics"][date]["minade"]
        and candidate_metrics["date_metrics"][date]["minfde"]
        < control_metrics["date_metrics"][date]["minfde"]
        for date in dates
    )
    result["date_count"] = len(dates)
    result["both_point_estimates_improve"] = all(
        result[metric]["relative_gain"] > 0 for metric in ("minade", "minfde")
    )
    return result


def build_p1_summary(results: dict[str, dict[str, object]]) -> dict[str, object]:
    b0 = results["B0_signed_coupled"]
    b2 = results["B2_decoupled_original"]
    comparisons_vs_b0 = {
        variant: compare(b0, result)
        for variant, result in results.items()
        if variant != "B0_signed_coupled"
    }
    comparisons_vs_b2 = {
        variant: compare(b2, result)
        for variant, result in results.items()
        if variant != "B2_decoupled_original"
    }
    candidate_gates = {}
    passing = []
    for variant in ("B5_single_combined", "B6_dual_oracle"):
        gates = {
            "both_metrics_better_than_B0": comparisons_vs_b0[variant][
                "both_point_estimates_improve"
            ],
            "both_metrics_better_than_B2": comparisons_vs_b2[variant][
                "both_point_estimates_improve"
            ],
        }
        passed = all(gates.values())
        candidate_gates[variant] = {"gates": gates, "passed": passed}
        if passed:
            passing.append(variant)
    selected = None
    if passing:
        b2_metrics = b2["validation_metrics"]["overall"]

        def minimax(variant: str) -> tuple[float, int]:
            metrics = results[variant]["validation_metrics"]["overall"]
            score = max(
                float(metrics["minade"]) / float(b2_metrics["minade"]),
                float(metrics["minfde"]) / float(b2_metrics["minfde"]),
            )
            return score, 0 if variant == "B6_dual_oracle" else 1

        selected = min(passing, key=minimax)
    metrics = {
        variant: {
            "minade": result["validation_metrics"]["overall"]["minade"],
            "minfde": result["validation_metrics"]["overall"]["minfde"],
            "p95_fde": result["validation_metrics"]["overall"]["minfde_p95"],
            "energy_score": result["validation_metrics"]["overall"]["energy_score"],
            "effective_modes": result["validation_metrics"]["overall"][
                "winner_distribution"
            ]["effective_modes"],
            "parameter_count": result["parameter_count"],
        }
        for variant, result in results.items()
    }
    return {
        "format_version": 1,
        "cycle": "C127_metric_exact_score_isolated_ascent",
        "phase": "P1",
        "fold": 0,
        "seed": 42,
        "metrics": metrics,
        "comparisons_vs_B0": comparisons_vs_b0,
        "comparisons_vs_B2": comparisons_vs_b2,
        "candidate_gates": candidate_gates,
        "P2_selected_exact_candidate": selected,
        "decision": "P2_AUTHORIZED" if selected else "CLOSE_EXACT_CANDIDATES_AFTER_P1",
        "locked_test_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/experiments/metric_exact/p1_summary.json"),
    )
    args = parser.parse_args()
    protocol = load_protocol()
    protocol.assert_boundaries()
    results = {variant: load_result(result_path(variant)) for variant in VARIANTS}
    for result in results.values():
        if result["protocol_sha256"] != sha256(protocol.path):
            raise RuntimeError("C127 P1 result protocol hash mismatch")
    summary = build_p1_summary(results)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
