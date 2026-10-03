"""Aggregate registered E1 and external E11 result artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    "trajair_7days1",
    "trajair_7days2",
    "trajair_7days3",
    "trajair_7days4",
    "kagc_external_128",
    "kbtp_external_128",
)


def _gain(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-calibration", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/mabpt/e1_external_summary_v1.json")
    args = parser.parse_args()
    rows = {}
    total = 0
    sums = {arm: {} for arm in ("ascent_native", "mabpt")}
    fixed_sums = {arm: {} for arm in ("ascent", "mabpt")}
    metrics = ("top1_ade", "top1_fde", "minade", "minfde", "energy_score", "nll", "brier", "ece")
    for dataset in DATASETS:
        path = ROOT / f"artifacts/mabpt/e1_{dataset}_formal_v1.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        count = int(result["arms"]["mabpt"]["agents"])
        total += count
        row = {"actors": count, "gains": {}}
        for metric in metrics:
            reference = float(result["arms"]["ascent_native"][metric])
            candidate = float(result["arms"]["mabpt"][metric])
            row["gains"][metric] = _gain(reference, candidate)
            for arm, value in (("ascent_native", reference), ("mabpt", candidate)):
                sums[arm][metric] = sums[arm].get(metric, 0.0) + count * value
        if args.include_calibration:
            calibration_path = ROOT / f"artifacts/mabpt/e11_{dataset}_formal_v1.json"
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            row["fixed_event_gains"] = calibration["results"]["relative_gain_mabpt_vs_ascent"]
            for metric in (
                "event_nll",
                "event_brier",
                "event_ece",
                "mixture_nll",
                "energy_score",
                "effective_modes",
                "effective_events",
                "hard_support_zero_rate",
            ):
                for arm in ("ascent", "mabpt"):
                    value = float(calibration["results"][arm][metric])
                    fixed_sums[arm][metric] = fixed_sums[arm].get(metric, 0.0) + count * value
        rows[dataset] = row
    aggregate = {
        arm: {metric: value / total for metric, value in values.items()}
        for arm, values in sums.items()
    }
    aggregate["relative_gain_mabpt_vs_ascent"] = {
        metric: _gain(aggregate["ascent_native"][metric], aggregate["mabpt"][metric])
        for metric in metrics
    }
    payload = {
        "format_version": 1,
        "model": "MABPT",
        "experiment_ids": ["E1", "E11_external"] if args.include_calibration else ["E1"],
        "actors": total,
        "datasets": rows,
        "actor_weighted_aggregate": aggregate,
        "fresh_confirmatory_test": False,
    }
    if args.include_calibration:
        fixed_aggregate = {
            arm: {metric: value / total for metric, value in values.items()}
            for arm, values in fixed_sums.items()
        }
        fixed_aggregate["relative_gain_mabpt_vs_ascent"] = {
            metric: _gain(
                fixed_aggregate["ascent"][metric],
                fixed_aggregate["mabpt"][metric],
            )
            for metric in fixed_aggregate["ascent"]
        }
        payload["actor_weighted_fixed_event_aggregate"] = fixed_aggregate
    if args.output.exists():
        raise FileExistsError(f"MABPT refuses to overwrite {args.output}")
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
