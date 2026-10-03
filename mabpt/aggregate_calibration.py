"""Aggregate E11 fixed-event calibration across registered folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .aggregate import _atomic_json, _paired_date_bootstrap, _sha256
from .calibration import ROOT
from .events import PROTOCOL


METRICS = (
    "event_nll",
    "event_brier",
    "mixture_nll",
    "energy_score",
    "top1_ade",
    "top1_fde",
    "effective_modes",
    "effective_events",
    "duplicate_pair_rate",
    "hard_support_zero_rate",
)


def _combine(summaries: list[dict[str, object]]) -> dict[str, object]:
    actors = sum(int(summary["actors"]) for summary in summaries)
    dates = {}
    for summary in summaries:
        for date, metrics in summary["date_metrics"].items():
            if date in dates:
                raise RuntimeError(f"duplicate E11 date: {date}")
            dates[date] = metrics
    reliability = []
    for bin_index in range(len(summaries[0]["reliability_diagram"])):
        bins = [summary["reliability_diagram"][bin_index] for summary in summaries]
        count = sum(int(item["count"]) for item in bins)
        reliability.append(
            {
                "lower": bins[0]["lower"],
                "upper": bins[0]["upper"],
                "count": count,
                "mean_confidence": (
                    sum((item["mean_confidence"] or 0.0) * item["count"] for item in bins)
                    / count
                    if count
                    else None
                ),
                "accuracy": (
                    sum((item["accuracy"] or 0.0) * item["count"] for item in bins)
                    / count
                    if count
                    else None
                ),
            }
        )
    ece = sum(
        (item["count"] / actors) * abs(item["accuracy"] - item["mean_confidence"])
        for item in reliability
        if item["count"]
    )
    return {
        "actors": actors,
        **{
            metric: sum(float(summary[metric]) * int(summary["actors"]) for summary in summaries)
            / actors
            for metric in METRICS
        },
        "event_ece": ece,
        "reliability_diagram": reliability,
        "event_prevalence": [
            sum(float(summary["event_prevalence"][index]) for summary in summaries)
            for index in range(9)
        ],
        "fold_weighted_brier_decomposition": {
            name: sum(
                float(summary["brier_decomposition"][name]) * int(summary["actors"])
                for summary in summaries
            )
            / actors
            for name in ("reliability", "resolution", "uncertainty", "decomposed_brier")
        },
        "date_metrics": {date: dates[date] for date in sorted(dates)},
    }


def aggregate(paths: tuple[Path, Path]) -> dict[str, object]:
    folds = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if [fold["fold"] for fold in folds] != [1, 2]:
        raise RuntimeError("E11 requires ordered folds 1 and 2")
    protocol_sha = _sha256(PROTOCOL)
    if any(fold["protocol_sha256"] != protocol_sha for fold in folds):
        raise RuntimeError("E11 protocol hash mismatch")
    results = {
        model: _combine([fold["results"][model] for fold in folds])
        for model in ("ascent", "mabpt")
    }
    relative = {
        metric: (results["ascent"][metric] - results["mabpt"][metric])
        / results["ascent"][metric]
        for metric in ("event_nll", "event_brier", "mixture_nll", "energy_score")
    }
    bootstrap = {
        metric: _paired_date_bootstrap(
            results["ascent"]["date_metrics"],
            results["mabpt"]["date_metrics"],
            metric,
            replicates=10000,
            seed=11100 + index,
        )
        for index, metric in enumerate(("event_nll", "event_brier", "mixture_nll", "energy_score"))
    }
    results["relative_gain_mabpt_vs_ascent"] = relative
    return {
        "format_version": 2,
        "model": "MABPT",
        "experiment_id": "E11",
        "evidence_class": "legacy_development_only",
        "protocol_sha256": protocol_sha,
        "inputs": [
            {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)} for path in paths
        ],
        "results": results,
        "paired_date_bootstrap": bootstrap,
        "fold_results": {str(fold["fold"]): fold["results"] for fold in folds},
        "v1_hard_bin_smoke_retained": "artifacts/mabpt/e11_fold1_smoke_v1.json",
        "selection_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold1",
        type=Path,
        default=ROOT / "artifacts/mabpt/e11_fold1_formal_v2.json",
    )
    parser.add_argument(
        "--fold2",
        type=Path,
        default=ROOT / "artifacts/mabpt/e11_fold2_formal_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/mabpt/e11_summary_v2.json",
    )
    args = parser.parse_args()
    result = aggregate((args.fold1.resolve(), args.fold2.resolve()))
    _atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "actors": result["results"]["mabpt"]["actors"],
                "relative_gain": result["results"]["relative_gain_mabpt_vs_ascent"],
                "ece": {
                    model: result["results"][model]["event_ece"]
                    for model in ("ascent", "mabpt")
                },
                "hard_zero_rate": {
                    model: result["results"][model]["hard_support_zero_rate"]
                    for model in ("ascent", "mabpt")
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
