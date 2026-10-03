"""Aggregate registered E12 traffic-decision folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .aggregate import _atomic_json, _paired_date_bootstrap, _sha256
from .traffic import PROTOCOL, ROOT


def _combine(summaries: list[dict[str, object]]) -> dict[str, object]:
    pairs = sum(int(summary["pairs"]) for summary in summaries)
    positives = sum(int(summary["positive_pairs"]) for summary in summaries)
    negatives = pairs - positives
    alerts = sum(int(summary["alerts"]) for summary in summaries)
    true_positives = sum(int(summary["true_positive_alerts"]) for summary in summaries)
    dates = {}
    for summary in summaries:
        for date, metrics in summary["date_metrics"].items():
            if date in dates:
                raise RuntimeError(f"duplicate E12 date: {date}")
            dates[date] = metrics
    lead_denominator = max(true_positives, 1)
    return {
        "pairs": pairs,
        "positive_pairs": positives,
        "prevalence": positives / pairs,
        "brier": sum(summary["brier"] * summary["pairs"] for summary in summaries) / pairs,
        "nll": sum(summary["nll"] * summary["pairs"] for summary in summaries) / pairs,
        "fold_weighted_auprc": sum(summary["auprc"] * summary["pairs"] for summary in summaries) / pairs,
        "fold_weighted_ece": sum(summary["ece"] * summary["pairs"] for summary in summaries) / pairs,
        "recall_at_training_fixed_fpr": true_positives / max(positives, 1),
        "observed_fpr": (alerts - true_positives) / max(negatives, 1),
        "alerts": alerts,
        "true_positive_alerts": true_positives,
        "mean_warning_lead_seconds": sum(
            (summary["mean_warning_lead_seconds"] or 0.0) * summary["true_positive_alerts"]
            for summary in summaries
        )
        / lead_denominator,
        "fold_median_warning_lead_seconds": [
            summary["median_warning_lead_seconds"] for summary in summaries
        ],
        "hard_support_zero_rate_on_positive": sum(
            (summary["hard_support_zero_rate_on_positive"] or 0.0)
            * summary["positive_pairs"]
            for summary in summaries
        )
        / max(positives, 1),
        "fold_alert_thresholds": [summary["alert_threshold"] for summary in summaries],
        "date_metrics": {date: dates[date] for date in sorted(dates)},
    }


def aggregate(paths: tuple[Path, Path]) -> dict[str, object]:
    folds = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if [fold["fold"] for fold in folds] != [1, 2]:
        raise RuntimeError("E12 requires ordered folds 1 and 2")
    protocol_sha = _sha256(PROTOCOL)
    if any(fold["protocol_sha256"] != protocol_sha for fold in folds):
        raise RuntimeError("E12 protocol hash mismatch")
    results = {
        model: _combine([fold["results"][model] for fold in folds])
        for model in ("ascent", "mabpt")
    }
    results["relative_gain_mabpt_vs_ascent"] = {
        "brier": (results["ascent"]["brier"] - results["mabpt"]["brier"])
        / results["ascent"]["brier"],
        "nll": (results["ascent"]["nll"] - results["mabpt"]["nll"])
        / results["ascent"]["nll"],
        "auprc": (
            results["mabpt"]["fold_weighted_auprc"]
            - results["ascent"]["fold_weighted_auprc"]
        )
        / results["ascent"]["fold_weighted_auprc"],
    }
    bootstrap_dates = {
        model: {
            date: {"actors": metrics["pairs"], **metrics}
            for date, metrics in results[model]["date_metrics"].items()
        }
        for model in ("ascent", "mabpt")
    }
    bootstrap = {
        metric: _paired_date_bootstrap(
            bootstrap_dates["ascent"],
            bootstrap_dates["mabpt"],
            metric,
            replicates=10000,
            seed=11200 + index,
        )
        for index, metric in enumerate(("nll", "brier"))
    }
    return {
        "format_version": 1,
        "model": "MABPT",
        "experiment_id": "E12",
        "evidence_class": "legacy_development_only",
        "protocol_sha256": protocol_sha,
        "inputs": [
            {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)} for path in paths
        ],
        "results": results,
        "paired_date_bootstrap": bootstrap,
        "fold_results": {str(fold["fold"]): fold["results"] for fold in folds},
        "regulatory_claim": False,
        "selection_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold1",
        type=Path,
        default=ROOT / "artifacts/mabpt/e12_fold1_formal_v1.json",
    )
    parser.add_argument(
        "--fold2",
        type=Path,
        default=ROOT / "artifacts/mabpt/e12_fold2_formal_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/mabpt/e12_summary_v1.json",
    )
    args = parser.parse_args()
    result = aggregate((args.fold1.resolve(), args.fold2.resolve()))
    _atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "pairs": result["results"]["mabpt"]["pairs"],
                "relative_gain": result["results"]["relative_gain_mabpt_vs_ascent"],
                "recall": {
                    model: result["results"][model]["recall_at_training_fixed_fpr"]
                    for model in ("ascent", "mabpt")
                },
                "observed_fpr": {
                    model: result["results"][model]["observed_fpr"]
                    for model in ("ascent", "mabpt")
                },
                "lead_seconds": {
                    model: result["results"][model]["mean_warning_lead_seconds"]
                    for model in ("ascent", "mabpt")
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
