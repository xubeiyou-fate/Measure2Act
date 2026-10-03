"""Summarize frozen C127 P2 date-fold replication and authorize P3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .protocol import load_protocol, sha256
from .summarize import compare, load_result, paired_date_bootstrap


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/metric_exact"
ARTIFACT_ROOT = ROOT / "artifacts/metric_exact"
CONTROLS = ("B0_signed_coupled", "B2_decoupled_original")


def path_for(variant: str, fold: int) -> Path:
    phase = "P1" if fold == 0 else "P2"
    return RUN_ROOT / (
        f"{phase}_{variant}_fold{fold}_seed42_formal/training_summary.json"
    )


def merge_date_metrics(
    results: dict[int, dict[str, object]], variant: str
) -> dict[str, object]:
    merged: dict[str, object] = {}
    for fold, fold_results in sorted(results.items()):
        dates = fold_results[variant]["validation_metrics"]["date_metrics"]
        overlap = set(merged) & set(dates)
        if overlap:
            raise RuntimeError(f"C127 P2 date appears in multiple folds: {overlap}")
        merged.update(dates)
    return merged


def build_summary(
    fold_results: dict[int, dict[str, object]], candidate: str
) -> dict[str, object]:
    fold_comparisons = {
        str(fold): {
            "candidate_vs_B0": compare(results["B0_signed_coupled"], results[candidate]),
            "candidate_vs_B2": compare(results["B2_decoupled_original"], results[candidate]),
        }
        for fold, results in sorted(fold_results.items())
    }
    same_direction_folds = sum(
        values["candidate_vs_B2"]["both_point_estimates_improve"]
        for values in fold_comparisons.values()
    )
    b2_dates = merge_date_metrics(fold_results, "B2_decoupled_original")
    candidate_dates = merge_date_metrics(fold_results, candidate)
    bootstrap = {
        metric: paired_date_bootstrap(
            b2_dates,
            candidate_dates,
            metric,
            replicates=10000,
            seed=127,
        )
        for metric in ("minade", "minfde")
    }
    replication_results = {
        fold: results for fold, results in fold_results.items() if fold != 0
    }
    replication_b2_dates = merge_date_metrics(
        replication_results, "B2_decoupled_original"
    )
    replication_candidate_dates = merge_date_metrics(replication_results, candidate)
    replication_bootstrap = {
        metric: paired_date_bootstrap(
            replication_b2_dates,
            replication_candidate_dates,
            metric,
            replicates=10000,
            seed=127,
        )
        for metric in ("minade", "minfde")
    }
    replication_diagnostic = {
        "status": "posthoc_reporting_only_not_a_gate",
        "folds": sorted(replication_results),
        "excludes_candidate_selection_fold0": True,
        "same_direction_folds_vs_B2": sum(
            fold_comparisons[str(fold)]["candidate_vs_B2"][
                "both_point_estimates_improve"
            ]
            for fold in replication_results
        ),
        "paired_date_bootstrap_vs_B2": replication_bootstrap,
        "date_count": len(replication_b2_dates),
    }
    gates = {
        "both_metrics_better_than_B2_in_at_least_four_of_five_folds": (
            same_direction_folds >= 4
        ),
        "overall_paired_date_bootstrap_ci_lower_bound_positive_for_both": all(
            bootstrap[metric]["ci95"][0] > 0
            for metric in ("minade", "minfde")
        ),
        "all_74_effective_train_dates_evaluated_once": len(b2_dates) == 74
        and set(b2_dates) == set(candidate_dates),
    }
    passed = all(gates.values())
    return {
        "format_version": 1,
        "cycle": "C127_metric_exact_score_isolated_ascent",
        "phase": "P2",
        "candidate": candidate,
        "fold_comparisons": fold_comparisons,
        "same_direction_folds_vs_B2": same_direction_folds,
        "paired_date_bootstrap_vs_B2": bootstrap,
        "untouched_replication_diagnostic": replication_diagnostic,
        "gates": gates,
        "passed": passed,
        "P3_selected_exact_candidate": candidate if passed else None,
        "decision": "P3_AUTHORIZED" if passed else "CLOSE_C127_AFTER_P2",
        "locked_test_used": False,
    }


def run(output: Path) -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_boundaries()
    p1_path = ARTIFACT_ROOT / "p1_summary.json"
    if not p1_path.is_file():
        raise RuntimeError("C127 P1 summary is required before P2 summary")
    p1 = json.loads(p1_path.read_text(encoding="utf-8"))
    candidate = p1.get("P2_selected_exact_candidate")
    if p1.get("decision") != "P2_AUTHORIZED" or not candidate:
        raise RuntimeError("C127 P1 did not authorize a P2 candidate")
    variants = (*CONTROLS, candidate)
    fold_results = {
        fold: {variant: load_result(path_for(variant, fold)) for variant in variants}
        for fold in range(5)
    }
    protocol_hash = sha256(protocol.path)
    for results in fold_results.values():
        for result in results.values():
            if result["protocol_sha256"] != protocol_hash:
                raise RuntimeError("C127 P2 result protocol hash mismatch")
    summary = build_summary(fold_results, candidate)
    summary["protocol_sha256"] = protocol_hash
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_ROOT / "p2_summary.json",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
