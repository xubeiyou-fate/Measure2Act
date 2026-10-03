"""Machine decision for the frozen C7 TailQuery closure run."""

from __future__ import annotations


def relative_improvement(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / baseline


def compact(metrics: dict) -> dict:
    overall = metrics["overall"]
    tail = metrics["tail"]
    result = {
        "overall": {
            "minade": overall["minade"],
            "minfde": overall["minfde"],
            "top1_fde": overall["top1_fde"],
            "energy_score": overall["energy_score"],
            "endpoint_coverage": overall["endpoint_coverage"],
            "effective_modes": overall["winner_distribution"]["effective_modes"],
        },
        "tail": {
            "minade": tail["minade"],
            "minfde": tail["minfde"],
            "minfde_p95": tail["minfde_p95"],
            "endpoint_coverage": tail["endpoint_coverage"],
        },
    }
    if "pattern_prediction" in metrics:
        result["pattern_prediction"] = metrics["pattern_prediction"]
    return result


def summarize_c7_closure(baseline: dict, candidate: dict) -> dict:
    baseline_metrics = baseline["development"]
    candidate_metrics = candidate["development"]
    baseline_overall = baseline_metrics["overall"]
    candidate_overall = candidate_metrics["overall"]
    baseline_tail = baseline_metrics["tail"]
    candidate_tail = candidate_metrics["tail"]
    comparison = {
        "overall_minade_relative_improvement": relative_improvement(
            baseline_overall["minade"], candidate_overall["minade"]
        ),
        "overall_minfde_relative_improvement": relative_improvement(
            baseline_overall["minfde"], candidate_overall["minfde"]
        ),
        "tail_minfde_relative_improvement": relative_improvement(
            baseline_tail["minfde"], candidate_tail["minfde"]
        ),
        "tail_p95_relative_improvement": relative_improvement(
            baseline_tail["minfde_p95"], candidate_tail["minfde_p95"]
        ),
        "energy_score_relative_improvement": relative_improvement(
            baseline_overall["energy_score"], candidate_overall["energy_score"]
        ),
        "effective_mode_ratio": (
            candidate_overall["winner_distribution"]["effective_modes"]
            / baseline_overall["winner_distribution"]["effective_modes"]
        ),
    }
    gates = {
        "overall_minfde_improves_at_least_3pct": (
            comparison["overall_minfde_relative_improvement"] >= 0.03
        ),
        "tail_minfde_improves_at_least_8pct": (
            comparison["tail_minfde_relative_improvement"] >= 0.08
        ),
        "tail_p95_improves_at_least_8pct": (
            comparison["tail_p95_relative_improvement"] >= 0.08
        ),
        "overall_minade_not_worse_than_1pct": (
            comparison["overall_minade_relative_improvement"] >= -0.01
        ),
        "coverage_0_5_not_worse": (
            candidate_overall["endpoint_coverage"]["0.5"]
            >= baseline_overall["endpoint_coverage"]["0.5"]
        ),
        "coverage_1_0_not_worse": (
            candidate_overall["endpoint_coverage"]["1.0"]
            >= baseline_overall["endpoint_coverage"]["1.0"]
        ),
        "energy_score_not_worse": comparison["energy_score_relative_improvement"] >= 0,
        "effective_modes_at_least_95pct": comparison["effective_mode_ratio"] >= 0.95,
    }
    all_passed = all(gates.values())
    return {
        "format_version": 1,
        "cycle": "C7_tail_query_evidence_closure",
        "protocol": (
            "single seed-42 scratch closure on frozen C7 train/dev; effect gates added "
            "after P0 but before the first P1 result; no test/external"
        ),
        "gate_provenance": "borrowed unchanged from the later C8 fixed effect-size gate",
        "runs": {
            "baseline": {
                "best_epoch": baseline["best_epoch"],
                "checkpoint_sha256": baseline["checkpoint_sha256"],
                "parameter_count": baseline["parameter_count"],
                "metrics": compact(baseline_metrics),
            },
            "tail_query": {
                "best_epoch": candidate["best_epoch"],
                "checkpoint_sha256": candidate["checkpoint_sha256"],
                "parameter_count": candidate["parameter_count"],
                "metrics": compact(candidate_metrics),
            },
        },
        "candidate_vs_baseline": comparison,
        "gates": gates,
        "all_gates_passed": all_passed,
        "additional_seeds_authorized": False,
        "paper_algorithm_contribution_authorized": False,
        "fine_tuning_used": False,
        "learned_gate_used": False,
        "confirmation_test_external_used": False,
        "decision": (
            "candidate_signal_requires_independent_preregistration"
            if all_passed else "close_c7_no_algorithm_contribution"
        ),
    }
