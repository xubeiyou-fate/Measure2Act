"""Aggregate the paired 2x2x2 MABPT-ASCENT factorial experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate import CORE_METRICS, _atomic_json, _sha256
from .partc_design import PROTOCOL, load_protocol
from .partc_factorial import design_matrix, full_model_arm


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/mabpt_partc_20260811"
FACTOR_LEVELS = {
    "correspondence": ("hard_bijection", "exact_gibbs"),
    "assignment_cost_mass": ("uniform", "source_predicted_mass"),
    "projection": ("transported_prior", "energy_kl"),
}


def _combine(summaries: list[dict[str, object]]) -> dict[str, object]:
    actors = sum(int(summary["actors"]) for summary in summaries)
    result = {
        "actors": actors,
        **{
            metric: sum(
                float(summary[metric]) * int(summary["actors"])
                for summary in summaries
            )
            / actors
            for metric in CORE_METRICS
        },
    }
    dates: dict[str, object] = {}
    for summary in summaries:
        for date, values in summary["date_metrics"].items():
            if date in dates:
                raise RuntimeError(f"date appears in multiple validation folds: {date}")
            dates[date] = values
    result["date_metrics"] = {date: dates[date] for date in sorted(dates)}
    return result


def _combine_physical(summaries: list[dict[str, object]]) -> dict[str, object]:
    actors = sum(int(summary["actors"]) for summary in summaries)
    features = {}
    feature_names = list(summaries[0]["features"])
    for feature in feature_names:
        metrics = list(summaries[0]["features"][feature])
        features[feature] = {
            metric: sum(
                int(summary["actors"])
                * float(summary["features"][feature][metric])
                for summary in summaries
            )
            / actors
            for metric in metrics
        }
    return {"actors": actors, "features": features}


def _paired_date_bootstrap(
    control: dict[str, object],
    candidate: dict[str, object],
    metric: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    dates = sorted(control)
    if dates != sorted(candidate):
        raise RuntimeError("paired factorial date sets differ")
    weights = np.asarray([control[date]["actors"] for date in dates], dtype=np.float64)
    if not np.array_equal(
        weights, np.asarray([candidate[date]["actors"] for date in dates])
    ):
        raise RuntimeError("paired factorial date actor counts differ")
    effect = np.asarray(
        [float(control[date][metric]) - float(candidate[date][metric]) for date in dates],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(dates), size=(replicates, len(dates)))
    estimates = (effect[draws] * weights[draws]).sum(1) / weights[draws].sum(1)
    point = float((effect * weights).sum() / weights.sum())
    return {
        "effect_definition": "original_ASCENT_minus_full_MABPT_ASCENT; positive favors full model",
        "absolute_gain": point,
        "relative_gain": point
        / max(
            abs(
                sum(
                    float(control[date][metric]) * weights[index]
                    for index, date in enumerate(dates)
                )
                / weights.sum()
            ),
            1e-12,
        ),
        "ci95": [float(value) for value in np.quantile(estimates, [0.025, 0.975])],
        "raw_one_sided_p": (int((estimates <= 0).sum()) + 1) / (replicates + 1),
        "dates": dates,
        "replicates": replicates,
    }


def _codes() -> dict[str, dict[str, int]]:
    return {
        factor: {low: -1, high: 1}
        for factor, (low, high) in FACTOR_LEVELS.items()
    }


def factorial_effects(
    arms: dict[str, dict[str, object]],
    metric: str,
    *,
    replicates: int,
    random_seed: int,
) -> dict[str, object]:
    design = design_matrix()
    codes = _codes()
    dates = sorted(next(iter(arms.values()))["date_metrics"])
    for summary in arms.values():
        if sorted(summary["date_metrics"]) != dates:
            raise RuntimeError("factorial arms do not share date blocks")
    terms = {
        "correspondence": ("correspondence",),
        "assignment_cost_mass": ("assignment_cost_mass",),
        "projection": ("projection",),
        "correspondence_x_assignment_cost_mass": (
            "correspondence",
            "assignment_cost_mass",
        ),
        "correspondence_x_projection": ("correspondence", "projection"),
        "assignment_cost_mass_x_projection": (
            "assignment_cost_mass",
            "projection",
        ),
        "correspondence_x_assignment_cost_mass_x_projection": (
            "correspondence",
            "assignment_cost_mass",
            "projection",
        ),
    }
    date_weights = np.asarray(
        [next(iter(arms.values()))["date_metrics"][date]["actors"] for date in dates],
        dtype=np.float64,
    )
    date_effects = {term: np.zeros(len(dates), dtype=np.float64) for term in terms}
    for date_index, date in enumerate(dates):
        for term, factors in terms.items():
            contrast = 0.0
            for row in design:
                sign = np.prod([codes[factor][row[factor]] for factor in factors])
                contrast += sign * float(arms[row["arm"]]["date_metrics"][date][metric])
            date_effects[term][date_index] = contrast / 4.0
    rng = np.random.default_rng(random_seed)
    draws = rng.integers(0, len(dates), size=(replicates, len(dates)))
    results = {}
    for term, effects in date_effects.items():
        samples = (effects[draws] * date_weights[draws]).sum(1) / date_weights[draws].sum(1)
        results[term] = {
            "effect_definition": "mean high level minus mean low level",
            "estimate": float((effects * date_weights).sum() / date_weights.sum()),
            "ci95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
            "two_sided_bootstrap_p": min(
                1.0,
                2
                * min(
                    (int((samples <= 0).sum()) + 1) / (replicates + 1),
                    (int((samples >= 0).sum()) + 1) / (replicates + 1),
                ),
            ),
        }
    return results


def aggregate(payloads: list[dict[str, object]]) -> dict[str, object]:
    if [int(payload["fold"]) for payload in payloads] != [1, 2]:
        raise RuntimeError("ordered folds 1 and 2 are required")
    protocol = load_protocol()
    protocol_hash = _sha256(PROTOCOL)
    for payload in payloads:
        if payload.get("model") != "MABPT-ASCENT":
            raise RuntimeError("unexpected paper-model identity")
        if payload.get("evidence_class") != "retrospective_development_only":
            raise RuntimeError("factorial evidence class mismatch")
        if payload.get("partc_protocol_sha256") != protocol_hash:
            raise RuntimeError("Part C protocol hash mismatch")
        if len(payload.get("factorial_arms", {})) != 8:
            raise RuntimeError("incomplete 2x2x2 factorial payload")
    arm_names = [row["arm"] for row in design_matrix()]
    arms = {
        arm: _combine([payload["factorial_arms"][arm] for payload in payloads])
        for arm in arm_names
    }
    source = _combine([payload["original_ascent"] for payload in payloads])
    full = arms[full_model_arm()]
    physical = {
        model: _combine_physical([payload["physical"][model] for payload in payloads])
        for model in ("original_ascent", "mabpt_ascent")
    }
    replicates = int(protocol["analysis"]["bootstrap_replicates"])
    comparisons = {
        metric: _paired_date_bootstrap(
            source["date_metrics"],
            full["date_metrics"],
            metric,
            replicates=replicates,
            seed=20260900 + index,
        )
        for index, metric in enumerate(CORE_METRICS)
    }
    return {
        "format_version": 1,
        "model": "MABPT-ASCENT",
        "experiment_id": "factorial_fusion",
        "evidence_class": "retrospective_development_only",
        "partc_protocol_sha256": protocol_hash,
        "design": design_matrix(),
        "original_ascent": source,
        "factorial_arms": arms,
        "full_model_arm": full_model_arm(),
        "full_model_vs_original_ascent": comparisons,
        "factorial_effects": {
            metric: factorial_effects(
                arms,
                metric,
                replicates=replicates,
                random_seed=20261000 + index,
            )
            for index, metric in enumerate(CORE_METRICS)
        },
        "physical": physical,
        "integrity": {
            "all_eight_arms_reported": len(arms) == 8,
            "paired_same_support": True,
            "target_in_probability_forward": False,
            "selection_performed": False,
        },
        "claim_boundary": "Retrospective development evidence; not a fresh confirmatory test.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_ROOT / "factorial_physical_summary_v1.json",
    )
    args = parser.parse_args()
    paths = [
        ARTIFACT_ROOT / f"factorial_physical_fold{fold}_formal_v1.json"
        for fold in (1, 2)
    ]
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    result = aggregate(payloads)
    result["inputs"] = [
        {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
        for path in paths
    ]
    _atomic_json(args.output.resolve(), result)
    print(json.dumps({"output": str(args.output), "arms": len(result["factorial_arms"])}, indent=2))


if __name__ == "__main__":
    main()
