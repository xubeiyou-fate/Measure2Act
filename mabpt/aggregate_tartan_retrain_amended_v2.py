"""Amended Tartan aggregate for safety metrics that are not estimable.

The frozen v1 aggregator assumed every locked-test safety metric was numeric.
KAGC has no E12-positive actor pairs, so AUPRC is correctly null.  This module
preserves that null, marks the endpoint not estimable, and leaves the frozen
v1 implementation and all locked-test inputs untouched.
"""

from __future__ import annotations

import json

import numpy as np

from . import aggregate_tartan_retrain as frozen
from .aggregate_partc_seeds import hierarchical_paired_bootstrap


def _nullable_safety_seed_scalars(payloads, model: str) -> dict[str, object]:
    result: dict[str, object] = {}
    for metric in frozen.SAFETY_METRICS:
        values_by_seed = {
            str(seed): payloads[seed]["safety_proxy"]["models"][model]["overall"][metric]
            for seed in sorted(payloads)
        }
        estimable = [float(value) for value in values_by_seed.values() if value is not None]
        result[metric] = {
            "mean": float(np.mean(estimable)) if estimable else None,
            "sample_std": float(np.std(estimable, ddof=1)) if len(estimable) > 1 else None,
            "values_by_seed": values_by_seed,
            "n_estimable_seeds": len(estimable),
            "estimable_all_registered_seeds": len(estimable) == len(values_by_seed),
        }
    return result


def _nullable_safety_aggregate(
    payloads,
    seeds: list[int],
    *,
    replicates: int,
    random_seed: int,
) -> dict[str, object]:
    models = {
        model: _nullable_safety_seed_scalars(payloads, model)
        for model in frozen.MODELS
    }
    control = {
        seed: payloads[seed]["safety_proxy"]["models"]["original_ascent"]["per_date"]
        for seed in seeds
    }
    candidate = {
        seed: payloads[seed]["safety_proxy"]["models"]["mabpt_ascent"]["per_date"]
        for seed in seeds
    }
    effects: dict[str, object] = {}
    for index, metric in enumerate(frozen.SAFETY_METRICS):
        usable_control = {}
        usable_candidate = {}
        missing_seeds = []
        for seed in seeds:
            dates = [
                date
                for date in control[seed]
                if int(control[seed][date]["pairs"]) > 0
                and int(candidate[seed][date]["pairs"]) > 0
                and control[seed][date].get(metric) is not None
                and candidate[seed][date].get(metric) is not None
            ]
            if not dates:
                missing_seeds.append(seed)
                continue
            usable_control[seed] = {date: control[seed][date] for date in dates}
            usable_candidate[seed] = {date: candidate[seed][date] for date in dates}
        if missing_seeds:
            effects[metric] = {
                "status": "not_estimable",
                "reason": "No paired calendar date has a defined metric for every registered seed.",
                "missing_seeds": missing_seeds,
                "absolute_gain": None,
                "relative_gain": None,
                "ci95": None,
                "raw_one_sided_p": None,
            }
            continue
        higher_is_better = metric in frozen.SAFETY_HIGHER_IS_BETTER
        first, second = (
            (usable_candidate, usable_control)
            if higher_is_better
            else (usable_control, usable_candidate)
        )
        effect = hierarchical_paired_bootstrap(
            first,
            second,
            metric,
            replicates=replicates,
            random_seed=random_seed + index,
            count_key="pairs",
        )
        effect["status"] = "estimable"
        if higher_is_better:
            baseline_means = []
            for seed in seeds:
                total_pairs = sum(
                    int(value["pairs"]) for value in usable_control[seed].values()
                )
                baseline_means.append(
                    sum(
                        float(value[metric]) * int(value["pairs"])
                        for value in usable_control[seed].values()
                    )
                    / total_pairs
                )
            effect["effect_definition"] = (
                "MABPT_ASCENT_minus_original_ASCENT; positive favors MABPT-ASCENT"
            )
            effect["relative_gain"] = float(effect["absolute_gain"]) / max(
                abs(float(np.mean(baseline_means))), 1e-12
            )
        effects[metric] = effect
    return {
        "models": models,
        "paired_hierarchical_seed_date_bootstrap": effects,
        "effect_orientation": "positive always favors MABPT-ASCENT",
        "inference_unit": "calendar date with actor-pair weighting",
        "fixed_alert_thresholds_fit_on_training_only": True,
        "amendment": {
            "version": 2,
            "reason": "Preserve null metrics when the locked test has no positive E12 events.",
            "null_is_not_zero": True,
        },
    }


def main() -> None:
    # Reuse the frozen identity checks, geometric aggregation, CLI, and atomic
    # writer; replace only its safety summary in this process.
    frozen._safety_seed_scalars = _nullable_safety_seed_scalars
    frozen._safety_aggregate = _nullable_safety_aggregate
    frozen.main()


if __name__ == "__main__":
    main()
