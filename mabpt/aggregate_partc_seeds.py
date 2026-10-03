"""Aggregate the five matched-seed MABPT-ASCENT development evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate import CORE_METRICS, _atomic_json, _sha256
from .partc_design import PROTOCOL, load_protocol


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/mabpt_partc_20260811"
SCALAR_METRICS = (
    *CORE_METRICS,
    "ece_argmax",
    "effective_modes",
    "minfde_p95",
    "tail_minfde",
)


def _weighted_date_mean(
    values: dict[str, object], metric: str, draws, *, count_key: str
) -> float:
    dates = sorted(values)
    sums = np.asarray(
        [float(values[date][metric]) * int(values[date][count_key]) for date in dates],
        dtype=np.float64,
    )
    counts = np.asarray(
        [int(values[date][count_key]) for date in dates], dtype=np.float64
    )
    return float(sums[draws].sum() / counts[draws].sum())


def hierarchical_paired_bootstrap(
    control: dict[int, dict[str, object]],
    candidate: dict[int, dict[str, object]],
    metric: str,
    *,
    replicates: int = 10000,
    random_seed: int = 20260811,
    count_key: str = "actors",
) -> dict[str, object]:
    seeds = sorted(control)
    if seeds != sorted(candidate) or not seeds:
        raise RuntimeError("paired seed sets are empty or differ")
    point_effects: list[float] = []
    for seed in seeds:
        control_dates = control[seed]
        candidate_dates = candidate[seed]
        if set(control_dates) != set(candidate_dates) or not control_dates:
            raise RuntimeError(f"paired date sets differ for seed {seed}")
        if any(
            int(control_dates[date][count_key])
            != int(candidate_dates[date][count_key])
            for date in control_dates
        ):
            raise RuntimeError(f"paired date actor counts differ for seed {seed}")
        all_dates = np.arange(len(control_dates))
        point_effects.append(
            _weighted_date_mean(control_dates, metric, all_dates, count_key=count_key)
            - _weighted_date_mean(candidate_dates, metric, all_dates, count_key=count_key)
        )
    rng = np.random.default_rng(random_seed)
    samples = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected_seeds = rng.integers(0, len(seeds), len(seeds))
        effects = []
        for position in selected_seeds:
            seed = seeds[int(position)]
            date_count = len(control[seed])
            selected_dates = rng.integers(0, date_count, date_count)
            effects.append(
                _weighted_date_mean(
                    control[seed], metric, selected_dates, count_key=count_key
                )
                - _weighted_date_mean(
                    candidate[seed], metric, selected_dates, count_key=count_key
                )
            )
        samples[replicate] = float(np.mean(effects))
    point = float(np.mean(point_effects))
    control_means = [
        _weighted_date_mean(
            control[seed],
            metric,
            np.arange(len(control[seed])),
            count_key=count_key,
        )
        for seed in seeds
    ]
    directional_p = (int((samples <= 0).sum()) + 1) / (replicates + 1)
    return {
        "effect_definition": "original_ASCENT_minus_MABPT_ASCENT; positive favors MABPT-ASCENT",
        "seeds": seeds,
        "dates_per_seed": {str(seed): len(control[seed]) for seed in seeds},
        "replicates": replicates,
        "absolute_gain": point,
        "relative_gain": point / max(abs(float(np.mean(control_means))), 1e-12),
        "per_seed_absolute_gain": point_effects,
        "ci95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
        "raw_one_sided_p": directional_p,
    }


def paired_seed_bootstrap(
    control: dict[int, float],
    candidate: dict[int, float],
    *,
    higher_is_better: bool,
    replicates: int,
    random_seed: int,
) -> dict[str, object]:
    seeds = sorted(control)
    if seeds != sorted(candidate) or not seeds:
        raise RuntimeError("paired seed sets are empty or differ")
    effects = np.asarray(
        [
            (
                candidate[seed] - control[seed]
                if higher_is_better
                else control[seed] - candidate[seed]
            )
            for seed in seeds
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(random_seed)
    draws = rng.integers(0, len(seeds), size=(replicates, len(seeds)))
    samples = effects[draws].mean(1)
    return {
        "effect_definition": (
            "MABPT_ASCENT_minus_original_ASCENT; positive favors MABPT-ASCENT"
            if higher_is_better
            else "original_ASCENT_minus_MABPT_ASCENT; positive favors MABPT-ASCENT"
        ),
        "seeds": seeds,
        "replicates": replicates,
        "absolute_gain": float(effects.mean()),
        "per_seed_absolute_gain": effects.tolist(),
        "ci95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
        "raw_one_sided_p": (int((samples <= 0).sum()) + 1) / (replicates + 1),
    }


def _aggregate_scalars(
    payloads: dict[int, dict[str, object]], model: str
) -> dict[str, object]:
    result: dict[str, object] = {}
    for metric in SCALAR_METRICS:
        values = np.asarray(
            [payloads[seed]["models"][model][metric] for seed in sorted(payloads)],
            dtype=np.float64,
        )
        result[metric] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "values": values.tolist(),
        }
    return result


def _aggregate_section(
    payloads: dict[int, dict[str, object]],
    section: str,
    model: str,
    metrics: tuple[str, ...],
) -> dict[str, object]:
    result = {}
    for metric in metrics:
        values = np.asarray(
            [payloads[seed][section][model][metric] for seed in sorted(payloads)],
            dtype=np.float64,
        )
        result[metric] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "values": values.tolist(),
        }
    return result


def _combine_reliability(
    payloads: dict[int, dict[str, object]], model: str
) -> list[dict[str, object]]:
    seeds = sorted(payloads)
    diagrams = [
        payloads[seed]["fixed_event_calibration"][model]["reliability_diagram"]
        for seed in seeds
    ]
    if len({len(diagram) for diagram in diagrams}) != 1:
        raise RuntimeError("five-seed reliability bin counts differ")
    result = []
    for index in range(len(diagrams[0])):
        bins = [diagram[index] for diagram in diagrams]
        bounds = {(value["lower"], value["upper"]) for value in bins}
        if len(bounds) != 1:
            raise RuntimeError("five-seed reliability bin boundaries differ")
        count = sum(int(value["count"]) for value in bins)
        lower, upper = next(iter(bounds))
        result.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_confidence": (
                    sum(
                        int(value["count"]) * float(value["mean_confidence"])
                        for value in bins
                        if value["count"]
                    )
                    / count
                    if count
                    else None
                ),
                "accuracy": (
                    sum(
                        int(value["count"]) * float(value["accuracy"])
                        for value in bins
                        if value["count"]
                    )
                    / count
                    if count
                    else None
                ),
            }
        )
    return result


def aggregate(payloads: dict[int, dict[str, object]]) -> dict[str, object]:
    protocol = load_protocol()
    seeds = [int(value) for value in protocol["fixed_seeds"]]
    if sorted(payloads) != sorted(seeds):
        raise RuntimeError("all five frozen Part C seeds are required")
    protocol_hash = _sha256(PROTOCOL)
    for seed, payload in payloads.items():
        if payload.get("seed") != seed:
            raise RuntimeError("result seed and filename seed differ")
        if payload.get("model") != "MABPT-ASCENT":
            raise RuntimeError("unexpected paper-model identity")
        if payload.get("evidence_class") != "development_only":
            raise RuntimeError("only development evidence belongs in this aggregate")
        if payload.get("partc_protocol_sha256") != protocol_hash:
            raise RuntimeError("Part C protocol hash mismatch")
        integrity = payload.get("integrity", {})
        if not integrity.get("train_and_development_only"):
            raise RuntimeError("evaluation boundary is not verified")
        if integrity.get("historical_locked_test_used") is not False:
            raise RuntimeError("historical locked test entered development aggregate")
    control_dates = {
        seed: payloads[seed]["models"]["original_ascent"]["date_metrics"]
        for seed in seeds
    }
    candidate_dates = {
        seed: payloads[seed]["models"]["mabpt_ascent"]["date_metrics"]
        for seed in seeds
    }
    paired = {
        metric: hierarchical_paired_bootstrap(
            control_dates,
            candidate_dates,
            metric,
            replicates=int(protocol["analysis"]["bootstrap_replicates"]),
            random_seed=20260811 + index,
        )
        for index, metric in enumerate(CORE_METRICS)
    }
    calibration_dates = {
        model: {
            seed: payloads[seed]["fixed_event_calibration"][model]["date_metrics"]
            for seed in seeds
        }
        for model in ("original_ascent", "mabpt_ascent")
    }
    calibration_paired = {
        metric: hierarchical_paired_bootstrap(
            calibration_dates["original_ascent"],
            calibration_dates["mabpt_ascent"],
            metric,
            replicates=int(protocol["analysis"]["bootstrap_replicates"]),
            random_seed=20261200 + index,
        )
        for index, metric in enumerate(
            ("event_nll", "event_brier", "mixture_nll", "energy_score")
        )
    }
    conflict_dates = {
        model: {
            seed: payloads[seed]["conflict_risk"]["development"][model]["date_metrics"]
            for seed in seeds
        }
        for model in ("original_ascent", "mabpt_ascent")
    }
    conflict_paired = {
        metric: hierarchical_paired_bootstrap(
            conflict_dates["original_ascent"],
            conflict_dates["mabpt_ascent"],
            metric,
            replicates=int(protocol["analysis"]["bootstrap_replicates"]),
            random_seed=20261300 + index,
            count_key="pairs",
        )
        for index, metric in enumerate(("nll", "brier"))
    }
    recall_bootstrap = paired_seed_bootstrap(
        {
            seed: float(
                payloads[seed]["conflict_risk"]["development"]["original_ascent"][
                    "recall_at_fixed_fpr"
                ]
            )
            for seed in seeds
        },
        {
            seed: float(
                payloads[seed]["conflict_risk"]["development"]["mabpt_ascent"][
                    "recall_at_fixed_fpr"
                ]
            )
            for seed in seeds
        },
        higher_is_better=True,
        replicates=int(protocol["analysis"]["bootstrap_replicates"]),
        random_seed=20261320,
    )
    return {
        "format_version": 1,
        "model": "MABPT-ASCENT",
        "experiment_ids": [
            "main_accuracy_and_proper_scores",
            "calibration",
            "physical_plausibility",
            "runtime_and_memory",
        ],
        "evidence_class": "development_only",
        "partc_protocol_sha256": protocol_hash,
        "seeds": seeds,
        "aggregates": {
            model: _aggregate_scalars(payloads, model)
            for model in ("original_ascent", "mabpt_ascent")
        },
        "paired_hierarchical_bootstrap": paired,
        "fixed_event_calibration": {
            "aggregates": {
                model: _aggregate_section(
                    payloads,
                    "fixed_event_calibration",
                    model,
                    (
                        "event_nll",
                        "event_brier",
                        "mixture_nll",
                        "energy_score",
                        "top1_ade",
                        "top1_fde",
                        "effective_modes",
                        "effective_events",
                        "event_ece",
                    ),
                )
                for model in ("original_ascent", "mabpt_ascent")
            },
            "paired_hierarchical_bootstrap": calibration_paired,
            "pooled_seed_reliability_diagram": {
                model: _combine_reliability(payloads, model)
                for model in ("original_ascent", "mabpt_ascent")
            },
        },
        "conflict_risk": {
            "aggregates": {
                model: _aggregate_section(
                    {
                        seed: {
                            "conflict_development": payloads[seed]["conflict_risk"][
                                "development"
                            ]
                        }
                        for seed in seeds
                    },
                    "conflict_development",
                    model,
                    (
                        "brier",
                        "nll",
                        "auprc",
                        "ece",
                        "recall_at_fixed_fpr",
                        "observed_fpr",
                        "mean_warning_lead_seconds",
                    ),
                )
                for model in ("original_ascent", "mabpt_ascent")
            },
            "paired_hierarchical_bootstrap": conflict_paired,
            "paired_seed_bootstrap_recall_at_fixed_fpr": recall_bootstrap,
            "regulatory_claim": False,
        },
        "primary_development_tests": {
            "H1_energy": paired["energy_score"],
            "H2_top1_fde_superiority_component": paired["top1_fde"],
            "H2_minfde_noninferiority_component": {
                **paired["minfde"],
                "decision": "not_evaluated_without_prespecified_operational_margin",
            },
            "H3_fixed_event_brier": calibration_paired["event_brier"],
            "H3_fixed_event_nll_companion": calibration_paired["event_nll"],
            "H4_conflict_brier": conflict_paired["brier"],
            "H4_recall_at_fixed_fpr_companion": recall_bootstrap,
        },
        "multiplicity": {
            "holm_across_H1_to_H4": "pending_H3_and_H4_aggregation",
            "raw_p_values_are_not_final_claim_decisions": True,
        },
        "inputs": [
            {
                "path": str(
                    (
                        ARTIFACT_ROOT
                        / f"seed{seed}_development_formal_v1.json"
                    ).relative_to(ROOT)
                ),
                "sha256": _sha256(
                    ARTIFACT_ROOT / f"seed{seed}_development_formal_v1.json"
                ),
            }
            for seed in seeds
        ],
        "claim_boundary": (
            "Five-seed paired development evidence only. A never-opened later-period "
            "or airport cohort is still required for confirmatory Part C claims."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_ROOT / "five_seed_development_summary_v1.json",
    )
    args = parser.parse_args()
    protocol = load_protocol()
    payloads = {}
    for seed in protocol["fixed_seeds"]:
        path = ARTIFACT_ROOT / f"seed{seed}_development_formal_v1.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payloads[int(seed)] = json.loads(path.read_text(encoding="utf-8"))
    result = aggregate(payloads)
    _atomic_json(args.output.resolve(), result)
    print(json.dumps({"output": str(args.output), "seeds": result["seeds"]}, indent=2))


if __name__ == "__main__":
    main()
