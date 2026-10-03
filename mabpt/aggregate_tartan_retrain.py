"""Aggregate five-seed Tartan retraining results with date-block inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .aggregate_partc_seeds import hierarchical_paired_bootstrap


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("tartan_retrain_protocol_v1.json")
INPUT_ROOT = ROOT / "artifacts/partc_two_dataset_20260812/tartan_locked_test_v1"
MODELS = ("original_ascent", "mabpt_ascent")
METRICS = (
    "top1_ade",
    "top1_fde",
    "minade",
    "minfde",
    "energy_score",
    "nll",
    "brier",
    "ece",
    "tail_minfde",
)
PRIMARY = ("energy_score", "minfde")
SAFETY_METRICS = (
    "brier",
    "nll",
    "auprc",
    "ece_15_bin",
    "precision_at_training_fixed_fpr",
    "recall_at_training_fixed_fpr",
    "f1_at_training_fixed_fpr",
    "observed_fpr",
)
SAFETY_HIGHER_IS_BETTER = {
    "auprc",
    "precision_at_training_fixed_fpr",
    "recall_at_training_fixed_fpr",
    "f1_at_training_fixed_fpr",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def holm(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.get)
    adjusted = {}
    previous = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        value = min(1.0, raw[name] * (total - rank))
        previous = max(previous, value)
        adjusted[name] = previous
    return {name: adjusted[name] for name in raw}


def _seed_scalars(payloads, model: str) -> dict[str, object]:
    result = {}
    for metric in METRICS:
        values = np.asarray(
            [payloads[seed]["models"][model]["overall"][metric] for seed in sorted(payloads)],
            dtype=np.float64,
        )
        result[metric] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "values_by_seed": {
                str(seed): float(payloads[seed]["models"][model]["overall"][metric])
                for seed in sorted(payloads)
            },
        }
    return result


def _safety_seed_scalars(payloads, model: str) -> dict[str, object]:
    result = {}
    for metric in SAFETY_METRICS:
        values = np.asarray(
            [
                payloads[seed]["safety_proxy"]["models"][model]["overall"][metric]
                for seed in sorted(payloads)
            ],
            dtype=np.float64,
        )
        result[metric] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "values_by_seed": {
                str(seed): float(
                    payloads[seed]["safety_proxy"]["models"][model]["overall"][metric]
                )
                for seed in sorted(payloads)
            },
        }
    return result


def _efficiency_seed_scalars(payloads, model: str) -> dict[str, object]:
    result = {
        "parameter_count": {
            "values_by_seed": {
                str(seed): int(
                    payloads[seed]["efficiency"]["parameter_counts"][model]
                )
                for seed in sorted(payloads)
            }
        },
        "checkpoint_bytes": {
            stage: {
                "values_by_seed": {
                    str(seed): int(
                        payloads[seed]["efficiency"]["checkpoint_bytes"][stage]
                    )
                    for seed in sorted(payloads)
                }
            }
            for stage in ("ascent", "decision_support", "predicted_risk")
        },
    }
    throughput = np.asarray(
        [
            payloads[seed]["efficiency"]["actors_per_second"]
            for seed in sorted(payloads)
        ],
        dtype=np.float64,
    )
    result["joint_evaluator_actors_per_second"] = {
        "mean": float(throughput.mean()),
        "sample_std": float(throughput.std(ddof=1)),
        "values_by_seed": {
            str(seed): float(payloads[seed]["efficiency"]["actors_per_second"])
            for seed in sorted(payloads)
        },
        "scope": "Both ASCENT and MABPT arms plus CV and safety proxy in one process; not isolated per-model latency.",
    }
    result["peak_allocated_gpu_bytes_by_seed"] = {
        str(seed): int(payloads[seed]["efficiency"]["peak_allocated_gpu_bytes"])
        for seed in sorted(payloads)
    }
    return result


def _safety_aggregate(
    payloads,
    seeds: list[int],
    *,
    replicates: int,
    random_seed: int,
) -> dict[str, object]:
    models = {
        model: _safety_seed_scalars(payloads, model) for model in MODELS
    }
    control = {
        seed: payloads[seed]["safety_proxy"]["models"]["original_ascent"]["per_date"]
        for seed in seeds
    }
    candidate = {
        seed: payloads[seed]["safety_proxy"]["models"]["mabpt_ascent"]["per_date"]
        for seed in seeds
    }
    effects = {}
    for index, metric in enumerate(SAFETY_METRICS):
        # Dates without any actor pair have no defined proper score. Exclude
        # them symmetrically from both arms before pair-weighted inference.
        usable_control = {}
        usable_candidate = {}
        for seed in seeds:
            dates = [
                date
                for date in control[seed]
                if int(control[seed][date]["pairs"]) > 0
                and int(candidate[seed][date]["pairs"]) > 0
                and metric in control[seed][date]
                and metric in candidate[seed][date]
                and control[seed][date][metric] is not None
                and candidate[seed][date][metric] is not None
            ]
            if not dates:
                raise RuntimeError(f"no paired safety dates for seed {seed}, metric {metric}")
            usable_control[seed] = {date: control[seed][date] for date in dates}
            usable_candidate[seed] = {date: candidate[seed][date] for date in dates}
        higher_is_better = metric in SAFETY_HIGHER_IS_BETTER
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
    }


def aggregate(airport: str, regime: str, *, replicates: int = 10000) -> dict[str, object]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in protocol["training"]["seeds"]]
    payloads = {}
    inputs = []
    for seed in seeds:
        path = INPUT_ROOT / f"{airport}_{regime}_seed{seed}_locked_test_v1.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = (airport, regime, seed, "test")
        actual = (payload["airport"], payload["regime"], int(payload["seed"]), payload["split"])
        if actual != expected:
            raise RuntimeError(f"locked-test payload identity mismatch: {path}")
        if payload["evidence_class"] != "locked_retrospective_test_single_pass":
            raise RuntimeError(f"nonformal evidence in locked-test aggregate: {path}")
        payloads[seed] = payload
        inputs.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)})
    control = {
        seed: payloads[seed]["models"]["original_ascent"]["per_date"] for seed in seeds
    }
    candidate = {
        seed: payloads[seed]["models"]["mabpt_ascent"]["per_date"] for seed in seeds
    }
    effects = {
        metric: hierarchical_paired_bootstrap(
            control,
            candidate,
            metric,
            replicates=replicates,
            random_seed=20260813 + index,
            count_key="agents",
        )
        for index, metric in enumerate(METRICS)
    }
    raw_primary = {metric: float(effects[metric]["raw_one_sided_p"]) for metric in PRIMARY}
    adjusted = holm(raw_primary)
    for metric in PRIMARY:
        effects[metric]["holm_adjusted_one_sided_p_across_primary_endpoints"] = adjusted[metric]
    return {
        "format_version": 1,
        "experiment_id": "Tartan_retrain_locked_test_five_seed_aggregate",
        "airport": airport,
        "regime": regime,
        "seeds": seeds,
        "models": {model: _seed_scalars(payloads, model) for model in MODELS},
        "efficiency": {
            model: _efficiency_seed_scalars(payloads, model) for model in MODELS
        },
        "paired_hierarchical_seed_date_bootstrap": effects,
        "safety_proxy": _safety_aggregate(
            payloads, seeds, replicates=replicates, random_seed=20261013
        ),
        "primary_endpoints": list(PRIMARY),
        "multiple_comparison": "Holm adjustment across the two registered primary endpoints within this airport/regime comparison",
        "inputs": inputs,
        "protocol": {"path": PROTOCOL.relative_to(ROOT).as_posix(), "sha256": sha256(PROTOCOL)},
        "integrity": {
            "date_nested_within_seed_bootstrap": True,
            "overlapping_windows_not_independent": True,
            "matched_seed_pairing": True,
            "test_used_for_model_selection": False,
        },
        "claim_boundary": protocol["claim_boundaries"],
    }


def aggregate_cross_airport(
    training_airport: str,
    evaluation_airport: str,
    *,
    replicates: int = 10000,
) -> dict[str, object]:
    if training_airport == evaluation_airport:
        raise ValueError("cross-airport aggregate requires different airports")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in protocol["training"]["seeds"]]
    payloads = {}
    inputs = []
    for seed in seeds:
        path = INPUT_ROOT / (
            f"{training_airport}_to_{evaluation_airport}_target_only_seed{seed}_locked_test_v1.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = (training_airport, evaluation_airport, "target_only", seed, "test")
        actual = (
            payload["training_airport"],
            payload["evaluation_airport"],
            payload["regime"],
            int(payload["seed"]),
            payload["split"],
        )
        if actual != expected or payload.get("cross_airport") is not True:
            raise RuntimeError(f"cross-airport payload identity mismatch: {path}")
        payloads[seed] = payload
        inputs.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)})
    control = {
        seed: payloads[seed]["models"]["original_ascent"]["per_date"] for seed in seeds
    }
    candidate = {
        seed: payloads[seed]["models"]["mabpt_ascent"]["per_date"] for seed in seeds
    }
    effects = {
        metric: hierarchical_paired_bootstrap(
            control,
            candidate,
            metric,
            replicates=replicates,
            random_seed=20260913 + index,
            count_key="agents",
        )
        for index, metric in enumerate(METRICS)
    }
    raw_primary = {metric: float(effects[metric]["raw_one_sided_p"]) for metric in PRIMARY}
    adjusted = holm(raw_primary)
    for metric in PRIMARY:
        effects[metric]["holm_adjusted_one_sided_p_across_primary_endpoints"] = adjusted[metric]
    return {
        "format_version": 1,
        "experiment_id": "Tartan_cross_airport_locked_test_five_seed_aggregate",
        "training_airport": training_airport,
        "evaluation_airport": evaluation_airport,
        "regime": "target_only",
        "seeds": seeds,
        "models": {model: _seed_scalars(payloads, model) for model in MODELS},
        "efficiency": {
            model: _efficiency_seed_scalars(payloads, model) for model in MODELS
        },
        "paired_hierarchical_seed_date_bootstrap": effects,
        "safety_proxy": _safety_aggregate(
            payloads, seeds, replicates=replicates, random_seed=20261113
        ),
        "primary_endpoints": list(PRIMARY),
        "inputs": inputs,
        "protocol": {"path": PROTOCOL.relative_to(ROOT).as_posix(), "sha256": sha256(PROTOCOL)},
        "integrity": {
            "date_nested_within_seed_bootstrap": True,
            "source_airport_thresholds_used": True,
            "target_airport_labels_used_for_tuning": False,
            "overlapping_windows_not_independent": True,
        },
        "claim_boundary": protocol["claim_boundaries"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airport", choices=("KAGC", "KBTP"))
    parser.add_argument("--training-airport", choices=("KAGC", "KBTP"))
    parser.add_argument("--evaluation-airport", choices=("KAGC", "KBTP"))
    parser.add_argument("--regime", choices=("target_only", "full_finetune"), required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.replicates < 1000:
        raise ValueError("formal aggregate requires at least 1000 bootstrap replicates")
    cross = args.training_airport is not None or args.evaluation_airport is not None
    if cross:
        if args.airport is not None or args.training_airport is None or args.evaluation_airport is None:
            parser.error("cross-airport mode requires both airport roles and forbids --airport")
        if args.regime != "target_only":
            parser.error("cross-airport aggregate is registered for target_only")
        payload = aggregate_cross_airport(
            args.training_airport,
            args.evaluation_airport,
            replicates=args.replicates,
        )
    else:
        if args.airport is None:
            parser.error("domain evaluation requires --airport")
        payload = aggregate(args.airport, args.regime, replicates=args.replicates)
    atomic_json(args.output.resolve(), payload)
    print(json.dumps({"output": args.output.resolve().as_posix(), "experiment_id": payload["experiment_id"]}, indent=2))


if __name__ == "__main__":
    main()
