"""Build matched-horizon E1 social and no-social aviation baseline tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DAYS = ("7days1", "7days2", "7days3", "7days4")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _finite_measure_metrics(arm: dict[str, object]) -> dict[str, float]:
    return {
        "energy": float(arm["energy_score"]),
        "minade": float(arm["minade"]),
        "minfde": float(arm["minfde"]),
    }


def _official_metrics(metrics: dict[str, object]) -> dict[str, float]:
    return {
        "energy": float(metrics["energy"]),
        "minade": float(metrics["minade"]),
        "minfde": float(metrics["minfde"]),
        "sample1_ade": float(metrics["sample1_ade"]),
        "sample1_fde": float(metrics["sample1_fde"]),
        "expected_ade": float(metrics["expected_ade"]),
        "expected_fde": float(metrics["expected_fde"]),
        "scene_oracle_ade": float(metrics["scene_oracle_ade"]),
        "scene_oracle_fde": float(metrics["scene_oracle_fde"]),
    }


def _gain(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference


def _weighted(rows: dict[str, dict[str, object]]) -> dict[str, object]:
    actors = sum(int(row["actors"]) for row in rows.values())
    models = tuple(next(iter(rows.values()))["models"])
    aggregate = {"actors": actors, "models": {}}
    for model in models:
        metrics = tuple(next(iter(rows.values()))["models"][model])
        aggregate["models"][model] = {
            metric: sum(
                int(row["actors"]) * float(row["models"][model][metric])
                for row in rows.values()
            ) / actors
            for metric in metrics
        }
    aggregate["relative_energy_gain_mabpt"] = {
        model: _gain(
            aggregate["models"][model]["energy"],
            aggregate["models"]["MABPT"]["energy"],
        )
        for model in models if model != "MABPT"
    }
    return aggregate


def _condition(condition: str) -> tuple[dict[str, object], list[dict[str, str]]]:
    if condition == "social":
        dataset_prefix = "trajair_social"
        official_family = "trajairnet"
        official_label = "TrajAirNet"
        official_suffix = "fixed_epoch10_formal_v1"
    elif condition == "no_social":
        dataset_prefix = "act_no_social"
        official_family = "actrajnet"
        official_label = "ACTrajNet"
        official_suffix = "author_epoch1_formal_v1"
    else:
        raise ValueError(f"unknown E1 condition: {condition}")
    rows: dict[str, object] = {}
    receipts = []
    for day in DAYS:
        measure_path = (
            ROOT / "artifacts/mabpt"
            / f"e1_{dataset_prefix}_{day}_matched_formal_v2.json"
        )
        official_path = (
            ROOT / "artifacts/mabpt"
            / f"e1_{official_family}_{day}_{official_suffix}.json"
        )
        measure = json.loads(measure_path.read_text(encoding="utf-8"))
        official = json.loads(official_path.read_text(encoding="utf-8"))
        actors = int(measure["arms"]["mabpt"]["agents"])
        official_actors = int(official["metrics"]["actors"])
        if actors != official_actors:
            raise RuntimeError(
                f"E1 {condition}/{day} actor mismatch: {actors} != {official_actors}"
            )
        if int(official["evaluation"]["history_steps"]) != 11 or int(
            official["evaluation"]["forecast_stride_seconds"]
        ) != 10:
            raise RuntimeError("official baseline does not use the matched E1 grid")
        if int(measure["evaluation"]["observation_steps"]) != 11 or int(
            measure["evaluation"]["prediction_stride_seconds"]
        ) != 10:
            raise RuntimeError("MABPT result does not use the matched E1 grid")
        models = {
            "CV": _finite_measure_metrics(measure["arms"]["constant_velocity"]),
            "ASCENT": _finite_measure_metrics(measure["arms"]["ascent_native"]),
            "target_native": _finite_measure_metrics(
                measure["arms"]["target_native_logits"]
            ),
            "target_Energy": _finite_measure_metrics(
                measure["arms"]["target_energy_probabilities"]
            ),
            official_label: _official_metrics(official["metrics"]),
            "MABPT": _finite_measure_metrics(measure["arms"]["mabpt"]),
        }
        rows[day] = {
            "actors": actors,
            "models": models,
            "relative_energy_gain_mabpt": {
                model: _gain(values["energy"], models["MABPT"]["energy"])
                for model, values in models.items() if model != "MABPT"
            },
        }
        receipts.extend([
            {"path": measure_path.relative_to(ROOT).as_posix(), "sha256": _sha256(measure_path)},
            {"path": official_path.relative_to(ROOT).as_posix(), "sha256": _sha256(official_path)},
        ])
    return {
        "interaction_condition": condition,
        "official_baseline": official_label,
        "datasets": rows,
        "actor_weighted_aggregate": _weighted(rows),
    }, receipts


def aggregate() -> dict[str, object]:
    social, social_inputs = _condition("social")
    no_social, no_social_inputs = _condition("no_social")
    return {
        "format_version": 1,
        "model": "MABPT",
        "model_version": "1.0.0",
        "experiment_id": "E1",
        "evidence_class": "matched_horizon_retrospective_cross_dataset_reuse",
        "common_evaluation": {
            "history_steps": 11,
            "forecast_times_seconds": list(range(10, 121, 10)),
            "finite_measure_metrics": ["Energy", "minADE", "minFDE"],
            "official_stochastic_diagnostics_not_equated_to_MABPT_top1": [
                "sample1_ADE", "sample1_FDE", "expected_ADE", "expected_FDE"
            ],
        },
        "tables": {"social": social, "no_social": no_social},
        "inputs": social_inputs + no_social_inputs,
        "integrity": {
            "actor_counts_matched_per_dataset": True,
            "social_and_no_social_separate": True,
            "test_selected_checkpoint": False,
            "temperature_or_probability_fit": False,
            "fresh_confirmatory_test": False,
        },
        "claim_boundary": "All four public seven-day views were previously opened. The table is a matched-horizon reproducible comparison, not a sealed confirmatory test.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/mabpt/e1_matched_summary_v1.json",
    )
    args = parser.parse_args()
    result = aggregate()
    _atomic_json(args.output.resolve(), result)
    print(json.dumps({
        "output": str(args.output),
        "social": result["tables"]["social"]["actor_weighted_aggregate"],
        "no_social": result["tables"]["no_social"]["actor_weighted_aggregate"],
    }, indent=2))


if __name__ == "__main__":
    main()
