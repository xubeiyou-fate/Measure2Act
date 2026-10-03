"""Aggregate aWTA matched-baseline results on Tartan and TrajAir."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .evaluate_identity_disjoint_tartan import ROOT
from .train_awta_tartan import atomic_json


SEEDS = (42, 7, 123, 2024, 2026)
AIRPORTS = ("KAGC", "KBTP")
METRICS = (
    "energy_score",
    "nll",
    "brier",
    "ece",
    "minade",
    "minfde",
    "top1_ade",
    "top1_fde",
)
OUTPUT = ROOT / "artifacts/journal_extension_20260814/awta_summary_v1.json"


def _tartan_records(airport: str, split: str) -> list[dict[str, Any]]:
    records = []
    for seed in SEEDS:
        if split == "development":
            awta_path = ROOT / f"artifacts/journal_extension_20260814/awta_tartan/development/{airport}_seed{seed}_formal.json"
            parent_path = ROOT / f"artifacts/partc_two_dataset_20260812/probability_ablation_v1/development/{airport}_target_only_seed{seed}_development_v1.json"
            awta_payload = json.loads(awta_path.read_text(encoding="utf-8"))["development_metrics"]
            parent = json.loads(parent_path.read_text(encoding="utf-8"))["models"]
            models = {
                "ascent": {"overall": parent["ascent_native"]["overall"], "per_date": parent["ascent_native"]["per_date"]},
                "awta": {"overall": awta_payload["overall"], "per_date": awta_payload["date_metrics"]},
                "mabpt_selected": {"overall": parent["gibbs_unweighted_energy_kl"]["overall"], "per_date": parent["gibbs_unweighted_energy_kl"]["per_date"]},
            }
        else:
            awta_path = ROOT / f"artifacts/journal_extension_20260814/awta_tartan/test/{airport}_seed{seed}_test_v1.json"
            parent_path = ROOT / f"artifacts/partc_two_dataset_20260812/tartan_locked_test_v1/{airport}_target_only_seed{seed}_locked_test_v1.json"
            awta_payload = json.loads(awta_path.read_text(encoding="utf-8"))["test_metrics"]
            parent = json.loads(parent_path.read_text(encoding="utf-8"))["models"]
            models = {
                "ascent": parent["original_ascent"],
                "awta": {"overall": awta_payload["overall"], "per_date": awta_payload["date_metrics"]},
                "mabpt_selected": parent["mabpt_ascent"],
            }
        dates = set(models["ascent"]["per_date"])
        if any(set(value["per_date"]) != dates for value in models.values()):
            raise RuntimeError(f"aWTA date grid mismatch: {airport} {split} seed{seed}")
        records.append({"seed": seed, "models": models, "awta_path": awta_path.relative_to(ROOT).as_posix(), "parent_path": parent_path.relative_to(ROOT).as_posix()})
    return records


def _date_value(
    model: dict[str, Any],
    metric: str,
    date_keys: list[str],
    sampled: np.ndarray,
) -> float:
    rows = [model["per_date"][date] for date in date_keys]
    counts = np.asarray(
        [row["agents"] if "agents" in row else row["actors"] for row in rows],
        dtype=np.float64,
    )[sampled]
    values = np.asarray([row[metric] for row in rows], dtype=np.float64)[sampled]
    return float(np.sum(counts * values) / np.sum(counts))


def _hierarchical(
    records: list[dict[str, Any]],
    control: str,
    candidate: str,
    metric: str,
    *,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    values = np.empty(10000, dtype=np.float64)
    for replicate in range(len(values)):
        seed_draw = rng.integers(0, len(records), size=len(records))
        control_draw = []
        candidate_draw = []
        for seed_index in seed_draw:
            record = records[int(seed_index)]
            date_keys = sorted(record["models"][control]["per_date"])
            if set(date_keys) != set(record["models"][candidate]["per_date"]):
                raise RuntimeError("aWTA paired bootstrap date grid mismatch")
            date_count = len(date_keys)
            date_draw = rng.integers(0, date_count, size=date_count)
            control_draw.append(
                _date_value(record["models"][control], metric, date_keys, date_draw)
            )
            candidate_draw.append(
                _date_value(record["models"][candidate], metric, date_keys, date_draw)
            )
        values[replicate] = float(np.mean(control_draw) - np.mean(candidate_draw))
    return {
        "method": "paired_seed_outer_date_inner_bootstrap",
        "replicates": len(values),
        "absolute_improvement_ci95": np.quantile(values, (0.025, 0.975)).tolist(),
        "probability_improvement_gt_zero": float(np.mean(values > 0)),
    }


def _paired_seed_bootstrap(delta: np.ndarray, *, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(delta), size=(10000, len(delta)))
    values = delta[draws].mean(axis=1)
    return {
        "method": "paired_seed_bootstrap",
        "replicates": len(values),
        "absolute_improvement_ci95": np.quantile(values, (0.025, 0.975)).tolist(),
        "probability_improvement_gt_zero": float(np.mean(values > 0)),
    }


def _has_date_metric(records: list[dict[str, Any]], models: tuple[str, str], metric: str) -> bool:
    return all(
        record["models"][model]["per_date"]
        and all(metric in row for row in record["models"][model]["per_date"].values())
        for record in records
        for model in models
    )


def _tartan_summary(airport: str, split: str) -> dict[str, Any]:
    records = _tartan_records(airport, split)
    models = ("ascent", "awta", "mabpt_selected")
    result: dict[str, Any] = {"seeds": list(SEEDS), "inputs": [{key: record[key] for key in ("seed", "awta_path", "parent_path")} for record in records], "metrics": {}}
    offset = (0 if airport == "KAGC" else 1000) + (0 if split == "development" else 100)
    for metric_index, metric in enumerate(METRICS):
        arrays = {
            model: np.asarray([record["models"][model]["overall"][metric] for record in records], dtype=np.float64)
            for model in models
        }
        metric_result: dict[str, Any] = {
            model: {"mean": float(array.mean()), "seed_sd": float(array.std(ddof=1)), "per_seed": array.tolist()}
            for model, array in arrays.items()
        }
        for comparison_index, (control, candidate) in enumerate((("ascent", "awta"), ("awta", "mabpt_selected"))):
            delta = arrays[control] - arrays[candidate]
            bootstrap_seed = 20260814 + offset + 10 * metric_index + comparison_index
            bootstrap = (
                _hierarchical(records, control, candidate, metric, seed=bootstrap_seed)
                if _has_date_metric(records, (control, candidate), metric)
                else _paired_seed_bootstrap(delta, seed=bootstrap_seed)
            )
            metric_result[f"{candidate}_vs_{control}"] = {
                "absolute_improvement_mean": float(delta.mean()),
                "relative_improvement_mean": float((delta / arrays[control]).mean()),
                "improved_seed_count": int((delta > 0).sum()),
                "bootstrap": bootstrap,
            }
        result["metrics"][metric] = metric_result
    return result


def _trajair_summary() -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    inputs = []
    rng = np.random.default_rng(20260814)
    draws = rng.integers(0, len(SEEDS), size=(10000, len(SEEDS)))
    for metric in METRICS:
        ascent = []
        awta = []
        for seed in SEEDS:
            baseline_path = ROOT / f"artifacts/experiments/metric_exact/status/P3_B0_signed_coupled_all_train_seed{seed}_formal.json"
            awta_path = ROOT / f"artifacts/journal_extension_20260814/awta_trajair/development/seed{seed}_formal.json"
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["metrics"]
            candidate = json.loads(awta_path.read_text(encoding="utf-8"))["development_metrics"]["overall"]
            ascent.append(float(baseline[metric]))
            awta.append(float(candidate[metric]))
            if metric == METRICS[0]:
                inputs.append({"seed": seed, "ascent": baseline_path.relative_to(ROOT).as_posix(), "awta": awta_path.relative_to(ROOT).as_posix()})
        ascent_array = np.asarray(ascent)
        awta_array = np.asarray(awta)
        delta = ascent_array - awta_array
        bootstrap = delta[draws].mean(axis=1)
        metrics[metric] = {
            "ascent": {"mean": float(ascent_array.mean()), "seed_sd": float(ascent_array.std(ddof=1)), "per_seed": ascent},
            "awta": {"mean": float(awta_array.mean()), "seed_sd": float(awta_array.std(ddof=1)), "per_seed": awta},
            "awta_vs_ascent": {
                "absolute_improvement_mean": float(delta.mean()),
                "relative_improvement_mean": float((delta / ascent_array).mean()),
                "improved_seed_count": int((delta > 0).sum()),
                "paired_seed_bootstrap_ci95": np.quantile(bootstrap, (0.025, 0.975)).tolist(),
            },
        }
    return {"seeds": list(SEEDS), "inputs": inputs, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = {
        "format_version": 1,
        "experiment_id": "awta_matched_baseline_summary_v1",
        "tartan": {f"{airport}_{split}": _tartan_summary(airport, split) for split in ("development", "test") for airport in AIRPORTS},
        "trajair_development": _trajair_summary(),
        "uncertainty": {
            "tartan": "Paired seed-outer/date-inner actor-weighted bootstrap.",
            "trajair": "Paired seed bootstrap because the frozen B0 status artifacts retain no date table.",
        },
        "integrity": {"complete_grid": True, "fixed_final_epoch": True, "test_used_for_selection": False},
        "claim_boundary": "aWTA changes only the training assignment objective. Tartan test is internally locked and retrospective; TrajAir is development-only in this extension.",
    }
    atomic_json(args.output.resolve(), result)
    print(json.dumps({"output": args.output.resolve().as_posix(), "cells": list(result["tartan"])}, indent=2))


if __name__ == "__main__":
    main()
