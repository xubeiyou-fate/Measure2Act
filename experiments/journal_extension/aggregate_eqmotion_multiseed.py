"""Aggregate the matched five-seed EqMotion, ASCENT, and MABPT comparison."""

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
METRICS = ("minade", "minfde", "energy_score")
OUTPUT = ROOT / "artifacts/journal_extension_20260814/eqmotion_five_seed_summary_v1.json"


def _eqmotion_path(airport: str, seed: int, split: str) -> Path:
    base = ROOT / "artifacts/partc_two_dataset_20260812/modern_baseline"
    if split == "development":
        if seed == 42:
            return base / f"eqmotion_tartan_{airport}_target_only_seed42_formal.json"
        return base / f"eqmotion_tartan_multiseed_v2/{airport}_target_only_seed{seed}_formal.json"
    if seed == 42:
        return base / f"eqmotion_tartan_{airport}_target_only_seed42_locked_test_v1.json"
    return base / f"eqmotion_tartan_multiseed_v2/locked/{airport}_target_only_seed{seed}_locked_test_v2.json"


def _parent_path(airport: str, seed: int, split: str) -> Path:
    if split == "development":
        return ROOT / f"artifacts/partc_two_dataset_20260812/probability_ablation_v1/development/{airport}_target_only_seed{seed}_development_v1.json"
    return ROOT / f"artifacts/partc_two_dataset_20260812/tartan_locked_test_v1/{airport}_target_only_seed{seed}_locked_test_v1.json"


def _cell(airport: str, split: str) -> dict[str, Any]:
    values = {model: {metric: [] for metric in METRICS} for model in ("eqmotion", "ascent", "mabpt_selected")}
    inputs = []
    for seed in SEEDS:
        eq_path = _eqmotion_path(airport, seed, split)
        parent_path = _parent_path(airport, seed, split)
        eqmotion = json.loads(eq_path.read_text(encoding="utf-8"))
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        eq_metrics = eqmotion["development_metrics" if split == "development" else "metrics"]
        if split == "development":
            ascent_metrics = parent["models"]["ascent_native"]["overall"]
            mabpt_metrics = parent["models"]["gibbs_unweighted_energy_kl"]["overall"]
        else:
            ascent_metrics = parent["models"]["original_ascent"]["overall"]
            mabpt_metrics = parent["models"]["mabpt_ascent"]["overall"]
        for metric in METRICS:
            values["eqmotion"][metric].append(float(eq_metrics[metric]))
            values["ascent"][metric].append(float(ascent_metrics[metric]))
            values["mabpt_selected"][metric].append(float(mabpt_metrics[metric]))
        inputs.append({"seed": seed, "eqmotion": eq_path.relative_to(ROOT).as_posix(), "parent": parent_path.relative_to(ROOT).as_posix()})
    result: dict[str, Any] = {"seeds": list(SEEDS), "inputs": inputs, "metrics": {}}
    rng = np.random.default_rng(20260814 + (0 if split == "development" else 100) + (0 if airport == "KAGC" else 10))
    draws = rng.integers(0, len(SEEDS), size=(10000, len(SEEDS)))
    for metric in METRICS:
        arrays = {model: np.asarray(values[model][metric], dtype=np.float64) for model in values}
        summary = {
            model: {
                "mean": float(array.mean()),
                "seed_sd": float(array.std(ddof=1)),
                "per_seed": array.tolist(),
            }
            for model, array in arrays.items()
        }
        for model in ("ascent", "mabpt_selected"):
            relative = (arrays["eqmotion"] - arrays[model]) / arrays["eqmotion"]
            bootstrap = relative[draws].mean(axis=1)
            summary[f"{model}_improvement_vs_eqmotion"] = {
                "mean": float(relative.mean()),
                "improved_seed_count": int((relative > 0).sum()),
                "seed_bootstrap_ci95": np.quantile(bootstrap, (0.025, 0.975)).tolist(),
            }
        result["metrics"][metric] = summary
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    cells = {
        f"{airport}_{split}": _cell(airport, split)
        for split in ("development", "test")
        for airport in AIRPORTS
    }
    result = {
        "format_version": 1,
        "experiment_id": "eqmotion_tartan_five_seed_summary_v1",
        "cells": cells,
        "uncertainty": "Paired seed bootstrap only; EqMotion outputs do not retain date-level statistics.",
        "integrity": {"complete_seed_grid": True, "matched_k": 5, "test_used_for_selection": False},
        "claim_boundary": "EqMotion uses a uniform five-mode measure and no learned ranking; only minADE, minFDE, Energy, and seed uncertainty are compared.",
    }
    atomic_json(args.output.resolve(), result)
    print(json.dumps({"output": args.output.resolve().as_posix(), "cells": list(cells)}, indent=2))


if __name__ == "__main__":
    main()
