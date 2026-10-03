"""Aggregate the ten-cell E15 independent probabilistic baseline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from .run import AIRPORTS, PROTOCOL_PATH, ROOT, SEEDS, atomic_json, sha256


MABPT_ROOT = ROOT / "artifacts/journal_extension_20260814/probability_controls/test"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def e15_path(root: Path, airport: str, seed: int) -> Path:
    return root / f"{airport}_seed{seed}_test_v1.json"


def mabpt_path(airport: str, seed: int) -> Path:
    return MABPT_ROOT / f"{airport}_seed{seed}_test_v1.json"


def paired_effect(root: Path, metric: str, *, draws: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    airport_effects: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    point_airports = []
    for airport in AIRPORTS:
        pairs = []
        seed_points = []
        for run_seed in SEEDS:
            independent = load(e15_path(root, airport, run_seed))["test_calibrated"]["date_metrics"]
            mabpt = load(mabpt_path(airport, run_seed))["models"]["selected_mabpt_temp_0p75"]["per_date"]
            dates = sorted(set(independent) & set(mabpt))
            if not dates:
                raise RuntimeError(f"E15/MABPT have no paired dates for {airport}/seed{run_seed}")
            independent_key = "energy" if metric == "energy_score" else metric
            values = {
                date: float(independent[date][independent_key]) - float(mabpt[date][metric])
                for date in dates
            }
            pairs.append((values, {"seed": run_seed}))
            seed_points.append(float(np.mean(list(values.values()))))
        airport_effects[airport] = pairs
        point_airports.append(float(np.mean(seed_points)))
    samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        effects = []
        for airport in AIRPORTS:
            pairs = airport_effects[airport]
            selected = rng.integers(0, len(pairs), size=len(pairs))
            seed_effects = []
            for index in selected:
                values = pairs[int(index)][0]
                dates = sorted(values)
                sampled = rng.integers(0, len(dates), size=len(dates))
                seed_effects.append(float(np.mean([values[dates[int(i)]] for i in sampled])))
            effects.append(float(np.mean(seed_effects)))
        samples[draw] = float(np.mean(effects))
    return {
        "metric": metric,
        "effect_definition": "independent_GRU_minus_MABPT; positive favors MABPT",
        "absolute_effect": float(np.mean(point_airports)),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "bootstrap_replicates": draws,
    }


def aggregate(root: Path) -> dict[str, Any]:
    cells = []
    raw = []
    calibrated = []
    for airport in AIRPORTS:
        for seed in SEEDS:
            path = e15_path(root, airport, seed)
            payload = load(path)
            if (
                payload.get("complete") is not True
                or payload.get("formal") is not True
                or payload.get("airport") != airport
                or int(payload.get("seed", -1)) != seed
                or payload.get("protocol_sha256") != sha256(PROTOCOL_PATH)
                or payload.get("integrity", {}).get("test_used_for_selection") is not False
                or payload.get("integrity", {}).get("third_dataset_used") is not False
                or payload.get("integrity", {}).get("ascent_or_mabpt_weights_used") is not False
            ):
                raise RuntimeError(f"invalid E15 cell: {path}")
            cells.append({
                "airport": airport,
                "seed": seed,
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
            })
            raw.append(payload["test_raw"])
            calibrated.append(payload["test_calibrated"])
    probability_metrics = ("energy", "fixed_event_nll", "brier", "ece")
    return {
        "format_version": 1,
        "experiment_id": "E15",
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "model": "IndependentMixtureGRU",
        "identity": "independent architecture with directly learned K=5 categorical probabilities; not an official Trajectron++ reproduction",
        "cells": cells,
        "equal_cell_mean_raw": {metric: float(np.mean([cell[metric] for cell in raw])) for metric in probability_metrics},
        "equal_cell_mean_calibrated": {metric: float(np.mean([cell[metric] for cell in calibrated])) for metric in probability_metrics},
        "paired_vs_mabpt": {
            metric: paired_effect(root, metric, draws=10000, seed=20260817 + index)
            for index, metric in enumerate(("energy_score", "minade", "minfde"))
        },
        "integrity": {
            "formal_cells_complete": len(cells) == 10,
            "third_dataset_used": False,
            "test_used_for_temperature_selection": False,
            "cross_support_probability_metrics_compared": False,
        },
        "claim_boundary": "Retrospective independent architecture control on the two existing local datasets; not an official Trajectron++ reproduction.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    payload = aggregate(root)
    output = root / "e15_summary_v1.json"
    atomic_json(output, payload)
    print(json.dumps({"status": payload["status"], "output": output.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
