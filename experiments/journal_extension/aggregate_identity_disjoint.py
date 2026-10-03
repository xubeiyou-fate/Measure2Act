"""Aggregate identity-disjoint Tartan sensitivity with paired hierarchical bootstrap."""

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
SPLITS = ("development", "test")
MODELS = ("ascent_native", "mabpt_selected_unweighted_energy_kl")
METRICS = ("energy_score", "minade", "minfde", "top1_ade", "top1_fde")
DEFAULT_ROOT = ROOT / "artifacts/journal_extension_20260814/tartan_identity_disjoint"
DEFAULT_OUTPUT = ROOT / "artifacts/journal_extension_20260814/tartan_identity_disjoint_summary_v1.json"


def _load(root: Path, airport: str, split: str) -> list[dict[str, Any]]:
    payloads = []
    for seed in SEEDS:
        path = root / split / f"{airport}_seed{seed}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("airport") != airport or payload.get("split") != split or int(payload.get("seed", -1)) != seed:
            raise RuntimeError(f"identity-disjoint result identity mismatch: {path}")
        payloads.append(payload)
    return payloads


def _date_draw(
    payload: dict[str, Any],
    model: str,
    metric: str,
    indices: np.ndarray,
) -> float:
    records = list(payload["models"][model]["per_date"].values())
    counts = np.asarray([record["agents"] for record in records], dtype=np.float64)[indices]
    values = np.asarray([record[metric] for record in records], dtype=np.float64)[indices]
    return float(np.sum(counts * values) / np.sum(counts))


def _bootstrap(
    payloads: list[dict[str, Any]],
    metric: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    delta = np.empty(replicates, dtype=np.float64)
    relative = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled_seeds = rng.integers(0, len(payloads), size=len(payloads))
        ascent_values = []
        mabpt_values = []
        for seed_index in sampled_seeds:
            payload = payloads[int(seed_index)]
            date_count = len(payload["models"][MODELS[0]]["per_date"])
            sampled_dates = rng.integers(0, date_count, size=date_count)
            ascent_values.append(_date_draw(payload, MODELS[0], metric, sampled_dates))
            mabpt_values.append(_date_draw(payload, MODELS[1], metric, sampled_dates))
        ascent = float(np.mean(ascent_values))
        mabpt = float(np.mean(mabpt_values))
        delta[replicate] = ascent - mabpt
        relative[replicate] = (ascent - mabpt) / ascent
    return {
        "replicates": replicates,
        "seed": seed,
        "absolute_improvement_ci95": np.quantile(delta, (0.025, 0.975)).tolist(),
        "relative_improvement_ci95": np.quantile(relative, (0.025, 0.975)).tolist(),
        "probability_improvement_gt_zero": float(np.mean(delta > 0)),
    }


def summarize(payloads: list[dict[str, Any]], *, replicates: int, bootstrap_seed: int) -> dict[str, Any]:
    result: dict[str, Any] = {"seeds": list(SEEDS), "actors_per_seed": [payload["actors"] for payload in payloads], "metrics": {}}
    for metric_index, metric in enumerate(METRICS):
        values = {
            model: np.asarray([payload["models"][model]["overall"][metric] for payload in payloads], dtype=np.float64)
            for model in MODELS
        }
        delta = values[MODELS[0]] - values[MODELS[1]]
        relative = delta / values[MODELS[0]]
        result["metrics"][metric] = {
            model: {
                "mean": float(values[model].mean()),
                "seed_sd": float(values[model].std(ddof=1)),
                "per_seed": values[model].tolist(),
            }
            for model in MODELS
        }
        result["metrics"][metric].update({
            "absolute_improvement_mean": float(delta.mean()),
            "relative_improvement_mean": float(relative.mean()),
            "improved_seed_count": int((delta > 0).sum()),
            "bootstrap": _bootstrap(
                payloads,
                metric,
                replicates=replicates,
                seed=bootstrap_seed + metric_index,
            ),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260814)
    args = parser.parse_args()
    cells = {}
    for split_index, split in enumerate(SPLITS):
        for airport_index, airport in enumerate(AIRPORTS):
            payloads = _load(args.input_root.resolve(), airport, split)
            cells[f"{airport}_{split}"] = summarize(
                payloads,
                replicates=args.replicates,
                bootstrap_seed=args.bootstrap_seed + 100 * split_index + 10 * airport_index,
            )
    result = {
        "format_version": 1,
        "experiment_id": "tartan_identity_disjoint_summary_v1",
        "estimand": "Equal-seed mean within airport and split; paired seed-outer/date-inner actor-weighted bootstrap.",
        "cells": cells,
        "integrity": {
            "complete_grid": len(cells) == len(AIRPORTS) * len(SPLITS),
            "paired_models": True,
            "test_used_for_selection": False,
        },
        "claim_boundary": "Retrospective sensitivity on local identity-disjoint subgroups; not a prospective or external holdout.",
    }
    atomic_json(args.output.resolve(), result)
    print(json.dumps({
        "output": args.output.resolve().as_posix(),
        "energy": {
            name: {
                "relative_improvement_mean": cell["metrics"]["energy_score"]["relative_improvement_mean"],
                "ci95": cell["metrics"]["energy_score"]["bootstrap"]["relative_improvement_ci95"],
            }
            for name, cell in cells.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
