"""Aggregate the frozen Tartan matching, calibration, and weight controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .evaluate_identity_disjoint_tartan import ROOT
from .evaluate_probability_controls import ARMS
from .train_awta_tartan import atomic_json


SEEDS = (42, 7, 123, 2024, 2026)
AIRPORTS = ("KAGC", "KBTP")
METRICS = ("energy_score", "nll", "brier", "ece", "effective_modes")
INPUT_ROOT = ROOT / "artifacts/journal_extension_20260814/probability_controls"
RECEIPT = ROOT / "artifacts/journal_extension_20260814/probability_controls_receipt_v1.json"
OUTPUT = ROOT / "artifacts/journal_extension_20260814/probability_controls_summary_v1.json"


def _load(split: str) -> list[dict[str, Any]]:
    payloads = []
    for airport in AIRPORTS:
        for seed in SEEDS:
            path = INPUT_ROOT / split / f"{airport}_seed{seed}_{split}_v1.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("airport") != airport or int(payload.get("seed", -1)) != seed or payload.get("split") != split:
                raise RuntimeError(f"probability-control identity mismatch: {path}")
            payloads.append(payload)
    return payloads


def _paired_comparison(
    payloads: list[dict[str, Any]],
    control: str,
    candidate: str,
    metric: str,
    *,
    seed: int,
    replicates: int = 10000,
) -> dict[str, Any]:
    control_values = np.asarray([payload["models"][control]["overall"][metric] for payload in payloads], dtype=np.float64)
    candidate_values = np.asarray([payload["models"][candidate]["overall"][metric] for payload in payloads], dtype=np.float64)
    delta = control_values - candidate_values
    rng = np.random.default_rng(seed)
    grouped = {
        airport: [payload for payload in payloads if payload["airport"] == airport]
        for airport in AIRPORTS
    }
    if any(len(grouped[airport]) != len(SEEDS) for airport in AIRPORTS):
        raise RuntimeError("probability-control bootstrap requires five seeds per airport")
    bootstrap = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        airport_deltas = []
        for airport in AIRPORTS:
            records = grouped[airport]
            sampled_seeds = rng.integers(0, len(records), size=len(records))
            seed_deltas = []
            for seed_index in sampled_seeds:
                payload = records[int(seed_index)]
                control_dates = payload["models"][control]["per_date"]
                candidate_dates = payload["models"][candidate]["per_date"]
                if set(control_dates) != set(candidate_dates):
                    raise RuntimeError("probability-control date grid mismatch")
                date_keys = list(control_dates)
                sampled_dates = rng.integers(0, len(date_keys), size=len(date_keys))

                def sampled_value(rows: dict[str, Any]) -> float:
                    values = [rows[date_keys[int(index)]] for index in sampled_dates]
                    counts = np.asarray([value["agents"] for value in values], dtype=np.float64)
                    scores = np.asarray([value[metric] for value in values], dtype=np.float64)
                    return float(np.sum(counts * scores) / np.sum(counts))

                seed_deltas.append(sampled_value(control_dates) - sampled_value(candidate_dates))
            airport_deltas.append(float(np.mean(seed_deltas)))
        bootstrap[replicate] = float(np.mean(airport_deltas))
    per_airport = {
        airport: {
            "control_equal_seed_mean": float(np.mean([
                payload["models"][control]["overall"][metric] for payload in grouped[airport]
            ])),
            "candidate_equal_seed_mean": float(np.mean([
                payload["models"][candidate]["overall"][metric] for payload in grouped[airport]
            ])),
        }
        for airport in AIRPORTS
    }
    return {
        "control": control,
        "candidate": candidate,
        "metric": metric,
        "control_equal_cell_mean": float(control_values.mean()),
        "candidate_equal_cell_mean": float(candidate_values.mean()),
        "absolute_improvement_mean": float(delta.mean()),
        "relative_improvement_mean": float((delta / control_values).mean()),
        "improved_cell_count": int((delta > 0).sum()),
        "per_airport": per_airport,
        "paired_airport_stratified_seed_outer_date_inner_bootstrap_ci95": np.quantile(
            bootstrap, (0.025, 0.975)
        ).tolist(),
    }


def _split_summary(payloads: list[dict[str, Any]], receipt: dict[str, Any], split: str) -> dict[str, Any]:
    means = {
        arm: {
            metric: float(np.mean([payload["models"][arm]["overall"][metric] for payload in payloads]))
            for metric in METRICS
        }
        for arm in ARMS
    }
    native_selected = receipt["selection"]["target_native"]["selected_arm"]
    mabpt_selected = receipt["selection"]["gibbs_unweighted_energy_kl"]["selected_arm"]
    comparisons = {
        "soft_vs_hard_after_energy_kl": _paired_comparison(
            payloads,
            "hard_unweighted_energy_kl",
            "gibbs_unweighted_energy_kl",
            "energy_score",
            seed=20260814 + (0 if split == "development" else 100),
        ),
        "target_native_selected_temperature_vs_unscaled": _paired_comparison(
            payloads,
            "target_native_temp_1p00",
            native_selected,
            "nll",
            seed=20260815 + (0 if split == "development" else 100),
        ),
        "mabpt_selected_temperature_vs_unscaled": _paired_comparison(
            payloads,
            "selected_mabpt_temp_1p00",
            mabpt_selected,
            "nll",
            seed=20260816 + (0 if split == "development" else 100),
        ),
    }
    sensitivity_arms = [
        arm for arm in ARMS
        if arm.startswith("gibbs_cost_temp_") or arm.startswith("energy_kl_")
    ]
    default_energy = means["gibbs_unweighted_energy_kl"]["energy_score"]
    sensitivity = sorted(
        (
            {
                "arm": arm,
                "energy_score": means[arm]["energy_score"],
                "relative_change_vs_default": (means[arm]["energy_score"] - default_energy) / default_energy,
            }
            for arm in sensitivity_arms
        ),
        key=lambda row: row["energy_score"],
    )
    return {
        "cells": len(payloads),
        "equal_cell_means": means,
        "registered_comparisons": comparisons,
        "energy_hyperparameter_sensitivity": sensitivity,
        "geometry_invariance_max": {
            metric: max(float(payload["geometry_invariance_max_absolute_difference"][metric]) for payload in payloads)
            for metric in ("minade", "minfde", "top1_ade", "top1_fde")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    splits = {split: _split_summary(_load(split), receipt, split) for split in ("development", "test")}
    result = {
        "format_version": 1,
        "experiment_id": "tartan_probability_controls_summary_v1",
        "selection": receipt["selection"],
        "splits": splits,
        "uncertainty": "Equal airport-seed cell mean with paired fixed-airport, seed-outer/date-inner actor-weighted bootstrap.",
        "integrity": {"complete_grid": True, "test_used_for_selection": False, "shared_support_geometry_invariant": True},
        "claim_boundary": "Retrospective local-data probability sensitivity; no trajectory geometry or prospective generalization claim.",
    }
    atomic_json(args.output.resolve(), result)
    print(json.dumps({"output": args.output.resolve().as_posix(), "comparisons": {split: value["registered_comparisons"] for split, value in splits.items()}}, indent=2))


if __name__ == "__main__":
    main()
