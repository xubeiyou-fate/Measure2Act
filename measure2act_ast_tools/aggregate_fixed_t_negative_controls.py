"""Aggregate Measure2Act fixed-support negative controls."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np


LOWER_IS_BETTER = ("energy_score", "nll", "brier", "ece", "minade", "minfde", "top1_ade", "top1_fde")
REPORT_METRICS = ("energy_score", "nll", "brier", "ece", "effective_modes")
SEEDS = (42, 7, 123, 2024, 2026)
AIRPORTS = ("KAGC", "KBTP")
REGIMES = ("target_only", "full_finetune")


def load_payloads(root: Path, split: str) -> list[dict[str, Any]]:
    payloads = []
    for airport in AIRPORTS:
        for regime in REGIMES:
            for seed in SEEDS:
                path = root / split / f"{airport}_{regime}_seed{seed}_{split}_v1.json"
                if not path.is_file():
                    raise FileNotFoundError(path)
                payload = json.loads(path.read_text(encoding="utf-8"))
                if (
                    payload.get("airport") != airport
                    or payload.get("regime") != regime
                    or int(payload.get("seed")) != seed
                    or payload.get("split") != split
                ):
                    raise RuntimeError(f"identity mismatch: {path}")
                payload["_path"] = path.as_posix()
                payloads.append(payload)
    return payloads


def paired_bootstrap(
    payloads: list[dict[str, Any]],
    *,
    arm: str,
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        grouped[(payload["airport"], payload["regime"])].append(payload)
    observed = []
    for payload in payloads:
        selected = payload["models"]["selected_mabpt"]["overall"][metric]
        control = payload["models"][arm]["overall"][metric]
        observed.append(control - selected)
    draws = np.empty(replicates, dtype=np.float64)
    cells = sorted(grouped)
    for index in range(replicates):
        cell_values = []
        for cell in cells:
            records = grouped[cell]
            sampled = rng.integers(0, len(records), size=len(records))
            seed_values = []
            for sample_index in sampled:
                payload = records[int(sample_index)]
                selected_dates = payload["models"]["selected_mabpt"]["per_date"]
                control_dates = payload["models"][arm]["per_date"]
                if set(selected_dates) != set(control_dates):
                    raise RuntimeError("date grid mismatch")
                dates = list(selected_dates)
                date_sample = rng.integers(0, len(dates), size=len(dates))
                values = []
                weights = []
                for date_index in date_sample:
                    date = dates[int(date_index)]
                    control_row = control_dates[date]
                    selected_row = selected_dates[date]
                    values.append(float(control_row[metric]) - float(selected_row[metric]))
                    weights.append(float(control_row["agents"]))
                seed_values.append(float(np.average(values, weights=weights)))
            cell_values.append(float(np.mean(seed_values)))
        draws[index] = float(np.mean(cell_values))
    observed = np.asarray(observed, dtype=np.float64)
    return {
        "control": arm,
        "candidate": "selected_mabpt",
        "metric": metric,
        "absolute_improvement_mean_control_minus_selected": float(observed.mean()),
        "improved_cell_count": int((observed > 0).sum()),
        "cells": len(observed),
        "ci95": np.quantile(draws, (0.025, 0.975)).tolist(),
    }


def summarize_split(payloads: list[dict[str, Any]], split: str, replicates: int) -> dict[str, Any]:
    arms = tuple(payloads[0]["negative_control_arms"])
    for payload in payloads:
        if tuple(payload["negative_control_arms"]) != arms:
            raise RuntimeError("arm registry mismatch")
    equal_cell_means = {
        arm: {
            metric: float(np.mean([
                payload["models"][arm]["overall"][metric] for payload in payloads
            ]))
            for metric in REPORT_METRICS
        }
        for arm in arms
    }
    comparisons = {
        arm: {
            metric: paired_bootstrap(
                payloads,
                arm=arm,
                metric=metric,
                replicates=replicates,
                seed=20260916 + 1000 * list(arms).index(arm) + REPORT_METRICS.index(metric),
            )
            for metric in REPORT_METRICS
        }
        for arm in arms
        if arm != "selected_mabpt"
    }
    geometry = {
        metric: max(float(payload["geometry_invariance_max_absolute_difference"][metric]) for payload in payloads)
        for metric in ("minade", "minfde", "top1_ade", "top1_fde")
    }
    probability_sum_error = {
        arm: max(float(payload["probability_audit"]["sum_max_abs_error"][arm]) for payload in payloads)
        for arm in arms
    }
    return {
        "split": split,
        "cells": len(payloads),
        "arms": list(arms),
        "equal_cell_means": equal_cell_means,
        "selected_mabpt_vs_controls": comparisons,
        "geometry_invariance_max": geometry,
        "probability_sum_max_abs_error": probability_sum_error,
        "support_audit_sample_rows": sum(int(payload["support_audit"]["sample_row_count"]) for payload in payloads),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args()
    result = {
        "format_version": 1,
        "experiment_id": "measure2act_fixed_t_negative_controls_summary_v1",
        "input_root": args.input_root.resolve().as_posix(),
        "splits": {
            split: summarize_split(load_payloads(args.input_root, split), split, args.replicates)
            for split in ("development", "test")
        },
        "estimand": (
            "Equal airport-regime cell mean over paired seeds; bootstrap resamples "
            "seeds within each cell and dates within seed with actor weights."
        ),
        "integrity": {
            "shared_support_geometry_invariant": True,
            "positive_control": "selected_mabpt",
            "outputs_refuse_overwrite": True,
        },
        "claim_boundary": "Local retrospective fixed-support negative controls.",
    }
    atomic_json(args.output.resolve(), result)
    print(json.dumps({"output": args.output.resolve().as_posix(), "splits": list(result["splits"])}, indent=2))


if __name__ == "__main__":
    main()
