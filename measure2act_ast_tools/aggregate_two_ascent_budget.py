"""Aggregate total-budget two-ASCENT capacity-control outputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


AIRPORTS = ("KAGC", "KBTP")
REGIMES = ("target_only", "full_finetune")
SEEDS = (42, 7, 123, 2024, 2026)
SELECTED = "selected_mabpt"
CONTROLS = ("two_ascent_union10", "ascent_native", "target_native", "buddy_ascent_native")
REPORT_METRICS = (
    "energy_score",
    "energy_score_full_path",
    "minade",
    "minfde",
    "nll",
    "brier",
    "support_index_ece",
    "endpoint_grid_nll",
    "endpoint_grid_brier",
    "endpoint_grid_ece",
    "endpoint_radial_nll",
    "endpoint_radial_brier",
    "endpoint_radial_ece",
    "effective_modes",
)
LOWER_IS_BETTER = tuple(metric for metric in REPORT_METRICS if metric != "effective_modes")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


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


def metric_value(payload: dict[str, Any], arm: str, metric: str) -> float:
    return float(payload["models"][arm]["overall"][metric])


def weighted_mean(payloads: list[dict[str, Any]], arm: str, metric: str) -> float:
    values = []
    weights = []
    for payload in payloads:
        values.append(metric_value(payload, arm, metric))
        weights.append(float(payload["models"][arm]["overall"]["agents"]))
    return float(np.average(values, weights=weights))


def paired_bootstrap(
    payloads: list[dict[str, Any]],
    *,
    control: str,
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        grouped[(payload["airport"], payload["regime"])].append(payload)
    cells = sorted(grouped)
    observed = []
    for payload in payloads:
        selected = metric_value(payload, SELECTED, metric)
        other = metric_value(payload, control, metric)
        observed.append(other - selected)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        cell_values = []
        for cell in cells:
            records = grouped[cell]
            sampled_records = rng.integers(0, len(records), size=len(records))
            seed_values = []
            for record_index in sampled_records:
                payload = records[int(record_index)]
                selected_dates = payload["models"][SELECTED]["per_date"]
                control_dates = payload["models"][control]["per_date"]
                dates = list(selected_dates)
                sampled_dates = rng.integers(0, len(dates), size=len(dates))
                values = []
                weights = []
                for date_index in sampled_dates:
                    date = dates[int(date_index)]
                    values.append(
                        float(control_dates[date][metric])
                        - float(selected_dates[date][metric])
                    )
                    weights.append(float(selected_dates[date]["agents"]))
                seed_values.append(float(np.average(values, weights=weights)))
            cell_values.append(float(np.mean(seed_values)))
        draws[index] = float(np.mean(cell_values))
    observed_array = np.asarray(observed, dtype=np.float64)
    return {
        "control": control,
        "candidate": SELECTED,
        "metric": metric,
        "effect_definition": "control_minus_selected; positive favors selected for lower-better metrics",
        "absolute_improvement_mean": float(observed_array.mean()),
        "ci95": np.quantile(draws, (0.025, 0.975)).tolist(),
        "improved_cell_count": int((observed_array > 0).sum()),
        "cells": int(observed_array.size),
        "replicates": replicates,
    }


def summarize_split(payloads: list[dict[str, Any]], split: str, replicates: int) -> dict[str, Any]:
    arms = tuple(payloads[0]["arms"])
    for payload in payloads:
        if tuple(payload["arms"]) != arms:
            raise RuntimeError("arm registry mismatch")
    comparisons = {
        control: {
            metric: paired_bootstrap(
                payloads,
                control=control,
                metric=metric,
                replicates=replicates,
                seed=20260917 + 1000 * CONTROLS.index(control) + REPORT_METRICS.index(metric),
            )
            for metric in LOWER_IS_BETTER
        }
        for control in CONTROLS
    }
    means = {
        arm: {
            metric: weighted_mean(payloads, arm, metric)
            for metric in REPORT_METRICS
            if metric in payloads[0]["models"][arm]["overall"]
        }
        for arm in arms
    }
    positive_counts = {
        control: {
            metric: {
                "positive_cells": int(
                    sum(
                        1
                        for payload in payloads
                        if metric_value(payload, control, metric)
                        - metric_value(payload, SELECTED, metric)
                        > 0
                    )
                ),
                "cells": len(payloads),
            }
            for metric in LOWER_IS_BETTER
        }
        for control in CONTROLS
    }
    runtime = {
        "files": len(payloads),
        "actors": int(sum(int(payload["actors"]) for payload in payloads)),
        "total_elapsed_seconds": float(
            sum(float(payload["runtime"]["total_elapsed_seconds"]) for payload in payloads)
        ),
        "total_inference_seconds": float(
            sum(float(payload["runtime"]["inference_seconds"]) for payload in payloads)
        ),
        "actors_per_second_median": float(
            np.median([float(payload["runtime"]["actors_per_second"]) for payload in payloads])
        ),
        "peak_allocated_gpu_bytes_max": int(
            max(int(payload["runtime"]["peak_allocated_gpu_bytes"]) for payload in payloads)
        ),
    }
    return {
        "split": split,
        "cells": len(payloads),
        "arms": list(arms),
        "means_actor_weighted": means,
        "selected_vs_controls": comparisons,
        "positive_cell_counts": positive_counts,
        "parameter_counts_first_cell": payloads[0]["parameter_counts"],
        "support_modes": payloads[0]["support_modes"],
        "probability_sum_max_abs_error": {
            arm: max(float(payload["probability_sum_max_abs_error"][arm]) for payload in payloads)
            for arm in arms
        },
        "runtime": runtime,
    }


def build_summary(input_root: Path, replicates: int) -> dict[str, Any]:
    payloads = {
        split: load_payloads(input_root, split) for split in ("development", "test")
    }
    return {
        "format_version": 1,
        "experiment_id": "measure2act_total_budget_two_ascent_summary_v1",
        "input_root": input_root.resolve().as_posix(),
        "input_root_sha256s": {
            split: [
                {"path": payload["_path"], "sha256": sha256(Path(payload["_path"]))}
                for payload in records
            ]
            for split, records in payloads.items()
        },
        "splits": {
            split: summarize_split(records, split, replicates)
            for split, records in payloads.items()
        },
        "claim_boundary": [
            "The two-ASCENT control uses a fixed next-seed pairing and 0.5/0.5 model mass.",
            "It is a total-budget capacity control with K=10 support, not a fixed-support attribution arm.",
            "Positive control-minus-selected effects favor selected MABPT for lower-better metrics.",
        ],
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Total-Budget Two-ASCENT Capacity Control",
        "",
        "This audit compares selected MABPT with a fixed two-source K=10 union control. The control has two source models, fixed next-seed pairing, and 0.5/0.5 model mass.",
        "",
        "## Primary Test Results",
    ]
    test = summary["splits"]["test"]
    for metric in ("energy_score", "energy_score_full_path", "endpoint_radial_nll", "endpoint_radial_brier", "minfde"):
        comp = test["selected_vs_controls"]["two_ascent_union10"][metric]
        lines.append(
            f"- {metric}: two_ascent_union10 - selected_mabpt = "
            f"{comp['absolute_improvement_mean']:.9f}, 95% CI "
            f"[{comp['ci95'][0]:.9f}, {comp['ci95'][1]:.9f}], "
            f"positive cells {comp['improved_cell_count']}/{comp['cells']}."
        )
    lines.extend(["", "## Actor-Weighted Test Means"])
    for arm in ("selected_mabpt", "two_ascent_union10", "ascent_native", "target_native"):
        means = test["means_actor_weighted"][arm]
        lines.append(
            f"- {arm}: Energy {means['energy_score']:.9f}; "
            f"Full-path Energy {means['energy_score_full_path']:.9f}; "
            f"minFDE {means['minfde']:.9f}; endpoint-radial NLL {means['endpoint_radial_nll']:.9f}."
        )
    lines.extend(
        [
            "",
            "## Integrity",
            f"- Test files: {test['runtime']['files']}; actors: {test['runtime']['actors']}.",
            f"- Support modes: {test['support_modes']}.",
            f"- Probability sum max error: {test['probability_sum_max_abs_error']}.",
            f"- Parameter counts first cell: {test['parameter_counts_first_cell']}.",
            f"- Runtime: total elapsed {test['runtime']['total_elapsed_seconds']:.2f}s; median throughput {test['runtime']['actors_per_second_median']:.2f} actors/s.",
            "",
            "## Claim Boundary",
            "- This closes the practical total-budget capacity-control gap for the local retrospective KAGC/KBTP setting.",
            "- It does not turn the two-ASCENT arm into a fixed-support probability attribution control because it uses K=10 support.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args()
    summary = build_summary(args.input_root, args.replicates)
    atomic_json(args.output_json, summary)
    write_markdown(args.output_md, summary)
    print(json.dumps({"output": args.output_json.as_posix()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
