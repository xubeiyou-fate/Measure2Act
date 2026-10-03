"""Aggregate EqMotion-support transfer audit outputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


LOWER_IS_BETTER = (
    "energy_score",
    "nll",
    "brier",
    "top1_ade",
    "top1_fde",
)
GEOMETRY = ("minade", "minfde")
PRIMARY_ARMS = (
    "hard_unweighted_prior",
    "hard_mass_aware_prior",
    "row_softmax_prior",
    "sinkhorn_prior",
    "gibbs_unweighted_prior",
    "gibbs_mass_aware_prior",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def weighted_metric(records: list[dict[str, Any]], arm: str, metric: str) -> float:
    numerator = 0.0
    denominator = 0
    for record in records:
        overall = record["models"][arm]["overall"]
        agents = int(overall["agents"])
        numerator += agents * float(overall[metric])
        denominator += agents
    if denominator <= 0:
        raise RuntimeError("empty aggregate")
    return numerator / denominator


def scoped(records: list[dict[str, Any]]) -> dict[str, Any]:
    arms = list(records[0]["models"])
    metrics = [metric for metric in LOWER_IS_BETTER if metric in records[0]["models"]["eqmotion_uniform"]["overall"]]
    geometry = [metric for metric in GEOMETRY if metric in records[0]["models"]["eqmotion_uniform"]["overall"]]
    totals = {
        arm: {metric: weighted_metric(records, arm, metric) for metric in (*metrics, *geometry)}
        for arm in arms
    }
    comparisons = {}
    for arm in PRIMARY_ARMS:
        comparisons[f"uniform_minus_{arm}"] = {
            metric: totals["eqmotion_uniform"][metric] - totals[arm][metric]
            for metric in metrics
        }
    sign = {}
    for arm in PRIMARY_ARMS:
        sign[arm] = {
            metric: {
                "positive_cells": sum(
                    1
                    for record in records
                    if float(record["models"]["eqmotion_uniform"]["overall"][metric])
                    - float(record["models"][arm]["overall"][metric])
                    > 0
                ),
                "cells": len(records),
            }
            for metric in metrics
        }
    return {
        "files": len(records),
        "actors": sum(int(record["actors"]) for record in records),
        "scenes": sum(int(record["scenes"]) for record in records),
        "arms": totals,
        "comparisons_lower_is_better_gain": comparisons,
        "positive_cell_counts": sign,
        "geometry_invariance_max": {
            metric: max(
                float(record["geometry_invariance_max_absolute_difference"][metric])
                for record in records
            )
            for metric in geometry
        },
        "probability_sum_error_max": max(
            float(record["integrity"]["probability_sum_error_max"]) for record in records
        ),
        "coordinate_alignment_max_absolute_difference": max(
            float(
                record["integrity"]["coordinate_alignment_max_absolute_difference"]
            )
            for record in records
        ),
        "elapsed_seconds": sum(
            float(record["runtime"]["total_elapsed_seconds"]) for record in records
        ),
    }


def bootstrap_units(records: list[dict[str, Any]], arm: str, metric: str) -> list[tuple[int, float]]:
    units: list[tuple[int, float]] = []
    for record in records:
        uniform = record["models"]["eqmotion_uniform"]["per_date"]
        candidate = record["models"][arm]["per_date"]
        for date, values in uniform.items():
            agents = int(values["agents"])
            delta = float(values[metric]) - float(candidate[date][metric])
            units.append((agents, delta))
    return units


def bootstrap_gain(
    records: list[dict[str, Any]],
    arm: str,
    metric: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    units = bootstrap_units(records, arm, metric)
    if not units:
        raise RuntimeError("empty bootstrap units")
    weights = np.asarray([item[0] for item in units], dtype=np.float64)
    deltas = np.asarray([item[1] for item in units], dtype=np.float64)
    observed = float((weights * deltas).sum() / weights.sum())
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.integers(0, len(units), size=len(units))
        sampled_weights = weights[sampled]
        draws[index] = float(
            (sampled_weights * deltas[sampled]).sum() / sampled_weights.sum()
        )
    return {
        "observed_actor_weighted_gain_uniform_minus_arm": observed,
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "units": len(units),
        "replicates": replicates,
    }


def build_summary(records: list[dict[str, Any]], *, replicates: int) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_split_airport: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_split[record["split"]].append(record)
        by_split_airport[(record["split"], record["airport"])].append(record)
    summary = {
        "format_version": 1,
        "experiment_id": "EqMotion_support_probability_transfer_aggregate_v1",
        "record_count": len(records),
        "splits": {split: scoped(items) for split, items in sorted(by_split.items())},
        "split_airport": {
            f"{split}_{airport}": scoped(items)
            for (split, airport), items in sorted(by_split_airport.items())
        },
        "bootstrap": {},
        "claim_boundary": [
            "External EqMotion supplies K=5 supports but no MABPT learned risk features.",
            "This aggregate supports only source-probability transport onto external supports, not a full learned Energy-KL external-generator interface.",
            "Uniform EqMotion is the geometry baseline; minADE/minFDE are invariant across same-support probability arms.",
        ],
    }
    for split, items in sorted(by_split.items()):
        split_result = {}
        for arm in PRIMARY_ARMS:
            split_result[arm] = {
                metric: bootstrap_gain(
                    items,
                    arm,
                    metric,
                    replicates=replicates,
                    seed=20260917 + hash((split, arm, metric)) % 100000,
                )
                for metric in ("energy_score", "nll", "brier")
            }
        summary["bootstrap"][split] = split_result
    return summary


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# EqMotion Support Transfer Audit",
        "",
        "This audit evaluates ASCENT source-probability transport onto frozen EqMotion K=5 supports. It is not a full learned MABPT Energy-KL run on EqMotion because EqMotion does not provide MABPT mode features or a support-conditional learned risk head.",
        "",
        "## Primary Test Results",
    ]
    test = summary["splits"]["test"]
    for arm in ("gibbs_unweighted_prior", "gibbs_mass_aware_prior", "hard_unweighted_prior"):
        comp = test["comparisons_lower_is_better_gain"][f"uniform_minus_{arm}"]
        boot = summary["bootstrap"]["test"][arm]["energy_score"]
        lines.append(
            f"- {arm}: Energy gain uniform-arm = {comp['energy_score']:.9f}, "
            f"95% CI [{boot['ci95'][0]:.9f}, {boot['ci95'][1]:.9f}], "
            f"positive cells {test['positive_cell_counts'][arm]['energy_score']['positive_cells']}/"
            f"{test['positive_cell_counts'][arm]['energy_score']['cells']}."
        )
    lines.extend(["", "## Airport Split"])
    for key in ("test_KAGC", "test_KBTP"):
        scope = summary["split_airport"][key]
        gain = scope["comparisons_lower_is_better_gain"][
            "uniform_minus_gibbs_unweighted_prior"
        ]["energy_score"]
        count = scope["positive_cell_counts"]["gibbs_unweighted_prior"]["energy_score"]
        lines.append(
            f"- {key}: Gibbs-unweighted Energy gain {gain:.9f}, "
            f"positive cells {count['positive_cells']}/{count['cells']}, "
            f"actors {scope['actors']}."
        )
    lines.extend(
        [
            "",
            "## Integrity",
            f"- Test files: {test['files']}; actors: {test['actors']}; scenes: {test['scenes']}.",
            f"- Geometry invariance max: {test['geometry_invariance_max']}.",
            f"- Probability sum max error: {test['probability_sum_error_max']:.3e}.",
            f"- Coordinate alignment max absolute difference: {test['coordinate_alignment_max_absolute_difference']:.3e}.",
            "",
            "## Claim Boundary",
            "- Use as supplementary evidence that finite-measure probability transport can be applied to an external generator's fixed support.",
            "- Do not claim arbitrary-generator learned MABPT portability or full Energy-KL projection on EqMotion without a retrained support-conditional risk scorer.",
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
    records = [
        load_json(path)
        for path in sorted(args.input_root.glob("*/*.json"))
        if "smoke" not in path.name
    ]
    if not records:
        raise RuntimeError("no EqMotion support-transfer records found")
    summary = build_summary(records, replicates=args.replicates)
    write_json(args.output_json, summary)
    write_markdown(args.output_md, summary)
    print(json.dumps({"records": len(records), "output": args.output_json.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
