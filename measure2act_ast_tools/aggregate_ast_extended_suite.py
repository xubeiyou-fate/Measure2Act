"""Aggregate the extended AST review suite."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


AIRPORTS = ("KAGC", "KBTP")
REGIMES = ("target_only", "full_finetune")
SEEDS = (42, 7, 123, 2024, 2026)
SELECTED = "selected_mabpt"
PRIMARY_CONTROLS = (
    "target_native",
    "wrong_source_actor_shift_gibbs_energy_kl",
)
SENSITIVITY_ARMS = (
    "temp0p5_gibbs_energy_kl",
    "temp2_gibbs_energy_kl",
    "risk0_energy_kl",
    "risk0p5_energy_kl",
    "risk2_energy_kl",
    "kl0p5_energy_kl",
    "kl2_energy_kl",
    "diversity0_energy_kl",
    "diversity2_energy_kl",
)
REPORT_METRICS = (
    "energy_score",
    "energy_score_full_path",
    "nll",
    "brier",
    "support_index_ece",
    "endpoint_grid_nll",
    "endpoint_grid_brier",
    "endpoint_grid_ece",
    "endpoint_radial_nll",
    "endpoint_radial_brier",
    "endpoint_radial_ece",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
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
    observed = []
    cells = sorted(grouped)
    for payload in payloads:
        selected = float(payload["models"][SELECTED]["overall"][metric])
        other = float(payload["models"][control]["overall"][metric])
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
                    values.append(float(control_dates[date][metric]) - float(selected_dates[date][metric]))
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
    }


def mean_over_payloads(payloads: list[dict[str, Any]], path: tuple[str, ...]) -> float:
    values = []
    weights = []
    for payload in payloads:
        node: Any = payload
        for key in path:
            node = node[key]
        values.append(float(node))
        weights.append(float(payload["actors"]))
    return float(np.average(values, weights=weights))


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
                seed=20260916 + 1000 * list(arms).index(control) + REPORT_METRICS.index(metric),
            )
            for metric in REPORT_METRICS
        }
        for control in (*PRIMARY_CONTROLS, *SENSITIVITY_ARMS)
    }
    selected_means = {
        metric: mean_over_payloads(payloads, ("models", SELECTED, "overall", metric))
        for metric in REPORT_METRICS
    }
    target_means = {
        metric: mean_over_payloads(payloads, ("models", "target_native", "overall", metric))
        for metric in REPORT_METRICS
    }
    risk = {
        key: mean_over_payloads(payloads, ("risk_head_diagnostics", key))
        for key in (
            "mse",
            "mae",
            "pearson",
            "spearman",
            "best_minus_worst_true_risk",
            "true_risk_predicted_best",
            "true_risk_predicted_worst",
        )
    }
    physical_keys = sorted(
        key for key in payloads[0]["physical_selected_mabpt"] if key != "agents"
    )
    physical = {
        key: mean_over_payloads(payloads, ("physical_selected_mabpt", key))
        for key in physical_keys
    }
    geometry = {
        metric: max(float(payload["geometry_invariance_max_absolute_difference"][metric]) for payload in payloads)
        for metric in ("minade", "minfde", "top1_ade", "top1_fde")
    }
    probability = {
        arm: max(float(payload["probability_sum_max_abs_error"][arm]) for payload in payloads)
        for arm in arms
    }
    runtime = {
        "files": len(payloads),
        "actors": int(sum(int(payload["actors"]) for payload in payloads)),
        "total_elapsed_seconds": float(sum(float(payload["runtime"]["total_elapsed_seconds"]) for payload in payloads)),
        "total_inference_seconds": float(sum(float(payload["runtime"]["inference_seconds"]) for payload in payloads)),
        "actors_per_second_median": float(np.median([float(payload["runtime"]["actors_per_second"]) for payload in payloads])),
        "peak_allocated_gpu_bytes_max": int(max(int(payload["runtime"]["peak_allocated_gpu_bytes"]) for payload in payloads)),
    }
    return {
        "split": split,
        "cells": len(payloads),
        "arms": list(arms),
        "selected_means": selected_means,
        "target_native_means": target_means,
        "selected_vs_controls": comparisons,
        "risk_head": risk,
        "physical_selected_mabpt": physical,
        "geometry_invariance_max": geometry,
        "probability_sum_max_abs_error": probability,
        "runtime": runtime,
    }


def maybe_file(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def external_assets() -> dict[str, Any]:
    return {
        "eqmotion_summary": maybe_file(Path("$LOCAL_WORKSPACE/_paper_migration_build_20260906_v5/MABPT_ASCENT_Paper_Migration_20260906_v5/01_论文写作/历史交接/17_CURRENT_EVIDENCE/summaries/eqmotion_five_seed_summary_v1.json")),
        "independent_gru_summary": maybe_file(Path("$LOCAL_WORKSPACE/_paper_migration_build_20260906_v5/MABPT_ASCENT_Paper_Migration_20260906_v5/03_核心实验结果/journal_e14_e20_20260817/e15/e15_summary_v1.json")),
        "capacity_controls": maybe_file(Path("$LOCAL_WORKSPACE/ast_experiment_closure_20260916/capacity_and_baseline_controls.md")),
        "blind_holdout": maybe_file(Path("$LOCAL_WORKSPACE/_paper_migration_build_20260906_v5/MABPT_ASCENT_Paper_Migration_20260906_v5/03_核心实验结果/journal_completion_20260906/publication_gap_closure_v1/blind_evaluation_summary_v1.json")),
        "external_validation_boundary": "No unused third-airport or prospective confirmation cohort was found locally; E11 is represented by the retained blind holdout assets plus an explicit limitation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args()
    result = {
        "format_version": 1,
        "experiment_id": "measure2act_ast_extended_suite_summary_v1",
        "input_root": args.input_root.resolve().as_posix(),
        "splits": {
            split: summarize_split(load_payloads(args.input_root, split), split, args.replicates)
            for split in ("development", "test")
        },
        "external_assets_for_E2_E5_E11": external_assets(),
        "estimand": "Equal airport-regime cell mean over paired seeds; date bootstrap uses actor weights within seed.",
        "claim_boundary": "Local retrospective AST extended diagnostics; E11 has no unused local third-airport/prospective cohort.",
    }
    atomic_json(args.output.resolve(), result)
    print(json.dumps({"output": args.output.resolve().as_posix(), "splits": list(result["splits"])}, indent=2))


if __name__ == "__main__":
    main()
