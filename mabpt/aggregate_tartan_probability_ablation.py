"""Aggregate the registered Tartan probability ablation by seed and date."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .aggregate_partc_seeds import hierarchical_paired_bootstrap
from .evaluate_tartan_probability_ablation import ARMS, SHARED_SUPPORT_ARMS


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("tartan_probability_ablation_protocol_v1.json")
INPUT_ROOT = ROOT / "artifacts/partc_two_dataset_20260812/probability_ablation_v1"
AIRPORTS = ("KAGC", "KBTP")
REGIMES = ("target_only", "full_finetune")
SEEDS = (42, 7, 123, 2024, 2026)
INFERENCE_METRICS = ("energy_score", "nll", "brier", "ece")
SUMMARY_METRICS = (
    "top1_ade",
    "top1_fde",
    "minade",
    "minfde",
    *INFERENCE_METRICS,
    "oracle_ade_rank1",
    "oracle_fde_rank1",
    "effective_modes",
)
CONTROL = "gibbs_unweighted_energy_kl"
CANDIDATE = "gibbs_mass_aware_energy_kl"
# Development files were generated before the evaluator provenance field was
# added. Preserve their exact historical evaluator digest for aggregation.
LEGACY_EVALUATOR_SHA256 = "7ca7d94dca7e521d9271e80ceb7626a23447d54e0257eff969c9ae0045d8929c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"probability-ablation aggregate refuses to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _input_path(root: Path, split: str, airport: str, regime: str, seed: int) -> Path:
    # The evaluator's immutable output namespace is the registered v1 path.
    # Accept the older confirmatory suffix only for previously generated runs.
    primary = root / split / f"{airport}_{regime}_seed{seed}_{split}_v1.json"
    if primary.is_file():
        return primary
    return root / split / f"{airport}_{regime}_seed{seed}_{split}_confirmatory_v1.json"


def _load_payloads(
    *, root: Path, split: str, airport: str, regime: str
) -> tuple[dict[int, dict[str, Any]], list[dict[str, object]]]:
    payloads = {}
    inputs = []
    for seed in SEEDS:
        path = _input_path(root, split, airport, regime, seed)
        payload = json.loads(path.read_text(encoding="utf-8"))
        actual = (
            payload.get("airport"),
            payload.get("regime"),
            int(payload.get("seed", -1)),
            payload.get("split"),
        )
        if actual != (airport, regime, seed, split):
            raise RuntimeError(f"probability-ablation identity mismatch: {path}")
        if set(payload.get("models", {})) != set(ARMS):
            raise RuntimeError(f"probability-ablation arm grid incomplete: {path}")
        if payload.get("integrity", {}).get("target_in_probability_forward") is not False:
            raise RuntimeError(f"future target entered probability forward: {path}")
        if payload.get("integrity", {}).get("union10_excluded_from_component_attribution") is not True:
            raise RuntimeError(f"union-10 claim boundary missing: {path}")
        evaluator = payload.get("inputs", {}).get("evaluator", {})
        evaluator_path = ROOT / evaluator.get("path", "missing")
        if evaluator:
            if not evaluator_path.is_file() or (
                evaluator.get("sha256") != sha256(evaluator_path)
                and evaluator.get("sha256") != LEGACY_EVALUATOR_SHA256
            ):
                raise RuntimeError(f"evaluator provenance mismatch: {path}")
        elif payload.get("format_version") == 1:
            # These development outputs were generated before the provenance
            # field was added; their checkpoint/protocol hashes remain bound.
            evaluator = {"legacy_output_without_evaluator_hash": True}
        else:
            raise RuntimeError(f"evaluator provenance missing: {path}")
        if split == "development":
            expected_evidence = "development_only_operator_selection"
            if payload.get("inputs", {}).get("selection_receipt") is not None:
                raise RuntimeError(f"development payload unexpectedly depends on selection receipt: {path}")
        else:
            expected_evidence = "locked_retrospective_test_post_main_analysis"
            if not payload.get("inputs", {}).get("selection_receipt"):
                raise RuntimeError(f"test payload lacks development selection receipt: {path}")
        if payload.get("evidence_class") != expected_evidence:
            raise RuntimeError(f"probability-ablation evidence class mismatch: {path}")
        payloads[seed] = payload
        inputs.append({
            "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    return payloads, inputs


def _seed_scalars(
    payloads: dict[int, dict[str, Any]], arm: str
) -> dict[str, object]:
    result = {}
    for metric in SUMMARY_METRICS:
        values = np.asarray(
            [payloads[seed]["models"][arm]["overall"][metric] for seed in SEEDS],
            dtype=np.float64,
        )
        result[metric] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "values_by_seed": {
                str(seed): float(value) for seed, value in zip(SEEDS, values, strict=True)
            },
        }
    return result


def aggregate_cell(
    airport: str,
    regime: str,
    split: str,
    *,
    input_root: Path = INPUT_ROOT,
    replicates: int = 10000,
) -> dict[str, object]:
    payloads, inputs = _load_payloads(
        root=input_root, split=split, airport=airport, regime=regime
    )
    comparisons = {}
    reference_dates = {
        seed: payloads[seed]["models"][CONTROL]["per_date"] for seed in SEEDS
    }
    for arm in ARMS:
        if arm == CONTROL:
            continue
        candidate_dates = {
            seed: payloads[seed]["models"][arm]["per_date"] for seed in SEEDS
        }
        comparisons[arm] = {
            metric: hierarchical_paired_bootstrap(
                reference_dates,
                candidate_dates,
                metric,
                replicates=replicates,
                random_seed=20260813 + 100 * list(ARMS).index(arm) + index,
                count_key="agents",
            )
            for index, metric in enumerate(INFERENCE_METRICS)
        }
    geometry_max = {
        metric: max(
            abs(
                float(payloads[seed]["models"][arm]["overall"][metric])
                - float(payloads[seed]["models"]["target_native"]["overall"][metric])
            )
            for seed in SEEDS
            for arm in SHARED_SUPPORT_ARMS
        )
        for metric in ("top1_ade", "top1_fde", "minade", "minfde")
    }
    if any(value > 1e-12 for value in geometry_max.values()):
        raise RuntimeError("shared-support geometry differs in aggregate")
    return {
        "format_version": 1,
        "experiment_id": "Tartan_probability_ablation_five_seed_cell_v1",
        "evidence_class": (
            "development_only_operator_selection"
            if split == "development"
            else "locked_retrospective_test_post_main_analysis"
        ),
        "airport": airport,
        "regime": regime,
        "split": split,
        "seeds": list(SEEDS),
        "models": {arm: _seed_scalars(payloads, arm) for arm in ARMS},
        "paired_hierarchical_seed_date_bootstrap_vs_unweighted": comparisons,
        "comparison_effect_orientation": (
            "unweighted_exact_Gibbs_plus_EnergyKL_minus_arm; positive favors the named arm"
        ),
        "shared_support_geometry_max_absolute_difference": geometry_max,
        "capacity_control_arm": "source_target_native_union10",
        "inputs": inputs,
        "protocol": {
            "path": PROTOCOL.relative_to(ROOT).as_posix(),
            "sha256": sha256(PROTOCOL),
        },
        "integrity": {
            "date_nested_within_seed_bootstrap": True,
            "matched_seed_pairing": True,
            "shared_support_geometry_exactly_invariant": True,
            "union10_excluded_from_component_attribution": True,
            "test_used_for_operator_selection": False,
        },
    }


def _weighted_date_mean(values: dict[str, Any], metric: str, draws: np.ndarray) -> float:
    dates = sorted(values)
    sums = np.asarray(
        [float(values[date][metric]) * int(values[date]["agents"]) for date in dates],
        dtype=np.float64,
    )
    counts = np.asarray(
        [int(values[date]["agents"]) for date in dates], dtype=np.float64
    )
    return float(sums[draws].sum() / counts[draws].sum())


def pooled_hierarchical_bootstrap(
    cells: dict[str, dict[int, dict[str, Any]]],
    metric: str,
    *,
    replicates: int,
    random_seed: int,
) -> dict[str, object]:
    """Pair global seed draws and resample dates inside each airport-regime cell."""
    cell_names = sorted(cells)
    point_by_cell_seed = {}
    for cell in cell_names:
        for seed in SEEDS:
            control = cells[cell][seed]["models"][CONTROL]["per_date"]
            candidate = cells[cell][seed]["models"][CANDIDATE]["per_date"]
            if set(control) != set(candidate) or not control:
                raise RuntimeError(f"paired date sets differ for {cell}, seed {seed}")
            if any(
                int(control[date]["agents"]) != int(candidate[date]["agents"])
                for date in control
            ):
                raise RuntimeError(f"paired date actor counts differ for {cell}, seed {seed}")
            draws = np.arange(len(control))
            point_by_cell_seed[(cell, seed)] = (
                _weighted_date_mean(control, metric, draws)
                - _weighted_date_mean(candidate, metric, draws)
            )
    rng = np.random.default_rng(random_seed)
    samples = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected_seed_positions = rng.integers(0, len(SEEDS), len(SEEDS))
        effects = []
        for position in selected_seed_positions:
            seed = SEEDS[int(position)]
            for cell in cell_names:
                control = cells[cell][seed]["models"][CONTROL]["per_date"]
                candidate = cells[cell][seed]["models"][CANDIDATE]["per_date"]
                draws = rng.integers(0, len(control), len(control))
                effects.append(
                    _weighted_date_mean(control, metric, draws)
                    - _weighted_date_mean(candidate, metric, draws)
                )
        samples[replicate] = float(np.mean(effects))
    point = float(np.mean(list(point_by_cell_seed.values())))
    return {
        "effect_definition": "unweighted_exact_Gibbs_plus_EnergyKL_minus_mass_aware_exact_Gibbs_plus_EnergyKL; positive favors mass-aware",
        "estimand": "equal-weight mean over airport-regime cells and paired seeds; actor-weighted within calendar date",
        "cells": cell_names,
        "seeds": list(SEEDS),
        "replicates": replicates,
        "absolute_gain": point,
        "per_cell_seed_absolute_gain": {
            f"{cell}__seed{seed}": float(value)
            for (cell, seed), value in sorted(point_by_cell_seed.items())
        },
        "ci95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
        "raw_one_sided_p_mass_aware_not_better": (
            int((samples <= 0).sum()) + 1
        ) / (replicates + 1),
        "probability_within_practical_equivalence_margin": None,
    }


def aggregate_pooled(
    split: str,
    *,
    input_root: Path = INPUT_ROOT,
    replicates: int = 10000,
) -> dict[str, object]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    margin = float(
        protocol["shared_support_primary_comparison"][
            "practical_equivalence_margin_energy_score"
        ]
    )
    cells: dict[str, dict[int, dict[str, Any]]] = {}
    inputs = []
    for airport in AIRPORTS:
        for regime in REGIMES:
            payloads, cell_inputs = _load_payloads(
                root=input_root, split=split, airport=airport, regime=regime
            )
            cells[f"{airport}__{regime}"] = payloads
            inputs.extend(cell_inputs)
    effects = {
        metric: pooled_hierarchical_bootstrap(
            cells,
            metric,
            replicates=replicates,
            random_seed=20260813 + index,
        )
        for index, metric in enumerate(INFERENCE_METRICS)
    }
    energy = effects["energy_score"]
    lower, upper = energy["ci95"]
    gain = float(energy["absolute_gain"])
    energy["practical_equivalence_margin"] = margin
    energy["ci_entirely_within_equivalence_bounds"] = (
        lower >= -margin and upper <= margin
    )
    if lower > 0 and gain >= margin:
        decision = {
            "selected_arm": CANDIDATE,
            "algorithm_identity": "Mass-Aware Exact Gibbs Permutation Transport with Energy-KL Projection",
            "mass_aware_independent_contribution_established": True,
            "reason": "Pooled development CI is above zero and the point gain reaches the registered practical margin.",
        }
    else:
        if upper < 0:
            classification = "mass_aware_harmful"
        elif lower >= -margin and upper <= margin:
            classification = "practically_equivalent"
        else:
            classification = "mass_aware_advantage_not_established"
        decision = {
            "selected_arm": CONTROL,
            "algorithm_identity": "Exact Gibbs Permutation Transport with Energy-KL Projection",
            "mass_aware_independent_contribution_established": False,
            "classification": classification,
            "reason": "The registered development rule for retaining Mass-Aware as the leading identity was not satisfied.",
        }
    if split == "test":
        receipt = json.loads(
            next(
                path for path in (
                    input_root / "development_selection_receipt_final_v1.json",
                    input_root / "development_selection_receipt_v1.json",
                )
                if path.is_file()
            ).read_text(encoding="utf-8")
        )
        decision = {
            "frozen_selected_arm": receipt["selected_arm"],
            "frozen_algorithm_identity": receipt["algorithm_identity"],
            "test_did_not_change_selection": True,
            "descriptive_test_classification_only": decision,
        }
    return {
        "format_version": 1,
        "experiment_id": "Tartan_probability_ablation_pooled_v1",
        "evidence_class": (
            "development_only_operator_selection"
            if split == "development"
            else "locked_retrospective_test_post_main_analysis"
        ),
        "split": split,
        "seeds": list(SEEDS),
        "cells": sorted(cells),
        "primary_comparison": {
            "control": CONTROL,
            "candidate": CANDIDATE,
        },
        "pooled_hierarchical_seed_cell_date_bootstrap": effects,
        "registered_decision": decision,
        "inputs": inputs,
        "protocol": {
            "path": PROTOCOL.relative_to(ROOT).as_posix(),
            "sha256": sha256(PROTOCOL),
        },
        "integrity": {
            "global_seed_draws_paired_across_cells": True,
            "dates_resampled_within_seed_and_cell": True,
            "airport_regime_cells_equally_weighted": True,
            "test_used_for_operator_selection": False,
            "union10_excluded_from_component_attribution": True,
        },
        "claim_boundaries": protocol["claim_boundaries"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("development", "test"), required=True)
    parser.add_argument("--airport", choices=AIRPORTS)
    parser.add_argument("--regime", choices=REGIMES)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.airport is None) != (args.regime is None):
        parser.error("--airport and --regime must be provided together")
    if args.airport is None:
        result = aggregate_pooled(
            args.split, input_root=args.input_root, replicates=args.replicates
        )
    else:
        result = aggregate_cell(
            args.airport,
            args.regime,
            args.split,
            input_root=args.input_root,
            replicates=args.replicates,
        )
    atomic_json(args.output.resolve(), result)
    print(json.dumps({
        "output": args.output.resolve().as_posix(),
        "split": result["split"],
        "decision": result.get("registered_decision"),
    }, indent=2))


if __name__ == "__main__":
    main()
