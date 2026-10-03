"""Summarize C127 multi-seed development replication and freeze its gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .protocol import load_protocol, sha256
from .summarize import compare, load_result


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/metric_exact"
ARTIFACT_ROOT = ROOT / "artifacts/metric_exact"
SEEDS = (42, 7, 123, 2024, 2026)
CONTROLS = ("B0_signed_coupled", "B2_decoupled_original")


def path_for(variant: str, seed: int) -> Path:
    return RUN_ROOT / (
        f"P3_{variant}_all_train_seed{seed}_formal/training_summary.json"
    )


def hierarchical_bootstrap(
    control: dict[int, dict[str, object]],
    candidate: dict[int, dict[str, object]],
    metric: str,
    *,
    replicates: int = 10000,
    seed: int = 127,
) -> dict[str, object]:
    seeds = sorted(control)
    if seeds != sorted(candidate):
        raise RuntimeError("C127 P3 seed sets differ")
    rng = np.random.default_rng(seed)

    def seed_mean(values: dict[str, object], selected: np.ndarray) -> float:
        dates = sorted(values)
        sums = np.asarray(
            [values[date][metric] * values[date]["actors"] for date in dates],
            dtype=np.float64,
        )
        counts = np.asarray(
            [values[date]["actors"] for date in dates], dtype=np.float64
        )
        return float(sums[selected].sum() / counts[selected].sum())

    point_gains = []
    for value in seeds:
        control_dates = control[value]
        candidate_dates = candidate[value]
        if set(control_dates) != set(candidate_dates) or len(control_dates) != 11:
            raise RuntimeError("C127 P3 development date metrics are incomplete")
        all_dates = np.arange(len(control_dates))
        point_gains.append(
            seed_mean(control_dates, all_dates)
            - seed_mean(candidate_dates, all_dates)
        )

    gains = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected_seeds = rng.integers(0, len(seeds), len(seeds))
        sampled_gains = []
        for seed_position in selected_seeds:
            value = seeds[int(seed_position)]
            dates = len(control[value])
            selected_dates = rng.integers(0, dates, dates)
            sampled_gains.append(
                seed_mean(control[value], selected_dates)
                - seed_mean(candidate[value], selected_dates)
            )
        gains[replicate] = float(np.mean(sampled_gains))
    return {
        "seeds": seeds,
        "dates_per_seed": 11,
        "replicates": replicates,
        "point_absolute_gain": float(np.mean(point_gains)),
        "per_seed_absolute_gain": point_gains,
        "ci95": [float(value) for value in np.quantile(gains, [0.025, 0.975])],
    }


def aggregate(results: dict[int, dict[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for metric in ("minade", "minfde", "minfde_p95", "energy_score", "tail_minfde"):
        values = np.asarray(
            [result["validation_metrics"]["overall"][metric] for result in results.values()],
            dtype=np.float64,
        )
        output[metric] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "values": values.tolist(),
        }
    return output


def build_summary(
    results: dict[str, dict[int, dict[str, object]]], candidate: str
) -> dict[str, object]:
    comparisons = {
        str(seed): compare(
            results["B2_decoupled_original"][seed], results[candidate][seed]
        )
        for seed in SEEDS
    }
    same_direction_seeds = sum(
        value["both_point_estimates_improve"] for value in comparisons.values()
    )
    candidate_dates = {
        seed: results[candidate][seed]["validation_metrics"]["date_metrics"]
        for seed in SEEDS
    }
    b2_dates = {
        seed: results["B2_decoupled_original"][seed]["validation_metrics"]["date_metrics"]
        for seed in SEEDS
    }
    bootstrap = {
        metric: hierarchical_bootstrap(b2_dates, candidate_dates, metric)
        for metric in ("minade", "minfde")
    }
    aggregates = {variant: aggregate(values) for variant, values in results.items()}
    candidate_aggregate = aggregates[candidate]
    official = {"minade": 0.275548, "minfde": 0.477318}
    gates = {
        "both_metrics_better_than_B2_in_at_least_four_of_five_seeds": (
            same_direction_seeds >= 4
        ),
        "candidate_mean_better_than_official_epoch12_for_both": all(
            candidate_aggregate[metric]["mean"] < official[metric]
            for metric in ("minade", "minfde")
        ),
        "hierarchical_date_by_seed_bootstrap_ci_lower_bound_positive_for_both": all(
            bootstrap[metric]["ci95"][0] > 0
            for metric in ("minade", "minfde")
        ),
    }
    passed = all(gates.values())
    return {
        "format_version": 1,
        "cycle": "C127_metric_exact_score_isolated_ascent",
        "phase": "P3",
        "candidate": candidate,
        "per_seed_candidate_vs_B2": comparisons,
        "same_direction_seeds_vs_B2": same_direction_seeds,
        "aggregates": aggregates,
        "official_epoch12": official,
        "hierarchical_bootstrap_vs_B2": bootstrap,
        "gates": gates,
        "passed": passed,
        "decision": (
            "LOCKED_TEST_AUTHORIZED" if passed else "CLOSE_C127_ON_DEVELOPMENT"
        ),
        "locked_test_used": False,
    }


def run(output: Path, gate_output: Path) -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_boundaries()
    p2_path = ARTIFACT_ROOT / "p2_summary.json"
    if not p2_path.is_file():
        raise RuntimeError("C127 P2 summary is required before P3 summary")
    p2 = json.loads(p2_path.read_text(encoding="utf-8"))
    candidate = p2.get("P3_selected_exact_candidate")
    if p2.get("decision") != "P3_AUTHORIZED" or not candidate:
        raise RuntimeError("C127 P2 did not authorize a P3 candidate")
    variants = (*CONTROLS, candidate)
    results = {
        variant: {seed: load_result(path_for(variant, seed)) for seed in SEEDS}
        for variant in variants
    }
    protocol_hash = sha256(protocol.path)
    for values in results.values():
        for result in values.values():
            if result["protocol_sha256"] != protocol_hash:
                raise RuntimeError("C127 P3 result protocol hash mismatch")
    summary = build_summary(results, candidate)
    summary["protocol_sha256"] = protocol_hash
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    gate = {
        "format_version": 1,
        "cycle": protocol.payload["cycle"],
        "protocol_sha256": protocol_hash,
        "candidate": candidate,
        "P1_passed": True,
        "P2_passed": True,
        "P3_gates": summary["gates"],
        "passed": summary["passed"],
        "decision": summary["decision"],
        "checkpoint_policy": "all 15 frozen P3 checkpoints in one locked-test event",
        "locked_test_used": False,
    }
    gate_output.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ARTIFACT_ROOT / "p3_summary.json"
    )
    parser.add_argument(
        "--gate-output",
        type=Path,
        default=ARTIFACT_ROOT / "development_gate.json",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.gate_output), indent=2))


if __name__ == "__main__":
    main()
