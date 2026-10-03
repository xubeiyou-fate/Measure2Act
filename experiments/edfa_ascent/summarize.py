"""Apply the preregistered three-seed C96 development gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .protocol import load_protocol


def gain(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / baseline


def load_summary(root: Path, variant: str, seed: int) -> dict:
    path = root / f"{variant}_seed{seed}_formal" / "training_summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def bootstrap_date_gain(
    baseline_dates: dict, candidate_dates: dict, seed: int, samples: int = 10000
) -> dict:
    dates = sorted(set(baseline_dates) & set(candidate_dates))
    values = np.asarray([
        gain(baseline_dates[date]["minfde"], candidate_dates[date]["minfde"])
        for date in dates
    ])
    rng = np.random.default_rng(seed)
    bootstrap = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return {
        "dates": len(dates),
        "mean_gain": float(values.mean()),
        "ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("runs/edfa_ascent"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/experiments/edfa_ascent/development_gate.json"))
    args = parser.parse_args()
    protocol = load_protocol()
    thresholds = protocol.payload["development_gates"]
    seeds = protocol.payload["seeds"]
    per_seed = {}
    all_pass = True
    for seed in seeds:
        a0 = load_summary(args.run_root, "a0", seed)["best_controls"]["predicted"]
        a1 = load_summary(args.run_root, "a1", seed)["best_controls"]["predicted"]
        a3_summary = load_summary(args.run_root, "a3", seed)
        a3 = a3_summary["best_controls"]["predicted"]
        shuffled = a3_summary["best_controls"]["shuffled_neighbors"]
        metrics = {
            "overall_minfde_gain": gain(a0["overall"]["minfde"], a3["overall"]["minfde"]),
            "multi_agent_minfde_gain": gain(a0["multi_agent"]["minfde"], a3["multi_agent"]["minfde"]),
            "multi_agent_minade_gain": gain(a0["multi_agent"]["minade"], a3["multi_agent"]["minade"]),
            "interactive_minfde_gain": gain(a0["interactive"]["minfde"], a3["interactive"]["minfde"]),
            "joint_scene_minfde_gain": gain(a0["joint_multi_scene"]["minfde"], a3["joint_multi_scene"]["minfde"]),
            "overall_p95_minfde_gain": gain(a0["overall"]["minfde_p95"], a3["overall"]["minfde_p95"]),
            "overall_energy_gain": gain(a0["overall"]["energy_score"], a3["overall"]["energy_score"]),
            "a3_vs_a1_multi_minfde_gain": gain(a1["multi_agent"]["minfde"], a3["multi_agent"]["minfde"]),
        }
        denominator = a0["overall"]["minfde"] - a3["overall"]["minfde"]
        metrics["placebo_gain_retention"] = (
            (a0["overall"]["minfde"] - shuffled["overall"]["minfde"]) / denominator
            if denominator > 0 else float("inf")
        )
        date_bootstrap = bootstrap_date_gain(
            a0["date_metrics"], a3["date_metrics"], seed=20260729 + seed
        )
        gates = {
            "overall_minfde": metrics["overall_minfde_gain"] >= thresholds["overall_minfde_relative_gain"],
            "multi_agent_minfde": metrics["multi_agent_minfde_gain"] >= thresholds["multi_agent_minfde_relative_gain"],
            "multi_agent_minade": metrics["multi_agent_minade_gain"] >= thresholds["multi_agent_minade_relative_gain"],
            "interactive_minfde": metrics["interactive_minfde_gain"] >= thresholds["interactive_minfde_relative_gain"],
            "joint_scene_minfde": metrics["joint_scene_minfde_gain"] >= thresholds["joint_scene_minfde_relative_gain"],
            "p95_nonworse": metrics["overall_p95_minfde_gain"] >= thresholds["overall_p95_minfde_relative_gain_minimum"],
            "energy": metrics["overall_energy_gain"] >= thresholds["overall_energy_relative_gain"],
            "placebo": metrics["placebo_gain_retention"] <= thresholds["placebo_gain_retention_maximum"],
            "beats_a1": metrics["a3_vs_a1_multi_minfde_gain"] > 0,
            "date_bootstrap": date_bootstrap["ci95"][0] > 0,
        }
        seed_pass = all(gates.values())
        all_pass = all_pass and seed_pass
        per_seed[str(seed)] = {
            "metrics": metrics,
            "date_cluster_bootstrap": date_bootstrap,
            "gates": gates,
            "passed": seed_pass,
        }
    result = {
        "format_version": 1,
        "cycle": "C96_EDFA_ASCENT",
        "seeds": seeds,
        "per_seed": per_seed,
        "all_development_gates_passed": all_pass,
        "locked_test_authorized": all_pass,
        "locked_test_evaluations": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
