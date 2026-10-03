"""Apply the preregistered seed-42 stop/go gate before C96 expansion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .protocol import load_protocol
from .summarize import bootstrap_date_gain, gain


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root", type=Path, default=Path("runs/edfa_ascent")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/experiments/edfa_ascent/seed42_stage_gate.json"),
    )
    args = parser.parse_args()
    protocol = load_protocol()
    thresholds = protocol.payload["development_gates"]
    seed = 42
    baseline_summary = json.loads(
        (
            args.run_root / f"a0_seed{seed}_formal" / "training_summary.json"
        ).read_text(encoding="utf-8")
    )
    method_summary = json.loads(
        (
            args.run_root / f"a3_seed{seed}_formal" / "training_summary.json"
        ).read_text(encoding="utf-8")
    )
    baseline = baseline_summary["best_controls"]["predicted"]
    method = method_summary["best_controls"]["predicted"]
    shuffled = method_summary["best_controls"]["shuffled_neighbors"]
    oracle = method_summary["best_controls"]["oracle_graph"]
    metrics = {
        "overall_minfde_gain": gain(
            baseline["overall"]["minfde"], method["overall"]["minfde"]
        ),
        "multi_agent_minfde_gain": gain(
            baseline["multi_agent"]["minfde"], method["multi_agent"]["minfde"]
        ),
        "multi_agent_minade_gain": gain(
            baseline["multi_agent"]["minade"], method["multi_agent"]["minade"]
        ),
        "interactive_minfde_gain": gain(
            baseline["interactive"]["minfde"], method["interactive"]["minfde"]
        ),
        "joint_scene_minfde_gain": gain(
            baseline["joint_multi_scene"]["minfde"],
            method["joint_multi_scene"]["minfde"],
        ),
        "overall_p95_minfde_gain": gain(
            baseline["overall"]["minfde_p95"], method["overall"]["minfde_p95"]
        ),
        "overall_energy_gain": gain(
            baseline["overall"]["energy_score"], method["overall"]["energy_score"]
        ),
    }
    denominator = baseline["overall"]["minfde"] - method["overall"]["minfde"]
    metrics["placebo_gain_retention"] = (
        (baseline["overall"]["minfde"] - shuffled["overall"]["minfde"])
        / denominator
        if denominator > 0
        else None
    )
    date_bootstrap = bootstrap_date_gain(
        baseline["date_metrics"], method["date_metrics"], seed=20260729 + seed
    )
    gates = {
        "overall_minfde": metrics["overall_minfde_gain"]
        >= thresholds["overall_minfde_relative_gain"],
        "multi_agent_minfde": metrics["multi_agent_minfde_gain"]
        >= thresholds["multi_agent_minfde_relative_gain"],
        "multi_agent_minade": metrics["multi_agent_minade_gain"]
        >= thresholds["multi_agent_minade_relative_gain"],
        "interactive_minfde": metrics["interactive_minfde_gain"]
        >= thresholds["interactive_minfde_relative_gain"],
        "joint_scene_minfde": metrics["joint_scene_minfde_gain"]
        >= thresholds["joint_scene_minfde_relative_gain"],
        "p95_nonworse": metrics["overall_p95_minfde_gain"]
        >= thresholds["overall_p95_minfde_relative_gain_minimum"],
        "energy": metrics["overall_energy_gain"]
        >= thresholds["overall_energy_relative_gain"],
        "placebo": metrics["placebo_gain_retention"] is not None
        and metrics["placebo_gain_retention"]
        <= thresholds["placebo_gain_retention_maximum"],
        "date_bootstrap": date_bootstrap["ci95"][0] > 0,
    }
    passed = all(gates.values())
    result = {
        "format_version": 1,
        "cycle": "C96_EDFA_ASCENT",
        "seed": seed,
        "best_epoch": method_summary["best_epoch"],
        "best_checkpoint_sha256": method_summary["best_checkpoint_sha256"],
        "metrics": metrics,
        "date_cluster_bootstrap": date_bootstrap,
        "gates": gates,
        "seed42_stage_passed": passed,
        "followup_experiments_authorized": passed,
        "locked_test_authorized": False,
        "locked_test_evaluations": 0,
        "controls": {
            "baseline": baseline,
            "predicted": method,
            "shuffled_neighbors": shuffled,
            "oracle_graph": oracle,
        },
        "stage_decision": (
            "continue_to_a1_a2_and_remaining_seeds"
            if passed
            else "stop_before_a1_a2_remaining_seeds_and_locked_test"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
