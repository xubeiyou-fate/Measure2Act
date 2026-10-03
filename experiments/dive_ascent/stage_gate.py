"""Apply the frozen C99 seed-42 metric and causal-structure gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.edfa_ascent.summarize import bootstrap_date_gain

from .gates import compare, gain
from .model import C99_VARIANTS
from .protocol import load_protocol, sha256


def _summary(run_root: Path, variant: str, seed: int = 42) -> dict[str, object]:
    path = run_root / f"{variant}_seed{seed}_formal" / "training_summary.json"
    if not path.is_file():
        raise RuntimeError(f"missing C99 seed-42 result: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("complete") is not True or result.get("formal") is not True:
        raise RuntimeError(f"incomplete C99 seed-42 result: {path}")
    return result


def run(run_root: Path, output: Path) -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_development_sealed()
    protocol_hash = sha256(protocol.path)
    summaries = {variant: _summary(run_root, variant) for variant in C99_VARIANTS}
    for variant, summary in summaries.items():
        if summary.get("protocol_sha256") != protocol_hash:
            raise RuntimeError(f"C99 protocol mismatch for {variant}")
        if summary.get("locked_test_used") is not False:
            raise RuntimeError(f"C99 seed-42 summary used locked test: {variant}")
        if int(summary.get("fixed_final_epoch", -1)) != int(protocol.payload["training"]["epochs"]):
            raise RuntimeError(f"C99 fixed-final-epoch mismatch for {variant}")
    metrics = {
        variant: summary["development_metrics"]["overall"]
        for variant, summary in summaries.items()
    }
    gates = protocol.payload["gates"]
    primary = compare(
        metrics[protocol.payload["screen"]["primary_baseline"]],
        metrics[protocol.payload["screen"]["primary_candidate"]],
        gates,
    )
    increments = {
        "a2_over_a1_minfde_gain": gain(
            metrics["a1_shared_positive"]["minfde"],
            metrics["a2_shared_decoupled"]["minfde"],
        ),
        "a4_over_a3_minfde_gain": gain(
            metrics["a3_scaled_decoupled"]["minfde"],
            metrics["a4_independent_random"]["minfde"],
        ),
        "a5_over_a4_minfde_gain": gain(
            metrics["a4_independent_random"]["minfde"],
            metrics["a5_dive"]["minfde"],
        ),
    }
    increment_checks = {
        "score_shielding_increment": increments["a2_over_a1_minfde_gain"]
        >= float(gates["a2_over_a1_minfde_gain_minimum"]),
        "independent_expert_increment": increments["a4_over_a3_minfde_gain"]
        >= float(gates["a4_over_a3_minfde_gain_minimum"]),
        "expert_birth_increment": increments["a5_over_a4_minfde_gain"]
        >= float(gates["a5_over_a4_minfde_gain_minimum"]),
    }
    split_history = summaries["a5_dive"].get("split_history", [])
    expected_epochs = [int(value) for value in protocol.payload["training"]["dac"]["split_before_epochs"]]
    dac_check = (
        [int(item["before_epoch"]) for item in split_history] == expected_epochs
        and [int(item["active_experts"]) for item in split_history] == [2, 3, 4, 5]
        and all(float(item["perturbation_l2"]) > 0.0 for item in split_history)
    )
    audit_path = protocol.repository_root / "artifacts/experiments/dive_ascent/gradient_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_complete = int(audit.get("processed_batches", -1)) == int(
        protocol.payload["gradient_audit"]["batches"]
    )
    passed = bool(
        primary["passed"]
        and all(increment_checks.values())
        and dac_check
        and audit_complete
    )
    date_bootstrap = bootstrap_date_gain(
        summaries["a0_shared_signed"]["development_metrics"]["date_metrics"],
        summaries["a5_dive"]["development_metrics"]["date_metrics"],
        seed=20260731,
    )
    result = {
        "format_version": 1,
        "cycle": protocol.payload["cycle"],
        "protocol_sha256": protocol_hash,
        "manifest_sha256": sha256(protocol.manifest_path),
        "seed": 42,
        "gradient_audit": {
            "complete": audit_complete,
            "mechanism_supported": bool(audit.get("mechanism_supported", False)),
            "passing_date_blocks": int(audit.get("passing_date_blocks", 0)),
            "role": "diagnostic_only_not_a_positive_score_shielding_claim",
        },
        "primary": primary,
        "causal_increments": increments,
        "causal_increment_checks": increment_checks,
        "dac_split_history": split_history,
        "dac_curriculum_complete": dac_check,
        "date_cluster_bootstrap_reported_not_selected": date_bootstrap,
        "metrics": metrics,
        "screen_passed": passed,
        "replication_authorized": passed,
        "locked_test_authorized": False,
        "locked_test_evaluations": 0,
        "decision": (
            "advance_C99_to_four_remaining_seed_pairs"
            if passed
            else "close_C99_after_seed42_joint_gate_failed"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("runs/dive_ascent"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/experiments/dive_ascent/seed42_gate.json"))
    args = parser.parse_args()
    result = run(args.run_root, args.output)
    print(json.dumps({
        "screen_passed": result["screen_passed"],
        "decision": result["decision"],
        "primary": result["primary"],
        "causal_increments": result["causal_increments"],
    }, indent=2))


if __name__ == "__main__":
    main()
