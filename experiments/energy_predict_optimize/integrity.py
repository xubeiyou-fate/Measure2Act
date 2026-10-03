"""Terminal C134 identity, boundary, gate, source, and checkpoint review."""

from __future__ import annotations

import json
from pathlib import Path

from .model import VARIANT
from .protocol import load_protocol, sha256


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/energy_predict_optimize"
RUN_ROOT = ROOT / "runs/energy_predict_optimize"


def _read(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"missing C134 integrity input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run() -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_boundaries()
    protocol_hash = sha256(protocol.path)
    paths = {
        "preflight": ARTIFACT_ROOT / "preflight.json",
        "p0_a": ARTIFACT_ROOT / "p0_a_decision.json",
        "p0_b": ARTIFACT_ROOT / "p0_b_decision.json",
        "preflight_p1": ARTIFACT_ROOT / "preflight_p1.json",
        "p1": ARTIFACT_ROOT / "p1_decision.json",
        "p0_b_design": ARTIFACT_ROOT / "p0_b_design.json",
        "p1_design": ARTIFACT_ROOT / "p1_design.json",
        "literature": ARTIFACT_ROOT / "literature_provenance.json",
    }
    artifacts = {name: _read(path) for name, path in paths.items()}
    summary_path = RUN_ROOT / f"{VARIANT}_fold0_seed42_formal/training_summary.json"
    summary = _read(summary_path)
    checkpoint_path = ROOT / str(summary["checkpoint"])
    failed_gates = sorted(
        name for name, passed in artifacts["p1"]["checks"].items() if not passed
    )
    formal_summaries = sorted(
        path.relative_to(ROOT).as_posix()
        for path in RUN_ROOT.glob(f"{VARIANT}_fold*_seed42_formal/training_summary.json")
    )
    invalid_evidence = {
        "p0_a_nll_floor": (
            ARTIFACT_ROOT / "p0_a_invalid_nll_floor_20260807.json",
            "a4921b105687d13944c26c85a2ae581e09881a9c7625477b66582d75c66f34e8",
        ),
        "p0_b_violation_mask": (
            ARTIFACT_ROOT / "logs/p0_b_invalid_violation_mask_20260807.log",
            "1fe79b16fa49d666bda3e2038eb71b7f2a647347d672f4c3acbf04c2ab021697",
        ),
        "p1_solver_gradient": (
            ARTIFACT_ROOT / "logs/p1_invalid_solver_gradient_20260807.log",
            "997cd76f56692d22a73a69f464cff278b91e53faf7fcfb10b21a35e801a17f49",
        ),
    }
    checks = {
        "protocol_boundary_checks_pass": True,
        "all_primary_artifacts_match_protocol": all(
            payload.get("protocol_sha256") == protocol_hash
            for name, payload in artifacts.items()
            if name not in {"literature"}
        ),
        "p0_a_passed_without_training_or_oracle_probabilities": artifacts["p0_a"].get(
            "decision"
        )
        == "P0_B_AUTHORIZED"
        and artifacts["p0_a"].get("training_performed") is False
        and artifacts["p0_a"].get("oracle_probabilities_used") is False,
        "p0_b_passed_diagnostic_only": artifacts["p0_b"].get("decision")
        == "P1_FOLD0_AUTHORIZED"
        and artifacts["p0_b"].get("diagnostic_target_used") is True
        and artifacts["p0_b"].get("target_used_for_deployable_inference") is False,
        "preflights_passed": artifacts["preflight"].get("passed") is True
        and artifacts["preflight_p1"].get("passed") is True,
        "formal_identity": summary.get("complete") is True
        and summary.get("formal") is True
        and summary.get("variant") == VARIANT
        and summary.get("fold") == 0
        and summary.get("seed") == 42
        and summary.get("fixed_final_epoch") == 20,
        "formal_protocol_and_design_match": summary.get("protocol_sha256")
        == protocol_hash
        and summary.get("p1_design_sha256") == sha256(paths["p1_design"]),
        "formal_checkpoint_hash_matches": checkpoint_path.is_file()
        and summary.get("checkpoint_sha256") == sha256(checkpoint_path),
        "formal_boundary_preserved": summary.get("locked_test_used") is False
        and summary.get("development_used") is False,
        "frozen_fold0_gate_closed_only_on_nll_and_brier": artifacts["p1"].get(
            "decision"
        )
        == "C134_CLOSED_FOLD0_GATE_FAILED"
        and failed_gates
        == ["brier_not_worse_than_C127_B0", "nll_not_worse_than_C127_B0"],
        "no_unauthorized_replication": formal_summaries
        == [summary_path.relative_to(ROOT).as_posix()],
        "invalid_runs_preserved": all(
            path.is_file() and sha256(path) == expected
            for path, expected in invalid_evidence.values()
        ),
        "implementation_amendments_present": all(
            (ARTIFACT_ROOT / name).is_file()
            for name in (
                "implementation_amendment.json",
                "p0_b_implementation_amendment.json",
                "p1_implementation_amendment.json",
            )
        ),
        "literature_provenance_present": paths["literature"].is_file(),
    }
    core_files = [
        ROOT / "experiments/energy_predict_optimize/protocol.json",
        ROOT / "experiments/energy_predict_optimize/solver.py",
        ROOT / "experiments/energy_predict_optimize/evaluation.py",
        ROOT / "experiments/energy_predict_optimize/physical_oracle.py",
        ROOT / "experiments/energy_predict_optimize/model.py",
        ROOT / "experiments/energy_predict_optimize/objective.py",
        ROOT / "experiments/energy_predict_optimize/train.py",
        ROOT / "experiments/energy_predict_optimize/summarize.py",
    ]
    result = {
        "format_version": 1,
        "cycle": protocol.payload["cycle"],
        "phase": "terminal_integrity_review",
        "protocol_sha256": protocol_hash,
        "decision": artifacts["p1"]["decision"],
        "artifact_sha256": {
            name: sha256(path) for name, path in paths.items()
        },
        "formal_summary": summary_path.relative_to(ROOT).as_posix(),
        "formal_summary_sha256": sha256(summary_path),
        "formal_checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
        "formal_checkpoint_sha256": sha256(checkpoint_path),
        "formal_summaries": formal_summaries,
        "failed_gates": failed_gates,
        "invalid_evidence": {
            name: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": expected,
            }
            for name, (path, expected) in invalid_evidence.items()
        },
        "core_source_sha256": {
            path.relative_to(ROOT).as_posix(): sha256(path) for path in core_files
        },
        "checks": checks,
        "passed": all(checks.values()),
        "locked_test_used": False,
        "development_used": False,
    }
    if not result["passed"]:
        raise RuntimeError(f"C134 integrity review failed: {checks}")
    output = ARTIFACT_ROOT / "integrity_review.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
