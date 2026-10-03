"""Terminal identity, hash, gate, and data-boundary review for C133."""

from __future__ import annotations

import json
from pathlib import Path

from .model import VARIANT
from .protocol import load_protocol, sha256


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/decision_regret"
RUN_ROOT = ROOT / "runs/decision_regret"


def _read(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"missing C133 integrity input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run() -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_boundaries()
    protocol_hash = sha256(protocol.path)
    preflight = _read(ARTIFACT_ROOT / "preflight.json")
    p0 = _read(ARTIFACT_ROOT / "p0_decision.json")
    p1 = _read(ARTIFACT_ROOT / "p1_decision.json")
    replication_authorized = p1.get("decision") == "REPLICATION_AUTHORIZED"
    terminal = (
        _read(ARTIFACT_ROOT / "final_summary.json")
        if replication_authorized
        else p1
    )
    expected_folds = (0, 1, 2) if replication_authorized else (0,)
    run_checks = {}
    for fold in expected_folds:
        summary_path = RUN_ROOT / f"{VARIANT}_fold{fold}_seed42_formal/training_summary.json"
        summary = _read(summary_path)
        checkpoint = ROOT / str(summary["checkpoint"])
        run_checks[str(fold)] = {
            "identity": summary.get("variant") == VARIANT
            and summary.get("fold") == fold
            and summary.get("seed") == 42,
            "formal_complete_epoch20": summary.get("formal") is True
            and summary.get("complete") is True
            and summary.get("fixed_final_epoch") == 20,
            "protocol_match": summary.get("protocol_sha256") == protocol_hash,
            "checkpoint_match": checkpoint.is_file()
            and summary.get("checkpoint_sha256") == sha256(checkpoint),
            "boundaries": summary.get("locked_test_used") is False
            and summary.get("development_used") is False,
        }
    diagnostic_checks = {}
    for fold in (0, 1, 2):
        diagnostic = _read(ARTIFACT_ROOT / f"diagnostic_fold{fold}.json")
        diagnostic_checks[str(fold)] = (
            diagnostic.get("protocol_sha256") == protocol_hash
            and diagnostic.get("training_performed") is False
            and diagnostic.get("adaptive_selection_performed") is False
            and diagnostic.get("locked_test_used") is False
            and diagnostic.get("development_used") is False
        )
    terminal_decisions = {
        "C133_CLOSED_FOLD0_GATE_FAILED",
        "C133_CLOSED_REPLICATION_GATE_FAILED",
        "C133_SCREEN_PASSED",
    }
    checks = {
        "preflight_passed": preflight.get("passed") is True
        and preflight.get("protocol_sha256") == protocol_hash,
        "p0_authorized_fold0": p0.get("decision") == "FORMAL_FOLD0_AUTHORIZED"
        and p0.get("protocol_sha256") == protocol_hash,
        "all_zero_training_diagnostics_valid": all(diagnostic_checks.values()),
        "p1_protocol_match": p1.get("protocol_sha256") == protocol_hash,
        "terminal_decision_frozen": terminal.get("decision") in terminal_decisions,
        "expected_formal_runs_only": len(run_checks) == len(expected_folds),
        "all_run_checks_pass": all(all(values.values()) for values in run_checks.values()),
        "decision_boundaries": all(
            artifact.get("locked_test_used") is False
            and artifact.get("development_used") is False
            for artifact in (preflight, p0, p1, terminal)
        ),
    }
    result = {
        "format_version": 1,
        "cycle": protocol.payload["cycle"],
        "protocol_sha256": protocol_hash,
        "decision": terminal["decision"],
        "checks": checks,
        "diagnostics": diagnostic_checks,
        "runs": run_checks,
        "passed": all(checks.values()),
        "locked_test_used": False,
        "development_used": False,
    }
    if not result["passed"]:
        raise RuntimeError(f"C133 integrity review failed: {result}")
    output = ARTIFACT_ROOT / "integrity_review.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
