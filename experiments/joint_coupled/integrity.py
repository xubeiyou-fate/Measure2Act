"""Independent identity and boundary review for completed C129 artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from .model import VARIANT
from .protocol import load_protocol, sha256


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/joint_coupled"
RUN_ROOT = ROOT / "runs/joint_coupled"


def read(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"missing integrity input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run() -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_boundaries()
    protocol_hash = sha256(protocol.path)
    preflight = read(ARTIFACT_ROOT / "preflight.json")
    p1 = read(ARTIFACT_ROOT / "p1_decision.json")
    final = read(ARTIFACT_ROOT / "final_summary.json")
    run_checks = {}
    for fold in (0, 1, 2):
        summary_path = RUN_ROOT / f"{VARIANT}_fold{fold}_seed42_formal/training_summary.json"
        summary = read(summary_path)
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
    checks = {
        "preflight_passed": preflight.get("passed") is True
        and preflight.get("protocol_sha256") == protocol_hash,
        "p1_authorized_replication": p1.get("decision") == "REPLICATION_AUTHORIZED",
        "final_decision_frozen_failure": final.get("decision")
        == "C129_CLOSED_REPLICATION_GATE_FAILED",
        "three_folds_present": final.get("available_folds") == [0, 1, 2],
        "primary_gate_not_substituted": final.get("aggregate", {}).get(
            "both_top1_better"
        )
        is False,
        "decision_boundaries": all(
            artifact.get("locked_test_used") is False
            and artifact.get("development_used") is False
            for artifact in (p1, final)
        ),
        "all_run_checks_pass": all(
            all(values.values()) for values in run_checks.values()
        ),
    }
    result = {
        "format_version": 1,
        "cycle": protocol.payload["cycle"],
        "protocol_sha256": protocol_hash,
        "decision": final["decision"],
        "checks": checks,
        "runs": run_checks,
        "passed": all(checks.values()),
        "locked_test_used": False,
        "development_used": False,
    }
    if not result["passed"]:
        raise RuntimeError(f"C129 integrity review failed: {result}")
    output = ARTIFACT_ROOT / "integrity_review.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
