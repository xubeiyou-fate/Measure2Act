"""Apply the frozen C134 fold0 gate to E1 and matched controls."""

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
        raise RuntimeError(f"missing C134 gate input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _control(protocol, family: str) -> tuple[dict[str, object], Path]:
    entry = protocol.payload["controls"][family]
    path = ROOT / str(entry["fold0_summary"])
    if sha256(path) != str(entry["fold0_summary_sha256"]):
        raise RuntimeError(f"C134 {family} gate summary hash mismatch")
    return _read(path), path


def summarize() -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_boundaries()
    protocol_hash = sha256(protocol.path)
    candidate_path = (
        RUN_ROOT / f"{VARIANT}_fold0_seed42_formal/training_summary.json"
    )
    candidate_summary = _read(candidate_path)
    design_path = ARTIFACT_ROOT / "p1_design.json"
    if (
        candidate_summary.get("complete") is not True
        or candidate_summary.get("formal") is not True
        or candidate_summary.get("variant") != VARIANT
        or candidate_summary.get("fold") != 0
        or candidate_summary.get("seed") != 42
        or candidate_summary.get("fixed_final_epoch") != 20
        or candidate_summary.get("protocol_sha256") != protocol_hash
        or candidate_summary.get("p1_design_sha256") != sha256(design_path)
        or candidate_summary.get("locked_test_used") is not False
        or candidate_summary.get("development_used") is not False
    ):
        raise RuntimeError("C134 E1 formal summary identity mismatch")
    checkpoint = ROOT / str(candidate_summary["checkpoint"])
    if candidate_summary.get("checkpoint_sha256") != sha256(checkpoint):
        raise RuntimeError("C134 E1 checkpoint hash mismatch")
    controls = {}
    for family in ("C127_B0", "C129_J1", "C131_R2", "C133_D1"):
        summary, path = _control(protocol, family)
        controls[family] = {
            "summary": path.relative_to(ROOT).as_posix(),
            "summary_sha256": sha256(path),
            "metrics": summary["validation_metrics"]["overall"],
        }
    candidate = candidate_summary["validation_metrics"]["overall"]
    b0 = controls["C127_B0"]["metrics"]
    c129 = controls["C129_J1"]["metrics"]
    c131 = controls["C131_R2"]["metrics"]
    gate = protocol.payload["p1"]["fold0_gate"]
    checks = {
        "top1_ade_gain_vs_C131_at_least_1pct": (
            (float(c131["top1_ade"]) - float(candidate["top1_ade"]))
            / float(c131["top1_ade"])
            >= float(gate["top1_ade_relative_gain_vs_C131_at_least"])
        ),
        "top1_fde_gain_vs_C131_at_least_1pct": (
            (float(c131["top1_fde"]) - float(candidate["top1_fde"]))
            / float(c131["top1_fde"])
            >= float(gate["top1_fde_relative_gain_vs_C131_at_least"])
        ),
        "energy_not_worse_than_C131": float(candidate["energy_score"])
        <= float(c131["energy_score"])
        * float(gate["energy_not_worse_than_C131_factor"]),
        "minade_within_C129_guard": float(candidate["minade"])
        <= float(c129["minade"])
        * float(gate["minade_not_worse_than_C129_factor"]),
        "minfde_within_C129_guard": float(candidate["minfde"])
        <= float(c129["minfde"])
        * float(gate["minfde_not_worse_than_C129_factor"]),
        "minfde_p95_within_C129_guard": float(candidate["minfde_p95"])
        <= float(c129["minfde_p95"])
        * float(gate["minfde_p95_not_worse_than_C129_factor"]),
        "tail_minfde_within_C129_guard": float(candidate["tail_minfde"])
        <= float(c129["tail_minfde"])
        * float(gate["tail_minfde_not_worse_than_C129_factor"]),
        "nll_not_worse_than_C127_B0": float(candidate["nll"])
        <= float(b0["nll"]) * float(gate["nll_not_worse_than_C127_B0_factor"]),
        "brier_not_worse_than_C127_B0": float(candidate["brier"])
        <= float(b0["brier"])
        * float(gate["brier_not_worse_than_C127_B0_factor"]),
        "physical_violation_count_zero": int(candidate["physical_violation_count"])
        == int(gate["physical_violation_count"]),
    }
    passed = all(checks.values())
    result = {
        "format_version": 1,
        "cycle": protocol.payload["cycle"],
        "phase": "P1_fold0",
        "protocol_sha256": protocol_hash,
        "p1_design_sha256": sha256(design_path),
        "candidate": {
            "summary": candidate_path.relative_to(ROOT).as_posix(),
            "summary_sha256": sha256(candidate_path),
            "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
            "checkpoint_sha256": sha256(checkpoint),
            "metrics": candidate,
        },
        "controls": controls,
        "checks": checks,
        "decision": "REPLICATION_AUTHORIZED" if passed else "C134_CLOSED_FOLD0_GATE_FAILED",
        "locked_test_used": False,
        "development_used": False,
        "claim_boundary": protocol.payload["claim_boundary"],
    }
    output = ARTIFACT_ROOT / "p1_decision.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return result


if __name__ == "__main__":
    print(json.dumps(summarize(), indent=2))
