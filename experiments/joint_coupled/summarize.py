"""Summarize C129 train-date folds and enforce the sequential gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model import VARIANT
from .protocol import load_protocol, sha256


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/joint_coupled"
RUN_ROOT = ROOT / "runs/joint_coupled"
C127_ROOT = ROOT / "runs/metric_exact"
C127_PROTOCOL_SHA = sha256(ROOT / "experiments/metric_exact/protocol.json")


def _read(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"missing C129 summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_path(fold: int) -> Path:
    return RUN_ROOT / f"{VARIANT}_fold{fold}_seed42_formal/training_summary.json"


def control_path(fold: int, variant: str) -> Path:
    phase = "P1" if fold == 0 else "P2"
    return C127_ROOT / (
        f"{phase}_{variant}_fold{fold}_seed42_formal/training_summary.json"
    )


def _validate_identity(
    summary: dict[str, object],
    *,
    candidate: bool,
    fold: int,
    variant: str,
) -> None:
    protocol_path = (
        ROOT / "experiments/joint_coupled/protocol.json"
        if candidate
        else ROOT / "experiments/metric_exact/protocol.json"
    )
    protocol_payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected_protocol = sha256(protocol_path)
    if summary.get("complete") is not True or summary.get("formal") is not True:
        raise RuntimeError("summary is not a complete formal run")
    if summary.get("locked_test_used") is not False:
        raise RuntimeError("summary crossed the locked-test boundary")
    if (
        summary.get("variant") != variant
        or summary.get("fold") != fold
        or summary.get("seed") != 42
        or summary.get("fixed_final_epoch") != 20
    ):
        raise RuntimeError("summary seed or final epoch does not match C129 screening")
    checkpoint = ROOT / str(summary.get("checkpoint", ""))
    if not checkpoint.is_file() or summary.get("checkpoint_sha256") != sha256(checkpoint):
        raise RuntimeError("summary checkpoint identity mismatch")
    if candidate:
        if summary.get("development_used") is not False:
            raise RuntimeError("C129 candidate accessed development data")
        if summary.get("protocol_sha256") != expected_protocol:
            raise RuntimeError("C129 candidate protocol hash mismatch")
        if summary.get("manifest_sha256") != protocol_payload["dataset"]["manifest_sha256"]:
            raise RuntimeError("C129 candidate manifest hash mismatch")
        fold_path = ROOT / str(protocol_payload["dataset"]["date_folds"])
        if summary.get("fold_artifact_sha256") != sha256(fold_path):
            raise RuntimeError("C129 candidate fold-artifact hash mismatch")
    else:
        if summary.get("protocol_sha256") != expected_protocol:
            raise RuntimeError("C127 control protocol hash mismatch")
        if summary.get("manifest_sha256") != protocol_payload["dataset"]["manifest_sha256"]:
            raise RuntimeError("C127 control manifest hash mismatch")


def _metrics(summary: dict[str, object]) -> dict[str, float]:
    overall = summary["validation_metrics"]["overall"]
    required = ("agents", "top1_ade", "top1_fde", "minade", "minfde", "energy_score")
    if any(name not in overall for name in required):
        raise RuntimeError(f"summary lacks required metrics: {required}")
    return {name: float(overall[name]) for name in required}


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    total = sum(row["agents"] for row in rows)
    if total <= 0:
        raise RuntimeError("cannot aggregate empty validation rows")
    return {
        name: sum(row[name] * row["agents"] for row in rows) / total
        for name in ("top1_ade", "top1_fde", "minade", "minfde", "energy_score")
    } | {"agents": total}


def _comparison(candidate: dict[str, float], control: dict[str, float]) -> dict[str, object]:
    both_top1 = candidate["top1_ade"] < control["top1_ade"] and candidate["top1_fde"] < control["top1_fde"]
    return {
        "candidate": candidate,
        "control": control,
        "absolute_gain": {
            name: control[name] - candidate[name]
            for name in ("top1_ade", "top1_fde", "minade", "minfde", "energy_score")
        },
        "relative_gain": {
            name: (control[name] - candidate[name]) / control[name]
            for name in ("top1_ade", "top1_fde", "minade", "minfde", "energy_score")
        },
        "both_top1_better": both_top1,
        "geometry_guard_pass": candidate["minade"] <= control["minade"]
        and candidate["minfde"] <= control["minfde"],
    }


def summarize() -> dict[str, object]:
    protocol = load_protocol()
    protocol_hash = sha256(protocol.path)
    if protocol.payload["screening"]["control_protocol_sha256"] != C127_PROTOCOL_SHA:
        raise RuntimeError("frozen C127 control protocol hash no longer matches")
    available = []
    for fold in protocol.payload["screening"]["folds"]:
        path = candidate_path(int(fold))
        if path.is_file():
            candidate_summary = _read(path)
            _validate_identity(
                candidate_summary, candidate=True, fold=int(fold), variant=VARIANT
            )
            b0_summary = _read(control_path(int(fold), "B0_signed_coupled"))
            b6_summary = _read(control_path(int(fold), "B6_dual_oracle"))
            _validate_identity(
                b0_summary,
                candidate=False,
                fold=int(fold),
                variant="B0_signed_coupled",
            )
            _validate_identity(
                b6_summary,
                candidate=False,
                fold=int(fold),
                variant="B6_dual_oracle",
            )
            available.append(
                {
                    "fold": int(fold),
                    "candidate": _metrics(candidate_summary),
                    "B0_signed_coupled": _metrics(b0_summary),
                    "B6_dual_oracle": _metrics(b6_summary),
                    "candidate_summary": path.relative_to(ROOT).as_posix(),
                }
            )

    if not available:
        raise RuntimeError("no complete C129 candidate fold is available")
    available.sort(key=lambda row: row["fold"])
    fold0 = next((row for row in available if row["fold"] == 0), None)
    if fold0 is None:
        raise RuntimeError("C129 fold 0 is required before any replication summary")

    fold0_comparison = _comparison(fold0["candidate"], fold0["B0_signed_coupled"])
    fold0_pass = bool(
        fold0_comparison["both_top1_better"]
        and fold0_comparison["geometry_guard_pass"]
    )
    if len(available) == 1:
        decision = "REPLICATION_AUTHORIZED" if fold0_pass else "C129_CLOSED_FOLD0_GATE_FAILED"
        result = {
            "format_version": 1,
            "cycle": protocol.payload["cycle"],
            "phase": "P1_fold0",
            "protocol_sha256": protocol_hash,
            "available_folds": [row["fold"] for row in available],
            "fold0": fold0_comparison,
            "decision": decision,
            "locked_test_used": False,
            "development_used": False,
            "claim_boundary": protocol.payload["claim_boundary"],
        }
        path = ARTIFACT_ROOT / "p1_decision.json"
    elif len(available) == len(protocol.payload["screening"]["folds"]):
        comparisons = {
            str(row["fold"]): _comparison(row["candidate"], row["B0_signed_coupled"])
            for row in available
        }
        aggregate_candidate = _aggregate([row["candidate"] for row in available])
        aggregate_b0 = _aggregate([row["B0_signed_coupled"] for row in available])
        both_count = sum(
            int(comparison["both_top1_better"])
            for comparison in comparisons.values()
        )
        aggregate = _comparison(aggregate_candidate, aggregate_b0)
        required_count = int(
            protocol.payload["screening"]["replication_gate"]
            ["both_top1_metrics_better_than_C127_B0_in_at_least_folds"]
        )
        passed = bool(
            both_count >= required_count
            and aggregate["both_top1_better"]
            and aggregate["geometry_guard_pass"]
        )
        result = {
            "format_version": 1,
            "cycle": protocol.payload["cycle"],
            "phase": "P2_three_date_folds",
            "protocol_sha256": protocol_hash,
            "available_folds": [row["fold"] for row in available],
            "fold_comparisons": comparisons,
            "aggregate": aggregate,
            "both_top1_better_count": both_count,
            "required_both_top1_better_count": required_count,
            "decision": "C129_SCREEN_PASSED" if passed else "C129_CLOSED_REPLICATION_GATE_FAILED",
            "locked_test_used": False,
            "development_used": False,
            "claim_boundary": protocol.payload["claim_boundary"],
        }
        path = ARTIFACT_ROOT / "final_summary.json"
    else:
        raise RuntimeError("partial C129 replication summaries are not a decision stage")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(summarize(), indent=2))


if __name__ == "__main__":
    main()
