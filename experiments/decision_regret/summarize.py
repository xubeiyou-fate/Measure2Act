"""Summarize C133 folds against frozen B0 and C129 controls."""

from __future__ import annotations

import json
from pathlib import Path

from .model import VARIANT
from .protocol import load_protocol, sha256


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/decision_regret"
RUN_ROOT = ROOT / "runs/decision_regret"
C127_ROOT = ROOT / "runs/metric_exact"
C129_ROOT = ROOT / "runs/joint_coupled"


def _read(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"missing C133 summary input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_path(fold: int) -> Path:
    return RUN_ROOT / f"{VARIANT}_fold{fold}_seed42_formal/training_summary.json"


def b0_path(fold: int) -> Path:
    phase = "P1" if fold == 0 else "P2"
    return C127_ROOT / f"{phase}_B0_signed_coupled_fold{fold}_seed42_formal/training_summary.json"


def c129_path(fold: int) -> Path:
    return C129_ROOT / f"J1_joint_coupled_dual_fold{fold}_seed42_formal/training_summary.json"


def _validate(
    summary: dict[str, object], *, family: str, fold: int, protocol
) -> None:
    variants = {
        "candidate": VARIANT,
        "B0": "B0_signed_coupled",
        "C129": "J1_joint_coupled_dual",
    }
    protocol_paths = {
        "candidate": protocol.path,
        "B0": ROOT / "experiments/metric_exact/protocol.json",
        "C129": ROOT / "experiments/joint_coupled/protocol.json",
    }
    if (
        summary.get("complete") is not True
        or summary.get("formal") is not True
        or summary.get("variant") != variants[family]
        or summary.get("fold") != fold
        or summary.get("seed") != 42
        or summary.get("fixed_final_epoch") != 20
        or summary.get("locked_test_used") is not False
    ):
        raise RuntimeError(f"{family} fold {fold} identity mismatch")
    if family in {"candidate", "C129"} and summary.get("development_used") is not False:
        raise RuntimeError(f"{family} fold {fold} crossed the development boundary")
    protocol_payload = json.loads(
        protocol_paths[family].read_text(encoding="utf-8")
    )
    expected_protocol = sha256(protocol_paths[family])
    if summary.get("protocol_sha256") != expected_protocol:
        raise RuntimeError(f"{family} fold {fold} protocol hash mismatch")
    checkpoint = ROOT / str(summary.get("checkpoint", ""))
    if not checkpoint.is_file() or summary.get("checkpoint_sha256") != sha256(checkpoint):
        raise RuntimeError(f"{family} fold {fold} checkpoint hash mismatch")
    if summary.get("manifest_sha256") != protocol_payload["dataset"]["manifest_sha256"]:
        raise RuntimeError(f"{family} fold {fold} manifest hash mismatch")


METRICS = (
    "agents",
    "top1_ade",
    "top1_fde",
    "minade",
    "minfde",
    "energy_score",
    "oracle_ade_rank1",
    "oracle_fde_rank1",
    "minfde_p95",
    "tail_minfde",
    "effective_modes",
)


def _metrics(summary: dict[str, object]) -> dict[str, float]:
    overall = summary["validation_metrics"]["overall"]
    direct_metrics = tuple(name for name in METRICS if name != "effective_modes")
    if any(name not in overall for name in direct_metrics):
        raise RuntimeError("summary lacks a required C133 metric")
    winner_distribution = overall.get("winner_distribution")
    if (
        not isinstance(winner_distribution, dict)
        or "effective_modes" not in winner_distribution
    ):
        raise RuntimeError("summary lacks C133 effective mode diagnostics")
    return {
        name: (
            float(winner_distribution["effective_modes"])
            if name == "effective_modes"
            else float(overall[name])
        )
        for name in METRICS
    }


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    total = sum(row["agents"] for row in rows)
    if total <= 0:
        raise RuntimeError("cannot aggregate empty C133 rows")
    return {
        name: sum(row[name] * row["agents"] for row in rows) / total
        for name in METRICS
        if name != "agents"
    } | {"agents": total}


def _gains(candidate: dict[str, float], control: dict[str, float]) -> dict[str, object]:
    names = ("top1_ade", "top1_fde", "minade", "minfde", "energy_score")
    return {
        "candidate": candidate,
        "control": control,
        "absolute_gain": {name: control[name] - candidate[name] for name in names},
        "relative_gain": {
            name: (control[name] - candidate[name]) / control[name] for name in names
        },
        "both_top1_better": (
            candidate["top1_ade"] < control["top1_ade"]
            and candidate["top1_fde"] < control["top1_fde"]
        ),
    }


def _fold_gate(
    candidate: dict[str, float], b0: dict[str, float], c129: dict[str, float]
) -> dict[str, bool]:
    gate = load_protocol().payload["screening"]["fold0_gate"]
    return {
        "top1_ade_better_than_b0": candidate["top1_ade"] < b0["top1_ade"],
        "top1_fde_better_than_b0": candidate["top1_fde"] < b0["top1_fde"],
        "top1_fde_relative_gain_at_least": (
            (b0["top1_fde"] - candidate["top1_fde"]) / b0["top1_fde"]
            >= float(gate["top1_fde_relative_gain_at_least"])
        ),
        "minade_within_c129_guard": candidate["minade"]
        <= c129["minade"] * (1.0 + float(gate["minade_not_worse_than_C129_plus"])),
        "minfde_within_c129_guard": candidate["minfde"]
        <= c129["minfde"] * (1.0 + float(gate["minfde_not_worse_than_C129_plus"])),
        "energy_within_b0_guard": candidate["energy_score"]
        <= b0["energy_score"] * (1.0 + float(gate["energy_not_worse_than_B0_plus"])),
        "p95_within_b0_guard": candidate["minfde_p95"]
        <= b0["minfde_p95"] * (1.0 + float(gate["p95_and_tail_not_worse_than_B0_plus"])),
        "tail_within_b0_guard": candidate["tail_minfde"]
        <= b0["tail_minfde"] * (1.0 + float(gate["p95_and_tail_not_worse_than_B0_plus"])),
        "effective_modes_retained": candidate["effective_modes"]
        >= b0["effective_modes"] * float(gate["effective_modes_retention"]),
    }


def summarize() -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_boundaries()
    protocol_hash = sha256(protocol.path)
    screening = protocol.payload["screening"]
    if sha256(ROOT / "experiments/metric_exact/protocol.json") != screening["c127_control_protocol_sha256"]:
        raise RuntimeError("frozen C127 protocol hash mismatch")
    if sha256(ROOT / "experiments/joint_coupled/protocol.json") != screening["c129_protocol_sha256"]:
        raise RuntimeError("frozen C129 protocol hash mismatch")

    available = []
    for fold in screening["folds"]:
        path = candidate_path(fold)
        if not path.is_file():
            continue
        candidate_summary = _read(path)
        b0_summary = _read(b0_path(fold))
        c129_summary = _read(c129_path(fold))
        _validate(candidate_summary, family="candidate", fold=fold, protocol=protocol)
        _validate(b0_summary, family="B0", fold=fold, protocol=protocol)
        _validate(c129_summary, family="C129", fold=fold, protocol=protocol)
        candidate = _metrics(candidate_summary)
        b0 = _metrics(b0_summary)
        c129 = _metrics(c129_summary)
        available.append(
            {
                "fold": fold,
                "candidate": candidate,
                "B0": b0,
                "C129": c129,
                "vs_B0": _gains(candidate, b0),
                "vs_C129": _gains(candidate, c129),
                "gate": _fold_gate(candidate, b0, c129),
                "candidate_summary": path.relative_to(ROOT).as_posix(),
            }
        )
    if not available or available[0]["fold"] != 0:
        raise RuntimeError("C133 fold 0 must be the first complete formal fold")

    if len(available) == 1:
        passed = all(available[0]["gate"].values())
        result = {
            "format_version": 1,
            "cycle": protocol.payload["cycle"],
            "phase": "P1_fold0",
            "protocol_sha256": protocol_hash,
            "available_folds": [0],
            "fold0": available[0],
            "decision": "REPLICATION_AUTHORIZED" if passed else "C133_CLOSED_FOLD0_GATE_FAILED",
            "locked_test_used": False,
            "development_used": False,
            "claim_boundary": protocol.payload["claim_boundary"],
        }
        output = ARTIFACT_ROOT / "p1_decision.json"
    elif len(available) == 3 and [row["fold"] for row in available] == [0, 1, 2]:
        candidate = _aggregate([row["candidate"] for row in available])
        b0 = _aggregate([row["B0"] for row in available])
        c129 = _aggregate([row["C129"] for row in available])
        both_count = sum(
            int(
                row["gate"]["top1_ade_better_than_b0"]
                and row["gate"]["top1_fde_better_than_b0"]
            )
            for row in available
        )
        required = int(
            screening["replication_gate"]
            ["both_top1_metrics_better_than_B0_in_at_least_folds"]
        )
        final_checks = {
            "both_top1_better_than_b0_in_required_folds": both_count >= required,
            "aggregate_top1_ade_better_than_b0": candidate["top1_ade"] < b0["top1_ade"],
            "aggregate_top1_fde_better_than_b0": candidate["top1_fde"] < b0["top1_fde"],
            "aggregate_minade_within_c129_guard": candidate["minade"] <= c129["minade"] * 1.01,
            "aggregate_minfde_within_c129_guard": candidate["minfde"] <= c129["minfde"] * 1.01,
            "aggregate_energy_within_b0_guard": candidate["energy_score"] <= b0["energy_score"] * 1.01,
            "aggregate_p95_within_b0_guard": candidate["minfde_p95"] <= b0["minfde_p95"] * 1.01,
            "aggregate_tail_within_b0_guard": candidate["tail_minfde"] <= b0["tail_minfde"] * 1.01,
            "aggregate_effective_modes_retained": candidate["effective_modes"] >= b0["effective_modes"] * 0.95,
        }
        result = {
            "format_version": 1,
            "cycle": protocol.payload["cycle"],
            "phase": "P2_three_date_folds",
            "protocol_sha256": protocol_hash,
            "available_folds": [0, 1, 2],
            "fold_results": {str(row["fold"]): row for row in available},
            "aggregate": {
                "candidate": candidate,
                "B0": b0,
                "C129": c129,
                "vs_B0": _gains(candidate, b0),
                "vs_C129": _gains(candidate, c129),
            },
            "both_top1_better_than_both_controls_count": both_count,
            "required_fold_count": required,
            "final_checks": final_checks,
            "decision": "C133_SCREEN_PASSED" if all(final_checks.values()) else "C133_CLOSED_REPLICATION_GATE_FAILED",
            "locked_test_used": False,
            "development_used": False,
            "claim_boundary": protocol.payload["claim_boundary"],
        }
        output = ARTIFACT_ROOT / "final_summary.json"
    else:
        raise RuntimeError("partial C133 replication summaries are not a decision stage")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return result


if __name__ == "__main__":
    print(json.dumps(summarize(), indent=2))
