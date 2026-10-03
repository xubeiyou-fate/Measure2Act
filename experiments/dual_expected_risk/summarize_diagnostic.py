"""Aggregate C130 zero-training diagnostics and apply the frozen P0 gate."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .protocol import load_protocol, sha256


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/dual_expected_risk"


def _read(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"missing C130 diagnostic: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    total = sum(int(row["actors"]) for row in rows)
    horizons = {}
    for horizon in ("30", "60", "90", "120"):
        horizons[horizon] = {
            metric: sum(
                float(row["horizons"][horizon][metric]) * int(row["actors"])
                for row in rows
            )
            / total
            for metric in ("top1_fde", "minfde", "top1_regret")
        }
    scalar_names = (
        "ade_fde_winner_overlap",
        "combined_winner_top1_overlap",
        "pairwise_score_cost_concordance",
        "oracle_ade_rank",
        "oracle_fde_rank",
        "oracle_ade_rank1",
        "oracle_fde_rank1",
    )
    result = {
        "actors": total,
        "horizons": horizons,
        "oracle_winner_transition_rate": {
            name: sum(
                float(row["oracle_winner_transition_rate"][name]) * int(row["actors"])
                for row in rows
            )
            / total
            for name in ("30_to_60", "60_to_90", "90_to_120", "30_to_120")
        },
        **{
            name: sum(float(row[name]) * int(row["actors"]) for row in rows) / total
            for name in scalar_names
        },
    }
    return result


def summarize() -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_boundaries()
    protocol_hash = sha256(protocol.path)
    fold_payloads = []
    for fold in protocol.payload["diagnostic"]["folds"]:
        payload = _read(ARTIFACT_ROOT / f"diagnostic_fold{fold}.json")
        if (
            payload.get("fold") != fold
            or payload.get("protocol_sha256") != protocol_hash
            or payload.get("training_performed") is not False
            or payload.get("adaptive_selection_performed") is not False
            or payload.get("locked_test_used") is not False
            or payload.get("development_used") is not False
        ):
            raise RuntimeError(f"C130 diagnostic fold {fold} identity mismatch")
        fold_payloads.append(payload)

    aggregates = {
        family: _aggregate(
            [payload["families"][family]["metrics"] for payload in fold_payloads]
        )
        for family in ("C127_B0", "C129_J1")
    }
    c129_final_path = ROOT / "artifacts/experiments/joint_coupled/final_summary.json"
    if sha256(c129_final_path) != protocol.payload["screening"]["c129_final_summary_sha256"]:
        raise RuntimeError("frozen C129 final summary hash mismatch")
    c129_final = _read(c129_final_path)
    comparison = c129_final["aggregate"]
    c129_metrics = comparison["candidate"]
    b0_metrics = comparison["control"]
    minade_advantage = b0_metrics["minade"] - c129_metrics["minade"]
    minfde_advantage = b0_metrics["minfde"] - c129_metrics["minfde"]
    top1_fde_deficit = c129_metrics["top1_fde"] - b0_metrics["top1_fde"]
    required_capture_fraction = top1_fde_deficit / minfde_advantage

    finite = all(
        math.isfinite(float(value))
        for family in aggregates.values()
        for value in (
            family["horizons"]["120"]["top1_fde"],
            family["horizons"]["120"]["minfde"],
            family["pairwise_score_cost_concordance"],
            family["oracle_fde_rank1"],
        )
    )
    checks = {
        "all_three_folds_present": len(fold_payloads) == 3,
        "all_diagnostics_finite": finite,
        "c129_minade_better_than_b0": minade_advantage > 0,
        "c129_minfde_better_than_b0": minfde_advantage > 0,
        "c129_top1_fde_deficit_smaller_than_minfde_advantage": (
            top1_fde_deficit > 0 and top1_fde_deficit < minfde_advantage
        ),
        "no_training_or_adaptive_selection": all(
            payload["training_performed"] is False
            and payload["adaptive_selection_performed"] is False
            for payload in fold_payloads
        ),
    }
    passed = all(checks.values())
    result = {
        "format_version": 1,
        "cycle": protocol.payload["cycle"],
        "phase": "P0_zero_training_read_only",
        "protocol_sha256": protocol_hash,
        "folds": [payload["fold"] for payload in fold_payloads],
        "aggregate_diagnostics": aggregates,
        "feasibility": {
            "c129_minade_advantage": minade_advantage,
            "c129_minfde_advantage": minfde_advantage,
            "c129_top1_fde_deficit": top1_fde_deficit,
            "required_minfde_advantage_capture_fraction": required_capture_fraction,
        },
        "checks": checks,
        "decision": "FORMAL_FOLD0_AUTHORIZED" if passed else "C130_CLOSED_P0_FAILED",
        "training_performed": False,
        "adaptive_selection_performed": False,
        "locked_test_used": False,
        "development_used": False,
        "claim_boundary": protocol.payload["claim_boundary"],
    }
    output = ARTIFACT_ROOT / "p0_decision.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return result


if __name__ == "__main__":
    print(json.dumps(summarize(), indent=2))
