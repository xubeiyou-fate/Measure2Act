"""Audit publication experiment coverage for the unified MABPT-ASCENT model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARTC = ROOT / "artifacts/mabpt_partc_20260811"
LEGACY = ROOT / "artifacts/mabpt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _receipt(path: Path) -> dict[str, object]:
    exists = path.is_file()
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": exists,
        "bytes": path.stat().st_size if exists else None,
        "sha256": _sha256(path) if exists else None,
    }


def experiment_groups() -> dict[str, dict[str, object]]:
    return {
        "main_accuracy_and_proper_scores": {
            "required": [PARTC / "five_seed_development_summary_v1.json"],
            "evidence": "five_seed_development",
        },
        "baseline_fairness": {
            "required": [LEGACY / "e1_matched_summary_v1.json"],
            "evidence": "matched_horizon_retrospective_cross_dataset_reuse",
        },
        "module_ablations": {
            "required": [LEGACY / "e2_e5_summary_v1.json"],
            "evidence": "retrospective_development",
        },
        "factorial_fusion": {
            "required": [PARTC / "factorial_physical_summary_v1.json"],
            "evidence": "retrospective_development",
        },
        "finite_measure_diagnostics": {
            "required": [
                LEGACY / "e6_e8_summary_v1.json",
                LEGACY / "e9_summary_v1.json",
            ],
            "evidence": "retrospective_development",
        },
        "calibration": {
            "required": [
                PARTC / "five_seed_development_summary_v1.json",
                LEGACY / "e11_summary_v2.json",
            ],
            "evidence": "five_seed_development_plus_opened_external_reuse",
        },
        "physical_plausibility": {
            "required": [PARTC / "factorial_physical_summary_v1.json"],
            "evidence": "training_envelope_retrospective_development",
        },
        "robustness": {
            "required": [
                PARTC / "five_seed_robustness_summary_v1.json",
                LEGACY / "e10_summary_v1.json",
            ],
            "evidence": "five_seed_development_plus_retrospective_provenance",
        },
        "temporal_and_airport_transfer": {
            "required": [
                LEGACY / "e1_external_summary_v2.json",
                LEGACY / "e1_matched_summary_v1.json",
            ],
            "evidence": "opened_external_reuse_only",
        },
        "conflict_risk_operations": {
            "required": [
                PARTC / "five_seed_development_summary_v1.json",
                LEGACY / "e12_summary_v1.json",
            ],
            "evidence": "five_seed_non_regulatory_surrogate_development",
        },
        "runtime_and_memory": {
            "required": [
                LEGACY / "e9_scaling_v1.json",
                PARTC / "five_seed_development_summary_v1.json",
            ],
            "evidence": "synthetic_operator_plus_end_to_end_development",
        },
        "subgroups_and_failures": {
            "required": [LEGACY / "e13_summary_v1.json"],
            "evidence": "target_blind_retrospective_development",
        },
        "prespecified_qualitative_cases": {
            "required": [
                PARTC / "figures/e13_target_blind_cases_fold1.png",
                PARTC / "figures/e13_target_blind_cases_fold1.pdf",
                PARTC / "figures/e13_target_blind_cases_fold1.json",
                PARTC / "figures/e13_target_blind_cases_fold2.png",
                PARTC / "figures/e13_target_blind_cases_fold2.pdf",
                PARTC / "figures/e13_target_blind_cases_fold2.json",
            ],
            "evidence": "target_blind_retrospective_development",
        },
    }


def audit() -> dict[str, object]:
    groups = {}
    for name, definition in experiment_groups().items():
        receipts = [_receipt(path) for path in definition["required"]]
        groups[name] = {
            "local_computation_complete": all(item["exists"] for item in receipts),
            "evidence_class": definition["evidence"],
            "receipts": receipts,
        }
    completed = sum(
        bool(group["local_computation_complete"]) for group in groups.values()
    )
    return {
        "format_version": 1,
        "paper_model": "MABPT-ASCENT",
        "model_identity_policy": (
            "All route-labelled artifacts are internal provenance for modules of one "
            "new ASCENT-derived finite-measure multimodal trajectory prediction model."
        ),
        "groups": groups,
        "local_groups_complete": completed,
        "local_groups_total": len(groups),
        "all_local_computation_complete": completed == len(groups),
        "fresh_confirmatory_evidence": {
            "complete": False,
            "reason": (
                "The locally available C127 locked test was opened on 2026-08-05. "
                "A never-opened later-period or airport cohort is not present locally."
            ),
            "required_next_input": "new sealed temporal period or airport cohort",
        },
        "publication_interpretation": (
            "Local completion does not convert opened retrospective or external-reuse "
            "results into fresh confirmatory evidence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=PARTC / "experiment_audit.json"
    )
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "complete": result["local_groups_complete"],
                "total": result["local_groups_total"],
                "fresh_confirmatory": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
