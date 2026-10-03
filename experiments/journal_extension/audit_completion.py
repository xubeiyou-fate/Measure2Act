"""Audit the complete journal-extension evidence grid and write a hash manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mabpt.train_tartan_retrain import sha256

from .evaluate_identity_disjoint_tartan import ROOT
from .train_awta_tartan import atomic_json


SEEDS = (42, 7, 123, 2024, 2026)
AIRPORTS = ("KAGC", "KBTP")
OUTPUT = ROOT / "artifacts/journal_extension_20260814/completion_audit_v1.json"


def expected_paths() -> list[Path]:
    paths = [
        ROOT / "artifacts/journal_extension_20260814/social_context_invariance_v1.json",
        ROOT / "artifacts/journal_extension_20260814/tartan_identity_disjoint_data_v1/manifest.json",
        ROOT / "artifacts/journal_extension_20260814/tartan_identity_disjoint_index_v1/summary.json",
        ROOT / "artifacts/journal_extension_20260814/tartan_identity_disjoint_evaluation_receipt_v1.json",
        ROOT / "artifacts/journal_extension_20260814/tartan_identity_disjoint_summary_v1.json",
        ROOT / "artifacts/partc_two_dataset_20260812/modern_baseline/eqmotion_tartan_multiseed_v2_evaluation_receipt.json",
        ROOT / "artifacts/journal_extension_20260814/eqmotion_five_seed_summary_v1.json",
        ROOT / "artifacts/journal_extension_20260814/probability_controls_receipt_v1.json",
        ROOT / "artifacts/journal_extension_20260814/probability_controls_summary_v1.json",
        ROOT / "artifacts/journal_extension_20260814/awta_tartan_evaluation_receipt_v1.json",
        ROOT / "artifacts/journal_extension_20260814/awta_summary_v1.json",
        ROOT / "artifacts/partc_two_dataset_20260812/efficiency/ascent_vs_mabpt_e2e_RTX5090_formal_v1.json",
    ]
    for airport in AIRPORTS:
        for seed in SEEDS:
            paths.extend((
                ROOT / f"artifacts/journal_extension_20260814/tartan_identity_disjoint/development/{airport}_seed{seed}.json",
                ROOT / f"artifacts/journal_extension_20260814/tartan_identity_disjoint/test/{airport}_seed{seed}.json",
                ROOT / f"artifacts/journal_extension_20260814/probability_controls/development/{airport}_seed{seed}_development_v1.json",
                ROOT / f"artifacts/journal_extension_20260814/probability_controls/test/{airport}_seed{seed}_test_v1.json",
                ROOT / f"artifacts/journal_extension_20260814/awta_tartan/development/{airport}_seed{seed}_formal.json",
                ROOT / f"artifacts/journal_extension_20260814/awta_tartan/test/{airport}_seed{seed}_test_v1.json",
                ROOT / f"runs/journal_extension_20260814/awta_tartan/{airport}/seed{seed}_formal/last.pt",
            ))
    for airport in AIRPORTS:
        for seed in (7, 123, 2024, 2026):
            paths.append(ROOT / f"artifacts/partc_two_dataset_20260812/modern_baseline/eqmotion_tartan_multiseed_v2/locked/{airport}_target_only_seed{seed}_locked_test_v2.json")
    for seed in SEEDS:
        paths.extend((
            ROOT / f"artifacts/journal_extension_20260814/awta_trajair/development/seed{seed}_formal.json",
            ROOT / f"runs/journal_extension_20260814/awta_trajair/seed{seed}_formal/last.pt",
        ))
    return paths


def audit() -> dict[str, Any]:
    paths = expected_paths()
    missing = [path.relative_to(ROOT).as_posix() for path in paths if not path.is_file()]
    invalid_json = []
    files = {}
    for path in paths:
        if not path.is_file():
            continue
        if path.suffix == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                invalid_json.append(path.relative_to(ROOT).as_posix())
        files[path.relative_to(ROOT).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    complete = not missing and not invalid_json and len(files) == len(paths)
    return {
        "format_version": 1,
        "experiment_id": "journal_extension_completion_audit_v1",
        "complete": complete,
        "expected_file_count": len(paths),
        "verified_file_count": len(files),
        "missing": missing,
        "invalid_json": invalid_json,
        "files": files,
        "evidence_boundaries": {
            "local_datasets_only": True,
            "retrospective_test": True,
            "third_party_blind_test": False,
            "social_model_claim_supported": False,
            "large_k_scalability_claim_supported": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    result = audit()
    if not result["complete"] and not args.allow_incomplete:
        raise RuntimeError(json.dumps({"missing": result["missing"], "invalid_json": result["invalid_json"]}, indent=2))
    atomic_json(args.output.resolve(), result)
    print(json.dumps({key: result[key] for key in ("complete", "expected_file_count", "verified_file_count", "missing", "invalid_json")}, indent=2))


if __name__ == "__main__":
    main()
