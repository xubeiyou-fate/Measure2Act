"""Freeze identity-disjoint data, index, models, and evaluator before test use."""

from __future__ import annotations

import json
from pathlib import Path

from mabpt.evaluate_tartan_retrain import _selected_checkpoint_triplet
from mabpt.train_tartan_retrain import sha256

from .evaluate_identity_disjoint_tartan import DATA_ROOT, INDEX_ROOT, PARENT_PROTOCOL, PROTOCOL, RECEIPT, ROOT
from .train_awta_tartan import atomic_json


def file_record(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def freeze() -> dict[str, object]:
    if RECEIPT.exists():
        raise FileExistsError(RECEIPT)
    manifest = json.loads((DATA_ROOT / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("formal") is not True:
        raise RuntimeError("identity-disjoint formal data are incomplete")
    parent = json.loads(PARENT_PROTOCOL.read_text(encoding="utf-8"))
    files = [
        DATA_ROOT / "manifest.json",
        DATA_ROOT / "qc.json",
        INDEX_ROOT / "summary.json",
        PROTOCOL,
        Path(__file__).with_name("evaluate_identity_disjoint_tartan.py"),
        PARENT_PROTOCOL,
    ]
    for airport in ("KAGC", "KBTP"):
        for split in ("train", "development", "test"):
            files.append(INDEX_ROOT / f"{airport}_{split}_scene_dates.json")
        for seed in (42, 7, 123, 2024, 2026):
            checkpoints = _selected_checkpoint_triplet(
                root=ROOT,
                protocol=parent,
                airport=airport,
                regime="target_only",
                seed=seed,
                formal=True,
            )
            files.extend(ROOT / checkpoints[name]["path"] for name in ("ascent", "predicted_risk"))
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    test_root = ROOT / "artifacts/journal_extension_20260814/tartan_identity_disjoint/test"
    existing_test = list(test_root.glob("*.json")) if test_root.exists() else []
    if existing_test:
        raise RuntimeError("identity-disjoint test outputs already exist before freeze")
    unique = sorted(set(files), key=lambda path: path.as_posix())
    receipt = {
        "schema_version": 1,
        "receipt_id": "tartan_identity_disjoint_evaluation_receipt_v1",
        "frozen_at": "2026-08-14",
        "registered_airports": ["KAGC", "KBTP"],
        "registered_seeds": [42, 7, 123, 2024, 2026],
        "identity_disjoint_test_inference_completed_before_freeze": False,
        "historical_parent_test_previously_opened": True,
        "files": {path.relative_to(ROOT).as_posix(): file_record(path) for path in unique},
        "claim_boundary": "The new subgroup was frozen before its model inference, but the parent retrospective test cohort had already been opened.",
    }
    atomic_json(RECEIPT, receipt)
    return receipt


def main() -> None:
    receipt = freeze()
    print(json.dumps({"receipt": str(RECEIPT), "files": len(receipt["files"])}, indent=2))


if __name__ == "__main__":
    main()
