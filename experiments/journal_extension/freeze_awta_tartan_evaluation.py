"""Freeze all formal aWTA Tartan checkpoints before test inference."""

from __future__ import annotations

import json
from pathlib import Path

from mabpt import train_tartan_retrain as base
from mabpt.train_tartan_retrain import sha256

from .evaluate_awta_tartan_locked import AIRPORTS, LOCKED_ROOT, RECEIPT, SEEDS, checkpoint
from .train_awta_tartan import PROTOCOL, ROOT, atomic_json, expected_paths, load_protocol


def main() -> None:
    if RECEIPT.exists():
        raise FileExistsError(RECEIPT)
    existing_test = sorted(LOCKED_ROOT.glob("*.json")) if LOCKED_ROOT.exists() else []
    if existing_test:
        raise RuntimeError("aWTA test outputs exist before evaluation freeze")
    protocol = load_protocol()
    parent = json.loads((ROOT / protocol["inherits"]["tartan_protocol"]).read_text(encoding="utf-8"))
    tail_thresholds = {}
    index_paths = []
    for airport in AIRPORTS:
        train_dataset, _, train_index = base._dataset(parent, airport, "train")
        _, _, test_index = base._dataset(parent, airport, "test")
        tail_thresholds[airport] = base._tail_threshold(
            train_dataset, list(range(len(train_dataset)))
        )
        index_paths.extend((train_index, test_index))
    static = [
        PROTOCOL,
        Path(__file__),
        Path(__file__).with_name("evaluate_awta_tartan_locked.py"),
        Path(__file__).with_name("train_awta_tartan.py"),
        Path(__file__).with_name("awta.py"),
        ROOT / protocol["inherits"]["tartan_protocol"],
        ROOT / protocol["inherits"]["data_manifest"],
        ROOT / protocol["inherits"]["scene_index_summary"],
        *index_paths,
    ]
    formal = []
    for airport in AIRPORTS:
        for seed in SEEDS:
            checkpoint_path, _ = checkpoint(airport, seed)
            formal.extend((checkpoint_path, expected_paths(airport, seed)[1]))
    paths = sorted(set((*static, *formal)), key=lambda value: value.as_posix())
    payload = {
        "format_version": 1,
        "experiment_id": "awta_tartan_evaluation_freeze_v1",
        "purpose": "Bind the complete ten-cell aWTA final-epoch grid before any aWTA test inference.",
        "airports": list(AIRPORTS),
        "seeds": list(SEEDS),
        "formal_checkpoint_count": len(AIRPORTS) * len(SEEDS),
        "formal_result_count": len(AIRPORTS) * len(SEEDS),
        "awta_test_inference_completed_before_freeze": False,
        "tail_threshold_km": tail_thresholds,
        "files": {
            path.relative_to(ROOT).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in paths
        },
        "integrity": {
            "complete_development_grid": True,
            "fixed_final_epoch": True,
            "development_checkpoint_selection": False,
            "test_outputs_present_before_freeze": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    atomic_json(RECEIPT, payload)
    print(json.dumps({"receipt": RECEIPT.as_posix(), "files": len(paths)}, indent=2))


if __name__ == "__main__":
    main()
