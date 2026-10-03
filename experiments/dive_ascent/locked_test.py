"""One-event locked-test evaluation for an authorized C99 five-seed pair."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiments.edfa_ascent.data import build_scene_dates
from model.utils import TrajectoryDataset, seq_collate

from .evaluation import evaluate, target_tail_threshold
from .model import build_model
from .protocol import load_protocol, sha256


def run(output: Path | None = None, *, device_name: str | None = None) -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_development_sealed()
    gate_path = protocol.repository_root / str(
        protocol.payload["locked_test_policy"]["required_gate_artifact"]
    )
    if not gate_path.is_file():
        raise RuntimeError("C99 locked test requires development_gate.json")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("locked_test_authorized") is not True:
        raise RuntimeError("C99 development gate prohibits locked-test evaluation")
    receipt_path = protocol.repository_root / str(protocol.payload["locked_test_policy"]["receipt"])
    if receipt_path.exists():
        raise RuntimeError("C99 locked test has already been evaluated")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = TrajectoryDataset(
        protocol.split_path("locked_test").as_posix(),
        obs_len=16,
        obs_steps=1,
        pred_len=120,
        pred_step=5,
        delim=" ",
    )
    dates = build_scene_dates(
        protocol.split_path("locked_test"), protocol.manifest_path, "locked_test"
    )
    if len(dates) != len(dataset):
        raise RuntimeError("C99 locked-test scene-date index mismatch")
    loader = DataLoader(
        dataset,
        batch_size=int(protocol.payload["training"]["evaluation_batch_size"]),
        shuffle=False,
        collate_fn=seq_collate,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    threshold = target_tail_threshold(dataset)
    run_root = protocol.repository_root / str(protocol.payload["run_root"])
    results = {}
    for seed in protocol.payload["development"]["seeds"]:
        results[str(seed)] = {}
        for variant in protocol.payload["development"]["primary_variants"]:
            checkpoint_path = run_root / f"{variant}_seed{seed}_formal" / "last.pt"
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model = build_model(variant, batch_size=int(protocol.payload["training"]["batch_size"])).to(device)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            results[str(seed)][variant] = evaluate(
                model,
                loader,
                device,
                scene_dates=dates,
                tail_threshold=threshold,
            )
    result = {
        "format_version": 1,
        "cycle": protocol.payload["cycle"],
        "protocol_sha256": sha256(protocol.path),
        "manifest_sha256": sha256(protocol.manifest_path),
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "event": 1,
        "results": results,
    }
    output = output or protocol.repository_root / "artifacts/experiments/dive_ascent/locked_test_result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "format_version": 1,
        "cycle": protocol.payload["cycle"],
        "event": 1,
        "result": output.relative_to(protocol.repository_root).as_posix(),
        "result_sha256": sha256(output),
        "evaluated_utc": result["evaluated_utc"],
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device")
    args = parser.parse_args()
    result = run(args.output, device_name=args.device)
    print(json.dumps({"event": result["event"], "seeds": len(result["results"])}, indent=2))


if __name__ == "__main__":
    main()
