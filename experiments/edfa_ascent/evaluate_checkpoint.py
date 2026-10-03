"""Evaluate one C96 checkpoint while enforcing the locked-test policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model.utils import TrajectoryDataset, seq_collate

from .data import build_scene_dates
from .evaluation import evaluate
from .protocol import load_protocol, sha256
from .train import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "locked_test"), default="dev")
    parser.add_argument("--allow-locked-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol()
    protocol.assert_manifest_sealed()
    protocol.assert_split_allowed(args.split, allow_locked_test=args.allow_locked_test)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    model = build_model(config)
    model.load_state_dict(state["model_state_dict"])
    device = torch.device(args.device)
    model.to(device)
    dataset = TrajectoryDataset(
        protocol.split_path(args.split).as_posix(),
        obs_len=16, obs_steps=1, pred_len=120, pred_step=5, delim=" ",
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=seq_collate, num_workers=0,
    )
    dates = build_scene_dates(
        protocol.split_path(args.split), protocol.manifest_path, args.split
    )
    if len(dates) != len(dataset):
        raise RuntimeError("scene date reconstruction mismatch")
    result = {
        "format_version": 1,
        "cycle": "C96_EDFA_ASCENT",
        "variant": config["variant"],
        "split": args.split,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "metrics": evaluate(model, loader, device, scene_dates=dates),
        "locked_test_used": args.split == "locked_test",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.split == "locked_test":
        gate = json.loads(protocol.gate_artifact.read_text(encoding="utf-8"))
        gate["locked_test_evaluations"] = gate.get("locked_test_evaluations", 0) + 1
        gate["locked_test_result"] = str(args.output.resolve())
        protocol.gate_artifact.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
