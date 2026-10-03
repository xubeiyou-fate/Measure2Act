"""Attach the already-trained matched C12 ASCENT baseline to C96 metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model.ascent import Ascent
from model.utils import TrajectoryDataset, seq_collate

from .evaluation import evaluate
from .protocol import load_protocol, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("runs/c12_ascent/trajair_111day_c12_seed42/model_trajair_111day_c12_12.pt"),
    )
    parser.add_argument(
        "--training-summary", type=Path,
        default=Path("runs/c12_ascent/trajair_111day_c12_seed42/training_summary.json"),
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path("runs/c12_ascent/trajair_111day_c12_seed42/config.json"),
    )
    parser.add_argument("--scene-dates", type=Path, default=Path("artifacts/experiments/edfa_ascent/c12_dev_scene_dates.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/experiments/edfa_ascent/a0_seed42_formal"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    protocol = load_protocol()
    protocol.assert_manifest_sealed()
    source_summary = json.loads(args.training_summary.read_text(encoding="utf-8"))
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    source_config = json.loads(args.config.read_text(encoding="utf-8"))
    config = {**source_config, "variant": "a0", "cycle": "C96_EDFA_ASCENT"}
    model = Ascent(config)
    model.load_state_dict(state["model_state_dict"])
    device = torch.device(args.device)
    model.to(device)
    dataset = TrajectoryDataset(
        protocol.split_path("dev").as_posix(),
        obs_len=16, obs_steps=1, pred_len=120, pred_step=5, delim=" ",
    )
    dates = json.loads(args.scene_dates.read_text(encoding="utf-8"))["dates"]
    if len(dates) != len(dataset):
        raise RuntimeError("C96 scene-date index mismatch")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=seq_collate, num_workers=0,
    )
    metrics = evaluate(model, loader, device, scene_dates=dates)
    summary = {
        "format_version": 1,
        "cycle": "C96_EDFA_ASCENT",
        "variant": "a0",
        "seed": 42,
        "formal": True,
        "epochs_completed": source_summary["history"][-1]["epoch"],
        "best_epoch": source_summary["best_epoch"],
        "best_validation_minfde": metrics["overall"]["minfde"],
        "source_training_summary": str(args.training_summary.resolve()),
        "best_checkpoint": str(args.checkpoint.resolve()),
        "best_checkpoint_sha256": sha256(args.checkpoint),
        "locked_test_used": False,
        "best_controls": {"predicted": metrics},
        "history": source_summary["history"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
