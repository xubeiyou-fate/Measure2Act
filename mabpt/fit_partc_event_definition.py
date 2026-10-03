"""Fit the five-seed Part C event definition on the complete training split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model.utils import TrajectoryDataset

from .events import PROTOCOL as EVENT_PROTOCOL, motion_features, sha256
from .partc_design import PROTOCOL as PARTC_PROTOCOL, load_protocol


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "artifacts/mabpt_partc_20260811/event_definition_train_v1.json"
)


def fit(*, chunk_size: int) -> dict[str, object]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    protocol = load_protocol()
    dataset = TrajectoryDataset(
        (ROOT / protocol["data"]["train"]).as_posix(),
        obs_len=16,
        obs_steps=1,
        pred_len=120,
        pred_step=5,
        delim=" ",
    )
    turns = []
    altitudes = []
    squared_error = torch.zeros(24, 3, dtype=torch.float64)
    actor_count = int(dataset.obs_traj.shape[0])
    times = (5.0 * torch.arange(1, 25, dtype=torch.float64))[None, :, None]
    for start in range(0, actor_count, chunk_size):
        stop = min(start + chunk_size, actor_count)
        observation = dataset.obs_traj[start:stop].permute(0, 2, 1).to(torch.float64)
        future = dataset.pred_traj[start:stop].permute(0, 2, 1).to(torch.float64)
        turn, altitude = motion_features(observation, future[:, -1:, :])
        turns.append(turn.abs().reshape(-1).cpu())
        altitudes.append(altitude.abs().reshape(-1).cpu())
        velocity = observation[:, -1] - observation[:, -2]
        constant_velocity = observation[:, -1, None] + velocity[:, None] * times
        squared_error += (future - constant_velocity).square().sum(dim=0).cpu()
    turn_threshold = float(torch.quantile(torch.cat(turns), 0.5))
    altitude_threshold = float(torch.quantile(torch.cat(altitudes), 0.5))
    variance = (squared_error / actor_count).clamp_min(1e-6)
    if turn_threshold <= 0 or altitude_threshold <= 0:
        raise RuntimeError("degenerate complete-training event definition")
    return {
        "format_version": 1,
        "model": "MABPT-ASCENT",
        "experiment_id": "five_seed_fixed_event_definition",
        "fit_split": "complete_training_only",
        "training_scenes": len(dataset),
        "training_actors": actor_count,
        "turn_threshold_radians": turn_threshold,
        "altitude_threshold_km": altitude_threshold,
        "constant_velocity_residual_variance_km2": variance.tolist(),
        "minimum_variance_km2": 1e-6,
        "partc_protocol_sha256": sha256(PARTC_PROTOCOL),
        "event_protocol_sha256": sha256(EVENT_PROTOCOL),
        "development_or_test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = fit(chunk_size=args.chunk_size)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "training_actors": result["training_actors"],
                "turn_threshold_radians": result["turn_threshold_radians"],
                "altitude_threshold_km": result["altitude_threshold_km"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
