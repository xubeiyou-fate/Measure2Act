"""Training-only fixed-event and continuous-likelihood definitions for E11."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.ascent_recomparison.common import fold_subsets, load_dataset
from experiments.tpmo_ascent.protocol import load_protocol as load_legacy_data_protocol

from .evaluate import ROOT


PROTOCOL = Path(__file__).with_name("e11_protocol_v2.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"MABPT refuses to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _training_actor_indices(dataset, training_subset) -> torch.Tensor:
    scene_sizes = torch.tensor(
        [end - start for start, end in dataset.seq_start_end], dtype=torch.long
    )
    actor_scene = torch.repeat_interleave(
        torch.arange(len(scene_sizes), dtype=torch.long), scene_sizes
    )
    selected = torch.zeros(len(scene_sizes), dtype=torch.bool)
    selected[torch.as_tensor(training_subset.indices, dtype=torch.long)] = True
    return torch.nonzero(selected[actor_scene], as_tuple=False).flatten()


def motion_features(
    observations: torch.Tensor, trajectories: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return signed endpoint turn and altitude change.

    observations: [B,T,3]; trajectories: [B,...,H,3].
    """
    if observations.ndim != 3 or observations.shape[-1] != 3:
        raise ValueError("observations must have shape [B,T,3]")
    if trajectories.ndim < 3 or trajectories.shape[0] != observations.shape[0]:
        raise ValueError("trajectory batch is incompatible with observations")
    velocity = observations[:, -1, :2] - observations[:, -2, :2]
    observed_bearing = torch.atan2(velocity[:, 1], velocity[:, 0])
    endpoint = trajectories[..., -1, :]
    displacement = endpoint[..., :2] - observations[:, None, -1, :2] if trajectories.ndim == 4 else endpoint[..., :2] - observations[:, -1, :2]
    future_bearing = torch.atan2(displacement[..., 1], displacement[..., 0])
    delta = future_bearing - observed_bearing.reshape(
        (observations.shape[0],) + (1,) * (future_bearing.ndim - 1)
    )
    turn = torch.atan2(torch.sin(delta), torch.cos(delta))
    altitude = endpoint[..., 2] - observations[:, -1, 2].reshape(
        (observations.shape[0],) + (1,) * (endpoint[..., 2].ndim - 1)
    )
    return turn, altitude


def event_labels(
    observations: torch.Tensor,
    trajectories: torch.Tensor,
    *,
    turn_threshold: float,
    altitude_threshold: float,
) -> torch.Tensor:
    turn, altitude = motion_features(observations, trajectories)
    horizontal = torch.where(
        turn < -turn_threshold,
        torch.zeros_like(turn, dtype=torch.long),
        torch.where(
            turn > turn_threshold,
            torch.full_like(turn, 2, dtype=torch.long),
            torch.ones_like(turn, dtype=torch.long),
        ),
    )
    vertical = torch.where(
        altitude < -altitude_threshold,
        torch.zeros_like(altitude, dtype=torch.long),
        torch.where(
            altitude > altitude_threshold,
            torch.full_like(altitude, 2, dtype=torch.long),
            torch.ones_like(altitude, dtype=torch.long),
        ),
    )
    return horizontal * 3 + vertical


def _ordinal_membership(
    value: torch.Tensor, *, threshold: float, scale: float
) -> torch.Tensor:
    if threshold <= 0 or scale <= 0:
        raise ValueError("ordinal thresholds and scales must be positive")
    value = value.to(torch.float64)
    low_log = F.logsigmoid((-threshold - value) / scale)
    high_log = F.logsigmoid((value - threshold) / scale)
    middle_log = F.logsigmoid((threshold - value) / scale) + F.logsigmoid(
        (value + threshold) / scale
    )
    return torch.softmax(torch.stack((low_log, middle_log, high_log), dim=-1), dim=-1)


def event_membership(
    observations: torch.Tensor,
    trajectories: torch.Tensor,
    *,
    turn_threshold: float,
    altitude_threshold: float,
) -> torch.Tensor:
    """Map each trajectory to nine positive, model-independent event masses."""
    turn, altitude = motion_features(observations, trajectories)
    horizontal = _ordinal_membership(
        turn, threshold=turn_threshold, scale=turn_threshold / 4.0
    )
    vertical = _ordinal_membership(
        altitude, threshold=altitude_threshold, scale=altitude_threshold / 4.0
    )
    return torch.einsum("...i,...j->...ij", horizontal, vertical).flatten(-2)


def fit_definition(fold: int, *, chunk_size: int = 65536) -> dict[str, object]:
    protocol = load_legacy_data_protocol()
    protocol.assert_boundaries()
    dataset = load_dataset(protocol)
    training_data, _, _ = fold_subsets(protocol, dataset, fold)
    indices = _training_actor_indices(dataset, training_data)
    observations = dataset.obs_traj[indices].permute(0, 2, 1).to(torch.float64)
    target_endpoint = dataset.pred_traj[indices, :, -1].to(torch.float64)
    turn, altitude = motion_features(observations, target_endpoint[:, None])
    turn_threshold = float(torch.quantile(turn.abs(), 0.5))
    altitude_threshold = float(torch.quantile(altitude.abs(), 0.5))
    if turn_threshold <= 0 or altitude_threshold <= 0:
        raise RuntimeError("degenerate training-only E11 event threshold")
    squared_error = torch.zeros(24, 3, dtype=torch.float64)
    count = 0
    times = (5.0 * torch.arange(1, 25, dtype=torch.float64))[None, :, None]
    for start in range(0, indices.numel(), chunk_size):
        actor_index = indices[start : start + chunk_size]
        obs = dataset.obs_traj[actor_index].permute(0, 2, 1).to(torch.float64)
        future = dataset.pred_traj[actor_index].permute(0, 2, 1).to(torch.float64)
        velocity = obs[:, -1] - obs[:, -2]
        constant_velocity = obs[:, -1, None] + velocity[:, None] * times
        squared_error += (future - constant_velocity).square().sum(dim=0)
        count += int(actor_index.numel())
    variance = (squared_error / count).clamp_min(1e-6)
    return {
        "format_version": 1,
        "experiment_id": "E11",
        "fold": fold,
        "protocol_sha256": sha256(PROTOCOL),
        "fit_split": "fold_training_only",
        "training_actors": int(indices.numel()),
        "turn_threshold_radians": turn_threshold,
        "altitude_threshold_km": altitude_threshold,
        "constant_velocity_residual_variance_km2": variance.tolist(),
        "minimum_variance_km2": 1e-6,
        "validation_or_test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=(1, 2))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = fit_definition(args.fold)
    if args.output is None:
        args.output = ROOT / "artifacts/mabpt" / f"e11_definition_fold{args.fold}_v1.json"
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "fold": args.fold,
                "training_actors": result["training_actors"],
                "turn_threshold_radians": result["turn_threshold_radians"],
                "altitude_threshold_km": result["altitude_threshold_km"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
