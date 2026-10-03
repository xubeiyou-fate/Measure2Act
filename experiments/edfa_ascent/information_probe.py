"""Observed-only conditional neighbor information probe on C12 train/dev."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model.utils import TrajectoryDataset

from .protocol import load_protocol


def actor_features(
    dataset: TrajectoryDataset, maximum: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    counts = torch.tensor([end - start for start, end in dataset.seq_start_end])
    actor_sizes = torch.repeat_interleave(counts, counts)
    actor_scenes = torch.repeat_interleave(torch.arange(len(counts)), counts)
    multi_indices = torch.nonzero(actor_sizes > 1, as_tuple=False).flatten().numpy()
    if len(multi_indices) > maximum:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(multi_indices, size=maximum, replace=False))
    else:
        indices = multi_indices
    selected = torch.from_numpy(indices).to(torch.long)
    history = dataset.obs_traj[selected].permute(0, 2, 1).double()
    sampled = history[:, (0, 5, 10, 15)] - history[:, -1:, :]
    velocity = history[:, -1] - history[:, -2]
    acceleration = velocity - (history[:, -2] - history[:, -3])
    ego = torch.cat((sampled.flatten(1), velocity, acceleration), dim=-1)
    neighbor = torch.zeros((len(indices), 15), dtype=torch.float64)
    selected_scenes = actor_scenes[selected]
    unique_scenes, selected_counts = torch.unique_consecutive(
        selected_scenes, return_counts=True
    )
    cursor = 0
    for scene_tensor, selected_count_tensor in zip(unique_scenes, selected_counts):
        scene = int(scene_tensor)
        selected_count = int(selected_count_tensor)
        start, end = dataset.seq_start_end[scene]
        count = end - start
        scene_history = dataset.obs_traj[start:end].permute(0, 2, 1).double()
        position = scene_history[:, -1]
        scene_velocity = scene_history[:, -1] - scene_history[:, -2]
        relative_position = position[None] - position[:, None]
        relative_velocity = scene_velocity[None] - scene_velocity[:, None]
        distance = torch.linalg.vector_norm(relative_position[..., :2], dim=-1)
        distance.fill_diagonal_(float("inf"))
        nearest = distance.argmin(dim=-1)
        rows = torch.arange(count)
        rp = relative_position[rows, nearest]
        rv = relative_velocity[rows, nearest]
        rv2 = rv[..., :2].square().sum(dim=-1).clamp_min(1e-8)
        tcpa = (-(rp[..., :2] * rv[..., :2]).sum(dim=-1) / rv2).clamp(0, 120)
        closest = rp + tcpa[:, None] * rv
        values = torch.cat((
            rp,
            rv,
            distance[rows, nearest, None],
            rp[:, 2:].abs(),
            tcpa[:, None] / 120.0,
            torch.linalg.vector_norm(closest[..., :2], dim=-1, keepdim=True),
            closest[:, 2:].abs(),
            torch.tensor(float(count)).repeat(count, 1),
            relative_position.masked_fill(
                torch.eye(count, dtype=torch.bool)[:, :, None], 0.0
            ).sum(dim=1) / max(1, count - 1),
        ), dim=-1)
        selected_global = selected[cursor:cursor + selected_count]
        local = selected_global - start
        neighbor[cursor:cursor + selected_count] = values[local]
        cursor += selected_count
    return (
        ego.numpy(),
        neighbor.numpy(),
        actor_sizes[selected].numpy(),
        history.numpy(),
        indices,
    )


def matched_shuffle(features: np.ndarray, scene_size: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = features.copy()
    for size in np.unique(scene_size):
        indices = np.flatnonzero(scene_size == size)
        if len(indices) > 1:
            output[indices] = features[rng.permutation(indices)]
    return output


def error_summary(error: np.ndarray) -> dict:
    return {"mean_l2": float(error.mean()), "p95_l2": float(np.quantile(error, 0.95))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train-actors", type=int, default=100000)
    parser.add_argument("--max-dev-actors", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--output", type=Path, default=Path("artifacts/experiments/edfa_ascent/information_probe.json"))
    args = parser.parse_args()
    protocol = load_protocol()
    protocol.assert_manifest_sealed()
    datasets = {
        split: TrajectoryDataset(
            protocol.split_path(split).as_posix(),
            obs_len=16, obs_steps=1, pred_len=120, pred_step=5, delim=" ",
        )
        for split in ("train", "dev")
    }
    prepared = {}
    for split, dataset in datasets.items():
        maximum = args.max_train_actors if split == "train" else args.max_dev_actors
        ego, neighbor, sizes, history, indices = actor_features(
            dataset, maximum, args.seed + (split == "dev")
        )
        prepared[split] = {
            "ego": ego,
            "neighbor": neighbor,
            "sizes": sizes,
            "indices": indices,
            "history": history,
        }
    train = prepared["train"]
    dev = prepared["dev"]
    dev_shuffled = matched_shuffle(dev["neighbor"], dev["sizes"], args.seed + 11)
    results = {}
    future = datasets["train"].pred_traj.permute(0, 2, 1).numpy()[train["indices"]]
    dev_future = datasets["dev"].pred_traj.permute(0, 2, 1).numpy()[dev["indices"]]
    velocity = train["history"][:, -1] - train["history"][:, -2]
    dev_velocity = dev["history"][:, -1] - dev["history"][:, -2]
    for seconds in (30, 60, 90, 120):
        step = seconds // 5 - 1
        target = future[:, step] - (train["history"][:, -1] + velocity * seconds)
        dev_target = dev_future[:, step] - (dev["history"][:, -1] + dev_velocity * seconds)
        variants = {
            "ego": (train["ego"], dev["ego"]),
            "ego_plus_neighbors": (
                np.concatenate((train["ego"], train["neighbor"]), axis=1),
                np.concatenate((dev["ego"], dev["neighbor"]), axis=1),
            ),
            "ego_plus_shuffled_neighbors": (
                np.concatenate((train["ego"], train["neighbor"]), axis=1),
                np.concatenate((dev["ego"], dev_shuffled), axis=1),
            ),
        }
        horizon = {}
        for name, (fit_x, dev_x) in variants.items():
            model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            model.fit(fit_x, target)
            prediction = model.predict(dev_x)
            horizon[name] = error_summary(np.linalg.norm(prediction - dev_target, axis=1))
        baseline = horizon["ego"]
        for name in ("ego_plus_neighbors", "ego_plus_shuffled_neighbors"):
            horizon[name]["mean_gain"] = (
                baseline["mean_l2"] - horizon[name]["mean_l2"]
            ) / baseline["mean_l2"]
            horizon[name]["p95_gain"] = (
                baseline["p95_l2"] - horizon[name]["p95_l2"]
            ) / baseline["p95_l2"]
        results[str(seconds)] = horizon
    genuine = [results[str(seconds)]["ego_plus_neighbors"]["mean_gain"] for seconds in (30, 60, 90, 120)]
    shuffled = [results[str(seconds)]["ego_plus_shuffled_neighbors"]["mean_gain"] for seconds in (30, 60, 90, 120)]
    result = {
        "format_version": 1,
        "cycle": "C96_EDFA_ASCENT",
        "locked_test_used": False,
        "train_multi_actor_sample": len(train["indices"]),
        "dev_multi_actor_sample": len(dev["indices"]),
        "horizons": results,
        "gate": {
            "genuine_gain_at_least_5pct_on_three_horizons": sum(value >= 0.05 for value in genuine) >= 3,
            "genuine_p95_nonworse_on_three_horizons": sum(
                results[str(seconds)]["ego_plus_neighbors"]["p95_gain"] >= 0
                for seconds in (30, 60, 90, 120)
            ) >= 3,
            "mean_shuffled_retention": float(
                np.mean([
                    shuffled_value / genuine_value if genuine_value > 0 else float("inf")
                    for genuine_value, shuffled_value in zip(genuine, shuffled)
                ])
            ),
        },
    }
    result["gate"]["passed"] = (
        result["gate"]["genuine_gain_at_least_5pct_on_three_horizons"]
        and result["gate"]["genuine_p95_nonworse_on_three_horizons"]
        and result["gate"]["mean_shuffled_retention"] <= 0.2
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
