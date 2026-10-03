"""Shared deterministic data and artifact helpers for C161."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from experiments.metric_exact.folds import indices_for_fold
from experiments.joint_coupled.train import limited_indices, set_seed
from model.utils import TrajectoryDataset, seed_worker, seq_collate

from .protocol import Protocol


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_dataset(protocol: Protocol) -> TrajectoryDataset:
    dataset = TrajectoryDataset(
        protocol.split_path("train").as_posix(),
        obs_len=16,
        obs_steps=1,
        pred_len=120,
        pred_step=5,
        delim=" ",
    )
    expected = protocol.payload["dataset"]
    if (
        len(dataset) != int(expected["expected_train_scenes"])
        or int(dataset.obs_traj.shape[0]) != int(expected["expected_train_actors"])
    ):
        raise RuntimeError("C161 train cohort identity mismatch")
    return dataset


def fold_subsets(
    protocol: Protocol,
    dataset: TrajectoryDataset,
    fold: int,
    *,
    max_train_scenes: int | None = None,
    max_validation_scenes: int | None = None,
) -> tuple[Subset, Subset, list[str]]:
    root = protocol.repository_root
    dataset_spec = protocol.payload["dataset"]
    dates = json.loads(
        (root / str(dataset_spec["train_scene_dates"])).read_text(encoding="utf-8")
    )["dates"]
    folds = json.loads(
        (root / str(dataset_spec["date_folds"])).read_text(encoding="utf-8")
    )
    train_indices, validation_indices, validation_dates = indices_for_fold(
        dates, folds, fold
    )
    train_indices = limited_indices(train_indices, max_train_scenes)
    complete_validation = validation_indices
    validation_indices = limited_indices(validation_indices, max_validation_scenes)
    if len(validation_indices) != len(complete_validation):
        by_index = dict(zip(complete_validation, validation_dates, strict=True))
        validation_dates = [by_index[index] for index in validation_indices]
    return Subset(dataset, train_indices), Subset(dataset, validation_indices), validation_dates


def loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    prefetch: int,
    generator: torch.Generator | None = None,
) -> DataLoader:
    options = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "collate_fn": seq_collate,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
    }
    if generator is not None:
        options["generator"] = generator
    if workers > 0:
        options.update({"persistent_workers": True, "prefetch_factor": prefetch})
    return DataLoader(**options)


def authorize(protocol: Protocol, fold: int, seed: int) -> None:
    replication = protocol.payload["replication"]
    if fold not in [int(value) for value in replication["folds"]]:
        raise RuntimeError("C161 fold is outside the frozen replication set")
    if seed != int(replication["seed"]):
        raise RuntimeError("C161 seed is outside the frozen replication set")


__all__ = [
    "atomic_json",
    "authorize",
    "fold_subsets",
    "load_dataset",
    "loader",
    "set_seed",
]
