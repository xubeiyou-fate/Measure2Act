"""Deterministic date-block folds for C127 train-only model development."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np

from .protocol import Protocol, load_protocol, sha256


def assign_date_folds(
    scene_dates: list[str], *, fold_count: int = 5, seed: int = 127
) -> dict[str, object]:
    if fold_count < 2:
        raise ValueError("fold_count must be at least two")
    counts = Counter(scene_dates)
    if len(counts) < fold_count:
        raise ValueError("fewer dates than folds")
    rng = np.random.default_rng(seed)
    items = list(counts.items())
    rng.shuffle(items)
    items.sort(key=lambda item: item[1], reverse=True)
    folds: list[dict[str, object]] = [
        {"fold": index, "dates": [], "scene_count": 0} for index in range(fold_count)
    ]
    for date, count in items:
        minimum = min(int(fold["scene_count"]) for fold in folds)
        candidates = [
            index for index, fold in enumerate(folds)
            if int(fold["scene_count"]) == minimum
        ]
        selected = int(rng.choice(candidates))
        folds[selected]["dates"].append(date)
        folds[selected]["scene_count"] = int(folds[selected]["scene_count"]) + count
    for fold in folds:
        fold["dates"] = sorted(fold["dates"])
        fold["date_count"] = len(fold["dates"])
    fold_by_date = {
        date: int(fold["fold"])
        for fold in folds
        for date in fold["dates"]
    }
    return {
        "format_version": 1,
        "cycle": "C127_metric_exact_score_isolated_ascent",
        "assignment_seed": seed,
        "unit": "calendar_date",
        "scene_count": len(scene_dates),
        "date_count": len(counts),
        "fold_count": fold_count,
        "folds": folds,
        "fold_by_date": fold_by_date,
    }


def build_fold_artifact(
    protocol: Protocol | None = None, output: Path | None = None
) -> dict[str, object]:
    protocol = protocol or load_protocol()
    date_path = protocol.repository_root / str(
        protocol.payload["dataset"]["train_scene_dates"]
    )
    scene_dates = json.loads(date_path.read_text(encoding="utf-8"))["dates"]
    settings = protocol.payload["folds"]
    artifact = assign_date_folds(
        scene_dates,
        fold_count=int(settings["count"]),
        seed=int(settings["assignment_seed"]),
    )
    artifact["protocol_sha256"] = sha256(protocol.path)
    artifact["scene_dates_sha256"] = sha256(date_path)
    output = output or protocol.repository_root / str(settings["artifact"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


def indices_for_fold(
    scene_dates: list[str], artifact: dict[str, object], fold: int
) -> tuple[list[int], list[int], list[str]]:
    if not 0 <= fold < int(artifact["fold_count"]):
        raise ValueError("fold index outside the frozen range")
    validation_dates = set(artifact["folds"][fold]["dates"])
    train_indices = [
        index for index, date in enumerate(scene_dates) if date not in validation_dates
    ]
    validation_indices = [
        index for index, date in enumerate(scene_dates) if date in validation_dates
    ]
    validation_scene_dates = [scene_dates[index] for index in validation_indices]
    if set(train_indices) & set(validation_indices):
        raise RuntimeError("C127 train/validation fold overlap")
    if len(train_indices) + len(validation_indices) != len(scene_dates):
        raise RuntimeError("C127 fold does not cover every train scene")
    return train_indices, validation_indices, validation_scene_dates


__all__ = ["assign_date_folds", "build_fold_artifact", "indices_for_fold"]
