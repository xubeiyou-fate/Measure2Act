"""C96 data loading helpers, including exact source-date reconstruction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def build_scene_dates(
    split_directory: Path,
    manifest_path: Path,
    split: str,
    observation_length: int = 16,
    prediction_length: int = 120,
) -> list[str]:
    """Reproduce TrajectoryDataset's retained-window order and attach dates."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["partitions"][split]["records"]
    date_by_name = {record["name"]: record["assigned_date"] for record in records}
    sequence_length = observation_length + prediction_length
    dates: list[str] = []
    for path in sorted(item for item in split_directory.iterdir() if item.is_file()):
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    rows.append([float(value) for value in stripped.split(" ")])
        if not rows:
            continue
        data = np.asarray(rows)
        frames = np.unique(data[:, 0]).tolist()
        number = int(np.ceil(len(frames) - sequence_length + 1))
        for index in range(0, number + 1):
            window_frames = frames[index:index + sequence_length]
            if len(window_frames) != sequence_length:
                continue
            window = data[np.isin(data[:, 0], window_frames)]
            retained = 0
            for agent_id in np.unique(window[:, 1]):
                actor = window[window[:, 1] == agent_id]
                front = frames.index(actor[0, 0]) - index
                end = frames.index(actor[-1, 0]) - index + 1
                if end - front != sequence_length:
                    continue
                coordinates = actor[:, 2:].T
                observed = coordinates[:, :observation_length]
                predicted = coordinates[
                    :, observation_length + 5 - 1::5
                ]
                sampled = np.hstack((observed, predicted))
                if sampled.shape[1] == observation_length + int(
                    np.ceil(prediction_length / 5)
                ):
                    retained += 1
            if retained > 0:
                dates.append(date_by_name[path.name])
    return dates
