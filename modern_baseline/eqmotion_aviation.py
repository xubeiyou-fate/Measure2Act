"""Aircraft data and model adapters for the official EqMotion implementation.

The official source is kept unchanged under ``third_party/eqmotion_official``.
This module supplies the aviation-specific temporal grid, 3-D tensors, padding,
and a strict K=5 view of EqMotion's public diverse-prediction implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = ROOT / "third_party" / "eqmotion_official"


def _official_eqmotion_class():
    if not (OFFICIAL_ROOT / "eth_ucy" / "model_t.py").is_file():
        raise FileNotFoundError(
            "official EqMotion source is missing; expected "
            f"{OFFICIAL_ROOT / 'eth_ucy' / 'model_t.py'}"
        )
    official = str(OFFICIAL_ROOT)
    if official not in sys.path:
        sys.path.insert(0, official)
    # Keep the vendored official checkout byte-for-byte clean during imports.
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        from eth_ucy.model_t import EqMotion  # type: ignore[import-not-found]
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting

    return EqMotion


@dataclass(frozen=True)
class SceneRecord:
    history: torch.Tensor
    future: torch.Tensor
    source_file: str
    start_frame: float


class AviationSceneDataset(Dataset):
    """Load deterministic 5-second-grid multi-aircraft scenes from text files.

    Input rows are ``frame, actor_id, x, y, z, ...``. Files are inspected in
    descending-size order so a smoke run reaches valid long windows without
    scanning an entire dataset. This selection policy is deliberately not a
    formal sampling protocol.
    """

    history_steps = 16
    future_steps = 24
    temporal_stride_seconds = 5
    coordinate_dimensions = 3
    raw_window_steps = (history_steps + future_steps - 1) * temporal_stride_seconds + 1

    def __init__(
        self,
        data_dir: str | Path,
        *,
        delimiter: str,
        max_scenes: int,
        max_files: int | None = None,
        scene_stride_seconds: int = 1,
    ) -> None:
        super().__init__()
        if (
            max_scenes < 1
            or (max_files is not None and max_files < 1)
            or scene_stride_seconds < 1
        ):
            raise ValueError("scene and file limits and scene stride must be positive")
        self.data_dir = Path(data_dir).resolve()
        self.delimiter = delimiter
        self.max_scenes = int(max_scenes)
        self.max_files = int(max_files) if max_files is not None else None
        self.scene_stride_seconds = int(scene_stride_seconds)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(self.data_dir)
        files = sorted(
            (path for path in self.data_dir.iterdir() if path.is_file()),
            key=lambda path: (-path.stat().st_size, path.name),
        )
        if self.max_files is not None:
            files = files[: self.max_files]
        self.inspected_files = [path.name for path in files]
        self.records: list[SceneRecord] = []
        for path in files:
            self.records.extend(self._read_file(path, self.max_scenes - len(self.records)))
            if len(self.records) >= self.max_scenes:
                break
        if not self.records:
            raise RuntimeError(
                f"no valid {self.raw_window_steps}-second aircraft windows found in "
                f"the first {len(files)} inspected files under {self.data_dir}"
            )

    def _read_file(self, path: Path, remaining: int) -> list[SceneRecord]:
        try:
            rows = np.loadtxt(path, delimiter=self.delimiter, ndmin=2)
        except ValueError:
            return []
        if rows.shape[1] < 5 or rows.shape[0] < self.history_steps + self.future_steps:
            return []
        rows = rows[np.isfinite(rows[:, :5]).all(axis=1)]
        if not len(rows):
            return []
        frames = np.unique(rows[:, 0])
        if len(frames) < self.raw_window_steps:
            return []
        selected_offsets = np.arange(
            0,
            self.raw_window_steps,
            self.temporal_stride_seconds,
            dtype=np.int64,
        )
        if len(selected_offsets) != self.history_steps + self.future_steps:
            raise AssertionError("invalid 5-second temporal grid")
        output: list[SceneRecord] = []
        final_start = len(frames) - self.raw_window_steps
        for start in range(0, final_start + 1, self.scene_stride_seconds):
            raw_frames = frames[start : start + self.raw_window_steps]
            if not np.allclose(np.diff(raw_frames), 1.0, atol=0.0, rtol=0.0):
                continue
            selected_frames = raw_frames[selected_offsets]
            at_start = rows[rows[:, 0] == selected_frames[0], 1]
            actor_ids = np.unique(at_start)
            trajectories: list[np.ndarray] = []
            for actor_id in actor_ids:
                actor = rows[rows[:, 1] == actor_id]
                order = np.argsort(actor[:, 0], kind="stable")
                actor = actor[order]
                positions = np.searchsorted(actor[:, 0], selected_frames)
                if np.any(positions >= len(actor)):
                    continue
                matched = actor[positions]
                if not np.array_equal(matched[:, 0], selected_frames):
                    continue
                trajectories.append(matched[:, 2:5].astype(np.float32, copy=False))
            if not trajectories:
                continue
            scene = np.stack(trajectories, axis=0)
            history = torch.from_numpy(scene[:, : self.history_steps].copy())
            future = torch.from_numpy(scene[:, self.history_steps :].copy())
            output.append(
                SceneRecord(
                    history=history,
                    future=future,
                    source_file=path.name,
                    start_frame=float(selected_frames[0]),
                )
            )
            if len(output) >= remaining:
                break
        return output

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> SceneRecord:
        return self.records[index]

    def receipt(self) -> dict[str, object]:
        return {
            "data_dir": self.data_dir.as_posix(),
            "delimiter": self.delimiter,
            "selection": "descending_source_file_size_then_ascending_name",
            "inspected_files": self.inspected_files,
            "loaded_scenes": len(self.records),
            "loaded_actors": int(sum(record.history.shape[0] for record in self.records)),
            "source_records": [
                {
                    "file": record.source_file,
                    "start_frame": record.start_frame,
                    "actors": int(record.history.shape[0]),
                }
                for record in self.records
            ],
        }


class CachedTrajectorySceneDataset(Dataset):
    """Expose the repository's exact ASCENT temporal protocol to EqMotion."""

    def __init__(self, dataset: Dataset, indices: list[int] | None = None) -> None:
        self.dataset = dataset
        self.indices = indices if indices is not None else list(range(len(dataset)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> SceneRecord:
        source_index = self.indices[index]
        values = self.dataset[source_index]
        history = values[0].transpose(1, 2).contiguous()
        future = values[1].transpose(1, 2).contiguous()
        if history.shape[1:] != (16, 3) or future.shape[1:] != (24, 3):
            raise RuntimeError("cached scene differs from the 16x1s to 24x5s protocol")
        start = float(values[6][0]) if len(values[6]) else 0.0
        return SceneRecord(
            history=history,
            future=future,
            source_file=f"cached_scene_{source_index}",
            start_frame=start,
        )


def aviation_collate(records: Iterable[SceneRecord]) -> dict[str, object]:
    records = list(records)
    if not records:
        raise ValueError("cannot collate an empty batch")
    batch = len(records)
    maximum = max(record.history.shape[0] for record in records)
    history = torch.zeros(batch, maximum, 16, 3, dtype=torch.float32)
    future = torch.zeros(batch, maximum, 24, 3, dtype=torch.float32)
    valid = torch.zeros(batch, maximum, dtype=torch.bool)
    num_valid = torch.empty(batch, dtype=torch.long)
    metadata: list[dict[str, object]] = []
    for index, record in enumerate(records):
        actors = record.history.shape[0]
        if record.history.shape[1:] != (16, 3) or record.future.shape[1:] != (24, 3):
            raise ValueError("aviation scene does not match the registered 16-to-24 3-D grid")
        center = record.history[:, -1].mean(dim=0, keepdim=True)
        history[index, :actors] = record.history - center[:, None]
        future[index, :actors] = record.future - center[:, None]
        valid[index, :actors] = True
        num_valid[index] = actors
        metadata.append(
            {
                "source_file": record.source_file,
                "start_frame": record.start_frame,
                "actors": int(actors),
                "translation_center": center.squeeze(0).tolist(),
            }
        )
    return {
        "history": history,
        "future": future,
        "valid": valid,
        "num_valid": num_valid,
        "metadata": metadata,
    }


class EqMotionAviation(nn.Module):
    """Strict 3-D, K=5 wrapper around the unchanged official EqMotion source."""

    def __init__(
        self,
        *,
        device: torch.device,
        hidden_nf: int = 32,
        channels: int = 16,
        layers: int = 2,
        modes: int = 5,
    ) -> None:
        super().__init__()
        if modes != 5:
            raise ValueError("the registered aviation adapter fixes K=5")
        EqMotion = _official_eqmotion_class()
        self.modes = modes
        self.official_model = EqMotion(
            in_node_nf=16,
            in_edge_nf=2,
            hidden_nf=hidden_nf,
            in_channel=16,
            hid_channel=channels,
            out_channel=24,
            device=device,
            n_layers=layers,
            recurrent=True,
            norm_diff=False,
            tanh=False,
        )

    def forward(self, history: torch.Tensor, num_valid: torch.Tensor) -> torch.Tensor:
        if history.ndim != 4 or history.shape[2:] != (16, 3):
            raise ValueError("history must have shape [batch, agents, 16, 3]")
        velocity = torch.zeros_like(history)
        velocity[:, :, 1:] = history[:, :, 1:] - history[:, :, :-1]
        velocity[:, :, 0] = velocity[:, :, 1]
        node_features = torch.linalg.vector_norm(velocity, dim=-1).detach()
        predictions, _ = self.official_model(
            node_features,
            history.detach(),
            velocity,
            num_valid.to(torch.int64),
        )
        if predictions.shape[2] < self.modes:
            raise RuntimeError("official EqMotion returned fewer than five diverse heads")
        selected = predictions[:, :, : self.modes]
        agents = torch.arange(history.shape[1], device=history.device)[None]
        mask = (agents < num_valid[:, None]).to(selected.dtype)
        return selected * mask[:, :, None, None, None]


def best_of_k_ade_loss(
    predictions: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    if predictions.ndim != 5 or predictions.shape[2:] != (5, 24, 3):
        raise ValueError("predictions must have shape [batch, agents, 5, 24, 3]")
    displacement = torch.linalg.vector_norm(predictions - target[:, :, None], dim=-1)
    actor_loss = displacement.mean(dim=-1).min(dim=-1).values
    weights = valid.to(actor_loss.dtype)
    return (actor_loss * weights).sum() / weights.sum().clamp_min(1.0)


def valid_actor_tensors(
    predictions: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return predictions[valid], target[valid]


__all__ = [
    "AviationSceneDataset",
    "CachedTrajectorySceneDataset",
    "EqMotionAviation",
    "SceneRecord",
    "aviation_collate",
    "best_of_k_ade_loss",
    "valid_actor_tensors",
]
