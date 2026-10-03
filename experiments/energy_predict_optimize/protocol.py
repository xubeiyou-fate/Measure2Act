"""Frozen C134 protocol and input-boundary checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Protocol:
    repository_root: Path
    path: Path
    payload: dict[str, object]

    @property
    def dataset_root(self) -> Path:
        return self.repository_root / str(self.payload["dataset"]["root"])

    @property
    def manifest_path(self) -> Path:
        return self.repository_root / str(self.payload["dataset"]["manifest"])

    def split_path(self, split: str) -> Path:
        if split != "train":
            raise ValueError("C134 may access only the train split")
        return self.dataset_root / "processed_data" / split

    def control_path(self, family: str, field: str) -> Path:
        return self.repository_root / str(self.payload["controls"][family][field])

    def assert_boundaries(self) -> None:
        dataset = self.payload["dataset"]
        required = {
            self.manifest_path: str(dataset["manifest_sha256"]),
            self.repository_root / str(dataset["train_scene_dates"]): str(
                dataset["train_scene_dates_sha256"]
            ),
            self.repository_root / str(dataset["date_folds"]): str(
                dataset["date_folds_sha256"]
            ),
        }
        for family, entry in self.payload["controls"].items():
            for field in ("protocol", "fold0_summary"):
                required[self.repository_root / str(entry[field])] = str(
                    entry[f"{field}_sha256"]
                )
            if family == "C130_R1":
                required[self.repository_root / str(entry["fold0_checkpoint"])] = str(
                    entry["fold0_checkpoint_sha256"]
                )
        for path, expected in required.items():
            if not path.is_file() or sha256(path) != expected:
                raise RuntimeError(f"C134 frozen input mismatch: {path}")
        train = self.split_path("train")
        if not train.is_dir() or not any(train.glob("*.txt")):
            raise RuntimeError("missing or empty C134 train split")
        forbidden = self.payload["forbidden_mechanisms"]
        for mechanism in (
            "trajectory_or_control_residual",
            "learned_gate_router_or_mixture_of_experts",
            "development_split_access",
            "C127_locked_test_reuse",
        ):
            if mechanism not in forbidden:
                raise RuntimeError(f"C134 must prohibit {mechanism}")


def load_protocol() -> Protocol:
    path = Path(__file__).with_name("protocol.json").resolve()
    return Protocol(
        repository_root=path.parents[1],
        path=path,
        payload=json.loads(path.read_text(encoding="utf-8")),
    )


__all__ = ["Protocol", "load_protocol", "sha256"]
