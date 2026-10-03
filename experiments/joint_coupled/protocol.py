"""Frozen C129 protocol loader with a train-only data boundary."""

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
            raise ValueError("C129 may access only the train split")
        return self.dataset_root / "processed_data" / split

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
        for path, expected in required.items():
            if not path.is_file() or sha256(path) != expected:
                raise RuntimeError(f"C129 frozen input mismatch: {path.name}")
        train = self.split_path("train")
        if not train.is_dir() or not any(train.glob("*.txt")):
            raise RuntimeError("missing or empty C129 train split")
        if "C127_locked_test_reuse" not in self.payload["forbidden_mechanisms"]:
            raise RuntimeError("C129 must explicitly prohibit locked-test reuse")
        if "development_split_access" not in self.payload["forbidden_mechanisms"]:
            raise RuntimeError("C129 must explicitly prohibit development access")


def load_protocol() -> Protocol:
    path = Path(__file__).with_name("protocol.json").resolve()
    return Protocol(
        repository_root=path.parents[1],
        path=path,
        payload=json.loads(path.read_text(encoding="utf-8")),
    )


__all__ = ["Protocol", "load_protocol", "sha256"]
