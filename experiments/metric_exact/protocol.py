"""Frozen C127 protocol loading and boundary checks."""

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
        return self.dataset_root / "processed_data" / split

    def manifest(self) -> dict[str, object]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def assert_boundaries(self) -> None:
        dataset = self.payload["dataset"]
        required = {
            self.manifest_path: str(dataset["manifest_sha256"]),
            self.repository_root / str(dataset["train_scene_dates"]): str(
                dataset["train_scene_dates_sha256"]
            ),
            self.repository_root / str(dataset["development_scene_dates"]): str(
                dataset["development_scene_dates_sha256"]
            ),
        }
        for path, expected in required.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            if sha256(path) != expected:
                raise RuntimeError(f"C127 frozen input hash mismatch: {path.name}")
        manifest = self.manifest()
        if manifest.get("locked_test_evaluated") is not False:
            raise RuntimeError("C127 requires the locked test to remain sealed")
        receipt = self.repository_root / str(
            self.payload["locked_test_policy"]["receipt"]
        )
        if receipt.exists():
            raise RuntimeError("C127 locked-test receipt already exists")
        for split in ("train", "dev", "locked_test"):
            directory = self.split_path(split)
            if not directory.is_dir() or not any(directory.glob("*.txt")):
                raise RuntimeError(f"missing or empty C127 split: {split}")


def load_protocol() -> Protocol:
    path = Path(__file__).with_name("protocol.json").resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Protocol(repository_root=path.parents[1], path=path, payload=payload)


__all__ = ["Protocol", "load_protocol", "sha256"]
