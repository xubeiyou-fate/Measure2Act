"""Frozen protocol loading and split-boundary checks for C99."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Protocol:
    repository_root: Path
    path: Path
    payload: dict[str, object]

    @property
    def manifest_path(self) -> Path:
        return self.repository_root / str(self.payload["dataset"]["manifest"])

    def manifest(self) -> dict[str, object]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def split_path(self, split: str) -> Path:
        return self.repository_root / str(self.payload["dataset"]["root"]) / "processed_data" / split

    def assert_development_sealed(self) -> None:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        for split in ("train", "dev", "locked_test"):
            path = self.split_path(split)
            if not path.is_dir() or not any(path.glob("*.txt")):
                raise RuntimeError(f"missing or empty C99 split: {path}")
        locked_receipt = self.repository_root / str(
            self.payload["locked_test_policy"]["receipt"]
        )
        if locked_receipt.exists():
            raise RuntimeError("C99 locked-test receipt already exists")


def load_protocol() -> Protocol:
    path = Path(__file__).with_name("protocol.json").resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    root = path.parents[1]
    return Protocol(repository_root=root, path=path, payload=payload)


__all__ = ["Protocol", "load_protocol", "sha256"]
