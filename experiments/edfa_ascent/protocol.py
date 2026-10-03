"""Frozen C96 protocol and locked-test enforcement."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent
DEFAULT_PROTOCOL_PATH = PACKAGE_ROOT / "preregistration.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ExperimentProtocol:
    path: Path
    payload: dict[str, Any]

    @property
    def dataset_root(self) -> Path:
        return (REPOSITORY_ROOT / self.payload["dataset"]["root"]).resolve()

    @property
    def manifest_path(self) -> Path:
        return (REPOSITORY_ROOT / self.payload["dataset"]["manifest"]).resolve()

    @property
    def gate_artifact(self) -> Path:
        relative = self.payload["locked_test_policy"]["required_gate_artifact"]
        return (REPOSITORY_ROOT / relative).resolve()

    def manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def split_path(self, split: str) -> Path:
        return self.dataset_root / "processed_data" / split

    def assert_split_allowed(self, split: str, allow_locked_test: bool = False) -> None:
        locked = self.payload["dataset"]["locked_test_split"]
        if split != locked:
            return
        if not allow_locked_test:
            raise RuntimeError("C96 locked test is sealed until every development gate passes")
        if not self.gate_artifact.exists():
            raise RuntimeError(f"Missing locked-test gate artifact: {self.gate_artifact}")
        gate = json.loads(self.gate_artifact.read_text(encoding="utf-8"))
        if gate.get("all_development_gates_passed") is not True:
            raise RuntimeError("Development gate artifact does not authorize locked-test access")
        if gate.get("locked_test_evaluations", 0) >= int(
            self.payload["locked_test_policy"]["maximum_evaluations"]
        ):
            raise RuntimeError("C96 locked test has already been evaluated once")

    def assert_manifest_sealed(self) -> None:
        manifest = self.manifest()
        if manifest.get("date_overlap") != 0 or manifest.get("file_overlap") != 0:
            raise RuntimeError("C12 manifest contains split overlap")
        if manifest.get("locked_test_evaluated") is not False:
            raise RuntimeError("C12 locked-test flag is not sealed")


def load_protocol(path: Path | None = None) -> ExperimentProtocol:
    resolved = (path or DEFAULT_PROTOCOL_PATH).resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("cycle") != "C96_EDFA_ASCENT":
        raise ValueError("Unexpected experiment protocol")
    return ExperimentProtocol(path=resolved, payload=payload)
