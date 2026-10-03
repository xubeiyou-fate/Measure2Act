"""Frozen boundaries and artifact helpers for C162."""

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


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"C162 refuses to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class Protocol:
    repository_root: Path
    path: Path
    payload: dict[str, object]

    @property
    def dataset_root(self) -> Path:
        return self.repository_root / str(self.payload["dataset"]["root"])

    def split_path(self, split: str) -> Path:
        if split != "train":
            raise ValueError("C162 may access only the train split")
        return self.dataset_root / "processed_data" / split

    def fold_inputs(self, fold: int) -> dict[str, str]:
        folds = self.payload["frozen_inputs"]["folds"]
        if str(fold) not in folds:
            raise ValueError(f"fold {fold} is not frozen for C162")
        return folds[str(fold)]

    def assert_boundaries(self) -> None:
        root = self.repository_root
        dataset = self.payload["dataset"]
        required = {
            root / str(dataset["manifest"]): str(dataset["manifest_sha256"]),
            root / str(dataset["train_scene_dates"]): str(
                dataset["train_scene_dates_sha256"]
            ),
            root / str(dataset["date_folds"]): str(dataset["date_folds_sha256"]),
        }
        for spec in self.payload["frozen_inputs"].values():
            if not isinstance(spec, dict) or "path" not in spec:
                continue
            required[root / str(spec["path"])] = str(spec["sha256"])
        for fold_spec in self.payload["frozen_inputs"]["folds"].values():
            required[root / str(fold_spec["baseline_path"])] = str(
                fold_spec["baseline_sha256"]
            )
            required[root / str(fold_spec["candidate_path"])] = str(
                fold_spec["candidate_sha256"]
            )
        for path, expected in required.items():
            if not path.is_file():
                raise RuntimeError(f"missing frozen C162 input: {path}")
            observed = sha256(path)
            if observed != expected:
                raise RuntimeError(
                    f"frozen C162 input hash mismatch: {path} "
                    f"expected={expected} observed={observed}"
                )
        train = self.split_path("train")
        if not train.is_dir() or not any(train.glob("*.txt")):
            raise RuntimeError("missing or empty C162 train split")
        forbidden = set(self.payload["forbidden_mechanisms"])
        required_forbidden = {
            "trajectory_or_control_residual",
            "learned_gate_router_or_mixture_of_experts",
            "temperature_or_assignment_scale_search",
            "KL_or_energy_weight_search",
            "future_target_in_forward",
            "development_split_access",
            "C127_locked_test_reuse",
        }
        if not required_forbidden.issubset(forbidden):
            raise RuntimeError("C162 forbidden-mechanism boundary is incomplete")


def load_protocol() -> Protocol:
    path = Path(__file__).with_name("protocol.json").resolve()
    return Protocol(
        repository_root=path.parents[1],
        path=path,
        payload=json.loads(path.read_text(encoding="utf-8")),
    )


__all__ = ["Protocol", "atomic_json", "load_protocol", "sha256"]
