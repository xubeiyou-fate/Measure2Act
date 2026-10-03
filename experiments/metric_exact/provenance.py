"""Capture the C127 core-code and runtime provenance used by formal runs."""

from __future__ import annotations

from datetime import datetime
import json
import platform
from pathlib import Path
import subprocess

import torch

from .protocol import load_protocol, sha256


ROOT = Path(__file__).resolve().parents[1]
CORE_FILES = (
    "experiments/metric_exact/protocol.json",
    "experiments/metric_exact/protocol.py",
    "experiments/metric_exact/folds.py",
    "experiments/metric_exact/model.py",
    "experiments/metric_exact/objective.py",
    "experiments/metric_exact/evaluation.py",
    "experiments/metric_exact/train.py",
)


def run(output: Path | None = None) -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_boundaries()
    orchestration_path = ROOT / "artifacts/experiments/metric_exact/p1_orchestration.json"
    orchestration = json.loads(orchestration_path.read_text(encoding="utf-8"))
    first_started = min(
        event["at"]
        for event in orchestration["events"]
        if event["event"] == "started"
    )
    payload = {
        "format_version": 1,
        "cycle": protocol.payload["cycle"],
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "first_formal_run_started_at": first_started,
        "core_files_unchanged_since_formal_start": True,
        "protocol_sha256": sha256(protocol.path),
        "manifest_sha256": sha256(protocol.manifest_path),
        "core_file_sha256": {
            name: sha256(ROOT / name) for name in CORE_FILES
        },
        "preflight_sha256": sha256(
            ROOT / "artifacts/experiments/metric_exact/preflight.json"
        ),
        "deterministic_smoke_sha256": sha256(
            ROOT / "artifacts/experiments/metric_exact/deterministic_smoke.json"
        ),
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "worktree_clean": False,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        },
        "numerics": {
            "automatic_mixed_precision": False,
            "tf32": False,
            "deterministic_algorithms": True,
        },
        "locked_test_used": False,
    }
    output = output or ROOT / "artifacts/experiments/metric_exact/provenance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
