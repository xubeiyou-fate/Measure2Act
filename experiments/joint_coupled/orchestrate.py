"""Sequential C129 orchestration with a hard fold-0 stop gate."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from .protocol import load_protocol, sha256
from .summarize import summarize


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/joint_coupled"


def _command(fold: int, device: str, *, smoke: bool = False) -> list[str]:
    protocol = load_protocol()
    training = protocol.payload["training"]
    command = [
        sys.executable,
        "-m",
        "experiments.joint_coupled.train",
        "--fold",
        str(fold),
        "--seed",
        str(protocol.payload["screening"]["seed"]),
        "--device",
        device,
        "--num-workers",
        str(training["num_workers_per_run"]),
        "--prefetch-factor",
        str(training["prefetch_factor"]),
        "--epochs",
        str(training["epochs"]),
        "--batch-size",
        str(training["batch_size"]),
        "--eval-batch-size",
        str(training["evaluation_batch_size"]),
        "--resume",
    ]
    if smoke:
        command.extend(["--epochs", "1", "--batch-size", "8", "--eval-batch-size", "16", "--num-workers", "2", "--prefetch-factor", "2", "--max-train-scenes", "256", "--max-validation-scenes", "128"])
    return command


def _run_one(fold: int, device: str, *, log_name: str) -> int:
    log_path = ARTIFACT_ROOT / "logs" / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.run(
            _command(fold, device),
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    return int(process.returncode)


def _run_parallel(pairs: list[tuple[int, str]]) -> dict[str, int]:
    processes = []
    handles = []
    for fold, device in pairs:
        log_path = ARTIFACT_ROOT / "logs" / f"fold{fold}_{device.replace(':', '_')}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("a", encoding="utf-8")
        handles.append(handle)
        processes.append(
            (
                fold,
                device,
                subprocess.Popen(
                    _command(fold, device),
                    cwd=ROOT,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                ),
            )
        )
    status: dict[str, int] = {}
    try:
        for fold, device, process in processes:
            status[f"fold{fold}"] = int(process.wait())
    finally:
        for handle in handles:
            handle.close()
    return status


def run(phase: str, *, device: str, devices: tuple[str, str]) -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_boundaries()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = ARTIFACT_ROOT / "orchestration.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if phase == "P1":
            preflight = ARTIFACT_ROOT / "preflight.json"
            if not preflight.is_file():
                raise RuntimeError("C129 preflight must pass before P1")
            preflight_payload = json.loads(preflight.read_text(encoding="utf-8"))
            if (
                preflight_payload.get("passed") is not True
                or preflight_payload.get("protocol_sha256") != sha256(protocol.path)
                or preflight_payload.get("locked_test_used") is not False
                or preflight_payload.get("development_used") is not False
            ):
                raise RuntimeError("C129 preflight identity or boundary check failed")
            code = _run_one(0, device, log_name=f"fold0_{device.replace(':', '_')}.log")
            if code != 0:
                raise RuntimeError(f"C129 fold0 failed with exit code {code}")
            result = summarize()
            if result["decision"] != "REPLICATION_AUTHORIZED":
                return result
            return result

        if phase != "P2":
            raise ValueError("phase must be P1 or P2")
        decision_path = ARTIFACT_ROOT / "p1_decision.json"
        if not decision_path.is_file():
            raise RuntimeError("C129 P1 decision is required before P2")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if decision.get("decision") != "REPLICATION_AUTHORIZED":
            raise RuntimeError("C129 fold0 did not authorize replication")
        if decision.get("protocol_sha256") != sha256(protocol.path):
            raise RuntimeError("C129 P1 decision protocol hash mismatch")
        if (
            decision.get("locked_test_used") is not False
            or decision.get("development_used") is not False
        ):
            raise RuntimeError("C129 P1 decision crossed a frozen data boundary")
        status = _run_parallel([(1, devices[0]), (2, devices[1])])
        if any(code != 0 for code in status.values()):
            raise RuntimeError(f"C129 replication failed: {status}")
        return summarize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("P1", "P2"), required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--devices", nargs=2, default=("cuda:0", "cuda:1"))
    args = parser.parse_args()
    print(json.dumps(run(args.phase, device=args.device, devices=tuple(args.devices)), indent=2))


if __name__ == "__main__":
    main()
