"""Run the resumable ten-cell formal E15 training and evaluation queue."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from .run import AIRPORTS, PROTOCOL_PATH, ROOT, SEEDS, atomic_json, sha256


def training_summary(airport: str, seed: int) -> Path:
    return ROOT / f"runs/journal_e14_e20_20260817/e15/{airport}/seed{seed}/training_summary.json"


def checkpoint(airport: str, seed: int) -> Path:
    return training_summary(airport, seed).with_name("last.pt")


def evaluation(output_root: Path, airport: str, seed: int) -> Path:
    return output_root / f"{airport}_seed{seed}_test_v1.json"


def valid_json(path: Path, *, complete: bool = False) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return not complete or payload.get("complete") is True


def valid_training(airport: str, seed: int) -> bool:
    path = training_summary(airport, seed)
    if not path.is_file() or not checkpoint(airport, seed).is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("complete") is True
        and payload.get("formal") is True
        and payload.get("airport") == airport
        and int(payload.get("seed", -1)) == seed
        and int(payload.get("epochs", -1)) == 20
        and payload.get("protocol_sha256") == sha256(PROTOCOL_PATH)
        and payload.get("checkpoint") == checkpoint(airport, seed).relative_to(ROOT).as_posix()
        and payload.get("checkpoint_sha256") == sha256(checkpoint(airport, seed))
        and payload.get("integrity", {}).get("test_used") is False
        and payload.get("integrity", {}).get("third_dataset_used") is False
    )


def valid_evaluation(output_root: Path, airport: str, seed: int) -> bool:
    path = evaluation(output_root, airport, seed)
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("complete") is True
        and payload.get("formal") is True
        and payload.get("airport") == airport
        and int(payload.get("seed", -1)) == seed
        and payload.get("protocol_sha256") == sha256(PROTOCOL_PATH)
        and payload.get("checkpoint_sha256") == sha256(checkpoint(airport, seed))
        and payload.get("integrity", {}).get("test_used_for_selection") is False
        and payload.get("integrity", {}).get("third_dataset_used") is False
    )


def atomic_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for attempt in range(50):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 49:
                raise
            time.sleep(0.1)


def command_for(stage: str, output_root: Path, airport: str, seed: int, args: argparse.Namespace) -> list[str]:
    run_dir = training_summary(airport, seed).parent
    if stage == "train":
        command = [
            sys.executable, "-m", "mabpt.e15_independent_probabilistic_baseline.run", "train",
            "--airport", airport, "--seed", str(seed), "--epochs", "20",
            "--batch-size", str(args.batch_size), "--workers", str(args.workers),
            "--device", args.device, "--run-dir", str(run_dir),
        ]
        if checkpoint(airport, seed).is_file():
            command.append("--resume")
        return command
    return [
        sys.executable, "-m", "mabpt.e15_independent_probabilistic_baseline.run", "evaluate",
        "--airport", airport, "--seed", str(seed), "--checkpoint", str(checkpoint(airport, seed)),
        "--output", str(evaluation(output_root, airport, seed)), "--batch-size", str(args.batch_size),
        "--workers", str(args.workers), "--device", args.device,
        "--authorize-retrospective-test",
    ]


def run_stage(stage: str, output_root: Path, args: argparse.Namespace, parallel: int) -> None:
    tasks = [(airport, seed) for airport in AIRPORTS for seed in SEEDS]
    if stage == "train":
        tasks = [task for task in tasks if not valid_training(*task)]
    else:
        tasks = [task for task in tasks if not valid_evaluation(output_root, *task)]
    attempts = {task: 0 for task in tasks}
    running: dict[tuple[str, int], tuple[subprocess.Popen, Any, Any, Path]] = {}
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
    while tasks or running:
        while tasks and len(running) < parallel:
            task = tasks.pop(0)
            airport, seed = task
            attempts[task] += 1
            label = f"{stage}_{airport}_seed{seed}_attempt{attempts[task]}"
            stdout_path = logs / f"{label}.out.log"
            stderr_path = logs / f"{label}.err.log"
            stdout = stdout_path.open("a", encoding="utf-8")
            stderr = stderr_path.open("a", encoding="utf-8")
            process = subprocess.Popen(
                command_for(stage, output_root, airport, seed, args),
                cwd=ROOT, env=environment, stdout=stdout, stderr=stderr,
            )
            running[task] = (process, stdout, stderr, stderr_path)
        finished = []
        for task, (process, stdout, stderr, stderr_path) in running.items():
            returncode = process.poll()
            if returncode is None:
                continue
            stdout.close()
            stderr.close()
            airport, seed = task
            expected = training_summary(airport, seed) if stage == "train" else evaluation(output_root, airport, seed)
            valid = valid_training(airport, seed) if stage == "train" else valid_evaluation(output_root, airport, seed)
            if not valid:
                if attempts[task] < args.retries:
                    tasks.append(task)
                else:
                    tail = "\n".join(stderr_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
                    raise RuntimeError(f"E15 {stage} failed after {attempts[task]} attempts for {task}: {tail}")
            finished.append(task)
        for task in finished:
            del running[task]
        state = {
            "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "stage": stage,
            "pending": [f"{a}_seed{s}" for a, s in tasks],
            "running": [f"{a}_seed{s}" for a, s in running],
            "complete": sum(
                valid_training(a, s) if stage == "train" else valid_evaluation(output_root, a, s)
                for a in AIRPORTS for s in SEEDS
            ),
            "expected": len(AIRPORTS) * len(SEEDS),
        }
        atomic_status(output_root / "automated_flow_state_v1.json", state)
        if running and not finished:
            time.sleep(3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--parallel-train", type=int, default=4)
    parser.add_argument("--parallel-eval", type=int, default=2)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_stage("train", output_root, args, args.parallel_train)
    run_stage("evaluate", output_root, args, args.parallel_eval)
    cells = [evaluation(output_root, airport, seed) for airport in AIRPORTS for seed in SEEDS]
    manifest = {
        "format_version": 1,
        "experiment_id": "E15",
        "complete": all(valid_evaluation(output_root, airport, seed) for airport in AIRPORTS for seed in SEEDS),
        "cells": [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)} for path in cells],
    }
    atomic_json(output_root / "completion_manifest_v1.json", manifest)
    print(json.dumps({"complete": manifest["complete"], "cells": len(cells)}, indent=2))


if __name__ == "__main__":
    main()
