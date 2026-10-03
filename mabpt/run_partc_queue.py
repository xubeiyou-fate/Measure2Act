"""Run the remaining formal MABPT-ASCENT experiments when shared GPUs are idle."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/mabpt_partc_20260811"
ARTIFACT_ROOT = ROOT / "artifacts/mabpt_partc_20260811"
LOG_ROOT = RUN_ROOT / "queue_logs"
STATUS_PATH = ARTIFACT_ROOT / "formal_queue_status_win_gpu0.json"
SEEDS = (42, 7, 123, 2024, 2026)


@dataclass(frozen=True)
class Job:
    name: str
    command: Callable[[int], list[str]]
    complete: Callable[[], bool]
    dependencies: tuple[str, ...] = ()
    preferred_gpu: int | None = None
    environment: str = "default"
    share_gpu: bool = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_status(payload: dict[str, object]) -> None:
    path = STATUS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _valid_json(path: Path, **expected) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return all(payload.get(key) == value for key, value in expected.items())


def _latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = sorted(run_dir.glob("epoch_*.pt"))
    return checkpoints[-1] if checkpoints else None


def _target_command(stage: str, seed: int, gpu: int) -> list[str]:
    run_dir = RUN_ROOT / f"mabpt_ascent_{stage}_seed{seed}_formal"
    command = [
        sys.executable,
        "-m",
        "mabpt.train_partc_target",
        "--stage",
        stage,
        "--seed",
        str(seed),
        "--device",
        f"cuda:{gpu}",
        "--epochs",
        "20",
        "--workers",
        "0",
        "--run-dir",
        str(run_dir),
    ]
    latest = _latest_checkpoint(run_dir)
    if latest is not None:
        command.extend(("--resume", str(latest)))
    return command


def _target_complete(stage: str, seed: int) -> bool:
    summary = (
        RUN_ROOT
        / f"mabpt_ascent_{stage}_seed{seed}_formal"
        / "training_summary.json"
    )
    return _valid_json(
        summary,
        model="MABPT-ASCENT",
        stage=stage,
        seed=seed,
        formal=True,
        complete=True,
        fixed_final_epoch=20,
    )


def _seed_evaluation_command(seed: int, gpu: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "mabpt.partc_seed_evaluate",
        "--seed",
        str(seed),
        "--device",
        f"cuda:{gpu}",
        "--workers",
        "0",
        "--batch-size",
        "1024",
        "--output",
        str(ARTIFACT_ROOT / f"seed{seed}_development_formal_v1.json"),
    ]


def _seed_evaluation_complete(seed: int) -> bool:
    return _valid_json(
        ARTIFACT_ROOT / f"seed{seed}_development_formal_v1.json",
        model="MABPT-ASCENT",
        seed=seed,
        evidence_class="development_only",
    )


def _seed_robustness_command(seed: int, gpu: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "mabpt.partc_seed_robustness",
        "--seed",
        str(seed),
        "--device",
        f"cuda:{gpu}",
        "--workers",
        "0",
        "--batch-size",
        "1024",
        "--output",
        str(ARTIFACT_ROOT / f"seed{seed}_robustness_formal_v1.json"),
    ]


def _seed_robustness_complete(seed: int) -> bool:
    return _valid_json(
        ARTIFACT_ROOT / f"seed{seed}_robustness_formal_v1.json",
        model="MABPT-ASCENT",
        experiment_id="five_seed_robustness",
        seed=seed,
        evidence_class="development_only",
    )


def _factorial_command(fold: int, gpu: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "mabpt.partc_evaluate",
        "--fold",
        str(fold),
        "--device",
        f"cuda:{gpu}",
        "--workers",
        "0",
        "--batch-size",
        "512",
        "--output",
        str(ARTIFACT_ROOT / f"factorial_physical_fold{fold}_formal_v1.json"),
    ]


def _factorial_complete(fold: int) -> bool:
    return _valid_json(
        ARTIFACT_ROOT / f"factorial_physical_fold{fold}_formal_v1.json",
        model="MABPT-ASCENT",
        fold=fold,
        evidence_class="retrospective_development_only",
    )


def _baseline_run_dir() -> Path:
    return ROOT / (
        "runs/mabpt_official_baselines/"
        "trajairnet_7days1_seed42_formal"
    )


def _baseline_train_command(gpu: int) -> list[str]:
    run_dir = _baseline_run_dir()
    command = [
        sys.executable,
        "-m",
        "mabpt.train_official_baseline",
        "--family",
        "trajairnet",
        "--dataset",
        "7days1",
        "--seed",
        "42",
        "--device",
        f"cuda:{gpu}",
        "--epochs",
        "10",
        "--run-dir",
        str(run_dir),
    ]
    command.append("--compile-model")
    latest = _latest_checkpoint(run_dir)
    if latest is not None:
        # Epochs 1-6 were produced on physical CUDA 1. Windows continuation
        # checkpoints contain only the current physical CUDA 0 RNG stream.
        source_device = 1 if int(latest.stem.rsplit("_", 1)[1]) <= 6 else gpu
        command.extend(
            ("--resume", str(latest), "--rng-source-device", str(source_device))
        )
    return command


def _baseline_train_complete() -> bool:
    return _valid_json(
        _baseline_run_dir() / "training_summary.json",
        experiment_id="E1",
        family="trajairnet",
        dataset="7days1",
        seed=42,
        epochs=10,
    )


def _baseline_evaluation_command(dataset: str, gpu: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "mabpt.evaluate_official_baseline",
        "--family",
        "trajairnet",
        "--dataset",
        dataset,
        "--checkpoint",
        str(_baseline_run_dir() / "epoch_010.pt"),
        "--checkpoint-label",
        "fixed_epoch10",
        "--device",
        f"cuda:{gpu}",
        "--fuse-samples",
        "--compile-model",
        "--output",
        str(
            ROOT
            / "artifacts/mabpt"
            / f"e1_trajairnet_{dataset}_fixed_epoch10_formal_v1.json"
        ),
    ]


def _baseline_evaluation_complete(dataset: str) -> bool:
    path = (
        ROOT
        / "artifacts/mabpt"
        / f"e1_trajairnet_{dataset}_fixed_epoch10_formal_v1.json"
    )
    return _valid_json(path, experiment_id="E1", family="trajairnet", dataset=dataset)


def build_jobs() -> list[Job]:
    jobs = []
    for seed in SEEDS:
        jobs.append(
            Job(
                name=f"decision_seed{seed}",
                command=lambda gpu, seed=seed: _target_command(
                    "decision_support", seed, gpu
                ),
                complete=lambda seed=seed: _target_complete(
                    "decision_support", seed
                ),
                preferred_gpu=0 if seed == 42 else None,
                share_gpu=True,
            )
        )
    for seed in SEEDS:
        jobs.append(
            Job(
                name=f"risk_seed{seed}",
                command=lambda gpu, seed=seed: _target_command(
                    "predicted_risk", seed, gpu
                ),
                complete=lambda seed=seed: _target_complete("predicted_risk", seed),
                dependencies=(f"decision_seed{seed}",),
                share_gpu=True,
            )
        )
    for seed in SEEDS:
        jobs.append(
            Job(
                name=f"development_seed{seed}",
                command=lambda gpu, seed=seed: _seed_evaluation_command(seed, gpu),
                complete=lambda seed=seed: _seed_evaluation_complete(seed),
                dependencies=(f"risk_seed{seed}",),
            )
        )
    jobs.append(
        Job(
            name="trajairnet_train_epoch10",
            command=_baseline_train_command,
            complete=_baseline_train_complete,
            preferred_gpu=0,
            environment="baseline",
        )
    )
    for dataset in ("7days1", "7days2", "7days3", "7days4"):
        jobs.append(
            Job(
                name=f"trajairnet_eval_{dataset}",
                command=lambda gpu, dataset=dataset: _baseline_evaluation_command(
                    dataset, gpu
                ),
                complete=lambda dataset=dataset: _baseline_evaluation_complete(dataset),
                dependencies=("trajairnet_train_epoch10",),
                environment="baseline",
                share_gpu=True,
            )
        )
    for seed in SEEDS:
        jobs.append(
            Job(
                name=f"robustness_seed{seed}",
                command=lambda gpu, seed=seed: _seed_robustness_command(seed, gpu),
                complete=lambda seed=seed: _seed_robustness_complete(seed),
                dependencies=(f"risk_seed{seed}",),
            )
        )
    for fold in (1, 2):
        jobs.append(
            Job(
                name=f"factorial_fold{fold}",
                command=lambda gpu, fold=fold: _factorial_command(fold, gpu),
                complete=lambda fold=fold: _factorial_complete(fold),
            )
        )
    return jobs


def _gpu_state() -> dict[int, dict[str, int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True)
    result = {}
    for line in output.splitlines():
        index, memory_free, utilization = [int(value.strip()) for value in line.split(",")]
        result[index] = {"memory_free_mib": memory_free, "utilization_percent": utilization}
    return result


def _job_environment(job: Job) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if job.environment == "baseline" and platform.system() != "Windows":
        library = os.environ.get("ASCENT_LIBRARY_ROOT", "")
        current = environment.get("LD_LIBRARY_PATH")
        environment["LD_LIBRARY_PATH"] = library if not current else f"{library}:{current}"
    return environment


def _aggregate_inputs_current(path: Path) -> bool:
    """Allow restart reuse only when every recorded aggregate input is unchanged."""
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        inputs = payload["inputs"]
        if not isinstance(inputs, list) or not inputs:
            return False
        for receipt in inputs:
            input_path = ROOT / receipt["path"]
            if not input_path.is_file() or _sha256(input_path) != receipt["sha256"]:
                return False
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _run_cpu_aggregates() -> None:
    commands = (
        (
            [sys.executable, "-m", "mabpt.aggregate_partc_seeds"],
            ARTIFACT_ROOT / "five_seed_development_summary_v1.json",
        ),
        (
            [sys.executable, "-m", "mabpt.aggregate_partc_factorial"],
            ARTIFACT_ROOT / "factorial_physical_summary_v1.json",
        ),
        (
            [sys.executable, "-m", "mabpt.aggregate_partc_robustness"],
            ARTIFACT_ROOT / "five_seed_robustness_summary_v1.json",
        ),
        (
            [sys.executable, "-m", "mabpt.aggregate_e1_matched"],
            ROOT / "artifacts/mabpt/e1_matched_summary_v1.json",
        ),
        (
            [sys.executable, "-m", "mabpt.aggregate_partc_hypotheses"],
            ARTIFACT_ROOT / "hypothesis_summary_v1.json",
        ),
    )
    for command, output in commands:
        if _aggregate_inputs_current(output):
            print(
                json.dumps(
                    {
                        "event": "reused_current_aggregate",
                        "output": str(output.relative_to(ROOT)),
                    }
                ),
                flush=True,
            )
            continue
        subprocess.run(command, cwd=ROOT, check=True)
    subprocess.run(
        [sys.executable, "-m", "mabpt.audit_partc_experiments"],
        cwd=ROOT,
        check=True,
    )


def run(
    *,
    poll_seconds: int,
    stable_polls: int,
    maximum_attempts: int,
    gpu_ids: tuple[int, ...],
    minimum_free_mib: int,
    maximum_utilization: int,
    training_slots_per_gpu: int,
) -> None:
    jobs = build_jobs()
    completed = {job.name for job in jobs if job.complete()}
    attempts = {job.name: 0 for job in jobs}
    stable = {gpu: 0 for gpu in gpu_ids}
    running: dict[
        str, tuple[Job, subprocess.Popen, object, float, int]
    ] = {}
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    while len(completed) < len(jobs):
        running_names = set(running)
        for job in jobs:
            if (
                job.name not in completed
                and job.name not in running_names
                and job.complete()
            ):
                completed.add(job.name)
                print(
                    json.dumps({"event": "detected_completed", "job": job.name}),
                    flush=True,
                )
        for job_name, (job, process, handle, started, gpu) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            del running[job_name]
            if code == 0 and job.complete():
                completed.add(job.name)
                print(json.dumps({"event": "completed", "job": job.name, "gpu": gpu}), flush=True)
            elif attempts[job.name] >= maximum_attempts:
                raise RuntimeError(
                    f"job {job.name} failed after {attempts[job.name]} attempts; "
                    f"see {LOG_ROOT / (job.name + '.log')}"
                )
            else:
                print(
                    json.dumps(
                        {
                            "event": "retry_pending",
                            "job": job.name,
                            "gpu": gpu,
                            "return_code": code,
                            "elapsed_seconds": time.time() - started,
                        }
                    ),
                    flush=True,
                )
        state = _gpu_state()
        missing_gpus = sorted(set(gpu_ids) - set(state))
        if missing_gpus:
            raise RuntimeError(f"requested GPUs are unavailable: {missing_gpus}")
        for gpu in stable:
            gpu_jobs = [value for value in running.values() if value[4] == gpu]
            idle = (
                not gpu_jobs
                and state[gpu]["memory_free_mib"] >= minimum_free_mib
                and state[gpu]["utilization_percent"] <= maximum_utilization
            )
            stable[gpu] = stable[gpu] + 1 if idle else 0
        for gpu in sorted(stable):
            while True:
                gpu_jobs = [value for value in running.values() if value[4] == gpu]
                if gpu_jobs:
                    sharing_available = (
                        len(gpu_jobs) < training_slots_per_gpu
                        and all(value[0].share_gpu for value in gpu_jobs)
                    )
                    if not sharing_available:
                        break
                elif stable[gpu] < stable_polls:
                    break
                eligible = [
                    job
                    for job in jobs
                    if job.name not in completed
                    and all(dependency in completed for dependency in job.dependencies)
                    and job.name not in running
                    and (job.preferred_gpu is None or job.preferred_gpu == gpu)
                    and (not gpu_jobs or job.share_gpu)
                ]
                if not eligible:
                    break
                job = eligible[0]
                attempts[job.name] += 1
                log_path = LOG_ROOT / f"{job.name}.log"
                handle = log_path.open("ab")
                command = job.command(gpu)
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=_job_environment(job),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
                running[job.name] = (job, process, handle, time.time(), gpu)
                stable[gpu] = 0
                print(
                    json.dumps(
                        {
                            "event": "started",
                            "job": job.name,
                            "gpu": gpu,
                            "attempt": attempts[job.name],
                            "shared_gpu": bool(gpu_jobs),
                            "log": str(log_path.relative_to(ROOT)),
                        }
                    ),
                    flush=True,
                )
                if not job.share_gpu:
                    break
        _atomic_status(
            {
                "model": "MABPT-ASCENT",
                "phase": "formal_gpu_queue",
                "completed": sorted(completed),
                "running": {
                    job_name: {
                        "gpu": value[4],
                        "elapsed_seconds": time.time() - value[3],
                    }
                    for job_name, value in running.items()
                },
                "pending": [job.name for job in jobs if job.name not in completed],
                "gpu_state": state,
                "stable_idle_polls": stable,
                "updated_unix": time.time(),
            }
        )
        if len(completed) < len(jobs):
            time.sleep(poll_seconds)
    _run_cpu_aggregates()
    _atomic_status(
        {
            "model": "MABPT-ASCENT",
            "phase": "complete",
            "completed": sorted(completed),
            "pending": [],
            "updated_unix": time.time(),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--stable-polls", type=int, default=3)
    parser.add_argument("--maximum-attempts", type=int, default=3)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--minimum-free-mib", type=int, default=24000)
    parser.add_argument("--maximum-utilization", type=int, default=10)
    parser.add_argument("--training-slots-per-gpu", type=int, default=1)
    args = parser.parse_args()
    gpu_ids = tuple(int(value.strip()) for value in args.gpus.split(",") if value.strip())
    if (
        args.poll_seconds < 5
        or args.stable_polls < 1
        or args.maximum_attempts < 1
        or not gpu_ids
        or len(set(gpu_ids)) != len(gpu_ids)
        or args.minimum_free_mib < 1
        or not 0 <= args.maximum_utilization <= 100
        or args.training_slots_per_gpu < 1
    ):
        raise ValueError("invalid queue control")
    run(
        poll_seconds=args.poll_seconds,
        stable_polls=args.stable_polls,
        maximum_attempts=args.maximum_attempts,
        gpu_ids=gpu_ids,
        minimum_free_mib=args.minimum_free_mib,
        maximum_utilization=args.maximum_utilization,
        training_slots_per_gpu=args.training_slots_per_gpu,
    )


if __name__ == "__main__":
    main()
