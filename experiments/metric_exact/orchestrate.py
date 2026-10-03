"""Run frozen C127 formal tasks on at most one process per GPU."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from .locking import exclusive_process_lock
from .protocol import load_protocol, sha256


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/metric_exact"
RUN_ROOT = ROOT / "runs/metric_exact"
AUDIT_ROOT = ROOT.parents[1] / "审核文件"
AUTO_DOCUMENT = AUDIT_ROOT / "ASCENT_C127_运行状态_自动更新.md"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def p1_tasks(protocol) -> list[dict[str, object]]:
    order_path = ROOT / str(protocol.payload["run_order"]["artifact"])
    if not order_path.is_file():
        raise RuntimeError("C127 preflight/run-order artifact is missing")
    order = json.loads(order_path.read_text(encoding="utf-8"))
    if order.get("locked_test_used") is not False:
        raise RuntimeError("C127 run-order locked-test boundary violation")
    return [
        {"phase": "P1", "variant": variant, "fold": 0, "seed": 42}
        for variant in order["P1_fold0_seed42"]
    ]


def _phase_order(
    protocol,
    phase: str,
    blocks: list[tuple[str, list[dict[str, object]]]],
) -> list[dict[str, object]]:
    path = ARTIFACT_ROOT / f"{phase.lower()}_run_order.json"
    protocol_hash = sha256(protocol.path)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol_sha256") != protocol_hash:
            raise RuntimeError(f"C127 {phase} run-order protocol hash mismatch")
        return payload["tasks"]
    generator = torch.Generator().manual_seed(
        int(protocol.payload["run_order"]["seed"])
    )
    tasks = []
    block_orders = {}
    for label, values in blocks:
        order = torch.randperm(len(values), generator=generator).tolist()
        randomized = [values[index] for index in order]
        tasks.extend(randomized)
        block_orders[label] = [run_name(task) for task in randomized]
    payload = {
        "format_version": 1,
        "cycle": protocol.payload["cycle"],
        "phase": phase,
        "assignment_seed": int(protocol.payload["run_order"]["seed"]),
        "protocol_sha256": protocol_hash,
        "block_orders": block_orders,
        "tasks": tasks,
        "maximum_concurrent_runs_per_gpu": 1,
        "locked_test_used": False,
    }
    atomic_json(path, payload)
    return tasks


def p2_tasks(protocol) -> list[dict[str, object]]:
    summary_path = ARTIFACT_ROOT / "p1_summary.json"
    if not summary_path.is_file():
        raise RuntimeError("C127 P1 summary is required before P2")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidate = summary.get("P2_selected_exact_candidate")
    if summary.get("decision") != "P2_AUTHORIZED" or not candidate:
        raise RuntimeError("C127 P1 did not authorize P2")
    variants = ["B0_signed_coupled", "B2_decoupled_original", candidate]
    blocks = [
        (
            f"fold{fold}_seed42",
            [
                {"phase": "P2", "variant": variant, "fold": fold, "seed": 42}
                for variant in variants
            ],
        )
        for fold in (1, 2, 3, 4)
    ]
    return _phase_order(protocol, "P2", blocks)


def p3_tasks(protocol) -> list[dict[str, object]]:
    summary_path = ARTIFACT_ROOT / "p2_summary.json"
    if not summary_path.is_file():
        raise RuntimeError("C127 P2 summary is required before P3")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidate = summary.get("P3_selected_exact_candidate")
    if summary.get("decision") != "P3_AUTHORIZED" or not candidate:
        raise RuntimeError("C127 P2 did not authorize P3")
    variants = ["B0_signed_coupled", "B2_decoupled_original", candidate]
    blocks = [
        (
            f"all_train_seed{seed}",
            [
                {"phase": "P3", "variant": variant, "fold": None, "seed": seed}
                for variant in variants
            ],
        )
        for seed in protocol.payload["phases"]["P3"]["seeds"]
    ]
    return _phase_order(protocol, "P3", blocks)


def run_name(task: dict[str, object]) -> str:
    fold_label = "all_train" if task["phase"] == "P3" else f"fold{task['fold']}"
    return (
        f"{task['phase']}_{task['variant']}_{fold_label}_"
        f"seed{task['seed']}_formal"
    )


def complete(task: dict[str, object], protocol_hash: str) -> bool:
    summary = RUN_ROOT / run_name(task) / "training_summary.json"
    if not summary.is_file():
        return False
    payload = json.loads(summary.read_text(encoding="utf-8"))
    return (
        payload.get("complete") is True
        and payload.get("formal") is True
        and payload.get("protocol_sha256") == protocol_hash
        and payload.get("locked_test_used") is False
    )


def status_for(task: dict[str, object]) -> dict[str, object]:
    path = ARTIFACT_ROOT / "status" / f"{run_name(task)}.json"
    if not path.is_file():
        return {"phase": "queued"}
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_live_document(
    tasks: list[dict[str, object]],
    running: dict[int, dict[str, object]],
    events: list[dict[str, object]],
    protocol_hash: str,
) -> None:
    rows = []
    for position, task in enumerate(tasks, start=1):
        status = status_for(task)
        gpu = next(
            (
                gpu_index
                for gpu_index, active in running.items()
                if active["task"] == task
            ),
            "-",
        )
        active_gpu = next(
            (
                gpu_index
                for gpu_index, active in running.items()
                if active["task"] == task
            ),
            None,
        )
        if active_gpu is not None and status.get("phase") == "queued":
            state = "starting"
        else:
            state = str(
                status.get("phase", "running" if active_gpu is not None else "queued")
            )
        epoch = status.get("epoch", "-")
        metric = status.get("metrics", {})
        minade = metric.get("minade", "-") if isinstance(metric, dict) else "-"
        minfde = metric.get("minfde", "-") if isinstance(metric, dict) else "-"
        rows.append(
            f"| {position} | {task['variant']} | {gpu} | {state} | {epoch} | "
            f"{minade} | {minfde} |"
        )
    completed = sum(status_for(task).get("phase") == "complete" for task in tasks)
    lines = [
        "# ASCENT C127 运行状态（自动更新）",
        "",
        f"更新时间：{now()}",
        f"协议 SHA256：`{protocol_hash}`",
        f"正式任务：{completed}/{len(tasks)} complete；locked test 未使用。",
        "",
        "本文件由 `experiments.metric_exact.orchestrate` 根据机器可读 status 自动生成。"
        "训练中指标只表示优化轨迹，不用于选择 epoch；正式比较固定使用 epoch 20。",
        "",
        "| 顺序 | arm | GPU | 状态 | epoch | minADE | minFDE |",
        "|---:|---|---:|---|---:|---:|---:|",
        *rows,
        "",
        "## 调度事件",
        "",
    ]
    for event in events[-30:]:
        line = f"- {event.get('at', 'unknown')}：{event.get('event', 'unknown')}"
        if "run" in event:
            line += f" `{event['run']}`"
        if "gpu" in event:
            line += f" on GPU{event['gpu']}"
        if event.get("event") == "orchestration_resumed":
            line += (
                f"（已完成 {event.get('already_complete', '?')}，"
                f"剩余 {event.get('remaining', '?')}）"
            )
        lines.append(line)
    AUTO_DOCUMENT.parent.mkdir(parents=True, exist_ok=True)
    temporary = AUTO_DOCUMENT.with_suffix(".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(AUTO_DOCUMENT)


def command(task: dict[str, object], gpu: int) -> list[str]:
    values = [
        sys.executable,
        "-m",
        "experiments.metric_exact.train",
        "--phase",
        str(task["phase"]),
        "--variant",
        str(task["variant"]),
        "--seed",
        str(task["seed"]),
        "--device",
        f"cuda:{gpu}",
        "--num-workers",
        "4",
        "--epochs",
        "20",
        "--batch-size",
        "256",
        "--eval-batch-size",
        "512",
        "--resume",
    ]
    if task["fold"] is not None:
        values.extend(["--fold", str(task["fold"])])
    return values


def orchestrate(phase: str, poll_seconds: int = 10) -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_boundaries()
    protocol_hash = sha256(protocol.path)
    task_builders = {"P1": p1_tasks, "P2": p2_tasks, "P3": p3_tasks}
    tasks = task_builders[phase](protocol)
    state_path = ARTIFACT_ROOT / f"{phase.lower()}_orchestration.json"
    logs = ARTIFACT_ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    pending = [task for task in tasks if not complete(task, protocol_hash)]
    running: dict[int, dict[str, object]] = {}
    failures: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    if state_path.is_file():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            previous.get("phase") != phase
            or previous.get("protocol_sha256") != protocol_hash
        ):
            raise RuntimeError(f"C127 {phase} orchestration state mismatch")
        previous_events = previous.get("events", [])
        if not isinstance(previous_events, list):
            raise RuntimeError(f"C127 {phase} orchestration events are invalid")
        events.extend(previous_events)
        events.append(
            {
                "at": now(),
                "event": "orchestration_resumed",
                "phase": phase,
                "already_complete": len(tasks) - len(pending),
                "remaining": len(pending),
            }
        )

    state = {
        "format_version": 1,
        "cycle": protocol.payload["cycle"],
        "phase": phase,
        "protocol_sha256": protocol_hash,
        "updated_at": now(),
        "pending": [run_name(task) for task in pending],
        "running": {},
        "events": events,
        "failures": failures,
        "locked_test_used": False,
    }

    while pending or running:
        for gpu in (0, 1):
            if gpu in running or not pending:
                continue
            task = pending.pop(0)
            name = run_name(task)
            log_path = logs / f"{name}.stdout.log"
            handle = log_path.open("a", encoding="utf-8")
            process = subprocess.Popen(
                command(task, gpu),
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running[gpu] = {
                "task": task,
                "process": process,
                "handle": handle,
                "started_at": now(),
                "log": log_path.relative_to(ROOT).as_posix(),
            }
            events.append({"at": now(), "event": "started", "run": name, "gpu": gpu})
            print(json.dumps(events[-1]), flush=True)

        for gpu, active in list(running.items()):
            process = active["process"]
            returncode = process.poll()
            if returncode is None:
                continue
            active["handle"].close()
            task = active["task"]
            name = run_name(task)
            event = {
                "at": now(),
                "event": "completed" if returncode == 0 else "failed",
                "run": name,
                "gpu": gpu,
                "returncode": returncode,
            }
            events.append(event)
            print(json.dumps(event), flush=True)
            if returncode != 0 or not complete(task, protocol_hash):
                failures.append(event)
            del running[gpu]

        state = {
            "format_version": 1,
            "cycle": protocol.payload["cycle"],
            "phase": phase,
            "protocol_sha256": protocol_hash,
            "updated_at": now(),
            "pending": [run_name(task) for task in pending],
            "running": {
                str(gpu): {
                    "run": run_name(active["task"]),
                    "started_at": active["started_at"],
                    "log": active["log"],
                }
                for gpu, active in running.items()
            },
            "events": events,
            "failures": failures,
            "locked_test_used": False,
        }
        atomic_json(state_path, state)
        write_live_document(tasks, running, events, protocol_hash)
        if failures:
            for active in running.values():
                active["process"].terminate()
                active["handle"].close()
            raise RuntimeError(f"C127 formal orchestration failed: {failures}")
        if pending or running:
            time.sleep(poll_seconds)

    result = {
        "phase": phase,
        "tasks": len(tasks),
        "complete": sum(complete(task, protocol_hash) for task in tasks),
        "events": events,
        "failed": failures,
        "locked_test_used": False,
    }
    atomic_json(state_path, {**state, "result": result})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("P1", "P2", "P3"), default="P1")
    parser.add_argument("--poll-seconds", type=int, default=10)
    args = parser.parse_args()
    lock_path = ARTIFACT_ROOT / f"{args.phase.lower()}_orchestration.lock"
    with exclusive_process_lock(lock_path, f"{args.phase} orchestration"):
        print(json.dumps(orchestrate(args.phase, args.poll_seconds), indent=2))


if __name__ == "__main__":
    main()
