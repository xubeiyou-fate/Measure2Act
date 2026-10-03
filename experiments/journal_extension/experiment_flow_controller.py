"""Persistently supervise the registered journal-extension experiment flow."""

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


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/journal_extension_20260814"
STATE_PATH = ARTIFACT_ROOT / "automated_flow_state_v1.json"
DOC_PATH = ARTIFACT_ROOT / "AUTOMATED_EXPERIMENT_FLOW.md"
LEDGER_PATH = ARTIFACT_ROOT / "AUTOMATED_TASK_LEDGER.md"
REPORT_ROOT = ARTIFACT_ROOT / "stage_reports"
LOG_ROOT = ARTIFACT_ROOT / "logs"
SEEDS = (42, 7, 123, 2024, 2026)
AIRPORTS = ("KAGC", "KBTP")


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def json_ok(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return True


def expected_stage_files() -> dict[str, list[Path]]:
    p = ROOT / "artifacts"
    def existing_or(primary: Path, alternate: Path) -> Path:
        return primary if primary.is_file() or not alternate.is_file() else alternate

    files: dict[str, list[Path]] = {
        "eqmotion_training": [
            existing_or(
                p / "partc_two_dataset_20260812/modern_baseline/eqmotion_tartan_multiseed_v2" / f"{airport}_target_only_seed{seed}_formal.json",
                p / "partc_two_dataset_20260812/modern_baseline" / f"eqmotion_tartan_{airport}_target_only_seed{seed}_formal.json",
            )
            for airport in AIRPORTS for seed in (42, 7, 123, 2024, 2026)
        ],
        "eqmotion_locked": [
            p / "partc_two_dataset_20260812/modern_baseline/eqmotion_tartan_multiseed_v2/locked" / f"{airport}_target_only_seed{seed}_locked_test_v2.json"
            for airport in AIRPORTS for seed in (7, 123, 2024, 2026)
        ] + [p / "journal_extension_20260814/eqmotion_five_seed_summary_v1.json"],
        "probability_controls": [
            p / "journal_extension_20260814/probability_controls" / split / f"{airport}_seed{seed}_{split}_v1.json"
            for split in ("development", "test") for airport in AIRPORTS for seed in SEEDS
        ] + [
            p / "journal_extension_20260814/probability_controls_receipt_v1.json",
            p / "journal_extension_20260814/probability_controls_summary_v1.json",
        ],
        "awta_tartan": [
            p / "journal_extension_20260814/awta_tartan/development" / f"{airport}_seed{seed}_formal.json"
            for airport in AIRPORTS for seed in SEEDS
        ] + [
            ROOT / "runs/journal_extension_20260814/awta_tartan" / airport / f"seed{seed}_formal/last.pt"
            for airport in AIRPORTS for seed in SEEDS
        ],
        "awta_tartan_locked": [
            p / "journal_extension_20260814/awta_tartan/test" / f"{airport}_seed{seed}_test_v1.json"
            for airport in AIRPORTS for seed in SEEDS
        ] + [p / "journal_extension_20260814/awta_tartan_evaluation_receipt_v1.json"],
        "awta_trajair": [
            p / "journal_extension_20260814/awta_trajair/development" / f"seed{seed}_formal.json"
            for seed in SEEDS
        ] + [
            ROOT / "runs/journal_extension_20260814/awta_trajair" / f"seed{seed}_formal/last.pt"
            for seed in SEEDS
        ] + [p / "journal_extension_20260814/awta_summary_v1.json"],
        "completion_audit": [p / "journal_extension_20260814/completion_audit_v1.json"],
    }
    return files


def stage_snapshot(files: list[Path]) -> dict[str, Any]:
    present = [path for path in files if path.is_file()]
    valid = [path for path in present if path.suffix != ".json" or json_ok(path)]
    return {
        "expected": len(files),
        "present": len(present),
        "valid": len(valid),
        "complete": len(valid) == len(files),
        "missing": [path.relative_to(ROOT).as_posix() for path in files if not path.is_file()],
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "valid": path in valid,
            }
            for path in files
        ],
    }


def active_pipeline() -> bool:
    try:
        import psutil  # type: ignore
    except ImportError:
        psutil = None
    if psutil is not None:
        for process in psutil.process_iter(("cmdline",)):
            try:
                command = " ".join(process.info.get("cmdline") or [])
            except (psutil.Error, OSError):
                continue
            if "experiments.journal_extension.run_remaining_pipeline" in command or "experiments.journal_extension.train_" in command:
                return True
        return False
    # The bundled Windows environment does not always include psutil. Fail
    # closed and treat an unreadable process table as busy to avoid duplicate GPU jobs.
    probe = subprocess.run(
        ["powershell", "-NoProfile", "-Command", "& { $x = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'journal_extension\\.(run_remaining_pipeline|train_|evaluate_)' }; if ($x) { exit 0 } else { exit 1 } }"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return probe.returncode == 0


def first_incomplete(stages: dict[str, dict[str, Any]]) -> str | None:
    for name, snapshot in stages.items():
        if not snapshot["complete"]:
            return name
    return None


def write_state(stages: dict[str, dict[str, Any]], *, controller: str, launch: dict[str, Any] | None) -> dict[str, Any]:
    state = {
        "format_version": 1,
        "experiment_id": "automated_journal_extension_flow_v1",
        "updated_at": now(),
        "controller": controller,
        "pipeline_active": active_pipeline(),
        "stages": stages,
        "next_stage": first_incomplete(stages),
        "last_launch": launch,
        "claim_boundary": {
            "local_datasets_only": True,
            "retrospective_test": True,
            "third_party_blind_test": False,
            "trajair_awta_development_only": True,
        },
    }
    atomic_write(STATE_PATH, json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# 自动化实验流程状态",
        "",
        f"更新时间：{state['updated_at']}",
        f"当前未完成阶段：`{state['next_stage'] or '无，全部阶段完成'}`",
        f"主流水线活动：`{'是' if state['pipeline_active'] else '否'}`",
        "",
        "本控制器按阶段检查产物；阶段达到完整条件后立即登记，并由主流水线进入下一阶段。测试结果属于本地数据的内部回顾性证据，TrajAir aWTA 仅使用开发集。",
        "",
        "| 阶段 | 状态 | 有效/预期 | 缺失文件数 |",
        "|---|---|---:|---:|",
    ]
    for name, snapshot in stages.items():
        status = "已完成" if snapshot["complete"] else "进行中"
        lines.append(f"| `{name}` | {status} | {snapshot['valid']}/{snapshot['expected']} | {len(snapshot['missing'])} |")
        if snapshot["complete"]:
            report = [
                f"# 阶段完成：{name}",
                "",
                f"登记时间：{state['updated_at']}",
                f"有效产物：{snapshot['valid']}/{snapshot['expected']}",
                "",
                "该阶段由自动化控制器根据产物存在性、JSON 可解析性和固定目录约束登记。结果只使用本地数据；测试证据属于内部回顾性评估，不构成第三方盲测或前瞻性确认。",
            ]
            if snapshot["missing"]:
                report.extend(["", "缺失产物：", *[f"- `{path}`" for path in snapshot["missing"]]])
            atomic_write(REPORT_ROOT / f"{name}.md", "\n".join(report) + "\n")
    if launch:
        lines.extend(["", f"最近自动接管：`{launch['started_at']}`，阶段 `{launch['stage']}`，PID `{launch['pid']}`。"])
    atomic_write(DOC_PATH, "\n".join(lines) + "\n")
    ledger = [
        "# 自动化实验任务台账",
        "",
        f"更新时间：{state['updated_at']}",
        "",
        "每个正式产物对应一个可恢复任务。产物通过存在性检查，JSON 产物还必须能够解析，才登记为完成。",
        "",
        "| 阶段 | 任务产物 | 状态 |",
        "|---|---|---|",
    ]
    for name, snapshot in stages.items():
        for item in snapshot["files"]:
            status = "已完成" if item["valid"] else "待运行"
            ledger.append(f"| `{name}` | `{item['path']}` | {status} |")
    atomic_write(LEDGER_PATH, "\n".join(ledger) + "\n")
    return state


def launch_pipeline(stage: str, device: str, workers: int) -> dict[str, Any]:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stdout = LOG_ROOT / f"controller_{stage}_{stamp}.out.log"
    stderr = LOG_ROOT / f"controller_{stage}_{stamp}.err.log"
    command = [
        str(PYTHON), "-m", "experiments.journal_extension.run_remaining_pipeline",
        "--start-stage", stage, "--device", device, "--workers", str(workers),
    ]
    process = subprocess.Popen(command, cwd=ROOT, stdout=stdout.open("w", encoding="utf-8"), stderr=stderr.open("w", encoding="utf-8"), creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    return {"started_at": now(), "stage": stage, "pid": process.pid, "stdout": stdout.relative_to(ROOT).as_posix(), "stderr": stderr.relative_to(ROOT).as_posix()}


PYTHON = Path(sys.executable).resolve()


def run_once(device: str, workers: int, controller: str) -> dict[str, Any]:
    expected = expected_stage_files()
    stages = {name: stage_snapshot(files) for name, files in expected.items()}
    launch = None
    stage = first_incomplete(stages)
    if stage and not active_pipeline():
        launch = launch_pipeline(stage, device, workers)
    return write_state(stages, controller=controller, launch=launch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        state = run_once(args.device, args.workers, "experiment_flow_controller")
        print(json.dumps({"time": state["updated_at"], "next_stage": state["next_stage"], "pipeline_active": state["pipeline_active"]}, ensure_ascii=False), flush=True)
        if args.once or state["next_stage"] is None:
            return
        time.sleep(max(args.interval, 10))


if __name__ == "__main__":
    main()
