"""Advance the frozen C127 phase pipeline until a gate closes or confirms it."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

from .finalize import run as finalize
from .locking import exclusive_process_lock
from .protocol import load_protocol, sha256


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/metric_exact"
AUDIT_ROOT = ROOT.parents[1] / "审核文件"
LIVE = AUDIT_ROOT / "ASCENT_C127_流水线状态_自动更新.md"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_state(state: dict[str, object]) -> None:
    state["updated_at"] = now()
    path = ARTIFACT_ROOT / "pipeline_state.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    events = "\n".join(
        f"- {event['at']}：{event['event']}" for event in state["events"][-40:]
    )
    document = "\n".join(
        [
            "# ASCENT C127 流水线状态（自动更新）",
            "",
            f"更新时间：{state['updated_at']}",
            f"当前阶段：`{state['stage']}`",
            f"终端状态：`{state.get('terminal_decision')}`",
            "locked test 仅在 P1/P2/P3 全门通过后执行一次。",
            "",
            "## 事件",
            "",
            events,
        ]
    )
    temporary_doc = LIVE.with_suffix(".tmp")
    temporary_doc.write_text(document + "\n", encoding="utf-8")
    temporary_doc.replace(LIVE)


def event(state: dict[str, object], message: str) -> None:
    state["events"].append({"at": now(), "event": message})
    write_state(state)
    print(json.dumps(state["events"][-1]), flush=True)


def wait_for_orchestration(phase: str, state: dict[str, object]) -> None:
    path = ARTIFACT_ROOT / f"{phase.lower()}_orchestration.json"
    while True:
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if "result" in payload:
                if payload["result"]["failed"]:
                    raise RuntimeError(f"C127 {phase} orchestration failed")
                event(state, f"{phase} formal runs complete")
                return
        write_state(state)
        time.sleep(30)


def run_command(arguments: list[str], state: dict[str, object], label: str) -> None:
    event(state, f"starting {label}")
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)
    event(state, f"completed {label}")


def run() -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_boundaries()
    state: dict[str, object] = {
        "format_version": 1,
        "cycle": protocol.payload["cycle"],
        "protocol_sha256": sha256(protocol.path),
        "stage": "P1_RUNNING",
        "terminal_decision": None,
        "events": [{"at": now(), "event": "pipeline supervisor started"}],
        "locked_test_used": False,
    }
    write_state(state)
    wait_for_orchestration("P1", state)
    run_command(["-m", "experiments.metric_exact.summarize"], state, "P1 summary")
    p1 = json.loads((ARTIFACT_ROOT / "p1_summary.json").read_text(encoding="utf-8"))
    if p1["decision"] != "P2_AUTHORIZED":
        state["stage"] = "TERMINAL"
        state["terminal_decision"] = p1["decision"]
        finalize()
        event(state, f"pipeline stopped by P1 gate: {p1['decision']}")
        return state

    state["stage"] = "P2_RUNNING"
    run_command(
        ["-m", "experiments.metric_exact.orchestrate", "--phase", "P2"],
        state,
        "P2 formal runs",
    )
    run_command(["-m", "experiments.metric_exact.summarize_p2"], state, "P2 summary")
    p2 = json.loads((ARTIFACT_ROOT / "p2_summary.json").read_text(encoding="utf-8"))
    if p2["decision"] != "P3_AUTHORIZED":
        state["stage"] = "TERMINAL"
        state["terminal_decision"] = p2["decision"]
        finalize()
        event(state, f"pipeline stopped by P2 gate: {p2['decision']}")
        return state

    state["stage"] = "P3_RUNNING"
    run_command(
        ["-m", "experiments.metric_exact.orchestrate", "--phase", "P3"],
        state,
        "P3 formal runs",
    )
    run_command(["-m", "experiments.metric_exact.summarize_p3"], state, "P3 summary")
    p3 = json.loads((ARTIFACT_ROOT / "p3_summary.json").read_text(encoding="utf-8"))
    if p3["decision"] != "LOCKED_TEST_AUTHORIZED":
        state["stage"] = "TERMINAL"
        state["terminal_decision"] = p3["decision"]
        finalize()
        event(state, f"pipeline stopped by P3 gate: {p3['decision']}")
        return state

    state["stage"] = "LOCKED_TEST_EVENT"
    run_command(
        ["-m", "experiments.metric_exact.locked_test", "--device", "cuda:0"],
        state,
        "single locked-test event",
    )
    final = finalize()
    state["stage"] = "TERMINAL"
    state["terminal_decision"] = final["decision"]
    state["locked_test_used"] = True
    event(state, f"pipeline complete: {final['decision']}")
    return state


def main() -> None:
    with exclusive_process_lock(ARTIFACT_ROOT / "pipeline.lock", "pipeline"):
        print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
