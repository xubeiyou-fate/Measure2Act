"""Atomic human-readable progress reports for long C96 runs."""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GLOBAL_LIVE_REPORT = REPOSITORY_ROOT / "docs" / "edfa_ascent_live.md"


def _best_effort_atomic_write(
    destination: Path,
    content: str,
    retries: int = 8,
) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    for attempt in range(retries):
        temporary.write_text(content, encoding="utf-8")
        try:
            temporary.replace(destination)
            return True
        except PermissionError:
            if attempt + 1 < retries:
                time.sleep(0.05 * (attempt + 1))
    temporary.unlink(missing_ok=True)
    print(
        f"warning: live report is temporarily locked; skipped update for {destination}",
        file=sys.stderr,
        flush=True,
    )
    return False


def _metric(validation: dict | None, group: str, name: str) -> str:
    if not validation:
        return "-"
    value = validation.get(group, {}).get(name)
    return "-" if value is None else f"{float(value):.6f}"


def render_progress(state: dict[str, Any]) -> str:
    history = state.get("history", [])
    lines = [
        "# C96 EDFA-ASCENT Live Progress",
        "",
        f"Last update: {state['updated_at']}",
        "",
        f"- Status: {state['status']}",
        f"- Variant: {state['variant']}",
        f"- Seed: {state['seed']}",
        f"- Formal run: {str(state['formal']).lower()}",
        f"- Current epoch: {state.get('epoch', 0)}/{state['epochs']}",
        f"- Current batch: {state.get('batch', 0)}/{state.get('batches_per_epoch', 0)}",
        f"- Running mean loss: {state.get('running_loss', '-')}",
        f"- Best epoch: {state.get('best_epoch', -1)}",
        f"- Best development minFDE: {state.get('best_minfde', '-')}",
        f"- Locked test used: {str(state.get('locked_test_used', False)).lower()}",
        "",
        "## Epoch History",
        "",
        "| Epoch | Train loss | Overall minADE | Overall minFDE | Multi minFDE | Interactive minFDE | Joint minFDE | Relation accuracy |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in history:
        validation = record.get("validation")
        relation = validation.get("relation", {}) if validation else {}
        lines.append(
            "| {epoch} | {loss:.6f} | {minade} | {minfde} | {multi} | {interactive} | {joint} | {relation} |".format(
                epoch=record["epoch"],
                loss=float(record["train"]["loss"]),
                minade=_metric(validation, "overall", "minade"),
                minfde=_metric(validation, "overall", "minfde"),
                multi=_metric(validation, "multi_agent", "minfde"),
                interactive=_metric(validation, "interactive", "minfde"),
                joint=_metric(validation, "joint_multi_scene", "minfde"),
                relation=(
                    "-" if "accuracy" not in relation
                    else f"{float(relation['accuracy']):.4f}"
                ),
            )
        )
    controls = state.get("controls")
    if controls:
        lines.extend(("", "## Final Controls", ""))
        for name, metrics in controls.items():
            lines.append(
                f"- {name}: overall minFDE={metrics['overall']['minfde']:.6f}, "
                f"multi minFDE={metrics['multi_agent']['minfde']:.6f}, "
                f"joint minFDE={metrics['joint_multi_scene']['minfde']:.6f}"
            )
    lines.extend((
        "",
        "## Boundary",
        "",
        "C12 locked test remains sealed until the preregistered development gate authorizes one evaluation.",
        "",
    ))
    return "\n".join(lines)


def update_live_document(run_dir: Path, state: dict[str, Any]) -> None:
    payload = {
        **state,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    content = render_progress(payload)
    for destination in (GLOBAL_LIVE_REPORT, run_dir / "progress.md"):
        _best_effort_atomic_write(destination, content)
