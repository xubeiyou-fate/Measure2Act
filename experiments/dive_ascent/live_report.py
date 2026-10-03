"""Atomic C99 status artifacts and human-readable live report."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def update_status(repository_root: Path, run_name: str, status: dict[str, object]) -> None:
    artifact_root = repository_root / "artifacts/experiments/dive_ascent/status"
    artifact_root.mkdir(parents=True, exist_ok=True)
    payload = {
        **status,
        "run": run_name,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write(artifact_root / f"{run_name}.json", json.dumps(payload, indent=2) + "\n")
    rows = []
    for path in sorted(artifact_root.glob("*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows.append(
            "| {run} | {variant} | {phase} | {epoch}/{epochs} | {batch}/{batches} | {loss} | {metric} | {updated} |".format(
                run=item.get("run", path.stem),
                variant=item.get("variant", "-"),
                phase=item.get("phase", "-"),
                epoch=item.get("epoch", 0),
                epochs=item.get("epochs", "-"),
                batch=item.get("batch", 0),
                batches=item.get("batches_per_epoch", "-"),
                loss=item.get("running_loss", "-"),
                metric=item.get("development_minfde", "-"),
                updated=item.get("updated_utc", "-"),
            )
        )
    document = "\n".join(
        [
            "# C99 DIVE-ASCENT Live Status",
            "",
            "Generated from atomic per-run status artifacts. The C12 locked-test partition remains sealed.",
            "",
            "| Run | Variant | Phase | Epoch | Batch | Running loss | Dev minFDE | Updated UTC |",
            "|---|---|---|---:|---:|---:|---:|---|",
            *rows,
            "",
        ]
    )
    _atomic_write(repository_root / "docs/dive_ascent_live.md", document)
