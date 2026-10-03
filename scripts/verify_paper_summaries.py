#!/usr/bin/env python3
"""Verify the frozen small aggregate evidence shipped with the code release."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results/paper_tables"
MANIFEST = TABLES / "MANIFEST.sha256"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    failures: list[str] = []
    rows = 0
    expected_files: set[str] = set()
    for line_number, line in enumerate(MANIFEST.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)", line)
        if not match:
            failures.append(f"malformed manifest line {line_number}")
            continue
        expected, name = match.groups()
        expected_files.add(name)
        path = TABLES / name
        if not path.is_file() or sha256(path) != expected:
            failures.append(f"checksum mismatch: {name}")
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows += sum(1 for _ in csv.DictReader(handle))
    actual_files = {path.name for path in TABLES.glob("table*.csv")}
    if actual_files != expected_files:
        failures.append("table file set does not match MANIFEST.sha256")
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1
    print(f"OK: {len(expected_files)} aggregate table files, {rows} rows, all SHA256 values matched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
