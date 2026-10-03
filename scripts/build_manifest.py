"""Build a deterministic SHA256 manifest for the source-only GitHub tree."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def iter_files(root: Path, output: Path):
    ignored = {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "build",
        "dist",
    }
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == output:
            continue
        parts = path.relative_to(root).parts
        if any(part in ignored or part.endswith(".egg-info") for part in parts):
            continue
        yield path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "MANIFEST.sha256").resolve()
    lines = [
        f"{digest(path)}  {path.relative_to(root).as_posix()}"
        for path in iter_files(root, output)
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} entries to {output}")


if __name__ == "__main__":
    main()
