"""Fail when data/model payloads or developer-local paths enter the code release."""

from __future__ import annotations

import argparse
from pathlib import Path


FORBIDDEN_TOP_LEVEL = {
    "artifacts",
    "checkpoints",
    "data",
    "dataset",
    "deposits",
    "runs",
    "weights",
}
FORBIDDEN_DIRECTORY_NAMES = FORBIDDEN_TOP_LEVEL | {
    "derived_trajectory_pool",
    "processed_data",
    "raw_data",
    "third_party_raw",
}
FORBIDDEN_SUFFIXES = {
    ".7z",
    ".bin",
    ".bz2",
    ".ckpt",
    ".gz",
    ".npy",
    ".npz",
    ".parquet",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
    ".tar",
    ".tgz",
    ".xz",
    ".zip",
}
TEXT_SUFFIXES = {".cff", ".csv", ".json", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}
CACHE_DIRECTORY_NAMES = {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
# Assemble these markers so the audit source does not match its own rules.
PRIVATE_PATH_MARKERS = ("/data/" + "xudaoming", "/home/" + "cwadmin")
MAX_FILE_BYTES = 10 * 1024 * 1024


def audit(root: Path) -> list[str]:
    failures: list[str] = []
    for name in sorted(FORBIDDEN_TOP_LEVEL):
        if (root / name).exists():
            failures.append(f"forbidden top-level asset directory: {name}/")
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or ".git" in path.parts
            or any(part in CACHE_DIRECTORY_NAMES for part in path.parts)
        ):
            continue
        relative = path.relative_to(root)
        if any(part in FORBIDDEN_DIRECTORY_NAMES for part in relative.parts[:-1]):
            failures.append(f"forbidden nested asset directory: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"forbidden data/model/archive payload: {relative}")
        if path.stat().st_size > MAX_FILE_BYTES:
            failures.append(f"file exceeds 10 MiB code-repository limit: {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES:
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                failures.append(f"declared text file is not UTF-8: {relative}")
                continue
            for marker in PRIVATE_PATH_MARKERS:
                if marker in content:
                    failures.append(f"developer-local absolute path in {relative}: {marker}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    failures = audit(args.root.resolve())
    if failures:
        raise SystemExit("\n".join(f"ERROR: {item}" for item in failures))
    print("OK: source-only release boundary verified")


if __name__ == "__main__":
    main()
