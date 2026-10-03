"""Verify the immutable legacy C165 evidence boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("frozen_c165_manifest.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify() -> dict[str, object]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures: list[dict[str, str]] = []
    for relative, expected in payload["files"].items():
        path = ROOT / relative
        if not path.is_file():
            failures.append({"path": relative, "error": "missing"})
            continue
        observed = sha256(path)
        if observed != expected:
            failures.append(
                {
                    "path": relative,
                    "error": "sha256_mismatch",
                    "expected": expected,
                    "observed": observed,
                }
            )
    return {
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "verified_files": len(payload["files"]) - len(failures),
        "total_files": len(payload["files"]),
        "ok": not failures,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = verify()
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"C165 freeze: {result['verified_files']}/{result['total_files']} "
            f"verified, ok={result['ok']}"
        )
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
