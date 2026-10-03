#!/usr/bin/env python3
"""Resolve and optionally fetch official datasets outside the Git checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs/data_sources.json"


def load_registry() -> dict:
    with REGISTRY.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def source(registry: dict, source_id: str) -> dict:
    for item in registry["datasets"]:
        if item["id"] == source_id:
            return item
    raise KeyError(source_id)


def ensure_external_destination(path: Path) -> Path:
    destination = path.expanduser().resolve()
    try:
        destination.relative_to(ROOT)
    except ValueError:
        return destination
    raise ValueError(f"destination must be outside the Git checkout: {destination}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5(path: Path) -> str:
    """Return the upstream-published integrity digest (not a security primitive)."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Measure2Act-data-fetch/1.0"})
    with urllib.request.urlopen(request) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    partial.replace(target)


def print_sources(registry: dict) -> None:
    for item in registry["datasets"]:
        identifier = item.get("doi") or item.get("official_source_repository")
        print(f"{item['id']}: {item['name']} | {identifier} | hosted=false")


def fetch_trajair(item: dict, destination: Path, asset_name: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    assets = {asset["name"]: asset for asset in item["official_files"]}
    if asset_name not in assets:
        raise ValueError(f"unknown TrajAir asset {asset_name!r}; choose from {', '.join(sorted(assets))}")
    asset = assets[asset_name]
    target = destination / asset_name
    if target.exists():
        raise FileExistsError(target)
    download(asset["download_url"], target)
    actual_md5 = md5(target)
    actual_sha256 = sha256(target)
    if actual_md5 != asset["official_md5"] or actual_sha256 != asset["sha256"]:
        raise RuntimeError(f"checksum mismatch for {target}")
    print(f"downloaded={target}")
    print(f"official_md5={actual_md5}")
    print(f"sha256={actual_sha256}")


def fetch_tartan(item: dict, destination: Path, option: str, location: str) -> None:
    try:
        import boto3  # noqa: F401
        import requests  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("TartanAviation's official downloader requires: pip install boto3 requests") from exc
    destination.mkdir(parents=True, exist_ok=True)
    checkout = destination / "TartanAviation-upstream"
    if checkout.exists():
        raise FileExistsError(checkout)
    subprocess.run(["git", "clone", item["official_source_repository"], str(checkout)], check=True)
    subprocess.run(["git", "checkout", "--detach", item["paper_pinned_commit"]], cwd=checkout, check=True)
    data_root = destination / "tartanaviation-data"
    subprocess.run(
        [
            sys.executable,
            str(checkout / item["official_download_script"]),
            "--save_dir",
            str(data_root),
            "--option",
            option,
            "--location",
            location,
        ],
        cwd=checkout / "adsb",
        check=True,
    )
    print(f"official_source_commit={item['paper_pinned_commit']}")
    print(f"data_root={data_root}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="show registered official sources")
    parser.add_argument("--dataset", choices=("trajair", "tartanaviation"))
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--asset", default="111_days.zip", help="TrajAir asset name; see --list and docs/data_sources.json")
    parser.add_argument("--execute", action="store_true", help="perform the network download; default is dry-run")
    parser.add_argument("--accept-upstream-terms", action="store_true", help="confirm that the user reviewed upstream terms")
    parser.add_argument("--tartan-option", choices=("Sample", "Processed", "Raw", "All"), default="Processed")
    parser.add_argument("--location", choices=("kbtp", "kagc", "Both"), default="Both")
    args = parser.parse_args()
    registry = load_registry()
    if args.list:
        print_sources(registry)
        return 0
    if args.dataset is None or args.destination is None:
        parser.error("--dataset and --destination are required unless --list is used")
    destination = ensure_external_destination(args.destination)
    item = source(registry, args.dataset)
    print(json.dumps({"dataset": item["id"], "destination": str(destination), "redistributed_by_measure2act": False}, indent=2))
    if not args.execute:
        print("dry-run: add --accept-upstream-terms --execute to fetch from the official source")
        return 0
    if not args.accept_upstream_terms:
        parser.error("--execute requires --accept-upstream-terms")
    if args.dataset == "trajair":
        fetch_trajair(item, destination, args.asset)
    else:
        fetch_tartan(item, destination, args.tartan_option, args.location)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
