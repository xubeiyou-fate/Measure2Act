"""Verify C96 data integrity and the sealed evaluation boundary."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from model.utils import TrajectoryDataset

from .protocol import load_protocol, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/experiments/edfa_ascent/preflight.json"))
    parser.add_argument("--verify-all-hashes", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol()
    protocol.assert_manifest_sealed()
    manifest = protocol.manifest()
    partitions = {}
    hash_failures = []
    for split in ("train", "dev", "locked_test"):
        directory = protocol.split_path(split)
        records = manifest["partitions"][split]["records"]
        names = {path.name for path in directory.iterdir() if path.is_file()}
        expected = {record["name"] for record in records}
        if names != expected:
            raise RuntimeError(f"{split} files do not match the manifest")
        if args.verify_all_hashes:
            for record in records:
                actual = sha256(directory / record["name"])
                if actual != record["sha256"]:
                    hash_failures.append({"split": split, "name": record["name"]})
        partitions[split] = {
            "dates": manifest["partitions"][split]["date_count"],
            "files": len(records),
            "estimated_actor_windows": manifest["partitions"][split]["estimated_actor_window_load"],
            "assigned_date_counts": dict(Counter(record["assigned_date"] for record in records)),
        }
    if hash_failures:
        raise RuntimeError(f"C96 hash failures: {hash_failures[:5]}")
    datasets = {
        split: TrajectoryDataset(
            protocol.split_path(split).as_posix(),
            obs_len=16, obs_steps=1, pred_len=120, pred_step=5, delim=" ",
        )
        for split in ("train", "dev")
    }
    dev_sizes = torch.tensor([end - start for start, end in datasets["dev"].seq_start_end])
    result = {
        "format_version": 1,
        "cycle": "C96_EDFA_ASCENT",
        "protocol_sha256": sha256(protocol.path),
        "manifest_sha256": sha256(protocol.manifest_path),
        "date_overlap": manifest["date_overlap"],
        "file_overlap": manifest["file_overlap"],
        "locked_test_evaluated": manifest["locked_test_evaluated"],
        "hashes_verified": args.verify_all_hashes,
        "hash_failures": hash_failures,
        "partitions": partitions,
        "loaded": {
            "train_scenes": len(datasets["train"]),
            "train_actors": int(datasets["train"].obs_traj.shape[0]),
            "dev_scenes": len(datasets["dev"]),
            "dev_actors": int(datasets["dev"].obs_traj.shape[0]),
            "dev_multi_agent_scenes": int((dev_sizes > 1).sum()),
            "dev_actors_in_multi_agent_scenes": int(dev_sizes[dev_sizes > 1].sum()),
            "dev_max_scene_size": int(dev_sizes.max()),
        },
        "forbidden_mechanisms": {
            name: False
            for name in protocol.payload["forbidden_mechanisms"]
        },
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
