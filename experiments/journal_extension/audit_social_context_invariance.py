"""Audit adjacency invariance of the final actor-history-only models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from mabpt.evaluate_tartan_retrain import _dataset, _selected_checkpoint_triplet
from mabpt.partc_seed_evaluate import _load_model_pair
from mabpt.train_tartan_retrain import _limited_indices, sha256
from model.utils import seq_collate

from .train_awta_tartan import atomic_json


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("social_context_invariance_protocol_v1.json")


def load_protocol() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    parent = ROOT / protocol["data"]["parent_protocol"]
    if not parent.is_file() or sha256(parent) != protocol["data"]["parent_protocol_sha256"]:
        raise RuntimeError("social audit parent protocol mismatch")
    return protocol


def changed_adjacency(data: dict[str, Any], condition: str) -> dict[str, Any]:
    result = dict(data)
    actors = int(data["adj"].numel())
    if condition == "isolated":
        result["adj"] = torch.arange(actors, device=data["adj"].device)
    elif condition == "merged":
        result["adj"] = torch.zeros(actors, dtype=torch.long, device=data["adj"].device)
    else:
        raise ValueError(condition)
    return result


def tensors(source, target, data: dict[str, Any]) -> dict[str, torch.Tensor]:
    source_support, source_logits, _ = source(data)
    target_support, target_probability, target_decision, auxiliary = target(data)
    return {
        "source_support": source_support,
        "source_logits": source_logits,
        "target_support": target_support,
        "target_probability": target_probability,
        "target_decision": target_decision,
        "predicted_normalized_risk": auxiliary["predicted_normalized_ade_risk"],
    }


@torch.inference_mode()
def run(*, device: torch.device, batch_size: int, workers: int) -> dict[str, Any]:
    protocol = load_protocol()
    parent = json.loads((ROOT / protocol["data"]["parent_protocol"]).read_text(encoding="utf-8"))
    results = []
    overall_maximum = 0.0
    for airport in protocol["data"]["airports"]:
        dataset, _, index_path = _dataset(parent, airport, "development")
        indices = _limited_indices(len(dataset), int(protocol["data"]["maximum_scenes_per_airport"]))
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=seq_collate,
            num_workers=workers,
        )
        for seed in protocol["data"]["seeds"]:
            checkpoints = _selected_checkpoint_triplet(
                root=ROOT,
                protocol=parent,
                airport=airport,
                regime="target_only",
                seed=int(seed),
                formal=True,
            )
            source, target = _load_model_pair(
                source_checkpoint=ROOT / checkpoints["ascent"]["path"],
                target_checkpoint=ROOT / checkpoints["predicted_risk"]["path"],
                device=device,
                batch_size=batch_size,
            )
            maxima = {condition: {} for condition in ("isolated", "merged")}
            actors = 0
            multi_actor_scenes = 0
            for data in loader:
                data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
                native = tensors(source, target, data)
                counts = torch.bincount(data["adj"])
                multi_actor_scenes += int((counts > 1).sum().cpu())
                actors += int(data["adj"].numel())
                for condition in maxima:
                    altered = tensors(source, target, changed_adjacency(data, condition))
                    for name in native:
                        difference = float((native[name] - altered[name]).abs().max().cpu())
                        maxima[condition][name] = max(maxima[condition].get(name, 0.0), difference)
                        overall_maximum = max(overall_maximum, difference)
            results.append(
                {
                    "airport": airport,
                    "seed": seed,
                    "scenes": len(indices),
                    "actors": actors,
                    "multi_actor_scenes": multi_actor_scenes,
                    "maximum_absolute_difference": maxima,
                    "checkpoints": checkpoints,
                    "scene_index": index_path.relative_to(ROOT).as_posix(),
                    "scene_index_sha256": sha256(index_path),
                }
            )
    return {
        "format_version": 1,
        "experiment_id": "social_context_invariance_v1",
        "evidence_class": "development_input_invariance_audit",
        "overall_maximum_absolute_difference": overall_maximum,
        "passed": overall_maximum == 0.0,
        "results": results,
        "protocol": {"path": PROTOCOL.relative_to(ROOT).as_posix(), "sha256": sha256(PROTOCOL)},
        "implementation": {"path": Path(__file__).relative_to(ROOT).as_posix(), "sha256": sha256(Path(__file__))},
        "integrity": {"development_only": True, "locked_test_used": False, "all_registered_cells_reported": len(results) == 10},
        "interpretation": "The registered final models are adjacency-invariant and do not implement a social-interaction mechanism.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(device=torch.device(args.device), batch_size=args.batch_size, workers=args.workers)
    atomic_json(args.output, result)
    print(json.dumps({"output": str(args.output), "passed": result["passed"]}, indent=2))


if __name__ == "__main__":
    main()
