"""Fit Tartan near-conflict alert thresholds on registered training dates only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.utils import seed_worker, seq_collate

from .evaluate_tartan_retrain import (
    AIRPORTS,
    EVALUATED_ARMS,
    REGIMES,
    ROOT,
    _constant_velocity,
    _selected_checkpoint_triplet,
    _verify_freeze_receipt,
)
from .partc_seed_evaluate import _load_model_pair, _model_outputs
from .traffic import pair_conflict_probabilities
from .train_tartan_retrain import _dataset


PROTOCOL = Path(__file__).with_name("tartan_retrain_protocol_v1.json")
EVENT_PROTOCOL = Path(__file__).with_name("e12_protocol.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@torch.inference_mode()
def fit(
    *,
    airport: str,
    regime: str,
    seed: int,
    device: torch.device,
    workers: int,
    batch_size: int,
    root: Path = ROOT,
) -> dict[str, object]:
    protocol_path = root / "mabpt/tartan_retrain_protocol_v1.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if airport not in AIRPORTS or regime not in REGIMES:
        raise ValueError("airport or regime lies outside the frozen registry")
    if seed not in [int(value) for value in protocol["training"]["seeds"]]:
        raise ValueError("seed lies outside the frozen registry")
    freeze = _verify_freeze_receipt(
        root=root,
        receipt_path=root
        / "artifacts/partc_two_dataset_20260812/tartan_retrain_frozen_receipt_v1.json",
    )
    checkpoints = _selected_checkpoint_triplet(
        root=root,
        protocol=protocol,
        airport=airport,
        regime=regime,
        seed=seed,
        formal=True,
    )
    dataset, dates, index_path = _dataset(protocol, airport, "train")
    options = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": seq_collate,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }
    if workers > 0:
        options.update({"persistent_workers": True, "prefetch_factor": 4})
    loader = DataLoader(**options)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    source, target = _load_model_pair(
        source_checkpoint=root / checkpoints["ascent"]["path"],
        target_checkpoint=root / checkpoints["predicted_risk"]["path"],
        device=device,
        batch_size=batch_size,
    )
    event = json.loads((root / "mabpt/e12_protocol.json").read_text(encoding="utf-8"))["event"]
    probabilities = {name: [] for name in EVALUATED_ARMS}
    labels = {name: [] for name in EVALUATED_ARMS}
    started = time.perf_counter()
    actors = 0
    for data in loader:
        data = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in data.items()
        }
        truth = data["pred_traj"].transpose(1, 0)
        outputs = _model_outputs(source, target, data)
        outputs["constant_velocity"] = _constant_velocity(data)
        for model, values in outputs.items():
            result = pair_conflict_probabilities(
                values["support"],
                values["probability"],
                truth,
                data["adj"],
                horizontal_threshold=float(event["horizontal_threshold_km"]),
                vertical_threshold=float(event["vertical_threshold_km"]),
                horizontal_scale=float(event["horizontal_kernel_scale_km"]),
                vertical_scale=float(event["vertical_kernel_scale_km"]),
            )
            probabilities[model].append(result["probability"].detach().cpu().numpy())
            labels[model].append(result["label"].detach().cpu().numpy().astype(np.bool_))
        actors += int(truth.shape[0])
    thresholds = {}
    for model in EVALUATED_ARMS:
        probability = np.concatenate(probabilities[model]) if probabilities[model] else np.empty(0)
        label = np.concatenate(labels[model]) if labels[model] else np.empty(0, dtype=np.bool_)
        negatives = probability[~label]
        if not len(negatives):
            raise RuntimeError(f"{airport} training split has no negative pairs for {model}")
        threshold = float(np.quantile(negatives, 0.95, method="higher"))
        thresholds[model] = {
            "threshold": threshold,
            "pairs": int(len(label)),
            "positive_pairs": int(label.sum()),
            "negative_pairs": int(len(negatives)),
            "training_observed_fpr": float((negatives >= threshold).mean()),
        }
    return {
        "format_version": 1,
        "experiment_id": "Tartan_target_domain_training_only_safety_threshold",
        "airport": airport,
        "regime": regime,
        "seed": seed,
        "fit_split": "train",
        "target_false_positive_rate": 0.05,
        "threshold_rule": "95th percentile of negative-pair probability with higher quantile method",
        "thresholds": thresholds,
        "training_scenes": len(dataset),
        "training_dates": len(set(dates)),
        "training_actors": actors,
        "inputs": {
            "protocol": protocol_path.relative_to(root).as_posix(),
            "protocol_sha256": sha256(protocol_path),
            "event_protocol": "mabpt/e12_protocol.json",
            "event_protocol_sha256": sha256(root / "mabpt/e12_protocol.json"),
            "scene_date_index": index_path.relative_to(root).as_posix(),
            "scene_date_index_sha256": sha256(index_path),
            "freeze_receipt": freeze,
            "checkpoints": checkpoints,
        },
        "runtime": {"device": str(device), "elapsed_seconds": time.perf_counter() - started},
        "integrity": {
            "train_only": True,
            "development_accessed": False,
            "locked_test_accessed": False,
            "threshold_fitted_per_model_airport_regime_seed": True,
        },
        "claim_boundary": "Near-conflict alerting is a research proxy, not regulatory safety assurance.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airport", choices=AIRPORTS, required=True)
    parser.add_argument("--regime", choices=REGIMES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = fit(
        airport=args.airport,
        regime=args.regime,
        seed=args.seed,
        device=torch.device(args.device),
        workers=args.workers,
        batch_size=args.batch_size,
    )
    atomic_json(args.output.resolve(), result)
    print(json.dumps({"output": args.output.resolve().as_posix(), "thresholds": result["thresholds"]}, indent=2))


if __name__ == "__main__":
    main()
