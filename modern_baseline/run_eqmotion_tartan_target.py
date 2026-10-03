"""Train target-only EqMotion on frozen Tartan train dates and evaluate development only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.energy_predict_optimize.evaluation import compute_batch_metrics
from model.utils import TrajectoryDataset

from .eqmotion_aviation import (
    CachedTrajectorySceneDataset,
    EqMotionAviation,
    aviation_collate,
    best_of_k_ade_loss,
    valid_actor_tensors,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("eqmotion_tartan_target_protocol_v1.json")
AMENDMENT = Path(__file__).with_name("eqmotion_tartan_target_amendment_skip5_v2.json")
ALLOWED_SPLITS = frozenset({"train", "development"})
PUBLICATION_METRICS = ("minade", "minfde", "energy_score")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if set(protocol["data"]["allowed_splits"]) != ALLOWED_SPLITS:
        raise RuntimeError("EqMotion target runner permits train/development only")
    if protocol["data"]["forbidden_splits"] != ["test"]:
        raise RuntimeError("frozen protocol must explicitly forbid test")
    grid = protocol["grid"]
    expected = {
        "coordinates": 3,
        "history_points": 16,
        "history_interval_seconds": 1,
        "future_points": 24,
        "future_interval_seconds": 5,
        "future_horizon_seconds": 120,
        "modes": 5,
    }
    for key, value in expected.items():
        if grid.get(key) != value:
            raise RuntimeError(f"frozen EqMotion grid mismatch: {key}")
    return protocol


def resolve_split_path(
    data_root: Path,
    airport: str,
    split: str,
    protocol: dict[str, Any],
) -> Path:
    if airport not in protocol["data"]["airports"]:
        raise ValueError(f"unregistered airport: {airport}")
    if split not in ALLOWED_SPLITS:
        raise ValueError("only train and development are accessible; test is locked")
    path = (data_root / airport / split).resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    if "test" in {part.lower() for part in path.parts}:
        raise RuntimeError("refusing to resolve any test path")
    return path


def validate_manifest(manifest_path: Path, airport: str) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_id") != "partc_two_dataset_target_domain_v1":
        raise RuntimeError("unexpected Tartan data protocol")
    if manifest.get("history") != {"interval_seconds": 1, "points": 16}:
        raise RuntimeError("manifest history grid mismatch")
    future = manifest.get("future", {})
    if future.get("interval_seconds") != 5 or future.get("points") != 24:
        raise RuntimeError("manifest future grid mismatch")
    split = manifest.get("split", {})
    if (
        split.get("unit") != "calendar date"
        or split.get("rule") != "chronological earliest/middle/latest"
        or split.get("fractions") != [0.6, 0.2, 0.2]
    ):
        raise RuntimeError("manifest date split mismatch")
    registered = manifest.get("datasets", {}).get(airport, {}).get("splits", {})
    train_dates = list(registered.get("train", {}).get("dates", []))
    development_dates = list(registered.get("development", {}).get("dates", []))
    if not train_dates or not development_dates:
        raise RuntimeError("manifest lacks registered train/development dates")
    if set(train_dates) & set(development_dates) or max(train_dates) >= min(development_dates):
        raise RuntimeError("train/development dates are not disjoint chronological blocks")
    return {
        "train": registered["train"],
        "development": registered["development"],
    }


def verify_formal_sources(protocol: dict[str, Any]) -> None:
    checks = {
        ROOT / protocol["data"]["manifest"]: protocol["data"]["manifest_sha256"],
        ROOT / protocol["data"]["frozen_receipt"]: protocol["data"]["frozen_receipt_sha256"],
        ROOT / protocol["adapter"]["path"]: protocol["adapter"]["sha256"],
    }
    for path, expected in checks.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256(path) != expected:
            raise RuntimeError(f"frozen source hash mismatch: {path}")
    official = ROOT / protocol["official_source"]["model_file"]
    if not official.is_file():
        raise FileNotFoundError(official)
    if sha256(official) != protocol["official_source"]["model_file_sha256"]:
        raise RuntimeError("official EqMotion model file hash mismatch")
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    if sha256(PROTOCOL) != amendment["frozen_inputs"]["base_protocol_sha256"]:
        raise RuntimeError("EqMotion skip5 amendment base protocol mismatch")
    scene_index = ROOT / amendment["repair"]["expected_scene_counts_source"]
    if sha256(scene_index) != amendment["repair"]["expected_scene_index_summary_sha256"]:
        raise RuntimeError("EqMotion skip5 amendment scene-index mismatch")


def set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.Generator().manual_seed(seed)


def load_dataset(
    path: Path,
    *,
    cache_dir: Path,
    delimiter: str,
    maximum: int | None,
) -> CachedTrajectorySceneDataset:
    source = TrajectoryDataset(
        path.as_posix(),
        obs_len=16,
        obs_steps=1,
        pred_len=120,
        skip=5,
        pred_step=5,
        delim=delimiter,
        cache_dir=cache_dir,
    )
    indices = None
    if maximum is not None and maximum < len(source):
        indices = torch.linspace(0, len(source) - 1, maximum).round().long().unique().tolist()
    dataset = CachedTrajectorySceneDataset(source, indices)
    if not len(dataset):
        raise RuntimeError(f"no valid scenes in {path}")
    return dataset


def loader(dataset, *, batch_size: int, shuffle: bool, generator=None) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=aviation_collate,
        generator=generator,
    )


def move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


@torch.inference_mode()
def evaluate_development(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, float | int], dict[str, float | int]]:
    model.eval()
    totals = {name: 0.0 for name in PUBLICATION_METRICS}
    agents = 0
    batches = 0
    started = time.perf_counter()
    for batch in loader(dataset, batch_size=batch_size, shuffle=False):
        batch = move(batch, device)
        prediction = model(batch["history"], batch["num_valid"])
        prediction, truth = valid_actor_tensors(prediction, batch["future"], batch["valid"])
        count = int(prediction.shape[0])
        probability = torch.full((count, 5), 0.2, device=device, dtype=torch.float64)
        fixed_candidate0 = torch.zeros(count, device=device, dtype=torch.long)
        values = compute_batch_metrics(
            prediction.to(torch.float64),
            probability,
            fixed_candidate0,
            truth.to(torch.float64),
        )
        for name in PUBLICATION_METRICS:
            totals[name] += float(values[name].sum().cpu())
        agents += count
        batches += 1
    if not agents:
        raise RuntimeError("development evaluation produced no valid actors")
    elapsed = time.perf_counter() - started
    metrics = {name: totals[name] / agents for name in PUBLICATION_METRICS}
    metrics["agents"] = agents
    efficiency = {
        "development_elapsed_seconds": elapsed,
        "development_scenes_per_second": len(dataset) / max(elapsed, 1e-12),
        "development_actors_per_second": agents / max(elapsed, 1e-12),
        "development_batches": batches,
    }
    return metrics, efficiency


def checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    history: list[dict[str, Any]],
    epoch: int,
    airport: str,
    seed: int,
    protocol_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model": "EqMotion aviation Tartan target-only",
        "airport": airport,
        "regime": "target_only",
        "seed": seed,
        "epoch": epoch,
        "protocol_sha256": protocol_sha256,
        "manifest_sha256": manifest_sha256,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airport", choices=("KAGC", "KBTP"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-root-override", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--max-train-scenes", type=int)
    parser.add_argument("--max-dev-scenes", type=int)
    parser.add_argument("--max-train-batches", type=int)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    protocol_path = PROTOCOL.resolve()
    protocol = load_protocol(protocol_path)
    training = protocol["training"]
    if args.seed not in training["registered_seeds"]:
        raise ValueError("seed is outside the frozen EqMotion registry")

    epochs = args.epochs if args.epochs is not None else int(training["epochs"])
    batch_size = args.batch_size if args.batch_size is not None else int(training["batch_size"])
    eval_batch_size = (
        args.eval_batch_size
        if args.eval_batch_size is not None
        else int(training["evaluation_batch_size"])
    )
    limits = (args.max_train_scenes, args.max_dev_scenes, args.max_train_batches)
    if not args.smoke:
        verify_formal_sources(protocol)
        if args.data_root_override is not None:
            raise ValueError("formal training cannot override the frozen data root")
        if epochs != int(training["epochs"]):
            raise ValueError("formal training must use the frozen fixed epoch count")
        if batch_size != int(training["batch_size"]) or eval_batch_size != int(training["evaluation_batch_size"]):
            raise ValueError("formal batch sizes differ from the frozen protocol")
        if any(limit is not None for limit in limits):
            raise ValueError("formal training cannot use scene or batch limits")
    else:
        if torch.device(args.device).type != "cpu":
            raise ValueError("this smoke path is CPU-only while the formal GPU is occupied")
        if epochs < 1 or not all(limit is not None and limit > 0 for limit in limits):
            raise ValueError("smoke requires positive epoch, train/dev scene, and batch limits")

    run_dir = args.run_dir.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not args.resume and run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True, exist_ok=args.resume)

    data_root = (
        args.data_root_override.resolve()
        if args.data_root_override is not None
        else (ROOT / protocol["data"]["root"]).resolve()
    )
    manifest_path = data_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest_hash = sha256(manifest_path)
    if not args.smoke and manifest_hash != protocol["data"]["manifest_sha256"]:
        raise RuntimeError("formal data manifest hash mismatch")
    split_receipt = validate_manifest(manifest_path, args.airport)
    train_path = resolve_split_path(data_root, args.airport, "train", protocol)
    development_path = resolve_split_path(data_root, args.airport, "development", protocol)

    generator = set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but unavailable")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    compile_started = time.perf_counter()
    cache_dir = run_dir / "dataset_cache"
    train = load_dataset(
        train_path,
        cache_dir=cache_dir,
        delimiter=protocol["data"]["delimiter"],
        maximum=args.max_train_scenes,
    )
    development = load_dataset(
        development_path,
        cache_dir=cache_dir,
        delimiter=protocol["data"]["delimiter"],
        maximum=args.max_dev_scenes,
    )
    if not args.smoke:
        expected_train = int(split_receipt["train"]["scenes"])
        expected_development = int(split_receipt["development"]["scenes"])
        if len(train) != expected_train or len(development) != expected_development:
            raise RuntimeError(
                "EqMotion dataset differs from the frozen skip=5 scene index: "
                f"train {len(train)} != {expected_train}, development "
                f"{len(development)} != {expected_development}"
            )
    data_compile_seconds = time.perf_counter() - compile_started

    model = EqMotionAviation(
        device=device,
        hidden_nf=int(protocol["model"]["hidden_nf"]),
        channels=int(protocol["model"]["channels"]),
        layers=int(protocol["model"]["layers"]),
        modes=int(protocol["grid"]["modes"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(value) for value in training["milestones"]],
        gamma=float(training["gamma"]),
    )
    protocol_hash = sha256(protocol_path)
    last_checkpoint = run_dir / "last.pt"
    history: list[dict[str, Any]] = []
    start_epoch = 1
    if args.resume:
        if not last_checkpoint.is_file():
            raise FileNotFoundError(last_checkpoint)
        saved = torch.load(last_checkpoint, map_location=device, weights_only=False)
        if saved.get("protocol_sha256") != protocol_hash or saved.get("manifest_sha256") != manifest_hash:
            raise RuntimeError("resume provenance mismatch")
        if saved.get("airport") != args.airport or int(saved.get("seed", -1)) != args.seed:
            raise RuntimeError("resume airport/seed mismatch")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        history = list(saved["history"])
        start_epoch = int(saved["epoch"]) + 1

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_started = time.perf_counter()
        total_loss = 0.0
        batches = 0
        actors = 0
        scenes = 0
        epoch_generator = torch.Generator().manual_seed(args.seed * 1000 + epoch)
        for batch in loader(train, batch_size=batch_size, shuffle=True, generator=epoch_generator):
            if args.max_train_batches is not None and batches >= args.max_train_batches:
                break
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch["history"], batch["num_valid"])
            loss = best_of_k_ade_loss(prediction, batch["future"], batch["valid"])
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite EqMotion loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("non-finite EqMotion gradient")
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            actors += int(batch["valid"].sum().item())
            scenes += int(batch["history"].shape[0])
            batches += 1
        scheduler.step()
        record = {
            "epoch": epoch,
            "batches": batches,
            "actors": actors,
            "scenes": scenes,
            "mean_loss": total_loss / max(batches, 1),
            "learning_rate_after_epoch": scheduler.get_last_lr()[0],
            "elapsed_seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        temporary = last_checkpoint.with_suffix(".tmp")
        torch.save(
            checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                history=history,
                epoch=epoch,
                airport=args.airport,
                seed=args.seed,
                protocol_sha256=protocol_hash,
                manifest_sha256=manifest_hash,
            ),
            temporary,
        )
        temporary.replace(last_checkpoint)
        print(json.dumps(record, sort_keys=True), flush=True)
    training_elapsed = sum(float(record["elapsed_seconds"]) for record in history)
    train_actor_updates = sum(int(record["actors"]) for record in history)
    train_scene_updates = sum(int(record["scenes"]) for record in history)

    if not last_checkpoint.is_file():
        raise RuntimeError("fixed final checkpoint was not written")
    final = torch.load(last_checkpoint, map_location=device, weights_only=False)
    if int(final["epoch"]) != epochs:
        raise RuntimeError("development evaluation requires the fixed final epoch")
    model.load_state_dict(final["model_state_dict"])
    development_metrics, development_efficiency = evaluate_development(
        model, development, device, eval_batch_size
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    runtime = {
        "device": str(device),
        "data_compile_seconds": data_compile_seconds,
        "training_elapsed_seconds": training_elapsed,
        "training_scene_updates": train_scene_updates,
        "training_actor_updates": train_actor_updates,
        "training_scene_updates_per_second": train_scene_updates / max(training_elapsed, 1e-12),
        "training_actor_updates_per_second": train_actor_updates / max(training_elapsed, 1e-12),
        "parameter_count": parameters,
        "trainable_parameter_count": trainable_parameters,
        "checkpoint_bytes": last_checkpoint.stat().st_size,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        **development_efficiency,
    }
    payload = {
        "format_version": 1,
        "experiment_id": f"EqMotion_Tartan_{args.airport}_target_only_seed{args.seed}",
        "formal": not args.smoke,
        "airport": args.airport,
        "regime": "target_only",
        "seed": args.seed,
        "grid": protocol["grid"],
        "data": {
            "root": data_root.relative_to(ROOT).as_posix() if data_root.is_relative_to(ROOT) else data_root.as_posix(),
            "manifest_sha256": manifest_hash,
            "train_scenes": len(train),
            "development_scenes": len(development),
            "registered_train": split_receipt["train"],
            "registered_development": split_receipt["development"],
        },
        "history": history,
        "development_metrics": development_metrics,
        "publication_metrics": list(PUBLICATION_METRICS),
        "efficiency": runtime,
        "checkpoint": {
            "policy": training["checkpoint_policy"],
            "path": last_checkpoint.relative_to(ROOT).as_posix() if last_checkpoint.is_relative_to(ROOT) else last_checkpoint.as_posix(),
            "sha256": sha256(last_checkpoint),
            "epoch": epochs,
        },
        "protocol": {
            "path": protocol_path.relative_to(ROOT).as_posix() if protocol_path.is_relative_to(ROOT) else protocol_path.as_posix(),
            "sha256": protocol_hash,
        },
        "amendment": {
            "path": AMENDMENT.relative_to(ROOT).as_posix(),
            "sha256": sha256(AMENDMENT),
            "trajectory_dataset_skip": 5,
        },
        "official_source": protocol["official_source"],
        "probability_and_top1_boundary": {
            "probabilities": "uniform [0.2, 0.2, 0.2, 0.2, 0.2]",
            "decision": "fixed candidate 0",
            "energy_publication_comparable": True,
            "top1_publication_comparable": False,
            "omitted_metrics": ["top1_ade", "top1_fde", "nll", "brier", "ece", "rank"],
            "reason": protocol["evaluation"]["top1_limitation"],
        },
        "integrity": {
            "train_and_development_only": True,
            "locked_test_resolved": False,
            "locked_test_dataset_constructed": False,
            "locked_test_model_inference": False,
            "date_split": True,
            "fixed_final_epoch": True,
            "development_checkpoint_selection": False,
            "deterministic": True,
            "automatic_mixed_precision": False,
        },
        "command": [sys.executable, "-m", "modern_baseline.run_eqmotion_tartan_target", *sys.argv[1:]],
        "claim_boundary": protocol["claim_boundary"],
    }
    atomic_json(output, payload)
    print(
        json.dumps(
            {
                "output": output.as_posix(),
                "checkpoint": payload["checkpoint"],
                "development_metrics": development_metrics,
                "efficiency": runtime,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
