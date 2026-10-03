"""Train and evaluate EqMotion on the exact ASCENT temporal protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.energy_predict_optimize.evaluation import RankingMetricAccumulator, compute_batch_metrics
from model.utils import TrajectoryDataset

from .eqmotion_aviation import (
    CachedTrajectorySceneDataset,
    EqMotionAviation,
    aviation_collate,
    best_of_k_ade_loss,
    valid_actor_tensors,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("eqmotion_matched_protocol_v1.json")


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
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.Generator().manual_seed(seed)


def load_dataset(path: Path, delimiter: str, maximum: int | None):
    source = TrajectoryDataset(
        path.as_posix(), obs_len=16, obs_steps=1, pred_len=120, pred_step=5, delim=delimiter
    )
    indices = None
    if maximum is not None and maximum < len(source):
        indices = torch.linspace(0, len(source) - 1, maximum).round().long().unique().tolist()
    return CachedTrajectorySceneDataset(source, indices)


def move(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def loader(dataset, *, batch_size: int, shuffle: bool, generator=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=aviation_collate,
        generator=generator,
    )


@torch.inference_mode()
def evaluate(model, dataset, device: torch.device, batch_size: int) -> dict[str, object]:
    model.eval()
    state = RankingMetricAccumulator()
    for batch in loader(dataset, batch_size=batch_size, shuffle=False):
        batch = move(batch, device)
        prediction = model(batch["history"], batch["num_valid"])
        prediction, truth = valid_actor_tensors(prediction, batch["future"], batch["valid"])
        probability = torch.full((len(prediction), 5), 0.2, device=device, dtype=torch.float64)
        decision = torch.zeros(len(prediction), device=device, dtype=torch.long)
        state.update(compute_batch_metrics(
            prediction.to(torch.float64), probability, decision, truth.to(torch.float64)
        ))
    return state.summarize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-scenes", type=int)
    parser.add_argument("--max-eval-scenes", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    formal = not args.smoke
    frozen_eval_batch = int(protocol.get("evaluation", {}).get("batch_size", args.batch_size))
    eval_batch_size = args.eval_batch_size or frozen_eval_batch
    if formal and (
        args.epochs != protocol["training"]["epochs"]
        or args.batch_size != protocol["training"]["batch_size"]
        or eval_batch_size != frozen_eval_batch
    ):
        raise ValueError("formal settings differ from the frozen protocol")
    if formal and any(value is not None for value in (args.max_train_scenes, args.max_eval_scenes, args.max_train_batches)):
        raise ValueError("formal EqMotion run cannot use scene or batch limits")
    seed = int(protocol["training"]["seed"])
    set_seed(seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    train = load_dataset(ROOT / protocol["data"]["train"], " ", args.max_train_scenes)
    evaluation = {
        "trajair_development": load_dataset(ROOT / protocol["data"]["development"], " ", args.max_eval_scenes),
        "tartan_kagc_submission_1024": load_dataset(ROOT / protocol["data"]["external"]["kagc"], ",", args.max_eval_scenes),
        "tartan_kbtp_submission_1024": load_dataset(ROOT / protocol["data"]["external"]["kbtp"], ",", args.max_eval_scenes),
    }
    model = EqMotionAviation(
        device=device,
        hidden_nf=int(protocol["model"]["hidden_nf"]),
        channels=int(protocol["model"]["channels"]),
        layers=int(protocol["model"]["layers"]),
        modes=5,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(protocol["training"]["learning_rate"]))
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint = run_dir / "last.pt"
    history = []
    start_epoch = 1
    if args.resume:
        if not last_checkpoint.is_file():
            raise FileNotFoundError(last_checkpoint)
        saved = torch.load(last_checkpoint, map_location=device, weights_only=False)
        if saved.get("protocol_sha256") != sha256(protocol_path):
            raise RuntimeError("EqMotion resume protocol hash mismatch")
        if int(saved.get("seed", -1)) != seed:
            raise RuntimeError("EqMotion resume seed mismatch")
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        history = list(saved["history"])
        start_epoch = int(saved["epoch"]) + 1
    elif last_checkpoint.exists():
        raise FileExistsError(last_checkpoint)
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total = 0.0
        batches = 0
        epoch_started = time.perf_counter()
        epoch_generator = torch.Generator().manual_seed(seed * 1000 + epoch)
        for batch in loader(train, batch_size=args.batch_size, shuffle=True, generator=epoch_generator):
            if args.max_train_batches is not None and batches >= args.max_train_batches:
                break
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch["history"], batch["num_valid"])
            loss = best_of_k_ade_loss(prediction, batch["future"], batch["valid"])
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite EqMotion loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(protocol["training"]["gradient_clip_norm"]))
            optimizer.step()
            total += float(loss.detach().cpu())
            batches += 1
        history.append({
            "epoch": epoch,
            "batches": batches,
            "mean_loss": total / max(batches, 1),
            "elapsed_seconds": time.perf_counter() - epoch_started,
        })
        temporary = last_checkpoint.with_suffix(".tmp")
        torch.save({
            "format_version": 1,
            "model": "EqMotion aviation matched",
            "epoch": epoch,
            "seed": seed,
            "protocol_sha256": sha256(protocol_path),
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        }, temporary)
        temporary.replace(last_checkpoint)
        print(json.dumps(history[-1]), flush=True)
    checkpoint = run_dir / f"epoch_{args.epochs:03d}.pt"
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    torch.save({
        "format_version": 1,
        "model": "EqMotion aviation matched",
        "epoch": args.epochs,
        "seed": seed,
        "protocol_sha256": sha256(protocol_path),
        "state_dict": model.state_dict(),
        "history": history,
    }, checkpoint)
    metrics = {name: evaluate(model, dataset, device, eval_batch_size) for name, dataset in evaluation.items()}
    payload = {
        "format_version": 1,
        "experiment_id": "EqMotion_matched_TrajAir_to_Tartan",
        "formal": formal,
        "seed": seed,
        "train_scenes": len(train),
        "evaluation_scenes": {name: len(dataset) for name, dataset in evaluation.items()},
        "history": history,
        "metrics": metrics,
        "publication_metrics": protocol["publication_metrics"],
        "diagnostic_only_metrics": protocol["diagnostic_only_metrics"],
        "checkpoint": {
            "path": checkpoint.relative_to(ROOT).as_posix(),
            "sha256": sha256(checkpoint),
        },
        "protocol": {
            "path": protocol_path.relative_to(ROOT).as_posix(),
            "sha256": sha256(protocol_path),
        },
        "integrity": {
            "matched_ascent_temporal_grid": True,
            "external_finetuning_or_calibration": False,
            "fixed_final_epoch": True,
            "uniform_probability_measure": True,
            "learned_mode_ranking": False,
        },
        "runtime": {
            "device": str(device),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps({
        "output": args.output.resolve().as_posix(),
        "formal": formal,
        "train_scenes": len(train),
        "evaluation_scenes": payload["evaluation_scenes"],
        "metrics": {name: {key: value[key] for key in ("minade", "minfde", "energy_score")} for name, value in metrics.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
