"""Deterministically retrain frozen official aviation baselines for E1.

Run this module in a fresh process for exactly one family.  The official
repositories both expose a top-level ``model`` package, so mixing families in
one interpreter would make Python's module cache scientifically ambiguous.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("e1_official_baseline_protocol_v2.json")
SOURCE_RECEIPT = Path(__file__).with_name("e1_baseline_sources.json")
FAMILIES = ("trajairnet", "actrajnet")
DATASETS = ("7days1", "7days2", "7days3", "7days4")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _official_args() -> SimpleNamespace:
    return SimpleNamespace(
        input_channels=3,
        preds=120,
        preds_step=10,
        tcn_channel_size=256,
        tcn_layers=2,
        tcn_kernels=4,
        lstm_input_channels=1,
        lstm_hidden_size=256,
        lstm_layers=2,
        num_context_input_c=2,
        num_context_output_c=7,
        cnn_kernels=2,
        gat_heads=16,
        graph_hidden=256,
        dropout=0.05,
        alpha=0.2,
        cvae_hidden=128,
        cvae_channel_size=128,
        cvae_layers=2,
        mlp_layer=32,
        obs=11,
    )


def _load_official(family: str):
    if family not in FAMILIES:
        raise ValueError(f"unknown family: {family}")
    repository = ROOT / "external" / (
        "trajairnet_official" if family == "trajairnet" else "actrajnet_official"
    )
    repository_text = str(repository.resolve())
    if repository_text not in sys.path:
        sys.path.insert(0, repository_text)
    loaded_model = sys.modules.get("model")
    if loaded_model is not None:
        model_file = str(getattr(loaded_model, "__file__", ""))
        if repository_text not in model_file:
            raise RuntimeError(
                "a conflicting top-level model package was imported before the official baseline"
            )
    utils = importlib.import_module("model.utils")
    if family == "trajairnet":
        model_class = importlib.import_module("model.trajairnet").TrajAirNet
    else:
        model_class = importlib.import_module("model.CAF_tcn").ACTrajNet
    return repository, model_class, utils


def _dataset_path(family: str, dataset: str) -> Path:
    if family == "trajairnet":
        return (
            ROOT / "dataset" / f"{dataset}_trajair_reconstructed"
            / "processed_data" / "train"
        )
    return (
        ROOT
        / "external"
        / "actrajnet_official"
        / "dataset"
        / f"{dataset}_no_social"
        / "train"
    )


def _checkpoint_path(run_dir: Path, epoch: int) -> Path:
    return run_dir / f"epoch_{epoch:03d}.pt"


def _save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    updates: int,
    history: list[dict[str, float | int]],
    metadata: dict[str, Any],
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "updates": updates,
            "history": history,
            "metadata": metadata,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
        },
        temporary,
    )
    temporary.replace(path)


def _load_resume(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    expected: dict[str, Any],
    target_device: torch.device,
    rng_source_device: int | None,
) -> tuple[int, int, list[dict[str, float | int]]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})
    for key in ("family", "dataset", "seed", "protocol_sha256"):
        if metadata.get(key) != expected[key]:
            raise RuntimeError(f"resume metadata mismatch for {key}")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    rng_state = checkpoint.get("rng_state")
    if rng_state is None:
        raise RuntimeError("resume checkpoint lacks deterministic RNG state")
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch"])
    if torch.cuda.is_available() and rng_state["cuda"]:
        if rng_source_device is not None:
            if not 0 <= rng_source_device < len(rng_state["cuda"]):
                raise ValueError("resume RNG source device lies outside checkpoint state")
            torch.cuda.set_rng_state(
                rng_state["cuda"][rng_source_device], device=target_device
            )
        elif len(rng_state["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(rng_state["cuda"])
        else:
            raise ValueError(
                "checkpoint and runtime CUDA device counts differ; "
                "specify --rng-source-device"
            )
    return (
        int(checkpoint["epoch"]) + 1,
        int(checkpoint["updates"]),
        list(checkpoint.get("history", [])),
    )


def train(
    *,
    family: str,
    dataset_name: str,
    seed: int,
    device: torch.device,
    epochs: int,
    max_scenes: int | None,
    run_dir: Path,
    resume: Path | None,
    compile_model: bool,
    rng_source_device: int | None,
) -> dict[str, Any]:
    if dataset_name not in DATASETS:
        raise ValueError(f"dataset outside frozen registry: {dataset_name}")
    if epochs < 1 or epochs > 10:
        raise ValueError("the formal v2 TrajAirNet protocol is fixed to at most 10 epochs")
    _seed_everything(seed)
    repository, model_class, utils = _load_official(family)
    dataset_path = _dataset_path(family, dataset_name)
    dataset = utils.TrajectoryDataset(
        str(dataset_path), obs_len=11, pred_len=120, step=10, delim=" "
    )
    if max_scenes is not None:
        if max_scenes < 9:
            raise ValueError("a smoke run needs at least nine scenes")
        dataset = Subset(dataset, range(min(max_scenes, len(dataset))))
    model = model_class(_official_args()).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    protocol_sha256 = _sha256(PROTOCOL)
    metadata: dict[str, Any] = {
        "family": family,
        "dataset": dataset_name,
        "seed": seed,
        "protocol_sha256": protocol_sha256,
        "source_receipt_sha256": _sha256(SOURCE_RECEIPT),
        "official_repository": str(repository.relative_to(ROOT)),
        "train_data": str(dataset_path.relative_to(ROOT)),
        "smoke_max_scenes": max_scenes,
        "torch_compile": compile_model,
        "resume_rng_source_device": rng_source_device,
    }
    start_epoch = 1
    updates = 0
    history: list[dict[str, float | int]] = []
    if resume is not None:
        start_epoch, updates, history = _load_resume(
            resume,
            model=model,
            optimizer=optimizer,
            expected=metadata,
            target_device=device,
            rng_source_device=rng_source_device,
        )
    elif rng_source_device is not None:
        raise ValueError("--rng-source-device is valid only with --resume")
    training_model = (
        torch.compile(model, fullgraph=False, dynamic=False)
        if compile_model else model
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    if start_epoch > epochs:
        raise RuntimeError("resume checkpoint is already beyond the requested final epoch")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        # Epoch-indexed sampler seeds make interrupted/resumed ordering exact.
        generator = torch.Generator().manual_seed(seed * 1000 + epoch)
        loader = DataLoader(
            dataset,
            batch_size=9,
            num_workers=0,
            shuffle=True,
            collate_fn=utils.seq_collate,
            generator=generator,
        )
        training_model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_scenes = 0
        total_loss = 0.0
        scene_count = 0
        epoch_started = time.perf_counter()
        for batch in loader:
            batch = [tensor.to(device) for tensor in batch]
            obs, pred, _obs_rel, _pred_rel, context, _seq_start = batch
            scene_sizes = _seq_start[:, 1] - _seq_start[:, 0]
            if family == "actrajnet" and not bool((scene_sizes == 1).all()):
                raise RuntimeError("ACTrajNet no-social data contains a multi-actor scene")
            loss: torch.Tensor | int = 0
            for agents_tensor in scene_sizes.unique(sorted=True):
                agents = int(agents_tensor)
                scene_indices = (scene_sizes == agents_tensor).nonzero().flatten().tolist()
                scene_obs = torch.stack([
                    obs[:, int(_seq_start[index, 0]):int(_seq_start[index, 1])].transpose(1, 2)
                    for index in scene_indices
                ])
                scene_pred = torch.stack([
                    pred[:, int(_seq_start[index, 0]):int(_seq_start[index, 1])].transpose(1, 2)
                    for index in scene_indices
                ])
                scene_context = torch.stack([
                    context[:, int(_seq_start[index, 0]):int(_seq_start[index, 1])].transpose(1, 2)
                    for index in scene_indices
                ])

                def scene_forward(
                    one_obs: torch.Tensor,
                    one_pred: torch.Tensor,
                    one_context: torch.Tensor,
                ):
                    recon, mean, log_var = training_model(
                        one_obs,
                        one_pred,
                        torch.ones((agents,), device=device),
                        one_context,
                    )
                    return torch.cat(recon), torch.cat(mean), torch.cat(log_var)

                recon, mean, log_var = torch.vmap(
                    scene_forward, randomness="different"
                )(scene_obs, scene_pred, scene_context)
                target = scene_pred.permute(0, 3, 2, 1)
                trajectory_loss = torch.sqrt(
                    (recon - target).square().mean(dim=(2, 3))
                )
                kld = -0.5 * (
                    1 + log_var - mean.square() - log_var.exp()
                ).sum(dim=(2, 3))
                group_loss = (trajectory_loss + kld).sum()
                loss = loss + group_loss
                total_loss += float(group_loss.detach())
            batch_scenes = int(_seq_start.shape[0])
            scene_count += batch_scenes
            if batch_scenes == 9:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                updates += 1
                accumulated_scenes = 0
            else:
                accumulated_scenes = batch_scenes
        # The official scripts drop the final incomplete accumulation group.
        elapsed = time.perf_counter() - epoch_started
        record: dict[str, float | int] = {
            "epoch": epoch,
            "scenes": scene_count,
            "updates": updates,
            "mean_scene_loss": total_loss / scene_count,
            "dropped_remainder_scenes": accumulated_scenes,
            "elapsed_seconds": elapsed,
            "torch_compile": compile_model,
        }
        history.append(record)
        checkpoint = _checkpoint_path(run_dir, epoch)
        _save_checkpoint(
            checkpoint,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            updates=updates,
            history=history,
            metadata=metadata,
        )
        print(json.dumps(record), flush=True)
    result: dict[str, Any] = {
        "format_version": 1,
        "experiment_id": "E1",
        "family": family,
        "dataset": dataset_name,
        "seed": seed,
        "epochs": epochs,
        "scenes": len(dataset),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "updates": updates,
        "history": history,
        "training_backends": {
            "eager_epochs": [
                int(record["epoch"])
                for record in history
                if not bool(record.get("torch_compile", False))
            ],
            "compiled_epochs": [
                int(record["epoch"])
                for record in history
                if bool(record.get("torch_compile", False))
            ],
        },
        "protocol_sha256": protocol_sha256,
        "source_receipt_sha256": _sha256(SOURCE_RECEIPT),
        "checkpoint": str(_checkpoint_path(run_dir, epochs)),
        "checkpoint_sha256": _sha256(_checkpoint_path(run_dir, epochs)),
        "runtime": {
            "device": str(device),
            "total_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "integrity": {
            "fixed_final_epoch": True,
            "validation_or_test_used": False,
            "official_model_source_modified": False,
            "automatic_mixed_precision": False,
            "tf32": False,
            "torch_compile": compile_model,
        },
    }
    _atomic_json(run_dir / "training_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--seed", type=int, choices=(7, 42, 123), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--rng-source-device", type=int)
    args = parser.parse_args()
    suffix = "smoke" if args.max_scenes is not None else "formal"
    if args.run_dir is None:
        args.run_dir = (
            ROOT / "runs" / "mabpt_official_baselines"
            / f"{args.family}_{args.dataset}_seed{args.seed}_{suffix}"
        )
    result = train(
        family=args.family,
        dataset_name=args.dataset,
        seed=args.seed,
        device=torch.device(args.device),
        epochs=args.epochs,
        max_scenes=args.max_scenes,
        run_dir=args.run_dir.resolve(),
        resume=args.resume.resolve() if args.resume is not None else None,
        compile_model=args.compile_model,
        rng_source_device=args.rng_source_device,
    )
    print(json.dumps({
        "output": str(args.run_dir / "training_summary.json"),
        "checkpoint": result["checkpoint"],
        "updates": result["updates"],
    }, indent=2))


if __name__ == "__main__":
    main()
