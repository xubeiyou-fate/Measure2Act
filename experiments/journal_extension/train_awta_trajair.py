"""Train matched ASCENT with annealed WTA on the frozen TrajAir splits."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Subset

from experiments.metric_exact.evaluation import evaluate, target_tail_threshold
from experiments.metric_exact.model import build_model
from experiments.metric_exact.protocol import load_protocol as load_c127_protocol, sha256
from experiments.metric_exact.train import _capture_rng, _restore_rng, limited_indices, move, set_seed
from model.utils import TrajectoryDataset, seed_worker, seq_collate

from .awta import annealed_wta_objective
from .train_awta_tartan import atomic_json


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("awta_trajair_protocol_v1.json")
FORMAL_RUN_ROOT = ROOT / "runs/journal_extension_20260814/awta_trajair"
FORMAL_RESULT_ROOT = ROOT / "artifacts/journal_extension_20260814/awta_trajair/development"
SMOKE_RUN_ROOT = ROOT / "runs/journal_extension_20260814/awta_trajair_smoke"
SMOKE_RESULT_ROOT = ROOT / "artifacts/journal_extension_20260814/awta_trajair/smoke"


def load_protocol() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    for key in ("c127_protocol", "data_manifest", "train_scene_dates", "development_scene_dates"):
        path = ROOT / protocol["inherits"][key]
        if not path.is_file() or sha256(path) != protocol["inherits"][f"{key}_sha256"]:
            raise RuntimeError(f"frozen source mismatch: {path}")
    return protocol


def expected_paths(seed: int) -> tuple[Path, Path]:
    return (
        FORMAL_RUN_ROOT / f"seed{seed}_formal",
        FORMAL_RESULT_ROOT / f"seed{seed}_formal.json",
    )


def validate_paths(args: argparse.Namespace, smoke: bool) -> tuple[Path, Path]:
    run_dir = args.run_dir.resolve()
    output = args.output.resolve()
    if smoke:
        if not run_dir.is_relative_to(SMOKE_RUN_ROOT.resolve()):
            raise ValueError("smoke run path is outside the frozen root")
        if not output.is_relative_to(SMOKE_RESULT_ROOT.resolve()):
            raise ValueError("smoke output path is outside the frozen root")
    else:
        expected_run, expected_output = expected_paths(args.seed)
        if run_dir != expected_run.resolve() or output != expected_output.resolve():
            raise ValueError("formal TrajAir aWTA paths differ from the frozen protocol")
    return run_dir, output


def loader(dataset, *, batch_size: int, shuffle: bool, workers: int, generator=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=seq_collate,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    protocol = load_protocol()
    if args.seed not in protocol["training"]["seeds"]:
        raise ValueError("unregistered seed")
    smoke = args.max_train_scenes is not None or args.max_dev_scenes is not None
    if not smoke and (
        args.epochs != protocol["training"]["epochs"]
        or args.batch_size != protocol["training"]["batch_size"]
        or args.eval_batch_size != protocol["training"]["evaluation_batch_size"]
    ):
        raise RuntimeError("formal hyperparameters differ from the frozen protocol")
    run_dir, output = validate_paths(args, smoke)
    if output.exists():
        raise FileExistsError(output)
    if run_dir.exists() and not args.resume:
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True, exist_ok=args.resume)

    parent = load_c127_protocol()
    generator = set_seed(args.seed)
    train_dataset = TrajectoryDataset(
        parent.split_path("train").as_posix(),
        obs_len=16,
        obs_steps=1,
        pred_len=120,
        pred_step=5,
        delim=" ",
    )
    dev_dataset = TrajectoryDataset(
        parent.split_path("dev").as_posix(),
        obs_len=16,
        obs_steps=1,
        pred_len=120,
        pred_step=5,
        delim=" ",
    )
    train_dates = json.loads((ROOT / protocol["inherits"]["train_scene_dates"]).read_text())["dates"]
    dev_dates = json.loads((ROOT / protocol["inherits"]["development_scene_dates"]).read_text())["dates"]
    if len(train_dataset) != len(train_dates) or len(dev_dataset) != len(dev_dates):
        raise RuntimeError("TrajAir dataset/date index mismatch")
    train_indices = limited_indices(list(range(len(train_dataset))), args.max_train_scenes)
    dev_indices = limited_indices(list(range(len(dev_dataset))), args.max_dev_scenes)
    selected_dev_dates = [dev_dates[index] for index in dev_indices]
    train_data = Subset(train_dataset, train_indices)
    dev_data = Subset(dev_dataset, dev_indices)
    train_loader = loader(train_data, batch_size=args.batch_size, shuffle=True, workers=args.workers, generator=generator)
    dev_loader = loader(dev_data, batch_size=args.eval_batch_size, shuffle=False, workers=args.workers)

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build_model("B0_signed_coupled", batch_size=args.batch_size).to(device)
    optimizer = Adam(model.parameters(), lr=float(protocol["training"]["learning_rate"]))
    scheduler = MultiStepLR(optimizer, milestones=protocol["training"]["milestones"], gamma=float(protocol["training"]["gamma"]))
    checkpoint_path = run_dir / "last.pt"
    history: list[dict[str, Any]] = []
    start_epoch = 1
    if args.resume:
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if saved.get("protocol_sha256") != sha256(PROTOCOL):
            raise RuntimeError("resume protocol mismatch")
        if int(saved.get("seed", -1)) != args.seed:
            raise RuntimeError("resume seed mismatch")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        _restore_rng(saved["rng_state"], generator)
        history = list(saved["history"])
        start_epoch = int(saved["epoch"]) + 1

    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        totals: dict[str, float] = {}
        batches = 0
        epoch_started = time.perf_counter()
        for data in train_loader:
            data = move(data, device)
            target = data["pred_traj"].transpose(1, 0)
            optimizer.zero_grad(set_to_none=True)
            predictions, logits, _ = model(data)
            loss, diagnostics = annealed_wta_objective(
                predictions,
                logits,
                target,
                epoch=epoch,
                total_epochs=args.epochs,
                initial_temperature=float(protocol["model"]["initial_temperature_km"]),
                final_temperature=float(protocol["model"]["final_temperature_km"]),
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            if not all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
                raise RuntimeError(f"non-finite gradient at epoch {epoch}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(protocol["training"]["gradient_clip_norm"]))
            optimizer.step()
            batches += 1
            for name, value in diagnostics.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
        scheduler.step()
        record = {
            "epoch": epoch,
            "batches": batches,
            "learning_rate_after_epoch": optimizer.param_groups[0]["lr"],
            "train": {name: value / max(batches, 1) for name, value in totals.items()},
            "elapsed_seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save(
            {
                "format_version": 1,
                "model": "ASCENT-aWTA",
                "dataset": "TrajAir",
                "seed": args.seed,
                "epoch": epoch,
                "protocol_sha256": sha256(PROTOCOL),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "rng_state": _capture_rng(generator),
                "history": history,
                "locked_test_used": False,
            },
            temporary,
        )
        temporary.replace(checkpoint_path)
        print(json.dumps(record), flush=True)

    metrics = evaluate(
        model,
        dev_loader,
        device,
        scene_dates=selected_dev_dates,
        tail_threshold=target_tail_threshold(train_dataset),
    )
    result = {
        "format_version": 1,
        "experiment_id": f"aWTA_TrajAir_seed{args.seed}_development",
        "formal": not smoke,
        "dataset": "TrajAir",
        "seed": args.seed,
        "train_scenes": len(train_data),
        "development_scenes": len(dev_data),
        "development_dates": sorted(set(selected_dev_dates)),
        "development_metrics": metrics,
        "checkpoint": {"path": checkpoint_path.relative_to(ROOT).as_posix(), "sha256": sha256(checkpoint_path), "epoch": args.epochs},
        "protocol": {"path": PROTOCOL.relative_to(ROOT).as_posix(), "sha256": sha256(PROTOCOL)},
        "implementation": {
            "runner": Path(__file__).relative_to(ROOT).as_posix(),
            "runner_sha256": sha256(Path(__file__)),
            "objective": "experiments/journal_extension/awta.py",
            "objective_sha256": sha256(Path(__file__).with_name("awta.py")),
        },
        "history": history,
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        },
        "integrity": {"train_and_development_only": True, "locked_test_used": False, "fixed_final_epoch": True, "only_training_assignment_changed": True},
        "command": [sys.executable, "-m", "experiments.experiments.journal_extension.train_awta_trajair", *sys.argv[1:]],
    }
    atomic_json(output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-train-scenes", type=int)
    parser.add_argument("--max-dev-scenes", type=int)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = train(args)
    print(json.dumps({"output": str(args.output), "formal": result["formal"]}, indent=2))


if __name__ == "__main__":
    main()
