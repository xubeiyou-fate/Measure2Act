"""Train a matched ASCENT baseline with frozen annealed WTA assignment."""

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
from torch.utils.data import Subset

from experiments.metric_exact.evaluation import evaluate as evaluate_decision
from experiments.metric_exact.model import build_model as build_ascent_model
from mabpt import train_tartan_retrain as base

from .awta import annealed_wta_objective


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("awta_tartan_protocol_v1.json")
FORMAL_RUN_ROOT = ROOT / "runs/journal_extension_20260814/awta_tartan"
FORMAL_RESULT_ROOT = ROOT / "artifacts/journal_extension_20260814/awta_tartan/development"
SMOKE_RUN_ROOT = ROOT / "runs/journal_extension_20260814/awta_tartan_smoke"
SMOKE_RESULT_ROOT = ROOT / "artifacts/journal_extension_20260814/awta_tartan/smoke"


def load_protocol() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    for key in ("tartan_protocol", "data_manifest", "scene_index_summary", "prior_local_awta_objective"):
        path = ROOT / protocol["inherits"][key]
        if not path.is_file():
            raise FileNotFoundError(path)
        if base.sha256(path) != protocol["inherits"][f"{key}_sha256"]:
            raise RuntimeError(f"frozen source hash mismatch: {path}")
    return protocol


def expected_paths(airport: str, seed: int) -> tuple[Path, Path]:
    return (
        FORMAL_RUN_ROOT / airport / f"seed{seed}_formal",
        FORMAL_RESULT_ROOT / f"{airport}_seed{seed}_formal.json",
    )


def atomic_json(path: Path, payload: dict[str, Any], *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_paths(args: argparse.Namespace, smoke: bool) -> tuple[Path, Path]:
    run_dir = args.run_dir.resolve()
    output = args.output.resolve()
    if smoke:
        if not run_dir.is_relative_to(SMOKE_RUN_ROOT.resolve()):
            raise ValueError(f"smoke run must be below {SMOKE_RUN_ROOT}")
        if not output.is_relative_to(SMOKE_RESULT_ROOT.resolve()):
            raise ValueError(f"smoke output must be below {SMOKE_RESULT_ROOT}")
    else:
        expected_run, expected_output = expected_paths(args.airport, args.seed)
        if run_dir != expected_run.resolve() or output != expected_output.resolve():
            raise ValueError("formal aWTA paths differ from the frozen protocol")
    return run_dir, output


def train(args: argparse.Namespace) -> dict[str, Any]:
    protocol = load_protocol()
    if args.airport not in protocol["data"]["airports"]:
        raise ValueError("unregistered airport")
    if args.seed not in protocol["training"]["seeds"]:
        raise ValueError("unregistered seed")
    smoke = args.max_train_scenes is not None or args.max_dev_scenes is not None
    if not smoke and (
        args.epochs != protocol["training"]["epochs"]
        or args.batch_size != protocol["training"]["batch_size"]
        or args.eval_batch_size != protocol["training"]["evaluation_batch_size"]
    ):
        raise RuntimeError("formal aWTA hyperparameters differ from the frozen protocol")
    run_dir, output = validate_paths(args, smoke)
    if output.exists():
        raise FileExistsError(output)
    if run_dir.exists() and not args.resume:
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True, exist_ok=args.resume)

    tartan_protocol = json.loads((ROOT / protocol["inherits"]["tartan_protocol"]).read_text(encoding="utf-8"))
    generator = base.set_seed(args.seed)
    train_dataset, train_dates, train_index = base._dataset(tartan_protocol, args.airport, "train")
    dev_dataset, dev_dates, dev_index = base._dataset(tartan_protocol, args.airport, "development")
    train_indices = base._limited_indices(len(train_dataset), args.max_train_scenes)
    dev_indices = base._limited_indices(len(dev_dataset), args.max_dev_scenes)
    train_data = Subset(train_dataset, train_indices)
    dev_data = Subset(dev_dataset, dev_indices)
    train_loader = base._loader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        workers=args.workers,
        generator=generator,
    )
    dev_loader = base._loader(
        dev_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        workers=args.workers,
    )

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build_ascent_model("B0_signed_coupled", batch_size=args.batch_size).to(device)
    optimizer = Adam(model.parameters(), lr=float(protocol["training"]["learning_rate"]))
    scheduler = MultiStepLR(
        optimizer,
        milestones=protocol["training"]["milestones"],
        gamma=float(protocol["training"]["gamma"]),
    )
    checkpoint_path = run_dir / "last.pt"
    history: list[dict[str, Any]] = []
    start_epoch = 1
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if saved.get("protocol_sha256") != base.sha256(PROTOCOL):
            raise RuntimeError("resume protocol hash mismatch")
        if saved.get("airport") != args.airport or int(saved.get("seed", -1)) != args.seed:
            raise RuntimeError("resume identity mismatch")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        base._restore_rng(saved["rng_state"], generator)
        history = list(saved["history"])
        start_epoch = int(saved["epoch"]) + 1

    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        totals: dict[str, float] = {}
        batches = 0
        epoch_started = time.perf_counter()
        for data in train_loader:
            data = base.move(data, device)
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
                "airport": args.airport,
                "seed": args.seed,
                "epoch": epoch,
                "protocol_sha256": base.sha256(PROTOCOL),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "rng_state": base._capture_rng(generator),
                "history": history,
                "locked_test_used": False,
            },
            temporary,
        )
        temporary.replace(checkpoint_path)
        print(json.dumps(record), flush=True)

    tail = base._tail_threshold(train_dataset, train_indices)
    selected_dev_dates = [dev_dates[index] for index in dev_indices]
    validation = evaluate_decision(
        model,
        dev_loader,
        device,
        scene_dates=selected_dev_dates,
        tail_threshold=tail,
    )
    result = {
        "format_version": 1,
        "experiment_id": f"aWTA_Tartan_{args.airport}_seed{args.seed}_development",
        "formal": not smoke,
        "airport": args.airport,
        "regime": "target_only",
        "seed": args.seed,
        "train_scenes": len(train_data),
        "development_scenes": len(dev_data),
        "train_dates": sorted(set(train_dates[index] for index in train_indices)),
        "development_dates": sorted(set(selected_dev_dates)),
        "development_metrics": validation,
        "checkpoint": {
            "path": checkpoint_path.relative_to(ROOT).as_posix(),
            "sha256": base.sha256(checkpoint_path),
            "epoch": args.epochs,
        },
        "protocol": {
            "path": PROTOCOL.relative_to(ROOT).as_posix(),
            "sha256": base.sha256(PROTOCOL),
        },
        "implementation": {
            "runner": Path(__file__).relative_to(ROOT).as_posix(),
            "runner_sha256": base.sha256(Path(__file__)),
            "objective": "experiments/journal_extension/awta.py",
            "objective_sha256": base.sha256(Path(__file__).with_name("awta.py")),
        },
        "inputs": {
            "train_index_sha256": base.sha256(train_index),
            "development_index_sha256": base.sha256(dev_index),
            "data_manifest_sha256": protocol["inherits"]["data_manifest_sha256"],
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
        "integrity": {
            "train_and_development_only": True,
            "locked_test_used": False,
            "fixed_final_epoch": True,
            "matched_architecture": True,
            "only_training_assignment_changed": True,
        },
        "command": [sys.executable, "-m", "experiments.experiments.journal_extension.train_awta_tartan", *sys.argv[1:]],
    }
    atomic_json(output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airport", choices=("KAGC", "KBTP"), required=True)
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
