"""Train the unchanged C133 decision model on a C161 replication fold."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from experiments.metric_exact.evaluation import evaluate, target_tail_threshold
from experiments.joint_coupled.train import _capture_rng, _restore_rng, move
from experiments.decision_regret.model import VARIANT, ascent_config, build_model
from experiments.decision_regret.objective import decision_regret_objective

from .common import (
    atomic_json,
    authorize,
    fold_subsets,
    load_dataset,
    loader,
    set_seed,
)
from .protocol import load_protocol, sha256


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/ascent_recomparison"
RUN_ROOT = ROOT / "runs/ascent_recomparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--max-train-scenes", type=int)
    parser.add_argument("--max-validation-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_boundaries()
    authorize(protocol, args.fold, args.seed)
    settings = protocol.payload["training"]["decision"]
    smoke = (
        args.max_train_scenes is not None
        or args.max_validation_scenes is not None
        or args.epochs != int(settings["epochs"])
    )
    expected = (
        int(settings["epochs"]),
        int(settings["batch_size"]),
        int(settings["evaluation_batch_size"]),
    )
    if not smoke and (args.epochs, args.batch_size, args.eval_batch_size) != expected:
        raise RuntimeError("formal C161 decision settings differ from protocol")

    generator = set_seed(args.seed)
    dataset = load_dataset(protocol)
    train_data, validation_data, validation_dates = fold_subsets(
        protocol,
        dataset,
        args.fold,
        max_train_scenes=args.max_train_scenes,
        max_validation_scenes=args.max_validation_scenes,
    )
    train_loader = loader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        workers=args.num_workers,
        prefetch=args.prefetch_factor,
        generator=generator,
    )
    validation_loader = loader(
        validation_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        workers=args.num_workers,
        prefetch=args.prefetch_factor,
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(batch_size=args.batch_size).to(device)
    optimizer = Adam(model.parameters(), lr=float(settings["learning_rate"]))
    scheduler = MultiStepLR(
        optimizer,
        milestones=[int(value) for value in settings["scheduler_milestones"]],
        gamma=float(settings["scheduler_gamma"]),
    )

    suffix = "smoke" if smoke else "formal"
    name = f"{VARIANT}_fold{args.fold}_seed{args.seed}_{suffix}"
    run_dir = RUN_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "training_summary.json"
    checkpoint_path = run_dir / "last.pt"
    if summary_path.is_file() and not smoke:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("complete") is True:
            return summary

    protocol_hash = sha256(protocol.path)
    config = ascent_config(batch_size=args.batch_size)
    config.update(
        {
            "cycle": protocol.payload["cycle"],
            "fold": args.fold,
            "seed": args.seed,
            "epochs": args.epochs,
            "protocol_sha256": protocol_hash,
            "objective": "unchanged_C133_exact_dual_plus_SPO_plus",
            "development_used": False,
            "locked_test_used": False,
        }
    )
    atomic_json(run_dir / "config.json", config)
    history: list[dict[str, object]] = []
    start_epoch = 1
    if args.resume and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("protocol_sha256") != protocol_hash:
            raise RuntimeError("C161 decision resume protocol mismatch")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        _restore_rng(checkpoint["rng_state"], generator)
        history = checkpoint["history"]
        start_epoch = int(checkpoint["epoch"]) + 1

    metric_names = (
        "loss",
        "geometry",
        "decision_regret_surrogate",
        "top1_decision_regret",
        "batch_minade",
        "batch_minfde",
        "oracle_overlap",
        "top1_ade_overlap",
        "top1_fde_overlap",
    )
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        totals = {name: 0.0 for name in metric_names}
        batches = 0
        epoch_started = time.perf_counter()
        for data in train_loader:
            data = move(data, device)
            target = data["pred_traj"].transpose(1, 0)
            optimizer.zero_grad(set_to_none=True)
            predictions, logits, auxiliary = model(data)
            loss, diagnostics = decision_regret_objective(
                predictions, logits, auxiliary["decision_costs"], target
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite C161 decision loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float(settings["gradient_clip_norm"])
            )
            optimizer.step()
            batches += 1
            for metric_name in metric_names:
                totals[metric_name] += float(
                    diagnostics[metric_name].detach().cpu()
                )
        scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": {
                metric_name: value / max(batches, 1)
                for metric_name, value in totals.items()
            },
            "elapsed_seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        checkpoint = {
            "format_version": 1,
            "cycle": protocol.payload["cycle"],
            "variant": VARIANT,
            "fold": args.fold,
            "seed": args.seed,
            "epoch": epoch,
            "protocol_sha256": protocol_hash,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "rng_state": _capture_rng(generator),
            "history": history,
            "development_used": False,
            "locked_test_used": False,
        }
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(checkpoint_path)
        atomic_json(
            ARTIFACT_ROOT / "status" / f"{name}.json",
            {"phase": "training", "fold": args.fold, "seed": args.seed, **record},
        )
        print(json.dumps(record), flush=True)

    metrics = evaluate(
        model,
        validation_loader,
        device,
        scene_dates=validation_dates,
        tail_threshold=target_tail_threshold(dataset),
    )
    summary = {
        "format_version": 1,
        "cycle": protocol.payload["cycle"],
        "variant": VARIANT,
        "fold": args.fold,
        "seed": args.seed,
        "formal": not smoke,
        "complete": True,
        "fixed_final_epoch": history[-1]["epoch"],
        "train_scenes": len(train_data),
        "validation_scenes": len(validation_data),
        "validation_dates": sorted(set(validation_dates)),
        "checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": sha256(checkpoint_path),
        "protocol_sha256": protocol_hash,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
        "validation_metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "development_used": False,
        "locked_test_used": False,
    }
    atomic_json(summary_path, summary)
    atomic_json(
        ARTIFACT_ROOT / "status" / f"{name}.json",
        {"phase": "complete", "metrics": metrics["overall"]},
    )
    print(json.dumps({"run": name, "metrics": metrics["overall"]}), flush=True)
    return summary


if __name__ == "__main__":
    run(parse_args())
