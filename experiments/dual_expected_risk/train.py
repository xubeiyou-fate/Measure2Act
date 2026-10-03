"""Train C130 on one frozen train-date fold without dev or locked-test access."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Subset

from experiments.metric_exact.evaluation import evaluate, target_tail_threshold
from experiments.metric_exact.folds import indices_for_fold
from experiments.joint_coupled.train import (
    _capture_rng,
    _restore_rng,
    limited_indices,
    move,
    set_seed,
)
from model.utils import TrajectoryDataset, seed_worker, seq_collate

from .model import VARIANT, ascent_config, build_model
from .objective import dual_expected_risk_objective
from .protocol import load_protocol, sha256


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/dual_expected_risk"
RUN_ROOT = ROOT / "runs/dual_expected_risk"


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_name(fold: int, seed: int, smoke: bool) -> str:
    suffix = "smoke" if smoke else "formal"
    return f"{VARIANT}_fold{fold}_seed{seed}_{suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--max-train-scenes", type=int)
    parser.add_argument("--max-validation-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _authorize(args: argparse.Namespace, protocol, smoke: bool) -> None:
    screening = protocol.payload["screening"]
    if args.fold not in screening["folds"] or args.seed != int(screening["seed"]):
        raise RuntimeError("run is outside the frozen C130 fold/seed set")
    training = protocol.payload["training"]
    expected = (
        int(training["epochs"]),
        int(training["batch_size"]),
        int(training["evaluation_batch_size"]),
    )
    if not smoke and (args.epochs, args.batch_size, args.eval_batch_size) != expected:
        raise RuntimeError("formal C130 settings must match the frozen protocol")
    if smoke:
        return
    diagnostic_path = ARTIFACT_ROOT / "p0_decision.json"
    if not diagnostic_path.is_file():
        raise RuntimeError("formal C130 training requires the zero-training P0 decision")
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if (
        diagnostic.get("decision") != "FORMAL_FOLD0_AUTHORIZED"
        or diagnostic.get("protocol_sha256") != sha256(protocol.path)
        or diagnostic.get("locked_test_used") is not False
        or diagnostic.get("development_used") is not False
    ):
        raise RuntimeError("C130 P0 did not authorize formal training")
    if args.fold in {1, 2}:
        gate_path = ARTIFACT_ROOT / "p1_decision.json"
        if not gate_path.is_file():
            raise RuntimeError("C130 fold 1/2 require the fold-0 gate artifact")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if (
            gate.get("decision") != "REPLICATION_AUTHORIZED"
            or gate.get("protocol_sha256") != sha256(protocol.path)
            or gate.get("locked_test_used") is not False
            or gate.get("development_used") is not False
        ):
            raise RuntimeError("C130 fold-0 gate did not authorize replication")


def _loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    prefetch: int,
    generator=None,
) -> DataLoader:
    options = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "collate_fn": seq_collate,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
    }
    if generator is not None:
        options["generator"] = generator
    if workers > 0:
        options.update({"persistent_workers": True, "prefetch_factor": prefetch})
    return DataLoader(**options)


def run(args: argparse.Namespace) -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_boundaries()
    smoke = (
        args.max_train_scenes is not None
        or args.max_validation_scenes is not None
        or args.epochs != int(protocol.payload["training"]["epochs"])
    )
    _authorize(args, protocol, smoke)
    generator = set_seed(args.seed)
    dataset = TrajectoryDataset(
        protocol.split_path("train").as_posix(),
        obs_len=16,
        obs_steps=1,
        pred_len=120,
        pred_step=5,
        delim=" ",
    )
    expected = protocol.payload["dataset"]
    if (
        len(dataset) != int(expected["expected_train_scenes"])
        or int(dataset.obs_traj.shape[0]) != int(expected["expected_train_actors"])
    ):
        raise RuntimeError("C130 train cohort identity mismatch")

    dates = json.loads(
        (ROOT / str(expected["train_scene_dates"])).read_text(encoding="utf-8")
    )["dates"]
    fold_path = ROOT / str(expected["date_folds"])
    folds = json.loads(fold_path.read_text(encoding="utf-8"))
    train_indices, validation_indices, validation_dates = indices_for_fold(
        dates, folds, args.fold
    )
    train_indices = limited_indices(train_indices, args.max_train_scenes)
    complete_validation_indices = validation_indices
    validation_indices = limited_indices(validation_indices, args.max_validation_scenes)
    if len(validation_indices) != len(complete_validation_indices):
        by_index = dict(zip(complete_validation_indices, validation_dates, strict=True))
        validation_dates = [by_index[index] for index in validation_indices]

    train_data = Subset(dataset, train_indices)
    validation_data = Subset(dataset, validation_indices)
    train_loader = _loader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        workers=args.num_workers,
        prefetch=args.prefetch_factor,
        generator=generator,
    )
    validation_loader = _loader(
        validation_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        workers=args.num_workers,
        prefetch=args.prefetch_factor,
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but PyTorch cannot use CUDA")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(batch_size=args.batch_size).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[10, 15], gamma=0.5)

    name = run_name(args.fold, args.seed, smoke)
    run_dir = RUN_ROOT / name
    summary_path = run_dir / "training_summary.json"
    if summary_path.is_file() and not smoke:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("complete") is True:
            return summary
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "last.pt"
    protocol_hash = sha256(protocol.path)
    config = ascent_config(batch_size=args.batch_size)
    config.update(
        {
            "fold": args.fold,
            "seed": args.seed,
            "epochs": args.epochs,
            "protocol_sha256": protocol_hash,
            "objective": "exact_dual_geometry_plus_centered_dual_risk_mse",
            "automatic_mixed_precision": False,
            "tf32": False,
            "deterministic_algorithms": True,
            "locked_test_used": False,
            "development_used": False,
        }
    )
    atomic_json(run_dir / "config.json", config)

    history: list[dict[str, object]] = []
    start_epoch = 1
    if args.resume and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("protocol_sha256") != protocol_hash:
            raise RuntimeError("C130 resume protocol hash mismatch")
        if checkpoint.get("locked_test_used") is not False:
            raise RuntimeError("C130 resume checkpoint violates the locked boundary")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        history = checkpoint["history"]
        _restore_rng(checkpoint["rng_state"], generator)
        start_epoch = int(checkpoint["epoch"]) + 1

    tail_threshold = target_tail_threshold(dataset)
    started = time.perf_counter()
    names = (
        "loss",
        "geometry",
        "risk_regression",
        "batch_minade",
        "batch_minfde",
        "oracle_overlap",
        "top1_ade_overlap",
        "top1_fde_overlap",
    )
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        totals = {key: 0.0 for key in names}
        batches = 0
        epoch_started = time.perf_counter()
        for data in train_loader:
            data = move(data, device)
            target = data["pred_traj"].transpose(1, 0)
            optimizer.zero_grad(set_to_none=True)
            predictions, logits, auxiliary = model(data)
            loss, diagnostics = dual_expected_risk_objective(
                predictions, logits, auxiliary["risk_predictions"], target
            )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite C130 loss at fold {args.fold} epoch {epoch}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batches += 1
            for key in names:
                totals[key] += float(diagnostics[key].cpu())
        scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": {key: value / max(batches, 1) for key, value in totals.items()},
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
            "locked_test_used": False,
            "development_used": False,
        }
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(checkpoint_path)
        atomic_json(
            ARTIFACT_ROOT / "status" / f"{name}.json",
            {
                "phase": "training",
                "variant": VARIANT,
                "fold": args.fold,
                "seed": args.seed,
                **record,
            },
        )
        print(json.dumps(record), flush=True)

    metrics = evaluate(
        model,
        validation_loader,
        device,
        scene_dates=validation_dates,
        tail_threshold=tail_threshold,
    )
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    summary = {
        "format_version": 1,
        "cycle": protocol.payload["cycle"],
        "variant": VARIANT,
        "fold": args.fold,
        "seed": args.seed,
        "formal": not smoke,
        "fixed_final_epoch": history[-1]["epoch"],
        "train_scenes": len(train_data),
        "validation_scenes": len(validation_data),
        "validation_dates": sorted(set(validation_dates)),
        "checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": sha256(checkpoint_path),
        "protocol_sha256": protocol_hash,
        "manifest_sha256": sha256(protocol.manifest_path),
        "fold_artifact_sha256": sha256(fold_path),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
        "validation_metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": peak_memory,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
            "persistent_workers": args.num_workers > 0,
        },
        "locked_test_used": False,
        "development_used": False,
        "complete": True,
    }
    atomic_json(summary_path, summary)
    atomic_json(
        ARTIFACT_ROOT / "status" / f"{name}.json",
        {
            "phase": "complete",
            "variant": VARIANT,
            "fold": args.fold,
            "seed": args.seed,
            "metrics": metrics["overall"],
        },
    )
    print(
        json.dumps(
            {
                "run": name,
                **{
                    key: metrics["overall"][key]
                    for key in ("top1_ade", "top1_fde", "minade", "minfde")
                },
            }
        ),
        flush=True,
    )
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
