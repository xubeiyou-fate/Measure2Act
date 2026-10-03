"""Train one frozen C127 arm on a train-date fold or full train split."""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Subset

from model.utils import TrajectoryDataset, seq_collate, seed_worker

from .evaluation import evaluate, target_tail_threshold
from .folds import build_fold_artifact, indices_for_fold
from .model import VARIANTS, ascent_config, build_model, is_score_isolated
from .objective import objective_for_variant
from .protocol import load_protocol, sha256


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/metric_exact"
RUN_ROOT = ROOT / "runs/metric_exact"


def set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.Generator().manual_seed(seed)


def move(data: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in data.items()
    }


def limited_indices(indices: list[int], maximum: int | None) -> list[int]:
    if maximum is None or maximum >= len(indices):
        return indices
    positions = (
        torch.linspace(0, len(indices) - 1, steps=maximum)
        .round()
        .long()
        .unique()
        .tolist()
    )
    return [indices[position] for position in positions]


def write_status(run_name: str, payload: dict[str, object]) -> None:
    status_dir = ARTIFACT_ROOT / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    path = status_dir / f"{run_name}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _capture_rng(generator: torch.Generator) -> dict[str, object]:
    return {
        "loader_generator": generator.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


def _restore_rng(state: dict[str, object], generator: torch.Generator) -> None:
    # A CUDA map_location also moves serialized CPU RNG tensors. The RNG APIs
    # require CPU ByteTensors, so normalize them before restoring a checkpoint.
    generator.set_state(state["loader_generator"].detach().cpu())
    torch.set_rng_state(state["torch"].detach().cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(
            [value.detach().cpu() for value in state["cuda"]]
        )
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--phase", choices=("P1", "P2", "P3"), required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--max-train-scenes", type=int)
    parser.add_argument("--max-validation-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _authorize(args: argparse.Namespace, protocol) -> None:
    if args.phase in {"P1", "P2"} and args.fold is None:
        raise ValueError("P1/P2 require --fold")
    if args.phase == "P3" and args.fold is not None:
        raise ValueError("P3 uses all train dates and does not accept --fold")
    phase = protocol.payload["phases"][args.phase]
    if args.phase == "P1":
        if args.fold not in phase["folds"] or args.seed not in phase["seeds"]:
            raise RuntimeError("run is outside the frozen P1 fold/seed set")
        if args.variant not in phase["variants"]:
            raise RuntimeError("variant is outside the frozen P1 arm set")
    elif args.phase == "P2":
        if args.fold not in phase["folds"] or args.seed not in phase["seeds"]:
            raise RuntimeError("run is outside the frozen P2 fold/seed set")
        allowed = {"B0_signed_coupled", "B2_decoupled_original"}
        summary_path = ARTIFACT_ROOT / "p1_summary.json"
        if summary_path.is_file():
            selected = json.loads(summary_path.read_text(encoding="utf-8")).get(
                "P2_selected_exact_candidate"
            )
            if selected:
                allowed.add(selected)
        if args.variant not in allowed:
            raise RuntimeError("P2 exact candidate has not been authorized by P1")
    else:
        if args.seed not in phase["seeds"]:
            raise RuntimeError("seed is outside the frozen P3 set")
        summary_path = ARTIFACT_ROOT / "p2_summary.json"
        if not summary_path.is_file():
            raise RuntimeError("C127 P2 summary is required before P3")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        selected = summary.get("P3_selected_exact_candidate")
        if summary.get("decision") != "P3_AUTHORIZED" or not selected:
            raise RuntimeError("C127 P2 did not authorize P3")
        if args.variant not in {
            "B0_signed_coupled",
            "B2_decoupled_original",
            selected,
        }:
            raise RuntimeError("variant is outside the frozen P3 arm set")


def run(args: argparse.Namespace) -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_boundaries()
    _authorize(args, protocol)
    if args.epochs != int(protocol.payload["training"]["epochs"]):
        smoke_requested = args.max_train_scenes is not None or args.max_validation_scenes is not None
        if not smoke_requested:
            raise RuntimeError("formal C127 epochs must match the frozen protocol")
    generator = set_seed(args.seed)
    train_dataset = TrajectoryDataset(
        protocol.split_path("train").as_posix(),
        obs_len=16,
        obs_steps=1,
        pred_len=120,
        pred_step=5,
        delim=" ",
    )
    train_date_path = ROOT / str(protocol.payload["dataset"]["train_scene_dates"])
    train_scene_dates = json.loads(train_date_path.read_text(encoding="utf-8"))["dates"]
    if len(train_scene_dates) != len(train_dataset):
        raise RuntimeError("C127 train scene-date index mismatch")
    fold_artifact = build_fold_artifact(protocol)
    if args.phase in {"P1", "P2"}:
        train_indices, validation_indices, validation_dates = indices_for_fold(
            train_scene_dates, fold_artifact, args.fold
        )
        validation_dataset = train_dataset
    else:
        train_indices = list(range(len(train_dataset)))
        validation_dataset = TrajectoryDataset(
            protocol.split_path("dev").as_posix(),
            obs_len=16,
            obs_steps=1,
            pred_len=120,
            pred_step=5,
            delim=" ",
        )
        validation_indices = list(range(len(validation_dataset)))
        development_date_path = ROOT / str(
            protocol.payload["dataset"]["development_scene_dates"]
        )
        validation_dates = json.loads(
            development_date_path.read_text(encoding="utf-8")
        )["dates"]
        if len(validation_dates) != len(validation_dataset):
            raise RuntimeError("C127 development scene-date index mismatch")

    train_indices = limited_indices(train_indices, args.max_train_scenes)
    full_validation_indices = validation_indices
    validation_indices = limited_indices(
        full_validation_indices, args.max_validation_scenes
    )
    if len(validation_dates) != len(full_validation_indices):
        raise RuntimeError("C127 validation dates are not aligned with validation indices")
    if len(validation_indices) != len(full_validation_indices):
        date_by_index = dict(zip(full_validation_indices, validation_dates, strict=True))
        validation_dates = [date_by_index[index] for index in validation_indices]
    smoke = (
        args.max_train_scenes is not None
        or args.max_validation_scenes is not None
        or args.epochs != int(protocol.payload["training"]["epochs"])
    )
    train_data = Subset(train_dataset, train_indices)
    validation_data = Subset(validation_dataset, validation_indices)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=seq_collate,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=seq_collate,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
    )
    tail_threshold = target_tail_threshold(train_dataset)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but PyTorch cannot use CUDA")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(args.variant, batch_size=args.batch_size).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[10, 15], gamma=0.5)
    fold_label = "all_train" if args.phase == "P3" else f"fold{args.fold}"
    run_name = (
        f"{args.phase}_{args.variant}_{fold_label}_seed{args.seed}_"
        f"{'smoke' if smoke else 'formal'}"
    )
    run_dir = RUN_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "training_summary.json"
    if summary_path.is_file() and not smoke:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("complete") is True:
            return summary
    config = ascent_config(args.variant, batch_size=args.batch_size)
    config.update(
        {
            "phase": args.phase,
            "fold": args.fold,
            "seed": args.seed,
            "epochs": args.epochs,
            "score_isolated": is_score_isolated(args.variant),
            "protocol_sha256": sha256(protocol.path),
        }
    )
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    checkpoint_path = run_dir / "last.pt"
    history: list[dict[str, object]] = []
    start_epoch = 1
    if args.resume and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint["protocol_sha256"] != sha256(protocol.path):
            raise RuntimeError("C127 resume protocol hash mismatch")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        history = checkpoint["history"]
        _restore_rng(checkpoint["rng_state"], generator)
        start_epoch = int(checkpoint["epoch"]) + 1

    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        totals = {
            "loss": 0.0,
            "regression": 0.0,
            "classification": 0.0,
            "batch_minade": 0.0,
            "batch_minfde": 0.0,
            "oracle_overlap": 0.0,
        }
        ade_winner_counts = np.zeros(5, dtype=np.int64)
        fde_winner_counts = np.zeros(5, dtype=np.int64)
        batches = 0
        epoch_started = time.perf_counter()
        for data in train_loader:
            data = move(data, device)
            target = data["pred_traj"].transpose(1, 0)
            optimizer.zero_grad(set_to_none=True)
            predictions, logits, _ = model(data)
            loss, diagnostics = objective_for_variant(
                args.variant, predictions, logits, target
            )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite C127 loss: {args.variant} seed {args.seed}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batches += 1
            for name in totals:
                totals[name] += float(diagnostics[name].cpu())
            ade_winner_counts += np.bincount(
                diagnostics["ade_winner"].cpu().numpy(), minlength=5
            )
            fde_winner_counts += np.bincount(
                diagnostics["fde_winner"].cpu().numpy(), minlength=5
            )
        scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": {
                name: value / max(batches, 1) for name, value in totals.items()
            },
            "ade_winner_counts": ade_winner_counts.tolist(),
            "fde_winner_counts": fde_winner_counts.tolist(),
            "elapsed_seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        checkpoint = {
            "format_version": 1,
            "cycle": protocol.payload["cycle"],
            "variant": args.variant,
            "phase": args.phase,
            "fold": args.fold,
            "seed": args.seed,
            "epoch": epoch,
            "protocol_sha256": sha256(protocol.path),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "rng_state": _capture_rng(generator),
            "history": history,
            "trajectory_residual": False,
            "learned_gate": False,
            "token_codebook": False,
            "future_autoregression": False,
            "post_generation_selector": False,
            "locked_test_used": False,
        }
        temporary_checkpoint = checkpoint_path.with_suffix(".tmp")
        torch.save(checkpoint, temporary_checkpoint)
        temporary_checkpoint.replace(checkpoint_path)
        write_status(
            run_name,
            {
                "phase": "training",
                "variant": args.variant,
                "fold": args.fold,
                "seed": args.seed,
                **record,
            },
        )
        print(json.dumps(record), flush=True)

    write_status(
        run_name,
        {
            "phase": "evaluating_validation",
            "variant": args.variant,
            "fold": args.fold,
            "seed": args.seed,
            "epoch": history[-1]["epoch"],
        },
    )
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
        "variant": args.variant,
        "phase": args.phase,
        "fold": args.fold,
        "fold_label": fold_label,
        "seed": args.seed,
        "formal": not smoke,
        "fixed_final_epoch": history[-1]["epoch"],
        "train_scenes": len(train_data),
        "validation_scenes": len(validation_data),
        "validation_dates": sorted(set(validation_dates)),
        "checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": sha256(checkpoint_path),
        "protocol_sha256": sha256(protocol.path),
        "manifest_sha256": sha256(protocol.manifest_path),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "score_isolated": is_score_isolated(args.variant),
        "history": history,
        "validation_metrics": metrics,
        "tail_threshold": tail_threshold,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": peak_memory,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "locked_test_used": False,
        "complete": True,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_status(
        run_name,
        {
            "phase": "complete",
            "variant": args.variant,
            "fold": args.fold,
            "seed": args.seed,
            "metrics": metrics["overall"],
        },
    )
    print(
        json.dumps(
            {
                "run": run_name,
                "minade": metrics["overall"]["minade"],
                "minfde": metrics["overall"]["minfde"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "peak_gpu_memory_bytes": peak_memory,
            }
        ),
        flush=True,
    )
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
