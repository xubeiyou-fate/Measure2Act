"""Train the preregistered E6-E8 ASCENT controls on one date fold."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from experiments.joint_coupled.train import _capture_rng, _restore_rng, move
from experiments.ascent_recomparison.common import (
    fold_subsets,
    load_dataset,
    loader,
    set_seed,
)
from experiments.tpmo_ascent.protocol import load_protocol as load_data_protocol

from .controls import CONTROL_VARIANTS, build_control_model


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = Path(__file__).with_name("e6_e8_protocol.json")
RUN_ROOT = ROOT / "runs/mabpt"
STATUS_ROOT = ROOT / "artifacts/mabpt/status"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def original_ascent_objective(
    predictions: torch.Tensor, logits: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    displacement = torch.linalg.vector_norm(
        predictions - target[:, None], dim=-1
    )
    ade = displacement.mean(dim=-1)
    winner = ade.detach().argmin(dim=1)
    rows = torch.arange(target.shape[0], device=target.device)
    regression = F.smooth_l1_loss(predictions[rows, winner], target)
    classification = F.cross_entropy(logits, winner)
    loss = regression + classification
    return loss, {
        "loss": loss.detach(),
        "regression": regression.detach(),
        "classification": classification.detach(),
        "minade": ade.min(dim=1).values.mean().detach(),
        "minfde": displacement[..., -1].min(dim=1).values.mean().detach(),
    }


def _registered_epochs(variant: str) -> tuple[int, tuple[int, ...]]:
    if variant == "b0_extended":
        return 25, (20, 25)
    return 20, (20,)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.fold not in (1, 2):
        raise ValueError("E6-E8 controls are frozen to folds 1 and 2")
    if args.variant not in CONTROL_VARIANTS:
        raise ValueError(f"unknown control {args.variant}")
    registered_epochs, snapshot_epochs = _registered_epochs(args.variant)
    smoke = args.max_train_scenes is not None or args.epochs != registered_epochs
    if not smoke and args.seed != 7:
        raise RuntimeError("formal E6-E8 controls require registered seed 7")
    if not smoke and args.batch_size != 256:
        raise RuntimeError("formal E6-E8 controls require batch size 256")

    data_protocol = load_data_protocol()
    data_protocol.assert_boundaries()
    generator = set_seed(args.seed)
    dataset = load_dataset(data_protocol)
    train_data, _, _ = fold_subsets(
        data_protocol,
        dataset,
        args.fold,
        max_train_scenes=args.max_train_scenes,
    )
    train_loader = loader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        workers=args.workers,
        prefetch=args.prefetch,
        generator=generator,
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    model = build_control_model(args.variant, batch_size=args.batch_size).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[10, 15], gamma=0.5)

    suffix = "smoke" if smoke else "formal"
    name = f"{args.variant}_fold{args.fold}_seed{args.seed}_{suffix}"
    run_dir = RUN_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path = run_dir / "last.pt"
    summary_path = run_dir / "training_summary.json"
    if summary_path.is_file() and not smoke:
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("complete") is True:
            return existing

    config = {
        "model": args.variant,
        "fold": args.fold,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trajectory_or_control_residual": False,
        "learned_gate": False,
        "target_in_model_forward": False,
    }
    config_path = run_dir / "config.json"
    if not config_path.exists():
        _atomic_json(config_path, config)

    history: list[dict[str, object]] = []
    start_epoch = 1
    if args.resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint["protocol_sha256"] != sha256(PROTOCOL_PATH):
            raise RuntimeError("E6-E8 resume protocol hash mismatch")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        _restore_rng(checkpoint["rng_state"], generator)
        history = checkpoint["history"]
        start_epoch = int(checkpoint["epoch"]) + 1

    started = time.perf_counter()
    total_updates = 0 if not history else int(history[-1]["cumulative_updates"])
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        totals = {name: 0.0 for name in ("loss", "regression", "classification", "minade", "minfde")}
        batches = 0
        epoch_started = time.perf_counter()
        for data in train_loader:
            data = move(data, device)
            target = data["pred_traj"].transpose(1, 0)
            optimizer.zero_grad(set_to_none=True)
            predictions, logits, _ = model(data)
            loss, diagnostics = original_ascent_objective(
                predictions, logits, target
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite {args.variant} loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batches += 1
            for metric in totals:
                totals[metric] += float(diagnostics[metric].cpu())
        scheduler.step()
        total_updates += batches
        record = {
            "epoch": epoch,
            "batches": batches,
            "cumulative_updates": total_updates,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": {key: value / max(batches, 1) for key, value in totals.items()},
            "elapsed_seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        checkpoint = {
            "format_version": 1,
            "variant": args.variant,
            "fold": args.fold,
            "seed": args.seed,
            "epoch": epoch,
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "rng_state": _capture_rng(generator),
            "history": history,
            "training_updates": total_updates,
            "trajectory_or_control_residual": False,
            "learned_gate": False,
        }
        temporary = last_path.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(last_path)
        if epoch in snapshot_epochs or (smoke and epoch == args.epochs):
            snapshot = run_dir / f"epoch{epoch}.pt"
            if not snapshot.exists():
                torch.save(checkpoint, snapshot)
        _atomic_json(STATUS_ROOT / f"{name}.json", {"phase": "training", **record})
        print(json.dumps(record), flush=True)

    snapshots = {
        str(epoch): {
            "path": (run_dir / f"epoch{epoch}.pt").relative_to(ROOT).as_posix(),
            "sha256": sha256(run_dir / f"epoch{epoch}.pt"),
        }
        for epoch in snapshot_epochs
        if (run_dir / f"epoch{epoch}.pt").is_file()
    }
    summary = {
        "format_version": 1,
        "model": args.variant,
        "fold": args.fold,
        "seed": args.seed,
        "formal": not smoke,
        "complete": True,
        "train_scenes": len(train_data),
        "epochs": args.epochs,
        "batches_per_epoch": math.ceil(len(train_data) / args.batch_size),
        "training_updates": total_updates,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "snapshots": snapshots,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
        },
        "locked_test_used": False,
    }
    _atomic_json(summary_path, summary)
    _atomic_json(STATUS_ROOT / f"{name}.json", {"phase": "complete", "training_updates": total_updates})
    print(json.dumps({"run": name, "updates": total_updates, "snapshots": snapshots}), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=CONTROL_VARIANTS, required=True)
    parser.add_argument("--fold", type=int, choices=(1, 2), required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-train-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.epochs is None:
        args.epochs = _registered_epochs(args.variant)[0]
    run(args)


if __name__ == "__main__":
    main()
