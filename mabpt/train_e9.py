"""Train one native-cardinality E9 MABPT component on one date fold."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from experiments.joint_coupled.train import _capture_rng, _restore_rng, move
from experiments.ascent_recomparison.common import fold_subsets, load_dataset, loader, set_seed
from experiments.ascent_recomparison.protocol import load_protocol as load_data_protocol
from model.ascent import Ascent

from .train_controls import original_ascent_objective
from .scalable import (
    ScalableDecisionAscent,
    ScalableEnergyAscent,
    scalable_ascent_config,
    scalable_decision_objective,
    scalable_energy_objective,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = Path(__file__).with_name("e9_training_protocol.json")
RUN_ROOT = ROOT / "runs/mabpt/e9"
STATUS_ROOT = ROOT / "artifacts/mabpt/status"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_model(stage: str, modes: int, batch_size: int):
    if stage == "source":
        return Ascent(scalable_ascent_config(modes, role="source", batch_size=batch_size))
    if stage == "decision":
        return ScalableDecisionAscent(modes, batch_size=batch_size)
    if stage == "energy":
        return ScalableEnergyAscent(modes, batch_size=batch_size)
    raise ValueError(f"unknown E9 stage: {stage}")


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.modes not in (3, 7):
        raise ValueError("E9 training is frozen to K=3 or K=7")
    if args.fold not in (1, 2):
        raise ValueError("E9 uses folds 1 and 2")
    if args.stage not in ("source", "decision", "energy"):
        raise ValueError("unknown E9 stage")
    smoke = args.max_train_scenes is not None or args.epochs != 20
    if not smoke and args.seed != 42:
        raise RuntimeError("formal E9 training requires seed 42")
    batch_size = 1024 if args.stage == "energy" else 256
    expected_epochs = 20
    if not smoke and (args.epochs, args.batch_size) != (expected_epochs, batch_size):
        raise RuntimeError("formal E9 settings differ from the frozen protocol")
    if args.batch_size != batch_size and not smoke:
        raise RuntimeError("formal E9 batch size differs from stage protocol")

    protocol_hash = sha256(PROTOCOL_PATH)
    generator = set_seed(args.seed)
    data_protocol = load_data_protocol()
    data_protocol.assert_boundaries()
    dataset = load_dataset(data_protocol)
    train_data, _, _ = fold_subsets(data_protocol, dataset, args.fold, max_train_scenes=args.max_train_scenes)
    train_loader = loader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        workers=args.num_workers,
        prefetch=args.prefetch,
        generator=generator,
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(args.stage, args.modes, args.batch_size).to(device)
    source_checkpoint = None
    decision_checkpoint = None
    if args.stage == "energy":
        if args.decision_checkpoint is None:
            raise ValueError("energy stage requires --decision-checkpoint")
        decision_checkpoint = args.decision_checkpoint.resolve()
        model.load_backbone(decision_checkpoint, device)
        trainable = list(model.energy_cost_operator.parameters())
    else:
        trainable = list(model.parameters())
    optimizer = Adam(trainable, lr=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[10, 15], gamma=0.5)

    suffix = "smoke" if smoke else "formal"
    name = f"K{args.modes}_{args.stage}_fold{args.fold}_seed{args.seed}_{suffix}"
    run_dir = (args.run_dir or (RUN_ROOT / name)).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path = run_dir / "last.pt"
    summary_path = run_dir / "training_summary.json"
    if summary_path.is_file() and not smoke:
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("complete") is True:
            return existing
    config = {
        "stage": args.stage,
        "modes": args.modes,
        "fold": args.fold,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "protocol_sha256": protocol_hash,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "decision_checkpoint": str(decision_checkpoint.relative_to(ROOT)) if decision_checkpoint else None,
        "trajectory_residual": False,
        "learned_gate": False,
        "validation_target_used": False,
    }
    config_path = run_dir / "config.json"
    if not config_path.exists():
        atomic_json(config_path, config)
    history: list[dict[str, object]] = []
    start_epoch = 1
    updates = 0
    if args.resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("protocol_sha256") != protocol_hash:
            raise RuntimeError("E9 resume protocol mismatch")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        _restore_rng(checkpoint["rng_state"], generator)
        history = checkpoint["history"]
        updates = int(checkpoint["updates"])
        start_epoch = int(checkpoint["epoch"]) + 1

    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "primary": 0.0, "secondary": 0.0, "minade": 0.0, "minfde": 0.0}
        batches = 0
        epoch_started = time.perf_counter()
        for data in train_loader:
            data = move(data, device)
            target = data["pred_traj"].transpose(1, 0)
            optimizer.zero_grad(set_to_none=True)
            if args.stage == "energy":
                predictions, probabilities, _decision, auxiliary = model(data)
                loss, diagnostics = scalable_energy_objective(
                    predictions, probabilities, auxiliary["predicted_normalized_ade_risk"], target
                )
                metric_map = {
                    "loss": diagnostics["loss"],
                    "primary": diagnostics["risk_regression"],
                    "secondary": diagnostics["normalized_energy"],
                    "minade": torch.zeros((), device=device),
                    "minfde": torch.zeros((), device=device),
                }
            elif args.stage == "decision":
                predictions, logits, auxiliary = model(data)
                loss, diagnostics = scalable_decision_objective(
                    predictions, logits, auxiliary["decision_costs"], target
                )
                metric_map = {
                    "loss": diagnostics["loss"],
                    "primary": diagnostics["geometry"],
                    "secondary": diagnostics["decision_regret_surrogate"],
                    "minade": diagnostics["minade"],
                    "minfde": diagnostics["minfde"],
                }
            else:
                predictions, logits, _auxiliary = model(data)
                loss, diagnostics = original_ascent_objective(predictions, logits, target)
                metric_map = {
                    "loss": diagnostics["loss"],
                    "primary": diagnostics["regression"],
                    "secondary": diagnostics["classification"],
                    "minade": diagnostics["minade"],
                    "minfde": diagnostics["minfde"],
                }
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite E9 {args.stage} loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            updates += 1
            batches += 1
            for name_metric, value in metric_map.items():
                totals[name_metric] += float(value.detach().cpu())
        scheduler.step()
        record = {
            "epoch": epoch,
            "batches": batches,
            "updates": updates,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": {key: value / max(batches, 1) for key, value in totals.items()},
            "elapsed_seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        checkpoint = {
            "format_version": 1,
            "stage": args.stage,
            "modes": args.modes,
            "fold": args.fold,
            "seed": args.seed,
            "epoch": epoch,
            "protocol_sha256": protocol_hash,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "rng_state": _capture_rng(generator),
            "history": history,
            "updates": updates,
            "trajectory_residual": False,
            "learned_gate": False,
        }
        temporary = last_path.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(last_path)
        print(json.dumps(record), flush=True)

    final_path = run_dir / f"epoch{args.epochs}.pt"
    if not final_path.exists():
        torch.save(checkpoint, final_path)
    summary = {
        "format_version": 1,
        "experiment_id": "E9",
        "stage": args.stage,
        "modes": args.modes,
        "fold": args.fold,
        "seed": args.seed,
        "formal": not smoke,
        "complete": True,
        "epochs": args.epochs,
        "train_scenes": len(train_data),
        "batches_per_epoch": len(train_loader),
        "updates": updates,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "protocol_sha256": protocol_hash,
        "checkpoint": str(final_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256(final_path),
        "decision_checkpoint": str(decision_checkpoint.relative_to(ROOT)) if decision_checkpoint else None,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "device": str(device)},
        "trajectory_residual": False,
        "learned_gate": False,
        "validation_target_used": False,
    }
    atomic_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("source", "decision", "energy"), required=True)
    parser.add_argument("--modes", type=int, choices=(3, 7), required=True)
    parser.add_argument("--fold", type=int, choices=(1, 2), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-train-scenes", type=int)
    parser.add_argument("--decision-checkpoint", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.batch_size is None:
        args.batch_size = 1024 if args.stage == "energy" else 256
    result = run(args)
    print(json.dumps({"stage": args.stage, "modes": args.modes, "fold": args.fold, "updates": result["updates"]}, indent=2))


if __name__ == "__main__":
    main()
