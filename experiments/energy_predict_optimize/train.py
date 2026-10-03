"""Train the C134 E1 expected-distance operator on one frozen date fold."""

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

from experiments.metric_exact.evaluation import target_tail_threshold
from experiments.metric_exact.folds import indices_for_fold
from experiments.joint_coupled.train import (
    _capture_rng,
    _restore_rng,
    limited_indices,
    move,
    set_seed,
)
from model.utils import TrajectoryDataset, seed_worker, seq_collate

from .evaluate import evaluate
from .model import VARIANT, build_model
from .objective import energy_predict_optimize_objective
from .protocol import load_protocol, sha256


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/energy_predict_optimize"
RUN_ROOT = ROOT / "runs/energy_predict_optimize"
DESIGN_PATH = ARTIFACT_ROOT / "p1_design.json"


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--max-train-scenes", type=int)
    parser.add_argument("--max-validation-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _loader(dataset, *, batch_size, shuffle, workers, prefetch, generator=None):
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
    protocol_hash = sha256(protocol.path)
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    smoke = (
        args.max_train_scenes is not None
        or args.max_validation_scenes is not None
        or args.epochs != int(design["training"]["epochs"])
    )
    if args.fold != 0 or args.seed != 42:
        raise RuntimeError("C134 E1 currently authorizes only fold0 seed42")
    expected_settings = (
        int(design["training"]["epochs"]),
        int(design["training"]["batch_size"]),
        int(design["training"]["evaluation_batch_size"]),
        int(design["training"]["num_workers"]),
        int(design["training"]["prefetch_factor"]),
    )
    actual_settings = (
        args.epochs,
        args.batch_size,
        args.eval_batch_size,
        args.num_workers,
        args.prefetch_factor,
    )
    if not smoke and actual_settings != expected_settings:
        raise RuntimeError("formal C134 E1 settings differ from the frozen design")
    p0_b = json.loads((ARTIFACT_ROOT / "p0_b_decision.json").read_text(encoding="utf-8"))
    preflight = json.loads((ARTIFACT_ROOT / "preflight_p1.json").read_text(encoding="utf-8"))
    if (
        p0_b.get("decision") != "P1_FOLD0_AUTHORIZED"
        or p0_b.get("protocol_sha256") != protocol_hash
        or preflight.get("passed") is not True
        or preflight.get("p1_design_sha256") != sha256(DESIGN_PATH)
    ):
        raise RuntimeError("C134 E1 authorization artifacts are invalid")
    generator = set_seed(args.seed)
    expected = protocol.payload["dataset"]
    dataset = TrajectoryDataset(
        protocol.split_path("train").as_posix(),
        obs_len=16,
        obs_steps=1,
        pred_len=120,
        pred_step=5,
        delim=" ",
    )
    if (
        len(dataset) != int(expected["expected_train_scenes"])
        or int(dataset.obs_traj.shape[0]) != int(expected["expected_train_actors"])
    ):
        raise RuntimeError("C134 E1 train cohort identity mismatch")
    dates = json.loads(
        (ROOT / str(expected["train_scene_dates"])).read_text(encoding="utf-8")
    )["dates"]
    folds = json.loads(
        (ROOT / str(expected["date_folds"])).read_text(encoding="utf-8")
    )
    train_indices, validation_indices, _ = indices_for_fold(dates, folds, args.fold)
    train_indices = limited_indices(train_indices, args.max_train_scenes)
    validation_indices = limited_indices(validation_indices, args.max_validation_scenes)
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
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    model = build_model(batch_size=args.batch_size).to(device)
    backbone_path = ROOT / str(design["backbone"]["checkpoint"])
    if sha256(backbone_path) != design["backbone"]["checkpoint_sha256"]:
        raise RuntimeError("C134 E1 backbone checkpoint hash mismatch")
    model.load_backbone(backbone_path, device)
    trainable = list(model.energy_cost_operator.parameters())
    optimizer = Adam(trainable, lr=float(design["training"]["learning_rate"]))
    scheduler = MultiStepLR(
        optimizer,
        milestones=design["training"]["scheduler_milestones"],
        gamma=float(design["training"]["scheduler_gamma"]),
    )
    suffix = "smoke" if smoke else "formal"
    name = f"{VARIANT}_fold{args.fold}_seed{args.seed}_{suffix}"
    run_dir = RUN_ROOT / name
    summary_path = run_dir / "training_summary.json"
    if summary_path.is_file() and not smoke:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("complete") is True:
            return summary
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "last.pt"
    config = {
        **design["algorithm"],
        **design["training"],
        "fold": args.fold,
        "seed": args.seed,
        "variant": VARIANT,
        "protocol_sha256": protocol_hash,
        "p1_design_sha256": sha256(DESIGN_PATH),
        "locked_test_used": False,
        "development_used": False,
    }
    atomic_json(run_dir / "config.json", config)
    history = []
    start_epoch = 1
    if args.resume and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if (
            checkpoint.get("protocol_sha256") != protocol_hash
            or checkpoint.get("p1_design_sha256") != sha256(DESIGN_PATH)
        ):
            raise RuntimeError("C134 E1 resume identity mismatch")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        _restore_rng(checkpoint["rng_state"], generator)
        history = checkpoint["history"]
        start_epoch = int(checkpoint["epoch"]) + 1
    tail_threshold = target_tail_threshold(dataset)
    started = time.perf_counter()
    names = ("loss", "risk_regression", "normalized_energy", "energy_score")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        totals = {name: 0.0 for name in names}
        batches = 0
        epoch_started = time.perf_counter()
        for data in train_loader:
            data = move(data, device)
            target = data["pred_traj"].transpose(1, 0)
            optimizer.zero_grad(set_to_none=True)
            predictions, probabilities, _, auxiliary = model(data)
            loss, diagnostics = energy_predict_optimize_objective(
                predictions,
                probabilities,
                auxiliary["predicted_normalized_ade_risk"],
                target,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite C134 E1 loss at epoch {epoch}")
            loss.backward()
            if not all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in trainable
            ):
                raise RuntimeError(
                    f"non-finite C134 E1 operator gradient at epoch {epoch}"
                )
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
            optimizer.step()
            batches += 1
            for key in names:
                totals[key] += float(diagnostics[key])
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
            "p1_design_sha256": sha256(DESIGN_PATH),
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
            {"phase": "training", "variant": VARIANT, **record},
        )
        print(json.dumps(record), flush=True)
    metrics = evaluate(model, validation_loader, device, tail_threshold=tail_threshold)
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
        "checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": sha256(checkpoint_path),
        "protocol_sha256": protocol_hash,
        "p1_design_sha256": sha256(DESIGN_PATH),
        "manifest_sha256": sha256(protocol.manifest_path),
        "backbone_checkpoint_sha256": sha256(backbone_path),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "history": history,
        "validation_metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "persistent_workers": args.num_workers > 0,
        },
        "locked_test_used": False,
        "development_used": False,
    }
    atomic_json(summary_path, summary)
    atomic_json(
        ARTIFACT_ROOT / "status" / f"{name}.json",
        {"phase": "complete", "variant": VARIANT, "metrics": metrics["overall"]},
    )
    print(json.dumps({"run": name, "metrics": metrics["overall"]}), flush=True)
    return summary


if __name__ == "__main__":
    run(parse_args())
