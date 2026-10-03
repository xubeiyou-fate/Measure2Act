"""Train one frozen C99 variant and evaluate its fixed final checkpoint."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Subset

from model.utils import TrajectoryDataset, seq_collate

from .evaluation import evaluate, target_tail_threshold
from .live_report import update_status
from .model import C99_VARIANTS, DecoupledAscent, ascent_config, build_model
from .objective import coupled_wta_loss, geometry_wta_loss, voronoi_score_loss
from .protocol import load_protocol, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=C99_VARIANTS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--phase", choices=("screen", "replication"), required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-train-scenes", type=int)
    parser.add_argument("--max-dev-scenes", type=int)
    parser.add_argument("--live-update-interval", type=int, default=250)
    return parser.parse_args()


def set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.Generator().manual_seed(seed)


def _subset(dataset, maximum: int | None):
    if maximum is None or maximum >= len(dataset):
        return dataset, None
    indices = (
        torch.linspace(0, len(dataset) - 1, steps=maximum)
        .round()
        .long()
        .unique()
        .tolist()
    )
    return Subset(dataset, indices), indices


def _move(data: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in data.items()
    }


def _restore_generator_state(
    generator: torch.Generator, state: torch.Tensor | object
) -> None:
    state_tensor = state if torch.is_tensor(state) else torch.as_tensor(state)
    generator.set_state(state_tensor.detach().to(device="cpu", dtype=torch.uint8))


def _authorize(args: argparse.Namespace, protocol) -> None:
    if args.phase == "screen":
        screen = protocol.payload["screen"]
        if args.seed != int(screen["seed"]) or args.variant not in screen["variants"]:
            raise RuntimeError("C99 screen permits only frozen seed-42 variants")
    else:
        development = protocol.payload["development"]
        if args.variant not in development["primary_variants"]:
            raise RuntimeError("C99 replication permits only the primary A0/A5 pair")
        if args.seed not in set(development["seeds"]) - {int(protocol.payload["screen"]["seed"])}:
            raise RuntimeError("C99 replication seed is not preregistered")
        gate_path = protocol.repository_root / "artifacts/experiments/dive_ascent/seed42_gate.json"
        if not gate_path.is_file():
            raise RuntimeError("C99 replication requires seed42_gate.json")
        if json.loads(gate_path.read_text(encoding="utf-8")).get("replication_authorized") is not True:
            raise RuntimeError("C99 seed-42 gate prohibits replication")
    if args.variant in C99_VARIANTS[2:]:
        audit_path = protocol.repository_root / "artifacts/experiments/dive_ascent/gradient_audit.json"
        if not audit_path.is_file():
            raise RuntimeError("C99 decoupled training requires the frozen P0 gradient audit")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("processed_batches") != int(protocol.payload["gradient_audit"]["batches"]):
            raise RuntimeError("C99 gradient audit is incomplete")


def _make_optimizer(module_or_parameters, learning_rate: float) -> Adam:
    parameters = (
        module_or_parameters.parameters()
        if isinstance(module_or_parameters, torch.nn.Module)
        else module_or_parameters
    )
    return Adam(list(parameters), lr=learning_rate)


def _optimizer_payload(optimizers: dict[str, Adam]) -> dict[str, object]:
    return {name: optimizer.state_dict() for name, optimizer in optimizers.items()}


def _scheduler_payload(schedulers: dict[str, MultiStepLR]) -> dict[str, object]:
    return {name: scheduler.state_dict() for name, scheduler in schedulers.items()}


def run(args: argparse.Namespace) -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_development_sealed()
    _authorize(args, protocol)
    settings = protocol.payload["training"]
    epochs = int(settings["epochs"])
    batch_size = int(settings["batch_size"])
    evaluation_batch_size = int(settings["evaluation_batch_size"])
    generator = set_seed(args.seed)
    datasets = {
        split: TrajectoryDataset(
            protocol.split_path(split).as_posix(),
            obs_len=16,
            obs_steps=1,
            pred_len=120,
            pred_step=5,
            delim=" ",
        )
        for split in ("train", "dev")
    }
    train_data, train_indices = _subset(datasets["train"], args.max_train_scenes)
    dev_data, dev_indices = _subset(datasets["dev"], args.max_dev_scenes)
    smoke = train_indices is not None or dev_indices is not None
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=seq_collate,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    dev_loader = DataLoader(
        dev_data,
        batch_size=evaluation_batch_size,
        shuffle=False,
        collate_fn=seq_collate,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    dates_path = protocol.repository_root / str(protocol.payload["development_scene_dates_artifact"])
    dates = json.loads(dates_path.read_text(encoding="utf-8"))["dates"]
    if dev_indices is not None:
        dates = [dates[index] for index in dev_indices]
    if len(dates) != len(dev_data):
        raise RuntimeError("C99 scene-date count does not match development data")
    tail_threshold = target_tail_threshold(datasets["dev"])

    device = torch.device(args.device)
    model = build_model(args.variant, batch_size=batch_size).to(device)
    decoupled = isinstance(model, DecoupledAscent)
    learning_rate = float(settings["learning_rate"])
    if decoupled:
        optimizers = {
            "geometry": _make_optimizer(model.geometry_parameters(), learning_rate),
            "scorer": _make_optimizer(model.scorer_parameters(), float(settings["score_learning_rate"])),
        }
    else:
        optimizers = {"coupled": _make_optimizer(model, learning_rate)}
    schedulers = {
        name: MultiStepLR(
            optimizer,
            milestones=[int(value) for value in settings["scheduler_milestones"]],
            gamma=float(settings["scheduler_gamma"]),
        )
        for name, optimizer in optimizers.items()
    }
    suffix = "smoke" if smoke else "formal"
    run_name = f"{args.variant}_seed{args.seed}_{suffix}"
    run_dir = protocol.repository_root / str(protocol.payload["run_root"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "training_summary.json"
    if summary_path.is_file() and not smoke:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("complete") is True:
            return summary
    config = ascent_config(args.variant, batch_size=batch_size)
    config["complete_independent_experts"] = bool(decoupled and model.independent)
    config["score_gradient_detached"] = decoupled
    config["dac_curriculum"] = bool(decoupled and model.dac)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    checkpoint_path = run_dir / "last.pt"
    start_epoch = 1
    history: list[dict[str, object]] = []
    split_history: list[dict[str, object]] = []
    if args.resume and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint["protocol_sha256"] != sha256(protocol.path):
            raise RuntimeError("C99 resume checkpoint protocol hash mismatch")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(checkpoint["optimizer_state_dicts"][name])
        for name, scheduler in schedulers.items():
            scheduler.load_state_dict(checkpoint["scheduler_state_dicts"][name])
        _restore_generator_state(generator, checkpoint["generator_state"])
        history = checkpoint.get("history", [])
        split_history = checkpoint.get("split_history", [])
        start_epoch = int(checkpoint["epoch"]) + 1

    split_epochs = set(int(value) for value in settings["dac"]["split_before_epochs"])
    started = time.perf_counter()
    final_epoch = 1 if smoke else epochs
    for epoch in range(start_epoch, final_epoch + 1):
        if decoupled and model.dac and epoch in split_epochs:
            if not history:
                raise RuntimeError("DIVE split requires previous-epoch distortion statistics")
            previous = history[-1]["train"]["winner_distortion_sum"]
            active_before = int(model.active_experts.item())
            distortions = torch.tensor(previous[:active_before], device=device)
            split_record = model.split_highest_distortion(
                distortions,
                perturbation_scale=float(settings["dac"]["perturbation_scale"]),
                seed=int(settings["dac"]["perturbation_seed"]) + epoch + args.seed,
            )
            split_history.append({"before_epoch": epoch, **split_record})
        model.train()
        totals = {"loss": 0.0, "regression": 0.0, "classification": 0.0}
        winner_counts = torch.zeros(5, dtype=torch.long)
        distortion_sum = torch.zeros(5, dtype=torch.float64)
        batches = 0
        for data in train_loader:
            data = _move(data, device)
            target = data["pred_traj"].transpose(1, 0)
            if decoupled:
                active = int(model.active_experts.item())
                optimizers["geometry"].zero_grad(set_to_none=True)
                predictions, _ = model.predict_geometry(data)
                regression, diagnostics = geometry_wta_loss(
                    predictions, target, active_modes=active
                )
                if not torch.isfinite(regression):
                    raise RuntimeError(f"non-finite C99 geometry loss: {run_name} epoch {epoch}")
                regression.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.geometry_parameters()),
                    max_norm=float(settings["gradient_clip_norm"]),
                )
                optimizers["geometry"].step()

                optimizers["scorer"].zero_grad(set_to_none=True)
                logits, _ = model.score(data, predictions.detach())
                classification = voronoi_score_loss(
                    logits, diagnostics["winner"], active_modes=active
                )
                if not torch.isfinite(classification):
                    raise RuntimeError(f"non-finite C99 score loss: {run_name} epoch {epoch}")
                classification.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.scorer_parameters()),
                    max_norm=float(settings["gradient_clip_norm"]),
                )
                optimizers["scorer"].step()
                loss = regression.detach() + classification.detach()
            else:
                optimizers["coupled"].zero_grad(set_to_none=True)
                predictions, logits, _ = model(data)
                loss, diagnostics = coupled_wta_loss(predictions, logits, target)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite C99 coupled loss: {run_name} epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=float(settings["gradient_clip_norm"])
                )
                optimizers["coupled"].step()
                regression = diagnostics["regression"]
                classification = diagnostics["classification"]
            active = int(model.active_experts.item()) if decoupled else 5
            winner_counts[:active] += diagnostics["winner_counts"].detach().cpu()
            distortion_sum[:active] += diagnostics["winner_distortion_sum"].detach().double().cpu()
            batches += 1
            totals["loss"] += float(loss.detach().cpu())
            totals["regression"] += float(regression.detach().cpu())
            totals["classification"] += float(classification.detach().cpu())
            if batches % args.live_update_interval == 0 or batches == len(train_loader):
                update_status(
                    protocol.repository_root,
                    run_name,
                    {
                        "phase": "training",
                        "variant": args.variant,
                        "seed": args.seed,
                        "epoch": epoch,
                        "epochs": final_epoch,
                        "batch": batches,
                        "batches_per_epoch": len(train_loader),
                        "running_loss": f"{totals['loss'] / batches:.6f}",
                        "development_minfde": "-",
                    },
                )
        for scheduler in schedulers.values():
            scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rates": {
                name: optimizer.param_groups[0]["lr"]
                for name, optimizer in optimizers.items()
            },
            "active_experts": int(model.active_experts.item()) if decoupled else 5,
            "train": {
                **{name: value / batches for name, value in totals.items()},
                "winner_counts": winner_counts.tolist(),
                "winner_distortion_sum": distortion_sum.tolist(),
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(record)
        torch.save(
            {
                "format_version": 1,
                "cycle": protocol.payload["cycle"],
                "protocol_sha256": sha256(protocol.path),
                "manifest_sha256": sha256(protocol.manifest_path),
                "variant": args.variant,
                "seed": args.seed,
                "phase": args.phase,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dicts": _optimizer_payload(optimizers),
                "scheduler_state_dicts": _scheduler_payload(schedulers),
                "generator_state": generator.get_state(),
                "history": history,
                "split_history": split_history,
                "trajectory_residual": False,
                "learned_gate": False,
                "token_codebook": False,
                "future_autoregression": False,
                "post_generation_selector": False,
            },
            checkpoint_path,
        )
        print(json.dumps(record), flush=True)

    update_status(
        protocol.repository_root,
        run_name,
        {
            "phase": "evaluating_development",
            "variant": args.variant,
            "seed": args.seed,
            "epoch": history[-1]["epoch"],
            "epochs": final_epoch,
            "batch": len(train_loader),
            "batches_per_epoch": len(train_loader),
            "running_loss": f"{history[-1]['train']['loss']:.6f}",
            "development_minfde": "-",
        },
    )
    metrics = evaluate(
        model,
        dev_loader,
        device,
        scene_dates=dates,
        tail_threshold=tail_threshold,
    )
    summary = {
        "format_version": 1,
        "cycle": protocol.payload["cycle"],
        "protocol_sha256": sha256(protocol.path),
        "manifest_sha256": sha256(protocol.manifest_path),
        "variant": args.variant,
        "seed": args.seed,
        "phase": args.phase,
        "formal": not smoke,
        "fixed_final_epoch": history[-1]["epoch"],
        "checkpoint": checkpoint_path.relative_to(protocol.repository_root).as_posix(),
        "checkpoint_sha256": sha256(checkpoint_path),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
        "split_history": split_history,
        "development_metrics": metrics,
        "locked_test_used": False,
        "complete": True,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    update_status(
        protocol.repository_root,
        run_name,
        {
            "phase": "complete",
            "variant": args.variant,
            "seed": args.seed,
            "epoch": history[-1]["epoch"],
            "epochs": final_epoch,
            "batch": len(train_loader),
            "batches_per_epoch": len(train_loader),
            "running_loss": f"{history[-1]['train']['loss']:.6f}",
            "development_minfde": f"{metrics['overall']['minfde']:.6f}",
        },
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = run(args)
    print(
        json.dumps(
            {
                "variant": summary["variant"],
                "seed": summary["seed"],
                "minfde": summary["development_metrics"]["overall"]["minfde"],
                "complete": summary["complete"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
