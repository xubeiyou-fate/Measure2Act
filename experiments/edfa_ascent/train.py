"""Train matched C96 controls and EDFA-ASCENT on C12 development data."""

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
from tqdm import tqdm

from model.ascent import Ascent
from model.utils import TrajectoryDataset, seq_collate

from .data import build_scene_dates
from .evaluation import evaluate
from .live_report import update_live_document
from .model import EDFAAscent
from .objective import (
    per_agent_wta_loss,
    relation_classification_loss,
    relation_supervision,
    scene_wta_loss,
)
from .protocol import load_protocol, sha256


VARIANTS = ("a0", "a1", "a2", "a3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--relation-weight", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", type=Path, default=Path("runs/edfa_ascent"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/edfa_ascent"))
    parser.add_argument("--max-train-scenes", type=int)
    parser.add_argument("--max-dev-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--live-update-interval", type=int, default=250)
    parser.add_argument("--disable-progress", action="store_true")
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


def build_config(args: argparse.Namespace) -> dict:
    config = {
        "lr": args.learning_rate,
        "epochs": args.epochs,
        "k": 5,
        "obs_len": 16,
        "obs_steps": 1,
        "pred_len": 120,
        "pred_step": 5,
        "use_runway": False,
        "use_social": False,
        "use_weather": False,
        "ground_vs_airbourne": False,
        "split_xy_z": True,
        "normalize_coords": True,
        "global_pos_embedding": True,
        "mamba": False,
        "dataset_name": "trajair_111day_c12",
        "batch_size": args.batch_size,
        "loss_local": False,
        "flight_param_loss": False,
        "decoder": "new",
        "variant": args.variant,
        "cycle": "C96_EDFA_ASCENT",
        "trajectory_residual": False,
        "score_residual": False,
        "learned_fusion_gate": False,
        "temporal_autoregression": False,
        "post_generation_selector": False,
    }
    if args.variant == "a1":
        config.update({
            "scene_interaction": True,
            "scene_interaction_stage": "actor",
            "scene_attention_heads": 4,
        })
    elif args.variant in {"a2", "a3"}:
        config.update({
            "edfa_factorized": args.variant == "a3",
            "edfa_attention_heads": 4,
        })
    return config


def build_model(config: dict):
    return EDFAAscent(config) if config["variant"] in {"a2", "a3"} else Ascent(config)


def subset_indices(length: int, maximum: int | None) -> list[int] | None:
    if maximum is None or maximum >= length:
        return None
    return torch.linspace(0, length - 1, steps=maximum).round().to(torch.long).unique().tolist()


def load_scene_dates(args: argparse.Namespace, protocol, expected: int) -> list[str]:
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    path = args.artifact_root / "c12_dev_scene_dates.json"
    if path.exists():
        dates = json.loads(path.read_text(encoding="utf-8"))["dates"]
    else:
        dates = []
    if len(dates) != expected:
        dates = build_scene_dates(
            protocol.split_path("dev"), protocol.manifest_path, "dev"
        )
        path.write_text(json.dumps({"dates": dates}, indent=2) + "\n", encoding="utf-8")
    if len(dates) != expected:
        raise RuntimeError(f"scene-date index has {len(dates)} entries, expected {expected}")
    return dates


def teacher_forcing_probability(epoch: int, epochs: int) -> float:
    transition_end = max(2, int(round(epochs * 0.6)))
    if epoch >= transition_end:
        return 0.0
    return max(0.0, 1.0 - (epoch - 1) / max(1, transition_end - 1))


def main() -> None:
    args = parse_args()
    generator = set_seed(args.seed)
    protocol = load_protocol()
    protocol.assert_manifest_sealed()
    config = build_config(args)
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
    train_indices = subset_indices(len(datasets["train"]), args.max_train_scenes)
    dev_indices = subset_indices(len(datasets["dev"]), args.max_dev_scenes)
    train_data = Subset(datasets["train"], train_indices) if train_indices else datasets["train"]
    dev_data = Subset(datasets["dev"], dev_indices) if dev_indices else datasets["dev"]
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=seq_collate,
        num_workers=args.num_workers,
        generator=generator,
    )
    dev_loader = DataLoader(
        dev_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=seq_collate,
        num_workers=args.num_workers,
    )
    scene_dates = load_scene_dates(args, protocol, len(datasets["dev"]))
    if dev_indices:
        scene_dates = [scene_dates[index] for index in dev_indices]

    device = torch.device(args.device)
    model = build_model(config).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    milestones = sorted(set((max(1, args.epochs // 2), max(1, 3 * args.epochs // 4))))
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    suffix = "smoke" if train_indices or dev_indices else "formal"
    run_dir = args.output_root / f"{args.variant}_seed{args.seed}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    start_epoch = 1
    history = []
    best_minfde = float("inf")
    best_epoch = -1
    last_path = run_dir / "last.pt"
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = int(state["epoch"]) + 1
        history = state.get("history", [])
        best_minfde = float(state.get("best_minfde", best_minfde))
        best_epoch = int(state.get("best_epoch", best_epoch))

    started = time.perf_counter()
    live_state = {
        "status": "initializing",
        "variant": args.variant,
        "seed": args.seed,
        "formal": suffix == "formal",
        "epochs": args.epochs,
        "epoch": start_epoch - 1,
        "batch": 0,
        "batches_per_epoch": len(train_loader),
        "running_loss": "-",
        "best_epoch": best_epoch,
        "best_minfde": "-" if best_minfde == float("inf") else f"{best_minfde:.6f}",
        "history": history,
        "locked_test_used": False,
    }
    update_live_document(run_dir, live_state)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        totals: dict[str, float] = {}
        batches = 0
        oracle_probability = teacher_forcing_probability(epoch, args.epochs)
        for data in tqdm(
            train_loader,
            desc=f"C96 {args.variant} epoch {epoch}",
            disable=args.disable_progress,
        ):
            for key, value in data.items():
                if torch.is_tensor(value):
                    data[key] = value.to(device)
            target = data["pred_traj"].transpose(1, 0)
            labels = valid_pairs = None
            if args.variant in {"a2", "a3"}:
                labels, valid_pairs, _ = relation_supervision(target, data["adj"])
                if args.variant == "a3" and random.random() < oracle_probability:
                    data["graph_source"] = "oracle"
                    data["relation_labels"] = labels
            optimizer.zero_grad(set_to_none=True)
            predictions, logits, auxiliary = model(data)
            if args.variant in {"a0", "a1"}:
                loss, diagnostics = per_agent_wta_loss(predictions, logits, target)
            else:
                trajectory_loss, diagnostics = scene_wta_loss(
                    predictions, logits, target, data["adj"]
                )
                relation_loss, relation_diagnostics = relation_classification_loss(
                    auxiliary["relation_logits"], labels, valid_pairs
                )
                loss = trajectory_loss + args.relation_weight * relation_loss
                diagnostics = {
                    **diagnostics,
                    **relation_diagnostics,
                    "relation_loss": relation_loss.detach(),
                }
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batches += 1
            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach().cpu())
            for name, value in diagnostics.items():
                if name in {"winner", "scene_winner"}:
                    continue
                totals[name] = totals.get(name, 0.0) + float(value.cpu())
            if batches % args.live_update_interval == 0 or batches == len(train_loader):
                update_live_document(run_dir, {
                    **live_state,
                    "status": "training",
                    "epoch": epoch,
                    "batch": batches,
                    "running_loss": f"{totals['loss'] / batches:.6f}",
                    "best_epoch": best_epoch,
                    "best_minfde": "-" if best_minfde == float("inf") else f"{best_minfde:.6f}",
                    "history": history,
                })
        scheduler.step()

        validation = evaluate(model, dev_loader, device, scene_dates=scene_dates)
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "oracle_graph_probability": oracle_probability,
            "train": {name: value / batches for name, value in totals.items()},
            "validation": validation,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True))
        score = validation["overall"]["minfde"]
        if score < best_minfde:
            best_minfde = score
            best_epoch = epoch
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
            "history": history,
            "best_minfde": best_minfde,
            "best_epoch": best_epoch,
            "validation": validation,
        }
        torch.save(state, last_path)
        if best_epoch == epoch:
            torch.save(state, run_dir / "best.pt")
        summary = {
            "format_version": 1,
            "cycle": "C96_EDFA_ASCENT",
            "variant": args.variant,
            "seed": args.seed,
            "formal": suffix == "formal",
            "epochs_completed": epoch,
            "best_epoch": best_epoch,
            "best_validation_minfde": best_minfde,
            "elapsed_seconds": time.perf_counter() - started,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "history": history,
            "locked_test_used": False,
        }
        (run_dir / "training_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        live_state = {
            **live_state,
            "status": "validated",
            "epoch": epoch,
            "batch": len(train_loader),
            "running_loss": f"{record['train']['loss']:.6f}",
            "best_epoch": best_epoch,
            "best_minfde": f"{best_minfde:.6f}",
            "history": history,
        }
        update_live_document(run_dir, live_state)

    best_path = run_dir / "best.pt"
    state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    controls = {"predicted": evaluate(model, dev_loader, device, scene_dates=scene_dates)}
    if args.variant == "a3":
        controls["shuffled_neighbors"] = evaluate(
            model, dev_loader, device, scene_dates=scene_dates, permute_neighbors=True
        )
        controls["oracle_graph"] = evaluate(
            model, dev_loader, device, scene_dates=scene_dates, graph_source="oracle"
        )
    summary.update({
        "best_checkpoint": str(best_path.resolve()),
        "best_checkpoint_sha256": sha256(best_path),
        "best_controls": controls,
    })
    (run_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    update_live_document(run_dir, {
        **live_state,
        "status": "complete",
        "epoch": args.epochs,
        "batch": len(train_loader),
        "best_epoch": best_epoch,
        "best_minfde": f"{best_minfde:.6f}",
        "history": history,
        "controls": controls,
    })
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
