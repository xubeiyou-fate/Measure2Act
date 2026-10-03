"""Train ASCENT and MABPT-ASCENT on frozen Tartan airport-date splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

# Required by torch deterministic algorithms for CUDA matrix multiplication.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Subset

from experiments.metric_exact.evaluation import evaluate as evaluate_decision
from experiments.metric_exact.evaluation import target_tail_threshold
from experiments.metric_exact.model import build_model as build_ascent_model
from experiments.metric_exact.objective import objective_for_variant
from experiments.metric_exact.train import _capture_rng, _restore_rng, move, set_seed
from experiments.decision_regret.model import build_model as build_decision_model
from experiments.decision_regret.objective import decision_regret_objective
from experiments.energy_predict_optimize.evaluate import evaluate as evaluate_risk
from experiments.energy_predict_optimize.model import build_model as build_risk_model
from experiments.energy_predict_optimize.objective import energy_predict_optimize_objective
from model.utils import TrajectoryDataset, seed_worker, seq_collate


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("tartan_retrain_protocol_v1.json")
RUN_ROOT = ROOT / "runs/partc_tartan_retrain_20260812"
STATUS_ROOT = ROOT / "artifacts/partc_two_dataset_20260812/retrain_status_v1"
AIRPORTS = ("KAGC", "KBTP")
REGIMES = ("target_only", "full_finetune")
STAGES = ("ascent", "decision_support", "predicted_risk")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any], *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def _source_checkpoint(kind: str, seed: int, protocol: dict[str, Any]) -> Path:
    key = {
        "ascent": ("ASCENT", "source_checkpoint"),
        "decision_support": ("MABPT-ASCENT", "source_decision_checkpoint"),
        "predicted_risk": ("MABPT-ASCENT", "source_risk_checkpoint"),
    }[kind]
    return ROOT / protocol["registered_models"][key[0]][key[1]].format(seed=seed)


def _index_payload(protocol: dict[str, Any], airport: str, split: str) -> tuple[Path, dict[str, Any]]:
    path = ROOT / protocol["data"]["scene_index_root"] / f"{airport}_{split}_scene_dates.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def _dataset(protocol: dict[str, Any], airport: str, split: str) -> tuple[TrajectoryDataset, list[str], Path]:
    root = ROOT / protocol["data"]["root"]
    cache = ROOT / "dataset/_cache/partc_tartan_target_v4" / airport / split
    dataset = TrajectoryDataset(
        (root / airport / split).as_posix(),
        obs_len=16,
        obs_steps=1,
        pred_len=120,
        skip=5,
        pred_step=5,
        delim=" ",
        cache_dir=cache,
    )
    index_path, index = _index_payload(protocol, airport, split)
    dates = list(index["dates"])
    if len(dataset) != len(dates):
        raise RuntimeError(f"{airport}/{split} dataset/index mismatch: {len(dataset)} != {len(dates)}")
    return dataset, dates, index_path


def _fraction_dates(values: list[str], fraction: float) -> set[str]:
    unique = sorted(set(values))
    if fraction >= 1.0:
        return set(unique)
    count = max(1, int(round(len(unique) * fraction)))
    positions = np.linspace(0, len(unique) - 1, num=count).round().astype(int)
    return {unique[int(position)] for position in positions}


def _limited_indices(length: int, maximum: int | None) -> list[int]:
    if maximum is None or maximum >= length:
        return list(range(length))
    if maximum < 1:
        raise ValueError("scene maximum must be positive")
    return torch.linspace(0, length - 1, steps=maximum).round().long().unique().tolist()


def _tail_threshold(dataset: TrajectoryDataset, scene_indices: list[int], quantile: float = 0.75) -> float:
    distances = []
    for scene_index in scene_indices:
        start, end = dataset.seq_start_end[scene_index]
        displacement = dataset.pred_traj[start:end, :, -1] - dataset.obs_traj[start:end, :, -1]
        distances.append(torch.linalg.vector_norm(displacement, dim=1).float())
    if not distances:
        raise RuntimeError("cannot fit a tail threshold without training actors")
    return float(torch.quantile(torch.cat(distances), quantile))


def _loader(dataset, *, batch_size: int, shuffle: bool, workers: int, generator=None):
    options: dict[str, Any] = {
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
        options.update({"persistent_workers": True, "prefetch_factor": 4})
    return DataLoader(**options)


def _run_dir(airport: str, regime: str, fraction: float, stage: str, seed: int, smoke: bool) -> Path:
    percent = int(round(fraction * 100))
    return RUN_ROOT / airport / regime / f"p{percent:03d}" / f"{stage}_seed{seed}_{'smoke' if smoke else 'formal'}"


def _decision_checkpoint(airport: str, regime: str, fraction: float, seed: int, smoke: bool) -> Path:
    return _run_dir(airport, regime, fraction, "decision_support", seed, smoke) / "last.pt"


def _load_energy_operator(model, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    prefix = "energy_cost_operator."
    state = {
        key[len(prefix):]: value
        for key, value in checkpoint["model_state_dict"].items()
        if key.startswith(prefix)
    }
    model.energy_cost_operator.load_state_dict(state, strict=True)


def train(args: argparse.Namespace) -> dict[str, Any]:
    protocol = _protocol()
    if args.airport not in AIRPORTS or args.regime not in REGIMES or args.stage not in STAGES:
        raise ValueError("unregistered airport, regime, or stage")
    seeds = [int(value) for value in protocol["training"]["seeds"]]
    if args.seed not in seeds:
        raise RuntimeError("seed is outside the frozen Tartan registry")
    if args.fraction not in [float(value) for value in protocol["training"]["data_efficiency_fractions"]]:
        raise RuntimeError("training fraction is outside the frozen Tartan registry")
    smoke = args.max_train_scenes is not None or args.max_dev_scenes is not None or args.epochs != 20
    if not smoke and args.epochs != int(protocol["training"]["epochs"]):
        raise RuntimeError("formal training must use 20 epochs")
    if not smoke and args.fraction < 1.0 and args.seed != 42:
        raise RuntimeError("initial data-efficiency runs are registered for seed 42 only")
    formal_batches = protocol["training"]["batch_sizes"][args.stage]
    if not smoke and [args.batch_size, args.eval_batch_size] != formal_batches:
        raise RuntimeError(f"formal {args.stage} batches must be {formal_batches}")

    manifest = ROOT / protocol["data"]["root"] / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    generator = set_seed(args.seed)
    train_dataset, train_dates, train_index = _dataset(protocol, args.airport, "train")
    dev_dataset, dev_dates, dev_index = _dataset(protocol, args.airport, "development")
    selected_dates = _fraction_dates(train_dates, args.fraction)
    train_indices = [index for index, value in enumerate(train_dates) if value in selected_dates]
    train_indices = [train_indices[index] for index in _limited_indices(len(train_indices), args.max_train_scenes)]
    dev_indices = _limited_indices(len(dev_dataset), args.max_dev_scenes)
    selected_dev_dates = [dev_dates[index] for index in dev_indices]
    train_data = Subset(train_dataset, train_indices)
    dev_data = Subset(dev_dataset, dev_indices)
    train_loader = _loader(train_data, batch_size=args.batch_size, shuffle=True, workers=args.workers, generator=generator)
    dev_loader = _loader(dev_data, batch_size=args.eval_batch_size, shuffle=False, workers=args.workers)

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    source = _source_checkpoint(args.stage, args.seed, protocol)
    if args.regime == "full_finetune" and not source.is_file():
        raise FileNotFoundError(source)

    if args.stage == "ascent":
        model = build_ascent_model("B0_signed_coupled", batch_size=args.batch_size).to(device)
        if args.regime == "full_finetune":
            state = torch.load(source, map_location=device, weights_only=False)
            model.load_state_dict(state["model_state_dict"], strict=True)
        trainable = list(model.parameters())
    elif args.stage == "decision_support":
        model = build_decision_model(batch_size=args.batch_size).to(device)
        if args.regime == "full_finetune":
            state = torch.load(source, map_location=device, weights_only=False)
            model.load_state_dict(state["model_state_dict"], strict=True)
        trainable = list(model.parameters())
    else:
        decision_checkpoint = _decision_checkpoint(args.airport, args.regime, args.fraction, args.seed, smoke)
        if not decision_checkpoint.is_file():
            raise FileNotFoundError(f"matched decision checkpoint required: {decision_checkpoint}")
        model = build_risk_model(batch_size=args.batch_size).to(device)
        model.load_backbone(decision_checkpoint, device)
        if args.regime == "full_finetune":
            _load_energy_operator(model, source, device)
        trainable = list(model.energy_cost_operator.parameters())

    learning_rate = float(protocol["training"]["learning_rates"][args.regime])
    optimizer = Adam(trainable, lr=learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=protocol["training"]["milestones"], gamma=float(protocol["training"]["gamma"]))
    run_dir = args.run_dir.resolve() if args.run_dir else _run_dir(args.airport, args.regime, args.fraction, args.stage, args.seed, smoke)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "training_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("complete") is True:
            return summary
        raise RuntimeError(f"incomplete summary requires manual audit: {summary_path}")
    checkpoint_path = run_dir / "last.pt"
    config_path = run_dir / "config.json"
    if not config_path.exists():
        _atomic_json(config_path, {
            "airport": args.airport, "regime": args.regime, "stage": args.stage,
            "seed": args.seed, "fraction": args.fraction, "selected_train_dates": sorted(selected_dates),
            "epochs": args.epochs, "batch_size": args.batch_size, "eval_batch_size": args.eval_batch_size,
            "learning_rate": learning_rate, "protocol": str(PROTOCOL.relative_to(ROOT)),
            "protocol_sha256": sha256(PROTOCOL), "data_manifest_sha256": sha256(manifest),
            "source_checkpoint": str(source.relative_to(ROOT)) if args.regime == "full_finetune" else None,
            "source_checkpoint_sha256": sha256(source) if args.regime == "full_finetune" else None,
            "locked_test_used": False,
        })

    history: list[dict[str, Any]] = []
    start_epoch = 1
    if args.resume and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint["protocol_sha256"] != sha256(PROTOCOL):
            raise RuntimeError("resume protocol hash mismatch")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        _restore_rng(checkpoint["rng_state"], generator)
        history = list(checkpoint["history"])
        start_epoch = int(checkpoint["epoch"]) + 1

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
            if args.stage == "ascent":
                predictions, logits, _ = model(data)
                loss, diagnostics = objective_for_variant("B0_signed_coupled", predictions, logits, target)
            elif args.stage == "decision_support":
                predictions, logits, auxiliary = model(data)
                loss, diagnostics = decision_regret_objective(predictions, logits, auxiliary["decision_costs"], target)
            else:
                predictions, probabilities, _, auxiliary = model(data)
                loss, diagnostics = energy_predict_optimize_objective(predictions, probabilities, auxiliary["predicted_normalized_ade_risk"], target)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite {args.stage} loss at epoch {epoch}")
            loss.backward()
            if not all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in trainable):
                raise RuntimeError(f"non-finite {args.stage} gradient at epoch {epoch}")
            torch.nn.utils.clip_grad_norm_(trainable, float(protocol["training"]["gradient_clip_norm"]))
            optimizer.step()
            batches += 1
            for name, value in diagnostics.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
        scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": {name: value / max(batches, 1) for name, value in totals.items()},
            "elapsed_seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save({
            "format_version": 1, "model": "ASCENT" if args.stage == "ascent" else "MABPT-ASCENT",
            "airport": args.airport, "regime": args.regime, "stage": args.stage, "seed": args.seed,
            "fraction": args.fraction, "epoch": epoch, "protocol_sha256": sha256(PROTOCOL),
            "data_manifest_sha256": sha256(manifest), "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
            "rng_state": _capture_rng(generator), "history": history, "locked_test_used": False,
        }, temporary)
        temporary.replace(checkpoint_path)
        _atomic_json(STATUS_ROOT / f"{args.airport}_{args.regime}_p{int(args.fraction*100):03d}_{args.stage}_seed{args.seed}.json", {"phase": "training", **record}, replace=True)
        print(json.dumps(record), flush=True)

    tail = _tail_threshold(train_dataset, train_indices)
    if args.stage in {"ascent", "decision_support"}:
        validation = evaluate_decision(model, dev_loader, device, scene_dates=selected_dev_dates, tail_threshold=tail)
    else:
        validation = evaluate_risk(model, dev_loader, device, tail_threshold=tail)
    result = {
        "format_version": 1, "complete": True, "formal": not smoke,
        "airport": args.airport, "regime": args.regime, "stage": args.stage, "seed": args.seed,
        "fraction": args.fraction, "fixed_final_epoch": args.epochs,
        "train_scenes": len(train_data), "train_dates": sorted(selected_dates),
        "development_scenes": len(dev_data), "development_dates": sorted(set(selected_dev_dates)),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint_path),
        "protocol_sha256": sha256(PROTOCOL), "data_manifest_sha256": sha256(manifest),
        "train_index_sha256": sha256(train_index), "development_index_sha256": sha256(dev_index),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "history": history, "development_metrics": validation,
        "runtime": {"elapsed_seconds": time.perf_counter() - started, "python": platform.python_version(),
                    "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device),
                    "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0},
        "integrity": {"train_and_development_only": True, "locked_test_used": False,
                      "date_grouped_fraction": True, "fixed_final_epoch": True},
    }
    _atomic_json(summary_path, result)
    _atomic_json(STATUS_ROOT / f"{args.airport}_{args.regime}_p{int(args.fraction*100):03d}_{args.stage}_seed{args.seed}.json", {"phase": "complete", "checkpoint_sha256": result["checkpoint_sha256"]}, replace=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airport", choices=AIRPORTS, required=True)
    parser.add_argument("--regime", choices=REGIMES, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-train-scenes", type=int)
    parser.add_argument("--max-dev-scenes", type=int)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    formal = _protocol()["training"]["batch_sizes"][args.stage]
    args.batch_size = args.batch_size or int(formal[0])
    args.eval_batch_size = args.eval_batch_size or int(formal[1])
    result = train(args)
    print(json.dumps({"run": str(args.run_dir or _run_dir(args.airport, args.regime, args.fraction, args.stage, args.seed, not result["formal"])), "checkpoint": result["checkpoint"]}, indent=2))


if __name__ == "__main__":
    main()
