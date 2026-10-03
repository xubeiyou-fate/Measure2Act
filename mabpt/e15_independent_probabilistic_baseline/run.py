"""Train and evaluate the isolated E15 independent probability baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import time
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model.utils import TrajectoryDataset

from .model import IndependentMixtureGRU, parameter_count


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "protocol.json"
DATA_ROOT = ROOT / "artifacts/partc_two_dataset_20260812/target_domain_data_v4_formal"
INDEX_ROOT = ROOT / "artifacts/partc_two_dataset_20260812/target_domain_scene_index_v3"
AIRPORTS = ("KAGC", "KBTP")
SPLITS = ("train", "development", "test")
SEEDS = (42, 7, 123, 2024, 2026)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any], *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for attempt in range(50):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 49:
                raise
            time.sleep(0.1)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class ActorDataset(Dataset):
    """Flatten scene-level TrajectoryDataset records to independent actors."""

    def __init__(self, source: TrajectoryDataset, scene_dates: list[str], scene_limit: int | None = None):
        if len(source) != len(scene_dates):
            raise RuntimeError(f"scene/date mismatch: {len(source)} != {len(scene_dates)}")
        limit = len(source) if scene_limit is None else min(int(scene_limit), len(source))
        if limit < 1:
            raise ValueError("scene limit must be positive")
        actor_indices: list[int] = []
        dates: list[str] = []
        for scene_idx, (start, end) in enumerate(source.seq_start_end[:limit]):
            actor_indices.extend(range(int(start), int(end)))
            dates.extend([str(scene_dates[scene_idx])] * (int(end) - int(start)))
        self.obs = source.obs_traj[actor_indices].contiguous()
        self.future = source.pred_traj[actor_indices].contiguous()
        self.dates = dates
        self.scene_count = limit

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        return self.obs[index], self.future[index], self.dates[index]


def collate_actor(batch: list[tuple[torch.Tensor, torch.Tensor, str]]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    obs, future, dates = zip(*batch)
    return torch.stack(obs), torch.stack(future), list(dates)


def load_actor_dataset(airport: str, split: str, scene_limit: int | None = None) -> ActorDataset:
    if airport not in AIRPORTS or split not in SPLITS:
        raise ValueError("airport or split outside frozen E15 registry")
    cache = ROOT / "dataset/_cache/e15_independent" / airport / split
    source = TrajectoryDataset(
        (DATA_ROOT / airport / split).as_posix(),
        obs_len=16,
        obs_steps=1,
        pred_len=120,
        skip=5,
        pred_step=5,
        delim=" ",
        cache_dir=cache,
    )
    index_path = INDEX_ROOT / f"{airport}_{split}_scene_dates.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    return ActorDataset(source, list(index["dates"]), scene_limit)


def local_frame(obs: torch.Tensor, future: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Dataset storage is [B,3,T]. All predictions are made relative to the last observation.
    observed = obs.transpose(1, 2).contiguous()
    target = future.transpose(1, 2).contiguous()
    origin = observed[:, -1:, :]
    return observed - origin, target - origin, origin


def objective(trajectories: torch.Tensor, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    displacement = torch.linalg.vector_norm(trajectories - target[:, None], dim=-1)
    ade = displacement.mean(dim=-1)
    best = ade.argmin(dim=1)
    ce = nn.functional.cross_entropy(logits, best)
    rows = torch.arange(target.shape[0], device=target.device)
    selected = trajectories[rows, best]
    mse = (selected - target).square().mean()
    loss = ce + 0.5 * mse
    return loss, {"loss": float(loss.detach()), "cross_entropy": float(ce.detach()), "min_ade": float(ade.min(dim=1).values.mean().detach())}


def model_for(device: torch.device) -> IndependentMixtureGRU:
    return IndependentMixtureGRU(modes=5, hidden_dim=128, layers=2, dropout=0.05).to(device)


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.airport not in AIRPORTS or args.seed not in SEEDS:
        raise ValueError("airport or seed outside frozen E15 registry")
    formal = args.max_train_scenes is None and args.max_dev_scenes is None
    if args.epochs < 1 or args.epochs > 20:
        raise ValueError("E15 epochs must be in [1,20]")
    if formal and args.epochs != 20:
        raise RuntimeError("formal E15 training requires the frozen 20 epochs")
    if formal and args.batch_size != 512:
        raise RuntimeError("formal E15 training requires batch size 512")
    seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    train_data = load_actor_dataset(args.airport, "train", args.max_train_scenes)
    dev_data = load_actor_dataset(args.airport, "development", args.max_dev_scenes)
    model = model_for(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    run_dir = (args.run_dir or (ROOT / "runs/e15_independent_probabilistic" / args.airport / f"seed_{args.seed}")).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol_hash = sha256(PROTOCOL_PATH)
    config = {
        "experiment_id": "E15", "airport": args.airport, "seed": args.seed,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "max_train_scenes": args.max_train_scenes, "max_dev_scenes": args.max_dev_scenes,
        "protocol_sha256": protocol_hash, "model": "IndependentMixtureGRU",
        "third_dataset_used": False, "ascent_or_mabpt_weights_used": False,
    }
    config_path = run_dir / "config.json"
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError("existing E15 run config does not match the requested run")
    else:
        atomic_json(config_path, config)
    checkpoint_path = run_dir / "last.pt"
    history: list[dict[str, Any]] = []
    start_epoch = 1
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if (
            checkpoint.get("config") != config
            or int(checkpoint.get("epoch", -1)) < 1
            or int(checkpoint.get("epoch", -1)) > args.epochs
        ):
            raise RuntimeError("resume identity or protocol mismatch")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint["epoch"]) + 1
        random.setstate(checkpoint["rng_state"]["python"])
        np.random.set_state(checkpoint["rng_state"]["numpy"])
        torch.set_rng_state(checkpoint["rng_state"]["torch"])
        if device.type == "cuda" and checkpoint["rng_state"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(checkpoint["rng_state"]["cuda"])
        loader_generator.set_state(checkpoint["rng_state"]["loader_generator"])
    elif checkpoint_path.exists():
        raise RuntimeError("existing E15 checkpoint requires --resume")
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_actor,
    }
    if args.workers > 0:
        common.update({"persistent_workers": True, "prefetch_factor": 4})
    train_loader = DataLoader(
        train_data,
        shuffle=True,
        drop_last=formal,
        generator=loader_generator,
        **common,
    )
    dev_loader = DataLoader(dev_data, shuffle=False, **common)
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        totals: dict[str, float] = {}
        batches = 0
        for obs, future, _dates in train_loader:
            local_obs, local_target, _origin = local_frame(obs.to(device, non_blocking=True), future.to(device, non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            trajectories, logits = model(local_obs)
            loss, diagnostics = objective(trajectories, logits, local_target)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite E15 loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batches += 1
            for key, value in diagnostics.items():
                totals[key] = totals.get(key, 0.0) + value
        if batches == 0:
            raise RuntimeError("E15 training produced zero optimization batches")
        record = {"epoch": epoch, "batches": batches, "train": {key: value / max(batches, 1) for key, value in totals.items()}}
        history.append(record)
        torch.save({
            "format_version": 1, "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "history": history,
            "config": config,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                "loader_generator": loader_generator.get_state(),
            },
        }, checkpoint_path.with_suffix(".tmp"))
        checkpoint_path.with_suffix(".tmp").replace(checkpoint_path)
        print(json.dumps(record), flush=True)
    result = {
        "format_version": 1, "complete": True, "formal": formal, "experiment_id": "E15",
        "airport": args.airport, "seed": args.seed, "epochs": args.epochs,
        "protocol_sha256": protocol_hash,
        "train_scenes": train_data.scene_count, "train_actors": len(train_data),
        "development_scenes": dev_data.scene_count, "development_actors": len(dev_data),
        "checkpoint": checkpoint_path.relative_to(ROOT).as_posix(), "checkpoint_sha256": sha256(checkpoint_path),
        "parameter_count": parameter_count(model), "history": history,
        "runtime": {"device": str(device), "elapsed_seconds": time.perf_counter() - started,
                     "python": platform.python_version(), "torch": torch.__version__,
                     "cuda": torch.version.cuda if torch.cuda.is_available() else None,
                     "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0},
        "integrity": {"test_used": False, "development_used_for_training": False, "temperature_fit": False,
                      "third_dataset_used": False, "ascent_or_mabpt_weights_used": False},
    }
    atomic_json(run_dir / "training_summary.json", result)
    return result


def _metric_arrays(trajectories: torch.Tensor, logits: torch.Tensor, target: torch.Tensor, temperature: float) -> dict[str, np.ndarray]:
    probabilities = torch.softmax(logits / float(temperature), dim=-1)
    displacement = torch.linalg.vector_norm(trajectories - target[:, None], dim=-1)
    ade = displacement.mean(dim=-1)
    fde = displacement[..., -1]
    best = ade.argmin(dim=1)
    rows = torch.arange(target.shape[0], device=target.device)
    top1 = probabilities.argmax(dim=1)
    expected_ade = (probabilities * ade).sum(dim=1)
    expected_fde = (probabilities * fde).sum(dim=1)
    pairwise = torch.linalg.vector_norm(trajectories[:, :, None] - trajectories[:, None, :], dim=-1).mean(dim=-1)
    energy = expected_ade - 0.5 * (probabilities[:, :, None] * probabilities[:, None, :] * pairwise).sum(dim=(1, 2))
    one_hot = nn.functional.one_hot(best, num_classes=probabilities.shape[1]).to(probabilities.dtype)
    confidence, prediction = probabilities.max(dim=1)
    correct = (prediction == best).to(probabilities.dtype)
    return {
        "top1_ade": ade[rows, top1].detach().cpu().numpy(),
        "top1_fde": fde[rows, top1].detach().cpu().numpy(),
        "minade": ade.min(dim=1).values.detach().cpu().numpy(),
        "minfde": fde.min(dim=1).values.detach().cpu().numpy(),
        "expected_ade": expected_ade.detach().cpu().numpy(),
        "expected_fde": expected_fde.detach().cpu().numpy(),
        "energy": energy.detach().cpu().numpy(),
        "fixed_event_nll": (-torch.log(probabilities[rows, best].clamp_min(1e-8))).detach().cpu().numpy(),
        "brier": ((probabilities - one_hot).square().sum(dim=1)).detach().cpu().numpy(),
        "ece_confidence": confidence.detach().cpu().numpy(),
        "ece_correct": correct.detach().cpu().numpy(),
        "oracle_mode": best.detach().cpu().numpy(),
    }


def evaluate_split(model: IndependentMixtureGRU, loader: DataLoader, device: torch.device, temperature: float) -> dict[str, Any]:
    model.eval()
    arrays: dict[str, list[np.ndarray]] = {}
    dates: list[str] = []
    with torch.inference_mode():
        for obs, future, batch_dates in loader:
            local_obs, local_target, origin = local_frame(obs.to(device, non_blocking=True), future.to(device, non_blocking=True))
            local_predictions, logits = model(local_obs)
            predictions = local_predictions + origin[:, None]
            target = future.transpose(1, 2).to(device, non_blocking=True)
            values = _metric_arrays(predictions, logits, target, temperature)
            for key, value in values.items():
                arrays.setdefault(key, []).append(value)
            dates.extend(batch_dates)
    packed = {key: np.concatenate(value) for key, value in arrays.items()}
    metric_names = ("top1_ade", "top1_fde", "minade", "minfde", "expected_ade", "expected_fde", "energy", "fixed_event_nll", "brier")
    date_values: dict[str, dict[str, float]] = {}
    for date in sorted(set(dates)):
        mask = np.asarray([item == date for item in dates], dtype=bool)
        date_values[date] = {name: float(np.mean(packed[name][mask])) for name in metric_names}
    summary = {name: float(np.mean(packed[name])) for name in metric_names}
    confidence = packed["ece_confidence"]
    correct = packed["ece_correct"]
    ece = 0.0
    for lower, upper in zip(np.linspace(0.0, 1.0, 11)[:-1], np.linspace(0.0, 1.0, 11)[1:]):
        mask = (confidence >= lower) & ((confidence < upper) if upper < 1.0 else (confidence <= upper))
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(confidence[mask].mean()) - float(correct[mask].mean()))
    summary["ece"] = ece
    summary["actors"] = int(len(dates))
    summary["dates"] = int(len(date_values))
    summary["date_metrics"] = date_values
    return summary


def fit_temperature(model: IndependentMixtureGRU, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    grid = (0.5, 0.75, 1.0, 1.5, 2.0)
    scores = []
    for temperature in grid:
        metrics = evaluate_split(model, loader, device, temperature)
        scores.append({"temperature": temperature, "fixed_event_nll": metrics["fixed_event_nll"], "brier": metrics["brier"], "ece": metrics["ece"]})
    selected = min(scores, key=lambda item: (item["fixed_event_nll"], item["temperature"]))
    return {"grid": scores, "selected_temperature": float(selected["temperature"]), "selection_metric": "development fixed_event_nll"}


def bootstrap_date_metric(date_metrics: dict[str, dict[str, float]], metric: str, seed: int, draws: int = 10000) -> dict[str, float]:
    dates = sorted(date_metrics)
    values = np.asarray([date_metrics[date][metric] for date in dates], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "ci95_low": float(np.quantile(samples, 0.025)), "ci95_high": float(np.quantile(samples, 0.975)), "dates": int(len(values))}


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.airport not in AIRPORTS or args.seed not in SEEDS:
        raise ValueError("airport or seed outside frozen E15 registry")
    if not args.authorize_retrospective_test:
        raise RuntimeError("E15 test evaluation requires --authorize-retrospective-test")
    if args.max_test_scenes is not None:
        raise RuntimeError("formal E15 test evaluation forbids a partial test cohort")
    device = torch.device(args.device)
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    expected_config = checkpoint.get("config", {})
    if (
        expected_config.get("protocol_sha256") != sha256(PROTOCOL_PATH)
        or expected_config.get("airport") != args.airport
        or int(expected_config.get("seed", -1)) != args.seed
        or int(expected_config.get("epochs", -1)) != 20
        or expected_config.get("max_train_scenes") is not None
        or expected_config.get("max_dev_scenes") is not None
        or int(checkpoint.get("epoch", -1)) != 20
    ):
        raise RuntimeError("E15 evaluation checkpoint identity mismatch")
    model = model_for(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    dev_data = load_actor_dataset(args.airport, "development", args.max_dev_scenes)
    test_data = load_actor_dataset(args.airport, "test", args.max_test_scenes)
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_actor,
    }
    if args.workers > 0:
        common.update({"persistent_workers": True, "prefetch_factor": 4})
    dev_loader = DataLoader(dev_data, shuffle=False, **common)
    test_loader = DataLoader(test_data, shuffle=False, **common)
    started = time.perf_counter()
    calibration = fit_temperature(model, dev_loader, device)
    selected_temperature = float(calibration["selected_temperature"])
    dev_raw = evaluate_split(model, dev_loader, device, 1.0)
    dev_calibrated = evaluate_split(model, dev_loader, device, selected_temperature)
    test_raw = evaluate_split(model, test_loader, device, 1.0)
    test_calibrated = evaluate_split(model, test_loader, device, selected_temperature)
    bootstrap = {name: bootstrap_date_metric(test_calibrated["date_metrics"], name, args.seed + i) for i, name in enumerate(("energy", "minfde", "minade", "fixed_event_nll", "brier"))}
    result = {
        "format_version": 1, "complete": True, "formal": True, "experiment_id": "E15",
        "airport": args.airport, "seed": args.seed,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "checkpoint": checkpoint_path.relative_to(ROOT).as_posix(), "checkpoint_sha256": sha256(checkpoint_path),
        "parameter_count": parameter_count(model), "calibration": calibration,
        "development_raw": dev_raw, "development_calibrated": dev_calibrated,
        "test_raw": test_raw, "test_calibrated": test_calibrated,
        "test_date_bootstrap": bootstrap,
        "runtime": {"device": str(device), "elapsed_seconds": time.perf_counter() - started, "torch": torch.__version__},
        "integrity": {"test_selected_temperature": False, "test_used_for_selection": False, "third_dataset_used": False,
                      "ascent_or_mabpt_weights_used": False, "fixed_event_definition": "nearest-ADE mode index"},
    }
    output = args.output.resolve()
    atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--airport", choices=AIRPORTS, required=True)
    train_parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=512)
    train_parser.add_argument("--workers", type=int, default=12)
    train_parser.add_argument("--device", default="cuda:0")
    train_parser.add_argument("--max-train-scenes", type=int)
    train_parser.add_argument("--max-dev-scenes", type=int)
    train_parser.add_argument("--run-dir", type=Path)
    train_parser.add_argument("--resume", action="store_true")
    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--airport", choices=AIRPORTS, required=True)
    eval_parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--output", type=Path, required=True)
    eval_parser.add_argument("--device", default="cuda:0")
    eval_parser.add_argument("--batch-size", type=int, default=512)
    eval_parser.add_argument("--workers", type=int, default=12)
    eval_parser.add_argument("--max-dev-scenes", type=int)
    eval_parser.add_argument("--max-test-scenes", type=int)
    eval_parser.add_argument("--authorize-retrospective-test", action="store_true")
    args = parser.parse_args()
    if args.command == "train":
        result = train(args)
        print(json.dumps({"output": str((args.run_dir or (ROOT / 'runs/e15_independent_probabilistic' / args.airport / f'seed_{args.seed}')) / 'training_summary.json'), "checkpoint": result["checkpoint"]}, indent=2))
    else:
        result = evaluate(args)
        print(json.dumps({"output": str(args.output), "airport": args.airport, "seed": args.seed, "temperature": result["calibration"]["selected_temperature"], "test_energy": result["test_calibrated"]["energy"]}, indent=2))


if __name__ == "__main__":
    main()
