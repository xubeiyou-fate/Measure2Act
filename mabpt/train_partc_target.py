"""Train the decision-support and predicted-risk modules of MABPT-ASCENT."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import time
from typing import Any

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Subset

from experiments.metric_exact.evaluation import evaluate as evaluate_decision
from experiments.metric_exact.evaluation import target_tail_threshold
from experiments.metric_exact.train import _capture_rng, _restore_rng, move, set_seed
from experiments.decision_regret.model import build_model as build_decision_model
from experiments.decision_regret.objective import decision_regret_objective
from experiments.energy_predict_optimize.evaluate import evaluate as evaluate_risk
from experiments.energy_predict_optimize.model import build_model as build_risk_model
from experiments.energy_predict_optimize.objective import energy_predict_optimize_objective
from model.utils import TrajectoryDataset, seed_worker, seq_collate

from .partc_design import PROTOCOL, load_protocol, sha256


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/mabpt_partc_20260811"
ARTIFACT_ROOT = ROOT / "artifacts/mabpt_partc_20260811/status"
STAGES = ("decision_support", "predicted_risk")
FORMAL_BATCHES = {
    "decision_support": (256, 512),
    "predicted_risk": (1024, 2048),
}


def _file_sha256(path: Path) -> str:
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


def _limited(dataset, maximum: int | None):
    if maximum is None or maximum >= len(dataset):
        return dataset
    if maximum < 1:
        raise ValueError("maximum scene count must be positive")
    positions = (
        torch.linspace(0, len(dataset) - 1, steps=maximum)
        .round()
        .long()
        .unique()
        .tolist()
    )
    return Subset(dataset, positions)


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


def _load_data(max_train_scenes: int | None, max_dev_scenes: int | None):
    protocol = load_protocol()
    train = TrajectoryDataset(
        (ROOT / protocol["data"]["train"]).as_posix(),
        obs_len=16,
        obs_steps=1,
        pred_len=120,
        pred_step=5,
        delim=" ",
    )
    development = TrajectoryDataset(
        (ROOT / protocol["data"]["development"]).as_posix(),
        obs_len=16,
        obs_steps=1,
        pred_len=120,
        pred_step=5,
        delim=" ",
    )
    date_path = ROOT / "artifacts/c97_ikd_ascent/dev_scene_dates.json"
    dates = json.loads(date_path.read_text(encoding="utf-8"))["dates"]
    if len(dates) != len(development):
        raise RuntimeError("development scene-date index mismatch")
    if max_dev_scenes is not None and max_dev_scenes < len(development):
        positions = (
            torch.linspace(0, len(development) - 1, steps=max_dev_scenes)
            .round()
            .long()
            .unique()
            .tolist()
        )
        development = Subset(development, positions)
        dates = [dates[index] for index in positions]
    return _limited(train, max_train_scenes), development, dates, train, date_path


def _source_checkpoint(seed: int) -> Path:
    return (
        ROOT
        / "runs/metric_exact"
        / f"P3_B0_signed_coupled_all_train_seed{seed}_formal"
        / "last.pt"
    )


def _default_run_dir(stage: str, seed: int, smoke: bool) -> Path:
    suffix = "smoke" if smoke else "formal"
    return RUN_ROOT / f"mabpt_ascent_{stage}_seed{seed}_{suffix}"


def _checkpoint_path(run_dir: Path, epoch: int) -> Path:
    return run_dir / f"epoch_{epoch:03d}.pt"


def train(
    *,
    stage: str,
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    workers: int,
    max_train_scenes: int | None,
    max_dev_scenes: int | None,
    run_dir: Path,
    resume: Path | None,
    decision_checkpoint_override: Path | None,
) -> dict[str, Any]:
    protocol = load_protocol()
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    if seed not in [int(value) for value in protocol["fixed_seeds"]]:
        raise RuntimeError("seed lies outside the frozen Part C registry")
    smoke = max_train_scenes is not None or max_dev_scenes is not None
    if not smoke and epochs != 20:
        raise RuntimeError("formal target-module training is fixed to 20 epochs")
    if not smoke and (batch_size, eval_batch_size) != FORMAL_BATCHES[stage]:
        expected = FORMAL_BATCHES[stage]
        raise RuntimeError(
            f"formal {stage} batches are fixed to train={expected[0]}, "
            f"evaluation={expected[1]}"
        )
    if epochs < 1:
        raise ValueError("epochs must be positive")
    historical_receipt = ROOT / "artifacts/experiments/metric_exact/locked_test_receipt.json"
    if not historical_receipt.is_file():
        raise RuntimeError("expected historical C127 test receipt is missing")
    source_checkpoint = _source_checkpoint(seed)
    if not source_checkpoint.is_file():
        raise RuntimeError(f"missing matched source ASCENT checkpoint: {source_checkpoint}")
    source_summary = source_checkpoint.with_name("training_summary.json")
    if not source_summary.is_file():
        raise RuntimeError("source ASCENT checkpoint lacks a training summary")

    generator = set_seed(seed)
    train_data, dev_data, dev_dates, complete_train, date_path = _load_data(
        max_train_scenes, max_dev_scenes
    )
    train_loader = _loader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        workers=workers,
        generator=generator,
    )
    dev_loader = _loader(
        dev_data,
        batch_size=eval_batch_size,
        shuffle=False,
        workers=workers,
    )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    decision_checkpoint = (
        decision_checkpoint_override
        if decision_checkpoint_override is not None
        else (
            RUN_ROOT
            / f"mabpt_ascent_decision_support_seed{seed}_{'smoke' if smoke else 'formal'}"
            / f"epoch_{epochs:03d}.pt"
        )
    )
    if stage == "decision_support":
        model = build_decision_model(batch_size=batch_size).to(device)
        trainable = list(model.parameters())
    else:
        if not decision_checkpoint.is_file():
            raise RuntimeError(
                "predicted-risk training requires the matched fixed-epoch decision-support checkpoint"
            )
        model = build_risk_model(batch_size=batch_size).to(device)
        model.load_backbone(decision_checkpoint, device)
        trainable = list(model.energy_cost_operator.parameters())
    optimizer = Adam(trainable, lr=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[10, 15], gamma=0.5)

    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "training_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"completed run already exists: {summary_path}")
    protocol_hash = sha256(PROTOCOL)
    config_path = run_dir / "config.json"
    if not config_path.exists():
        _atomic_json(
            config_path,
            {
                "model": "MABPT-ASCENT",
                "stage": stage,
                "seed": seed,
                "epochs": epochs,
                "batch_size": batch_size,
                "evaluation_batch_size": eval_batch_size,
                "protocol": str(PROTOCOL.relative_to(ROOT)),
                "protocol_sha256": protocol_hash,
                "source_ascent_checkpoint": str(source_checkpoint.relative_to(ROOT)),
                "source_ascent_checkpoint_sha256": _file_sha256(source_checkpoint),
                "decision_support_checkpoint": (
                    str(decision_checkpoint.relative_to(ROOT))
                    if stage == "predicted_risk"
                    else None
                ),
                "train_scenes": len(train_data),
                "development_scenes": len(dev_data),
                "historical_test_receipt_sha256": _file_sha256(historical_receipt),
                "locked_test_used": False,
            },
        )

    history: list[dict[str, Any]] = []
    start_epoch = 1
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        if checkpoint["protocol_sha256"] != protocol_hash:
            raise RuntimeError("resume protocol hash mismatch")
        if checkpoint["stage"] != stage or int(checkpoint["seed"]) != seed:
            raise RuntimeError("resume stage or seed mismatch")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        _restore_rng(checkpoint["rng_state"], generator)
        history = list(checkpoint["history"])
        start_epoch = int(checkpoint["epoch"]) + 1

    metric_names = (
        ("loss", "geometry", "decision_regret_surrogate", "batch_minade", "batch_minfde")
        if stage == "decision_support"
        else ("loss", "risk_regression", "normalized_energy", "energy_score")
    )
    started = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        totals = {name: 0.0 for name in metric_names}
        batches = 0
        epoch_started = time.perf_counter()
        for data in train_loader:
            data = move(data, device)
            target = data["pred_traj"].transpose(1, 0)
            optimizer.zero_grad(set_to_none=True)
            if stage == "decision_support":
                predictions, logits, auxiliary = model(data)
                loss, diagnostics = decision_regret_objective(
                    predictions,
                    logits,
                    auxiliary["decision_costs"],
                    target,
                )
            else:
                predictions, probabilities, _, auxiliary = model(data)
                loss, diagnostics = energy_predict_optimize_objective(
                    predictions,
                    probabilities,
                    auxiliary["predicted_normalized_ade_risk"],
                    target,
                )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite {stage} loss at epoch {epoch}")
            loss.backward()
            if not all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in trainable
            ):
                raise RuntimeError(f"non-finite {stage} gradient at epoch {epoch}")
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            batches += 1
            for name in metric_names:
                totals[name] += float(diagnostics[name].detach().cpu())
        scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": {name: value / max(batches, 1) for name, value in totals.items()},
            "elapsed_seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        checkpoint_path = _checkpoint_path(run_dir, epoch)
        if checkpoint_path.exists():
            raise FileExistsError(f"refusing to overwrite {checkpoint_path}")
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save(
            {
                "format_version": 1,
                "model": "MABPT-ASCENT",
                "stage": stage,
                "seed": seed,
                "epoch": epoch,
                "protocol_sha256": protocol_hash,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "rng_state": _capture_rng(generator),
                "history": history,
                "locked_test_used": False,
            },
            temporary,
        )
        temporary.replace(checkpoint_path)
        _atomic_json(
            ARTIFACT_ROOT / f"{run_dir.name}.json",
            {"phase": "training", "stage": stage, "seed": seed, **record},
            replace=True,
        )
        print(json.dumps(record), flush=True)

    tail = target_tail_threshold(complete_train)
    if stage == "decision_support":
        validation = evaluate_decision(
            model,
            dev_loader,
            device,
            scene_dates=dev_dates,
            tail_threshold=tail,
        )
    else:
        validation = evaluate_risk(model, dev_loader, device, tail_threshold=tail)
    final_checkpoint = _checkpoint_path(run_dir, epochs)
    result: dict[str, Any] = {
        "format_version": 1,
        "model": "MABPT-ASCENT",
        "stage": stage,
        "seed": seed,
        "formal": not smoke,
        "complete": True,
        "fixed_final_epoch": epochs,
        "train_scenes": len(train_data),
        "development_scenes": len(dev_data),
        "checkpoint": str(final_checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": _file_sha256(final_checkpoint),
        "partc_protocol_sha256": protocol_hash,
        "development_dates_sha256": _file_sha256(date_path),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "history": history,
        "development_metrics": validation,
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        },
        "integrity": {
            "train_and_development_only": True,
            "locked_test_used": False,
            "fixed_epoch_checkpoint": True,
            "source_seed_matched": True,
        },
    }
    _atomic_json(summary_path, result)
    _atomic_json(
        ARTIFACT_ROOT / f"{run_dir.name}.json",
        {"phase": "complete", "stage": stage, "seed": seed},
        replace=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-train-scenes", type=int)
    parser.add_argument("--max-dev-scenes", type=int)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--decision-checkpoint", type=Path)
    args = parser.parse_args()
    smoke = args.max_train_scenes is not None or args.max_dev_scenes is not None
    formal_batch_size, formal_eval_batch_size = FORMAL_BATCHES[args.stage]
    batch_size = args.batch_size or formal_batch_size
    eval_batch_size = args.eval_batch_size or formal_eval_batch_size
    run_dir = (
        args.run_dir.resolve()
        if args.run_dir is not None
        else _default_run_dir(args.stage, args.seed, smoke)
    )
    result = train(
        stage=args.stage,
        seed=args.seed,
        device=torch.device(args.device),
        epochs=args.epochs,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        workers=args.workers,
        max_train_scenes=args.max_train_scenes,
        max_dev_scenes=args.max_dev_scenes,
        run_dir=run_dir,
        resume=args.resume.resolve() if args.resume is not None else None,
        decision_checkpoint_override=(
            args.decision_checkpoint.resolve()
            if args.decision_checkpoint is not None
            else None
        ),
    )
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "checkpoint": result["checkpoint"],
                "stage": result["stage"],
                "seed": result["seed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
