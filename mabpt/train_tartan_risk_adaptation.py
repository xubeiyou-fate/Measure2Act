"""Adapt only the MABPT risk head on one airport and test the other."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from experiments.energy_predict_optimize.evaluation import RankingMetricAccumulator, compute_batch_metrics
from experiments.energy_predict_optimize.model import build_model
from experiments.energy_predict_optimize.objective import energy_predict_optimize_objective
from model.utils import TrajectoryDataset, seed_worker, seq_collate

from .operator import DEFAULT_ADE_SCALE, pairwise_trajectory_distance, support_cost
from .partc_factorial import factorial_probability_arms, full_model_arm
from .partc_seed_evaluate import _default_source, _load_model_pair


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("tartan_risk_adaptation_protocol_v1.json")


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
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def loader(
    path: Path,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    generator=None,
    maximum_scenes: int | None = None,
):
    dataset = TrajectoryDataset(
        path.as_posix(), obs_len=16, obs_steps=1, pred_len=120, pred_step=5, delim=","
    )
    evaluation_dataset = (
        dataset
        if maximum_scenes is None or maximum_scenes >= len(dataset)
        else Subset(
            dataset,
            torch.linspace(0, len(dataset) - 1, maximum_scenes)
            .round()
            .long()
            .unique()
            .tolist(),
        )
    )
    options = {
        "dataset": evaluation_dataset,
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
    return evaluation_dataset, DataLoader(**options)


def move(data: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in data.items()
    }


@torch.inference_mode()
def evaluate(source, model, data_loader, device: torch.device) -> dict[str, object]:
    source.eval()
    model.eval()
    states = {name: RankingMetricAccumulator() for name in ("original_ascent", "adapted_mabpt")}
    for data in data_loader:
        data = move(data, device)
        truth = data["pred_traj"].transpose(1, 0).to(torch.float64)
        source_support, source_logits, _ = source(data)
        source_probability = source_logits.softmax(dim=1)
        target_support, _, target_decision, auxiliary = model(data)
        arms, _ = factorial_probability_arms(
            source_probability,
            support_cost(source_support, target_support),
            auxiliary["centered_predicted_normalized_ade_risk"].to(torch.float64),
            pairwise_trajectory_distance(target_support) / DEFAULT_ADE_SCALE,
        )
        measures = {
            "original_ascent": (source_support, source_probability, source_logits.argmax(dim=1)),
            "adapted_mabpt": (target_support, arms[full_model_arm()], target_decision),
        }
        for name, (support, probability, decision) in measures.items():
            states[name].update(compute_batch_metrics(
                support.to(torch.float64), probability.to(torch.float64), decision, truth
            ))
    return {name: state.summarize() for name, state in states.items()}


def run(args: argparse.Namespace) -> dict[str, object]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if args.direction not in protocol["directions"]:
        raise ValueError("direction lies outside the frozen registry")
    if args.epochs != protocol["training"]["epochs"] and not args.smoke:
        raise ValueError("formal epochs differ from frozen protocol")
    if args.max_test_scenes is not None and not args.smoke:
        raise ValueError("formal adaptation must evaluate the complete test airport")
    seed = int(protocol["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    generator = torch.Generator().manual_seed(seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    specification = protocol["directions"][args.direction]
    train_path = (ROOT / specification["train"]).resolve()
    test_path = (ROOT / specification["test"]).resolve()
    initial = (ROOT / protocol["initial_checkpoint"]).resolve()
    if sha256(initial) != protocol["initial_checkpoint_sha256"]:
        raise RuntimeError("initial target checkpoint hash mismatch")
    source_path = _default_source(seed).resolve()
    train_dataset, train_loader = loader(
        train_path,
        batch_size=args.batch_size,
        shuffle=True,
        workers=args.workers,
        generator=generator,
    )
    test_dataset, test_loader = loader(
        test_path,
        batch_size=args.eval_batch_size,
        shuffle=False,
        workers=args.workers,
        maximum_scenes=args.max_test_scenes,
    )
    source, _ = _load_model_pair(
        source_checkpoint=source_path,
        target_checkpoint=initial,
        device=device,
        batch_size=args.eval_batch_size,
    )
    model = build_model(batch_size=args.batch_size).to(device)
    checkpoint = torch.load(initial, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    trainable = list(model.energy_cost_operator.parameters())
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise RuntimeError("adaptation backbone is not frozen")
    optimizer = Adam(trainable, lr=float(protocol["training"]["learning_rate"]))
    epochs = 1 if args.smoke else args.epochs
    maximum_batches = args.max_train_batches if args.smoke else None
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {name: 0.0 for name in ("loss", "risk_regression", "normalized_energy", "energy_score")}
        batches = 0
        for data in train_loader:
            if maximum_batches is not None and batches >= maximum_batches:
                break
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
                raise RuntimeError("non-finite adaptation loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, float(protocol["training"]["gradient_clip_norm"]))
            optimizer.step()
            batches += 1
            for name in totals:
                totals[name] += float(diagnostics[name].detach().cpu())
        history.append({
            "epoch": epoch,
            "batches": batches,
            "train": {name: value / max(batches, 1) for name, value in totals.items()},
        })
    output_dir = args.run_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"epoch_{epochs:03d}.pt"
    if checkpoint_path.exists():
        raise FileExistsError(checkpoint_path)
    torch.save({
        "format_version": 1,
        "stage": "tartan_risk_head_adaptation",
        "direction": args.direction,
        "seed": seed,
        "epoch": epochs,
        "protocol_sha256": sha256(PROTOCOL),
        "model_state_dict": model.state_dict(),
        "history": history,
        "backbone_frozen": True,
    }, checkpoint_path)
    results = evaluate(source, model, test_loader, device)
    ascent = results["original_ascent"]
    adapted = results["adapted_mabpt"]
    return {
        "format_version": 1,
        "experiment_id": "Tartan_cross_airport_risk_head_adaptation",
        "evidence_class": protocol["evidence_class"],
        "direction": args.direction,
        "seed": seed,
        "formal": not args.smoke,
        "train_scenes": len(train_dataset),
        "test_scenes": len(test_dataset),
        "history": history,
        "results": results,
        "relative_gain_adapted_mabpt_vs_ascent": {
            metric: (ascent[metric] - adapted[metric]) / ascent[metric]
            for metric in protocol["evaluation"]["metrics"]
        },
        "inputs": {
            "protocol": PROTOCOL.relative_to(ROOT).as_posix(),
            "protocol_sha256": sha256(PROTOCOL),
            "initial_checkpoint": protocol["initial_checkpoint"],
            "initial_checkpoint_sha256": protocol["initial_checkpoint_sha256"],
            "source_checkpoint": source_path.relative_to(ROOT).as_posix(),
            "source_checkpoint_sha256": sha256(source_path),
            "train_path": specification["train"],
            "test_path": specification["test"],
        },
        "checkpoint": {
            "path": checkpoint_path.relative_to(ROOT).as_posix(),
            "sha256": sha256(checkpoint_path),
            "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        },
        "integrity": protocol["integrity"],
        "runtime": {
            "device": str(device),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        },
        "claim_boundary": protocol["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", choices=("kagc_to_kbtp", "kbtp_to_kagc"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=2)
    parser.add_argument("--max-test-scenes", type=int)
    args = parser.parse_args()
    result = run(args)
    atomic_json(args.output.resolve(), result)
    print(json.dumps({
        "output": args.output.resolve().as_posix(),
        "direction": result["direction"],
        "train_scenes": result["train_scenes"],
        "test_scenes": result["test_scenes"],
        "relative_gain": result["relative_gain_adapted_mabpt_vs_ascent"],
    }, indent=2))


if __name__ == "__main__":
    main()
