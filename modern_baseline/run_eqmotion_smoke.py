"""Run the registered deterministic EqMotion aviation adapter smoke gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import torch
from torch.utils.data import DataLoader

from experiments.energy_predict_optimize.evaluation import (
    RankingMetricAccumulator,
    compute_batch_metrics,
)
from modern_baseline.eqmotion_aviation import (
    AviationSceneDataset,
    EqMotionAviation,
    aviation_collate,
    best_of_k_ade_loss,
    valid_actor_tensors,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = Path(__file__).with_name("eqmotion_aviation_protocol.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-scenes", type=int, default=4)
    parser.add_argument("--eval-scenes", type=int, default=3)
    parser.add_argument("--max-files", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/partc_two_dataset_20260812/eqmotion_smoke_seed42"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/partc_two_dataset_20260812/modern_baseline/"
            "eqmotion_aviation_smoke_v1.json"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_fingerprint(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_new_json(path: Path, payload: dict[str, object]) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_receipt() -> dict[str, object]:
    source = ROOT / "third_party" / "eqmotion_official"
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    remote = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "repository": remote,
        "commit": commit,
        "working_tree_clean": status == "",
        "license": "MIT",
        "license_sha256": sha256(source / "LICENSE"),
    }


def make_dataset(path: str, delimiter: str, scenes: int, files: int) -> AviationSceneDataset:
    return AviationSceneDataset(
        ROOT / path,
        delimiter=delimiter,
        max_scenes=scenes,
        max_files=files,
        scene_stride_seconds=5,
    )


def move(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate(
    model: EqMotionAviation,
    dataset: AviationSceneDataset,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=aviation_collate,
    )
    accumulator = RankingMetricAccumulator()
    batch_shapes: list[dict[str, object]] = []
    for batch in loader:
        batch = move(batch, device)
        predictions = model(batch["history"], batch["num_valid"])
        actor_predictions, actor_target = valid_actor_tensors(
            predictions,
            batch["future"],
            batch["valid"],
        )
        probabilities = torch.full(
            (len(actor_predictions), 5),
            0.2,
            device=device,
            dtype=actor_predictions.dtype,
        )
        decision = torch.zeros(len(actor_predictions), device=device, dtype=torch.long)
        metrics = compute_batch_metrics(
            actor_predictions.to(torch.float64),
            probabilities.to(torch.float64),
            decision,
            actor_target.to(torch.float64),
        )
        accumulator.update(metrics)
        batch_shapes.append(
            {
                "history": list(batch["history"].shape),
                "future": list(batch["future"].shape),
                "predictions": list(predictions.shape),
                "valid_actors": int(batch["valid"].sum().item()),
            }
        )
    summary = accumulator.summarize()
    selected = {
        "actors": summary["agents"],
        "sample1_ADE": summary["top1_ade"],
        "sample1_FDE": summary["top1_fde"],
        "minADE_at_5": summary["minade"],
        "minFDE_at_5": summary["minfde"],
        "Energy": summary["energy_score"],
    }
    return {
        "metrics": selected,
        "all_finite": all(math.isfinite(float(value)) for value in selected.values()),
        "batch_shapes": batch_shapes,
        "data_receipt": dataset.receipt(),
    }


def main() -> None:
    args = parse_args()
    if min(
        args.train_scenes,
        args.eval_scenes,
        args.max_files,
        args.batch_size,
        args.train_batches,
    ) < 1:
        raise ValueError("all smoke limits must be positive")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if args.seed != protocol["smoke_training"]["seed"]:
        raise ValueError("seed differs from the frozen smoke protocol")
    if args.train_batches > protocol["smoke_training"]["maximum_batches"]:
        raise ValueError("train batch count exceeds the frozen smoke maximum")
    if protocol["grid"]["history_offsets_seconds"] != list(range(0, 80, 5)):
        raise ValueError("frozen EqMotion history grid does not match the adapter")
    if protocol["grid"]["future_offsets_seconds"] != list(range(5, 125, 5)):
        raise ValueError("frozen EqMotion future grid does not match the adapter")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    train_data = make_dataset(
        protocol["data"]["training_smoke"],
        " ",
        args.train_scenes,
        args.max_files,
    )
    evaluation_specs = {
        "trajair_development": (
            protocol["data"]["evaluation_smoke"]["trajair_development"],
            " ",
        ),
        "tartan_kagc_external_128": (
            protocol["data"]["evaluation_smoke"]["tartan_kagc_external_128"],
            ",",
        ),
        "tartan_kbtp_external_128": (
            protocol["data"]["evaluation_smoke"]["tartan_kbtp_external_128"],
            ",",
        ),
    }
    evaluation_data = {
        name: make_dataset(path, delimiter, args.eval_scenes, args.max_files)
        for name, (path, delimiter) in evaluation_specs.items()
    }
    model = EqMotionAviation(
        device=device,
        hidden_nf=protocol["model"]["hidden_nf"],
        channels=protocol["model"]["channels"],
        layers=protocol["model"]["layers"],
        modes=protocol["grid"]["modes"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=protocol["smoke_training"]["learning_rate"],
    )
    loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=aviation_collate,
    )
    losses: list[float] = []
    finite_nonzero_gradient = False
    train_shapes: list[dict[str, object]] = []
    model.train()
    for step, batch in enumerate(loader, start=1):
        if step > args.train_batches:
            break
        batch = move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(batch["history"], batch["num_valid"])
        loss = best_of_k_ade_loss(predictions, batch["future"], batch["valid"])
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite EqMotion smoke loss")
        loss.backward()
        finite_nonzero_gradient = any(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            and bool((parameter.grad != 0).any())
            for parameter in model.parameters()
        )
        if not finite_nonzero_gradient:
            raise RuntimeError("EqMotion smoke backward produced no finite nonzero gradient")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        train_shapes.append(
            {
                "history": list(batch["history"].shape),
                "future": list(batch["future"].shape),
                "predictions": list(predictions.shape),
                "valid_actors": int(batch["valid"].sum().item()),
            }
        )
    if len(losses) != min(args.train_batches, math.ceil(len(train_data) / args.batch_size)):
        raise RuntimeError("unexpected number of EqMotion smoke optimizer steps")

    evaluations = {
        name: evaluate(model, dataset, device, args.batch_size)
        for name, dataset in evaluation_data.items()
    }
    all_finite = all(bool(result["all_finite"]) for result in evaluations.values())
    if not all_finite:
        raise RuntimeError("EqMotion smoke evaluation produced non-finite metrics")

    run_dir = (ROOT / args.run_dir).resolve()
    checkpoint = run_dir / "checkpoint_after_smoke.pt"
    if checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite {checkpoint}")
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "seed": args.seed,
            "optimizer_steps": len(losses),
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "evidence_class": "deterministic engineering smoke only",
        },
        checkpoint,
    )
    elapsed = time.perf_counter() - started
    payload: dict[str, object] = {
        "format_version": 1,
        "status": "smoke_complete",
        "evidence_class": "deterministic engineering smoke only; not publication evidence",
        "protocol": {
            "path": PROTOCOL_PATH.relative_to(ROOT).as_posix(),
            "sha256": sha256(PROTOCOL_PATH),
        },
        "adapter_source": {
            "model_path": "modern_baseline/eqmotion_aviation.py",
            "model_sha256": sha256(Path(__file__).with_name("eqmotion_aviation.py")),
            "runner_path": "modern_baseline/run_eqmotion_smoke.py",
            "runner_sha256": sha256(Path(__file__)),
        },
        "official_source": source_receipt(),
        "environment": {
            "python_torch": torch.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "tf32": False,
        },
        "grid": protocol["grid"],
        "training": {
            "optimizer_steps": len(losses),
            "losses": losses,
            "finite_nonzero_gradient": finite_nonzero_gradient,
            "batch_shapes": train_shapes,
            "data_receipt": train_data.receipt(),
        },
        "evaluation": evaluations,
        "checkpoint": {
            "path": checkpoint.relative_to(ROOT).as_posix(),
            "sha256": sha256(checkpoint),
            "state_fingerprint": state_fingerprint(model.state_dict()),
        },
        "resources": {
            "elapsed_seconds": elapsed,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0,
        },
        "acceptance": {
            "data_batches_loaded": True,
            "forward_shape_valid": all(
                shape["predictions"][-3:] == [5, 24, 3]
                for shape in train_shapes
            ),
            "backward_finite_nonzero": finite_nonzero_gradient,
            "finite_metrics": all_finite,
            "passed": finite_nonzero_gradient and all_finite,
        },
        "limitations": protocol["limitations"],
        "reproduction_command": (
            "python -m modern_baseline.run_eqmotion_smoke "
            f"--device {args.device} --train-scenes {args.train_scenes} "
            f"--eval-scenes {args.eval_scenes} --max-files {args.max_files} "
            f"--batch-size {args.batch_size} --train-batches {args.train_batches} "
            f"--seed {args.seed} --run-dir {args.run_dir.as_posix()} "
            f"--output {args.output.as_posix()}"
        ),
    }
    write_new_json(ROOT / args.output, payload)
    print(json.dumps(payload["acceptance"], indent=2))
    print(f"artifact={ROOT / args.output}")


if __name__ == "__main__":
    main()
