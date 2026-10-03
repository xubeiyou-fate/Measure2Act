"""Evaluate native K=3/K=7 E9 models zero-shot on an external Tartan view."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import DataLoader, Subset

from experiments.energy_predict_optimize.evaluation import (
    RankingMetricAccumulator,
    compute_batch_metrics,
)
from model.utils import TrajectoryDataset, seed_worker, seq_collate

from .evaluate_e9 import TOP_M, _load_models, e9_probability_arms
from .operator import DEFAULT_ADE_SCALE, pairwise_trajectory_distance, support_cost


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("e9_tartan_external_protocol_v1.json")
NATIVE_MODES = (3, 7)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"E9 external evaluator refuses to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _dataset_fingerprint(dataset_dir: Path) -> dict[str, object]:
    paths = sorted(path for path in dataset_dir.iterdir() if path.is_file())
    if not paths:
        raise RuntimeError(f"external dataset directory has no files: {dataset_dir}")
    digest = hashlib.sha256()
    total_bytes = 0
    for path in paths:
        content_hash = _sha256(path)
        total_bytes += path.stat().st_size
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_hash.encode("ascii"))
        digest.update(b"\n")
    return {
        "files": len(paths),
        "bytes": total_bytes,
        "ordered_name_content_sha256": digest.hexdigest(),
    }


def _limited_dataset(dataset: TrajectoryDataset, maximum: int | None):
    if maximum is None or maximum >= len(dataset):
        return dataset
    if maximum < 1:
        raise ValueError("--max-scenes must be positive")
    indices = (
        torch.linspace(0, len(dataset) - 1, maximum)
        .round()
        .long()
        .unique()
        .tolist()
    )
    return Subset(dataset, indices)


def _assert_native_cardinality(
    source_support: torch.Tensor,
    target_support: torch.Tensor,
    modes: int,
) -> None:
    if modes not in NATIVE_MODES:
        raise ValueError("external E9 evaluation only registers native K=3 and K=7")
    if source_support.ndim != 4 or target_support.ndim != 4:
        raise RuntimeError("native E9 support tensors must have shape [B,K,T,3]")
    if source_support.shape[1] != modes or target_support.shape[1] != modes:
        raise RuntimeError(
            "checkpoint output cardinality differs from registered native cardinality; "
            "truncating K=5 or copying modes is prohibited"
        )


@torch.inference_mode()
def run(
    *,
    dataset_dir: Path,
    dataset_name: str,
    modes: int,
    fold: int,
    device: torch.device,
    workers: int,
    batch_size: int,
    max_scenes: int | None,
    delimiter: str = ",",
    protocol_path: Path = PROTOCOL,
) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if modes not in [int(value) for value in protocol["registered_native_modes"]]:
        raise ValueError("cardinality lies outside the frozen external E9 protocol")
    if fold not in [int(value) for value in protocol["registered_folds"]]:
        raise ValueError("fold lies outside the frozen external E9 protocol")
    if int(protocol["seed"]) != 42:
        raise RuntimeError("external E9 protocol must remain fixed to seed 42")
    if delimiter not in protocol["allowed_delimiters"]:
        raise ValueError("delimiter lies outside the frozen external E9 protocol")
    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(dataset_dir)
    fingerprint = _dataset_fingerprint(dataset_dir)

    load_started = time.perf_counter()
    dataset = TrajectoryDataset(
        dataset_dir.as_posix(),
        obs_len=16,
        obs_steps=1,
        pred_len=120,
        pred_step=5,
        delim=delimiter,
    )
    dataset_load_seconds = time.perf_counter() - load_started
    evaluation_dataset = _limited_dataset(dataset, max_scenes)
    loader_options = {
        "dataset": evaluation_dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": seq_collate,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }
    if workers > 0:
        loader_options.update({"persistent_workers": True, "prefetch_factor": 4})
    data_loader = DataLoader(**loader_options)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    source, energy, checkpoints = _load_models(modes, fold, device, batch_size)
    probability_names = (
        "mabpt_exact",
        *(f"mabpt_top{top_m}" for top_m in TOP_M[modes]),
        "mabpt_sinkhorn",
    )
    states = {
        name: RankingMetricAccumulator()
        for name in ("ascent_native", "target_energy_native", *probability_names)
    }
    diagnostic_sums: defaultdict[str, float] = defaultdict(float)
    diagnostic_count = 0
    batches = 0
    started = time.perf_counter()
    for data in data_loader:
        data = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in data.items()
        }
        truth = data["pred_traj"].transpose(1, 0).to(torch.float64)
        source_support, source_logits, _ = source(data)
        source_probability = source_logits.softmax(dim=1)
        target_support, target_probability, target_decision, auxiliary = energy(data)
        _assert_native_cardinality(source_support, target_support, modes)
        cross = support_cost(source_support, target_support)
        pairwise = pairwise_trajectory_distance(target_support) / DEFAULT_ADE_SCALE
        predicted_risk = auxiliary[
            "centered_predicted_normalized_ade_risk"
        ].to(torch.float64)
        probability_arms, diagnostics = e9_probability_arms(
            source_probability,
            cross,
            predicted_risk,
            pairwise,
            modes=modes,
        )
        measures = {
            "ascent_native": (
                source_support,
                source_probability,
                source_logits.argmax(dim=1),
            ),
            "target_energy_native": (
                target_support,
                target_probability,
                target_decision,
            ),
            **{
                name: (target_support, probability, target_decision)
                for name, probability in probability_arms.items()
            },
        }
        for name, (support, probability, decision) in measures.items():
            states[name].update(
                compute_batch_metrics(
                    support.to(torch.float64),
                    probability.to(torch.float64),
                    decision,
                    truth,
                )
            )
        for name, value in diagnostics.items():
            diagnostic_sums[name] += float(value.sum().cpu())
        diagnostic_count += int(truth.shape[0])
        batches += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started
    summaries = {name: state.summarize() for name, state in states.items()}
    actors = int(summaries["mabpt_exact"]["agents"])
    for name in probability_names[1:]:
        for metric in ("top1_ade", "top1_fde", "minade", "minfde"):
            if not math.isclose(
                summaries[name][metric],
                summaries["mabpt_exact"][metric],
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(f"probability arm changed native geometry: {name}/{metric}")
    return {
        "format_version": 1,
        "experiment_id": "E9_Tartan_external_native_cardinality",
        "evidence_class": protocol["evidence_class"],
        "dataset": dataset_name,
        "modes": modes,
        "fold": fold,
        "seed": 42,
        "scenes": len(evaluation_dataset),
        "actors": actors,
        "arms": summaries,
        "diagnostics": {
            name: value / diagnostic_count for name, value in diagnostic_sums.items()
        },
        "inputs": {
            "dataset_dir": dataset_dir.as_posix(),
            "dataset_fingerprint": fingerprint,
            "protocol": protocol_path.resolve().relative_to(ROOT).as_posix(),
            "protocol_sha256": _sha256(protocol_path),
            "checkpoints": checkpoints,
        },
        "integrity": {
            "retrospective_external_evaluation": True,
            "zero_shot_external_evaluation": True,
            "external_finetuning_or_calibration": False,
            "native_cardinality": True,
            "k5_truncation_or_mode_copy": False,
            "target_in_probability_forward": False,
            "external_result_based_selection": False,
            "fresh_prospective_or_hidden_test": False,
        },
        "runtime": {
            "device": str(device),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "dataset_load_seconds": dataset_load_seconds,
            "inference_seconds": elapsed_seconds,
            "batches": batches,
            "actors_per_second": actors / max(elapsed_seconds, 1e-12),
            "peak_allocated_gpu_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        },
        "claim_boundary": protocol["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--modes", type=int, choices=NATIVE_MODES, required=True)
    parser.add_argument("--fold", type=int, choices=(1, 2), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size is None:
        args.batch_size = {3: 512, 7: 32}[args.modes]
    if args.smoke and args.max_scenes is None:
        args.max_scenes = 8
    result = run(
        dataset_dir=args.dataset_dir,
        dataset_name=args.dataset_name,
        modes=args.modes,
        fold=args.fold,
        device=torch.device(args.device),
        workers=args.workers,
        batch_size=args.batch_size,
        max_scenes=args.max_scenes,
        delimiter=args.delimiter,
        protocol_path=args.protocol.resolve(),
    )
    _atomic_json(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "output": args.output.resolve().as_posix(),
                "dataset": result["dataset"],
                "modes": result["modes"],
                "fold": result["fold"],
                "scenes": result["scenes"],
                "actors": result["actors"],
                "energy": {
                    name: arm["energy_score"]
                    for name, arm in result["arms"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
