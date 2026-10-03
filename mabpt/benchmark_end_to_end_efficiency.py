"""Benchmark ASCENT and the complete two-network MABPT inference system."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from mabpt.operator import (
    DEFAULT_ADE_SCALE,
    energy_kl_projection,
    exact_gibbs_transport,
    pairwise_trajectory_distance,
    support_cost,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("end_to_end_efficiency_protocol_v1.json")
CARDINALITIES = (3, 5, 7)
K5_ROOT = ROOT / "runs/partc_tartan_retrain_20260812/KAGC/target_only/p100"
E9_ROOT = ROOT / "runs/mabpt/e9"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    """Atomically create a JSON artifact and never replace an existing file."""
    if path.exists():
        raise FileExistsError(f"efficiency benchmark refuses to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary benchmark output already exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def checkpoint_paths(modes: int) -> dict[str, Path]:
    """Return native checkpoints without truncating or replicating modes."""
    if modes not in CARDINALITIES:
        raise ValueError("cardinality must be one of K=3, K=5, K=7")
    if modes == 5:
        return {
            "source": K5_ROOT / "ascent_seed42_formal/last.pt",
            "decision": K5_ROOT / "decision_support_seed42_formal/last.pt",
            "energy": K5_ROOT / "predicted_risk_seed42_formal/last.pt",
        }
    return {
        stage: E9_ROOT / f"K{modes}_{stage}_fold1_seed42_formal/epoch20.pt"
        for stage in ("source", "decision", "energy")
    }


def _validate_paths(paths: dict[str, Path]) -> None:
    for stage in ("source", "decision", "energy"):
        path = paths[stage]
        if not path.is_file():
            raise FileNotFoundError(f"missing formal {stage} checkpoint: {path}")


def _validate_k5_summary(path: Path, stage: str) -> dict[str, Any]:
    summary_path = path.with_name("training_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_stage = "predicted_risk" if stage == "energy" else (
        "ascent" if stage == "source" else "decision_support"
    )
    identity = (
        summary.get("formal") is True,
        summary.get("airport") == "KAGC",
        summary.get("regime") == "target_only",
        summary.get("stage") == expected_stage,
        int(summary.get("seed", -1)) == 42,
        float(summary.get("fraction", -1.0)) == 1.0,
        int(summary.get("fixed_final_epoch", -1)) == 20,
        summary.get("integrity", {}).get("locked_test_used") is False,
    )
    if not all(identity):
        raise RuntimeError(f"K=5 formal summary identity mismatch: {summary_path}")
    if summary.get("checkpoint_sha256") != _sha256(path):
        raise RuntimeError(f"K=5 checkpoint hash differs from its summary: {path}")
    return summary


def _validate_e9_checkpoint(path: Path, stage: str, modes: int) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("stage") != stage
        or int(checkpoint.get("modes", -1)) != modes
        or int(checkpoint.get("fold", -1)) != 1
        or int(checkpoint.get("seed", -1)) != 42
        or int(checkpoint.get("epoch", -1)) != 20
    ):
        raise RuntimeError(f"E9 formal checkpoint identity mismatch: {path}")
    return checkpoint


def load_models(modes: int, *, batch_size: int) -> tuple[nn.Module, nn.Module]:
    """Load the registered native-K source and complete target-risk networks."""
    paths = checkpoint_paths(modes)
    _validate_paths(paths)
    if modes == 5:
        from mabpt.partc_seed_evaluate import _load_model_pair

        for stage, path in paths.items():
            _validate_k5_summary(path, stage)
        source, target = _load_model_pair(
            source_checkpoint=paths["source"],
            target_checkpoint=paths["energy"],
            device=torch.device("cpu"),
            batch_size=batch_size,
        )
    else:
        # This loader verifies protocol hashes and constructs the native K model.
        from mabpt.evaluate_e9 import _load_models

        for stage, path in paths.items():
            _validate_e9_checkpoint(path, stage, modes)
        source, target, loaded = _load_models(
            modes, fold=1, device=torch.device("cpu"), batch_size=batch_size
        )
        if Path(loaded["source"]) != paths["source"].relative_to(ROOT):
            raise RuntimeError("E9 source loader selected an unexpected checkpoint")
        if Path(loaded["energy"]) != paths["energy"].relative_to(ROOT):
            raise RuntimeError("E9 target loader selected an unexpected checkpoint")
    if int(getattr(source, "k", -1)) != modes:
        raise RuntimeError("source checkpoint is not native to the requested K")
    target_modes = getattr(target, "modes", getattr(target.backbone, "k", -1))
    if int(target_modes) != modes:
        raise RuntimeError("target checkpoint is not native to the requested K")
    return source.eval(), target.eval()


def synthetic_data(batch_size: int, *, seed: int = 42) -> dict[str, torch.Tensor]:
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed + batch_size)
    velocity = torch.randn(batch_size, 3, generator=generator) * torch.tensor(
        [0.004, 0.004, 0.0002]
    )
    noise = torch.randn(16, batch_size, 3, generator=generator) * torch.tensor(
        [0.0001, 0.0001, 0.00001]
    )
    observation = (velocity[None] + noise).cumsum(dim=0)
    return {"obs_traj": observation, "adj": torch.arange(batch_size)}


def move_data(data: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in data.items()}


def ascent_forward(source: nn.Module, data: dict[str, torch.Tensor]) -> torch.Tensor:
    support, logits, _ = source(data)
    if support.shape[:2] != logits.shape:
        raise RuntimeError("ASCENT support and logits disagree")
    return support


def full_mabpt_forward(
    source: nn.Module,
    target: nn.Module,
    data: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Execute both networks, exact transport, and registered projection."""
    source_support, source_logits, _ = source(data)
    target_support, _, _, auxiliary = target(data)
    if source_support.shape != target_support.shape:
        raise RuntimeError("source and target supports do not share native cardinality")
    source_probability = source_logits.softmax(dim=1)
    prior = exact_gibbs_transport(
        source_probability.to(torch.float64),
        support_cost(source_support, target_support),
        mass_weighted=True,
    )["transported"]
    probability, _ = energy_kl_projection(
        prior,
        auxiliary["centered_predicted_normalized_ade_risk"].to(torch.float64),
        pairwise_trajectory_distance(target_support) / DEFAULT_ADE_SCALE,
    )
    return probability


def parameter_counts(source: nn.Module, target: nn.Module) -> dict[str, object]:
    source_count = sum(parameter.numel() for parameter in source.parameters())
    target_count = sum(parameter.numel() for parameter in target.parameters())
    backbone_count = sum(parameter.numel() for parameter in target.backbone.parameters())
    risk_count = sum(
        parameter.numel() for parameter in target.energy_cost_operator.parameters()
    )
    if target_count != backbone_count + risk_count:
        raise RuntimeError("target parameter decomposition is incomplete")
    return {
        "ascent": {
            "source_network": source_count,
            "complete_system": source_count,
        },
        "mabpt": {
            "source_network": source_count,
            "target_network_total": target_count,
            "target_support_decision_backbone": backbone_count,
            "target_predicted_risk_operator": risk_count,
            "exact_transport_and_projection": 0,
            "complete_two_network_system": source_count + target_count,
            "relative_increase_vs_ascent": (source_count + target_count) / source_count - 1.0,
        },
    }


def checkpoint_accounting(modes: int) -> dict[str, object]:
    paths = checkpoint_paths(modes)
    _validate_paths(paths)
    records = {
        stage: {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for stage, path in paths.items()
    }
    return {
        "files": records,
        "ascent_runtime_bytes": records["source"]["bytes"],
        "mabpt_runtime_source_plus_target_bytes": (
            records["source"]["bytes"] + records["energy"]["bytes"]
        ),
        "mabpt_training_pipeline_three_checkpoint_bytes": sum(
            int(record["bytes"]) for record in records.values()
        ),
        "decision_checkpoint_is_not_an_additional_runtime_network": True,
    }


def _elapsed_from_history(history: list[dict[str, Any]]) -> float:
    values = [float(epoch["elapsed_seconds"]) for epoch in history]
    if not values or not all(math.isfinite(value) and value >= 0 for value in values):
        raise RuntimeError("training history has invalid elapsed time")
    return float(sum(values))


def training_time_summary(modes: int) -> dict[str, object]:
    paths = checkpoint_paths(modes)
    stages: dict[str, dict[str, object]] = {}
    for stage, path in paths.items():
        if modes == 5:
            payload = _validate_k5_summary(path, stage)
            history_seconds = _elapsed_from_history(payload["history"])
            process_seconds = float(payload["runtime"]["elapsed_seconds"])
            source = path.with_name("training_summary.json")
        else:
            payload = _validate_e9_checkpoint(path, stage, modes)
            history_seconds = _elapsed_from_history(payload["history"])
            process_seconds = None
            source = path
        stages[stage] = {
            "epoch_elapsed_seconds_sum": history_seconds,
            "recorded_process_elapsed_seconds": process_seconds,
            "source": source.relative_to(ROOT).as_posix(),
        }
    field = "recorded_process_elapsed_seconds" if modes == 5 else "epoch_elapsed_seconds_sum"
    ascent = float(stages["source"][field])
    mabpt = float(sum(float(stages[stage][field]) for stage in stages))
    return {
        "stages": stages,
        "comparison_field": field,
        "ascent_training_seconds": ascent,
        "mabpt_complete_three_stage_training_seconds": mabpt,
        "mabpt_to_ascent_ratio": mabpt / ascent,
        "historical_summary_not_new_training": True,
    }


def timing_statistics(elapsed_ms: list[float], batch_size: int) -> dict[str, float]:
    if not elapsed_ms or batch_size < 1:
        raise ValueError("timing samples and a positive batch size are required")
    values = np.asarray(elapsed_ms, dtype=np.float64)
    median = float(np.median(values))
    p95 = float(np.quantile(values, 0.95))
    return {
        "median_ms_batch": median,
        "p95_ms_batch": p95,
        "median_ms_actor": median / batch_size,
        "p95_ms_actor": p95 / batch_size,
        "median_derived_actors_per_second": 1000.0 * batch_size / median,
    }


@torch.inference_mode()
def benchmark_operation(
    operation: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    for _ in range(warmup):
        output = operation()
        del output
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        resident = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
        output = operation()
        torch.cuda.synchronize(device)
        del output
        peak = int(torch.cuda.max_memory_allocated(device))
        elapsed_ms = []
        for _ in range(repeats):
            started = torch.cuda.Event(enable_timing=True)
            ended = torch.cuda.Event(enable_timing=True)
            started.record()
            output = operation()
            ended.record()
            torch.cuda.synchronize(device)
            elapsed_ms.append(float(started.elapsed_time(ended)))
            del output
    else:
        resident = 0
        peak = 0
        elapsed_ms = []
        for _ in range(repeats):
            started_at = time.perf_counter()
            output = operation()
            elapsed_ms.append((time.perf_counter() - started_at) * 1000.0)
            del output
    return {
        **timing_statistics(elapsed_ms, batch_size),
        "samples": repeats,
        "resident_cuda_allocated_bytes": resident if device.type == "cuda" else None,
        "peak_cuda_allocated_bytes": peak if device.type == "cuda" else None,
        "incremental_peak_cuda_allocated_bytes": (
            peak - resident if device.type == "cuda" else None
        ),
    }


def _release_cuda(*models: nn.Module) -> None:
    for model in models:
        model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def benchmark_cardinality(
    modes: int,
    *,
    device: torch.device,
    batch_sizes: list[int],
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    source, target = load_models(modes, batch_size=max(batch_sizes))
    counts = parameter_counts(source, target)
    checkpoints = checkpoint_accounting(modes)
    training = training_time_summary(modes)

    source.to(device).eval()
    ascent = {}
    for batch_size in batch_sizes:
        data = move_data(synthetic_data(batch_size), device)
        ascent[str(batch_size)] = benchmark_operation(
            lambda data=data: ascent_forward(source, data),
            device=device,
            batch_size=batch_size,
            warmup=warmup,
            repeats=repeats,
        )
    _release_cuda(source)

    source.to(device).eval()
    target.to(device).eval()
    mabpt = {}
    for batch_size in batch_sizes:
        data = move_data(synthetic_data(batch_size), device)
        probability = full_mabpt_forward(source, target, data)
        if probability.shape != (batch_size, modes):
            raise RuntimeError("full MABPT output is not native to requested cardinality")
        del probability
        mabpt[str(batch_size)] = benchmark_operation(
            lambda data=data: full_mabpt_forward(source, target, data),
            device=device,
            batch_size=batch_size,
            warmup=warmup,
            repeats=repeats,
        )
    _release_cuda(source, target)
    return {
        "native_cardinality": modes,
        "exact_permutations": math.factorial(modes),
        "checkpoint_cohort": (
            "KAGC_target_only_p100_seed42_formal"
            if modes == 5
            else "E9_fold1_seed42_formal"
        ),
        "parameters": counts,
        "checkpoints": checkpoints,
        "training_time": training,
        "inference": {"ascent": ascent, "complete_mabpt": mabpt},
    }


def run(
    *,
    device: torch.device,
    batch_sizes: list[int],
    warmup: int,
    repeats: int,
    cardinalities: list[int],
) -> dict[str, object]:
    if sorted(set(cardinalities)) != sorted(cardinalities) or any(
        modes not in CARDINALITIES for modes in cardinalities
    ):
        raise ValueError("cardinalities must be unique members of K=3,5,7")
    if not batch_sizes or any(batch < 1 for batch in batch_sizes):
        raise ValueError("batch sizes must be positive")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
    return {
        "schema_version": 1,
        "benchmark": "ASCENT_vs_complete_MABPT_end_to_end_inference",
        "evidence_class": "retrospective_system_efficiency",
        "protocol": {
            "path": PROTOCOL.relative_to(ROOT).as_posix(),
            "sha256": _sha256(PROTOCOL),
            "history_points": 16,
            "history_interval_seconds": 1,
            "future_points": 24,
            "future_interval_seconds": 5,
            "future_horizon_seconds": 120,
            "exact_enumeration_max_k": 7,
            "batch_sizes": batch_sizes,
            "warmup": warmup,
            "repeats": repeats,
            "locked_test_used": False,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
        },
        "cardinalities": {
            str(modes): benchmark_cardinality(
                modes,
                device=device,
                batch_sizes=batch_sizes,
                warmup=warmup,
                repeats=repeats,
            )
            for modes in cardinalities
        },
        "claim_boundary": [
            "MABPT parameter count is the sum of two independent networks, not only the target network.",
            "K=3 and K=7 use native formal models; K=5 outputs are never truncated or copied.",
            "K=5 and K=3/K=7 were trained on different formal cohorts, so K growth supports systems-scaling claims only.",
            "This benchmark uses no locked-test inference and makes no accuracy claim.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32, 128])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--cardinalities", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        device=torch.device(args.device),
        batch_sizes=args.batch_sizes,
        warmup=args.warmup,
        repeats=args.repeats,
        cardinalities=args.cardinalities,
    )
    output = args.output.resolve()
    atomic_json(output, result)
    print(json.dumps({"output": str(output), "cardinalities": list(result["cardinalities"])}, indent=2))


if __name__ == "__main__":
    main()
