"""E9 operator scaling and approximation benchmark for K=3,5,7."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable

import torch

from .operator import exact_gibbs_transport, sinkhorn_transport, top_m_gibbs_transport


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("protocol.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"MABPT refuses to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(
    operation: Callable[[], dict[str, torch.Tensor]],
    *,
    device: torch.device,
    batch: int,
    warmup: int,
    repeats: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    for _ in range(warmup):
        result = operation()
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = operation()
        _synchronize(device)
        samples.append(time.perf_counter() - started)
    samples.sort()
    return result, {
        "median_batch_milliseconds": 1000.0 * samples[len(samples) // 2],
        "median_actor_microseconds": 1e6 * samples[len(samples) // 2] / batch,
        "minimum_batch_milliseconds": 1000.0 * samples[0],
        "peak_allocated_gpu_bytes": (
            float(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0.0
        ),
    }


def run(
    *, device: torch.device, warmup: int, repeats: int
) -> dict[str, object]:
    if device.type == "cuda":
        torch.cuda.set_device(device)
    batches = {3: 1024, 5: 256, 7: 32}
    mode_results = {}
    for modes, batch in batches.items():
        generator = torch.Generator(device=device).manual_seed(9000 + modes)
        probabilities = torch.rand(
            batch, modes, generator=generator, dtype=torch.float64, device=device
        )
        probabilities /= probabilities.sum(dim=1, keepdim=True)
        cost = 8.0 * torch.rand(
            batch, modes, modes, generator=generator, dtype=torch.float64, device=device
        )
        exact, exact_timing = _timed(
            lambda: exact_gibbs_transport(
                probabilities, cost, mass_weighted=True
            ),
            device=device,
            batch=batch,
            warmup=warmup,
            repeats=repeats,
        )
        factorial = math.factorial(modes)
        top_m_results = {}
        for top_m in sorted({min(value, factorial) for value in (8, 32, 128, 512)}):
            approximate, timing = _timed(
                lambda top_m=top_m: top_m_gibbs_transport(
                    probabilities,
                    cost,
                    top_m=top_m,
                    mass_weighted=True,
                ),
                device=device,
                batch=batch,
                warmup=warmup,
                repeats=repeats,
            )
            l1 = (
                approximate["transported"] - exact["transported"]
            ).abs().sum(dim=1)
            top_m_results[str(top_m)] = {
                **timing,
                "mean_L1_probability_error": float(l1.mean().cpu()),
                "max_L1_probability_error": float(l1.max().cpu()),
                "mean_retained_posterior_mass": float(
                    approximate["retained_posterior_mass"].mean().cpu()
                ),
                "backend": "enumerate_then_top_M",
            }
        sinkhorn, sinkhorn_timing = _timed(
            lambda: sinkhorn_transport(probabilities, cost),
            device=device,
            batch=batch,
            warmup=warmup,
            repeats=repeats,
        )
        sinkhorn_l1 = (
            sinkhorn["transported"] - exact["transported"]
        ).abs().sum(dim=1)
        mode_results[str(modes)] = {
            "batch": batch,
            "permutations": factorial,
            "exact": exact_timing,
            "top_M": top_m_results,
            "sinkhorn": {
                **sinkhorn_timing,
                "mean_L1_probability_error_vs_exact": float(sinkhorn_l1.mean().cpu()),
                "max_L1_probability_error_vs_exact": float(sinkhorn_l1.max().cpu()),
                "mean_row_error": float(sinkhorn["row_error"].mean().cpu()),
                "max_row_error": float(sinkhorn["row_error"].max().cpu()),
                "mean_column_error": float(sinkhorn["column_error"].mean().cpu()),
                "iterations": 64,
            },
        }
    return {
        "format_version": 1,
        "model": "MABPT",
        "experiment_id": "E9",
        "evidence_class": "synthetic_operator_benchmark",
        "protocol_sha256": _sha256(PROTOCOL),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "torch": torch.__version__,
        "warmup": warmup,
        "repeats": repeats,
        "modes": mode_results,
        "limitations": [
            "Top-M currently enumerates K! assignments before truncation; it measures approximation error but is not a Murty runtime claim.",
            "Synthetic cost tensors isolate operator scaling and do not replace K-specific model training results."
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/mabpt/e9_scaling_v1.json"
    )
    args = parser.parse_args()
    result = run(
        device=torch.device(args.device), warmup=args.warmup, repeats=args.repeats
    )
    _atomic_json(args.output, result)
    print(json.dumps({"output": str(args.output), "modes": result["modes"]}, indent=2))


if __name__ == "__main__":
    main()
