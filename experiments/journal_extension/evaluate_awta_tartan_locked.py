"""Evaluate frozen aWTA Tartan checkpoints after the ten-checkpoint receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch

from experiments.metric_exact.evaluation import evaluate
from experiments.metric_exact.model import build_model
from mabpt import train_tartan_retrain as base

from .train_awta_tartan import PROTOCOL, ROOT, atomic_json, expected_paths, load_protocol


RECEIPT = ROOT / "artifacts/journal_extension_20260814/awta_tartan_evaluation_receipt_v1.json"
LOCKED_ROOT = ROOT / "artifacts/journal_extension_20260814/awta_tartan/test"
SEEDS = (42, 7, 123, 2024, 2026)
AIRPORTS = ("KAGC", "KBTP")


def expected_output(airport: str, seed: int) -> Path:
    return LOCKED_ROOT / f"{airport}_seed{seed}_test_v1.json"


def verify_receipt() -> dict[str, Any]:
    if not RECEIPT.is_file():
        raise FileNotFoundError(RECEIPT)
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    if receipt.get("awta_test_inference_completed_before_freeze") is not False:
        raise RuntimeError("aWTA receipt does not precede test inference")
    if int(receipt.get("formal_checkpoint_count", -1)) != 10:
        raise RuntimeError("aWTA receipt does not bind ten checkpoints")
    for relative, expected in receipt["files"].items():
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size != int(expected["bytes"]) or base.sha256(path) != expected["sha256"]:
            raise RuntimeError(f"aWTA receipt mismatch: {relative}")
    return receipt


def checkpoint(airport: str, seed: int) -> tuple[Path, dict[str, Any]]:
    run_dir, result_path = expected_paths(airport, seed)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    checkpoint_path = run_dir / "last.pt"
    if (
        result.get("formal") is not True
        or result.get("airport") != airport
        or int(result.get("seed", -1)) != seed
        or int(result.get("checkpoint", {}).get("epoch", -1)) != 20
        or result.get("integrity", {}).get("locked_test_used") is not False
        or result.get("integrity", {}).get("fixed_final_epoch") is not True
    ):
        raise RuntimeError(f"aWTA formal result identity mismatch: {result_path}")
    if (ROOT / result["checkpoint"]["path"]).resolve() != checkpoint_path.resolve():
        raise RuntimeError("aWTA checkpoint path mismatch")
    if result["checkpoint"]["sha256"] != base.sha256(checkpoint_path):
        raise RuntimeError("aWTA checkpoint hash mismatch")
    return checkpoint_path, result


@torch.inference_mode()
def run(
    *,
    airport: str,
    seed: int,
    device: torch.device,
    batch_size: int,
    workers: int,
    authorized: bool,
) -> dict[str, Any]:
    if not authorized:
        raise RuntimeError("aWTA retrospective test requires explicit authorization")
    if airport not in AIRPORTS or seed not in SEEDS:
        raise ValueError("unregistered aWTA test cell")
    receipt = verify_receipt()
    protocol = load_protocol()
    parent_path = ROOT / protocol["inherits"]["tartan_protocol"]
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    checkpoint_path, development_result = checkpoint(airport, seed)

    # The test path is resolved only after authorization and complete receipt checks.
    test_dataset, test_dates, test_index = base._dataset(parent, airport, "test")
    test_loader = base._loader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        workers=workers,
    )
    tail_threshold = float(receipt["tail_threshold_km"][airport])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build_model("B0_signed_coupled", batch_size=batch_size).to(device)
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if (
        saved.get("model") != "ASCENT-aWTA"
        or saved.get("airport") != airport
        or int(saved.get("seed", -1)) != seed
        or int(saved.get("epoch", -1)) != 20
        or saved.get("protocol_sha256") != base.sha256(PROTOCOL)
        or saved.get("locked_test_used") is not False
    ):
        raise RuntimeError("aWTA checkpoint metadata mismatch")
    model.load_state_dict(saved["model_state_dict"], strict=True)
    started = time.perf_counter()
    metrics = evaluate(
        model,
        test_loader,
        device,
        scene_dates=test_dates,
        tail_threshold=tail_threshold,
    )
    elapsed = time.perf_counter() - started
    return {
        "format_version": 1,
        "experiment_id": f"aWTA_Tartan_{airport}_seed{seed}_test",
        "evidence_class": "internally_locked_retrospective_test",
        "airport": airport,
        "regime": "target_only",
        "seed": seed,
        "test_scenes": len(test_dataset),
        "test_metrics": metrics,
        "tail_threshold_km": tail_threshold,
        "inputs": {
            "checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
            "checkpoint_sha256": base.sha256(checkpoint_path),
            "development_result": expected_paths(airport, seed)[1].relative_to(ROOT).as_posix(),
            "development_result_sha256": base.sha256(expected_paths(airport, seed)[1]),
            "test_index": test_index.relative_to(ROOT).as_posix(),
            "test_index_sha256": base.sha256(test_index),
            "tail_threshold_source": RECEIPT.relative_to(ROOT).as_posix(),
            "receipt": RECEIPT.relative_to(ROOT).as_posix(),
            "receipt_sha256": base.sha256(RECEIPT),
            "receipt_file_count": len(receipt["files"]),
        },
        "runtime": {
            "device": str(device),
            "batch_size": batch_size,
            "workers": workers,
            "elapsed_seconds": elapsed,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        },
        "integrity": {
            "test_constructed_after_all_gates": True,
            "partial_test": False,
            "test_used_for_training_or_selection": False,
            "fixed_final_epoch": True,
            "only_training_assignment_changed": True,
        },
        "claim_boundary": protocol["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airport", choices=AIRPORTS, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--authorize-retrospective-test", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() != expected_output(args.airport, args.seed).resolve():
        raise ValueError(f"aWTA locked output path is frozen: {expected_output(args.airport, args.seed)}")
    result = run(
        airport=args.airport,
        seed=args.seed,
        device=torch.device(args.device),
        batch_size=args.batch_size,
        workers=args.workers,
        authorized=args.authorize_retrospective_test,
    )
    atomic_json(args.output.resolve(), result)
    print(json.dumps({"output": args.output.resolve().as_posix(), "metrics": result["test_metrics"]["overall"]}, indent=2))


if __name__ == "__main__":
    main()
