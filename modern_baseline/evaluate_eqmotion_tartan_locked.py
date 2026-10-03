"""Evaluate a frozen target-only EqMotion checkpoint on one locked Tartan test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import DataLoader

from model.utils import TrajectoryDataset

from .eqmotion_aviation import (
    CachedTrajectorySceneDataset,
    EqMotionAviation,
    aviation_collate,
    valid_actor_tensors,
)
from .run_eqmotion_tartan_target import PUBLICATION_METRICS, atomic_json, load_protocol


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("eqmotion_tartan_target_protocol_v1.json")
EVALUATION_RECEIPT = (
    ROOT
    / "artifacts/partc_two_dataset_20260812/tartan_retrain_evaluation_frozen_receipt_v1.json"
)
SCENE_INDEX_SUMMARY = (
    ROOT
    / "artifacts/partc_two_dataset_20260812/target_domain_scene_index_v3/summary.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_evaluation_receipt() -> dict[str, object]:
    receipt = json.loads(EVALUATION_RECEIPT.read_text(encoding="utf-8"))
    if receipt.get("locked_test_model_inference_completed_before_freeze") is not False:
        raise RuntimeError("evaluation receipt does not represent unopened test")
    for relative, expected in receipt["files"].items():
        path = ROOT / relative
        if sha256(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise RuntimeError(f"evaluation receipt mismatch: {relative}")
    return receipt


def expected_test_scene_count(
    airport: str,
    protocol: dict[str, object],
    receipt: dict[str, object],
    *,
    summary_path: Path = SCENE_INDEX_SUMMARY,
) -> int:
    relative = summary_path.resolve().relative_to(ROOT.resolve()).as_posix()
    if relative not in receipt.get("files", {}):
        raise RuntimeError("evaluation receipt does not bind the scene-index summary")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("skip", -1)) != 5:
        raise RuntimeError("scene-index summary does not use the matched skip=5")
    data = protocol["data"]
    if summary.get("data_manifest_sha256") != data["manifest_sha256"]:
        raise RuntimeError("scene-index summary data manifest differs from protocol")
    entry = summary.get("datasets", {}).get(airport, {}).get("test", {})
    count = int(entry.get("scene_count", 0))
    if count < 1:
        raise RuntimeError(f"scene-index summary lacks a positive {airport} test count")
    return count


def _checkpoint(airport: str) -> tuple[Path, dict[str, object]]:
    result_path = (
        ROOT
        / "artifacts/partc_two_dataset_20260812/modern_baseline"
        / f"eqmotion_tartan_{airport}_target_only_seed42_formal.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result.get("formal") is not True
        or result.get("airport") != airport
        or result.get("regime") != "target_only"
        or int(result.get("seed", -1)) != 42
        or result.get("integrity", {}).get("locked_test_model_inference") is not False
    ):
        raise RuntimeError(f"EqMotion formal training result identity mismatch: {result_path}")
    checkpoint = ROOT / result["checkpoint"]["path"]
    if sha256(checkpoint) != result["checkpoint"]["sha256"]:
        raise RuntimeError("EqMotion checkpoint hash mismatch")
    return checkpoint, {
        "training_result": result_path.relative_to(ROOT).as_posix(),
        "training_result_sha256": sha256(result_path),
        "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": sha256(checkpoint),
    }


@torch.inference_mode()
def run(airport: str, device: torch.device, batch_size: int) -> dict[str, object]:
    receipt = verify_evaluation_receipt()
    protocol = load_protocol(PROTOCOL)
    if airport not in protocol["data"]["airports"]:
        raise ValueError("unregistered airport")
    expected_scenes = expected_test_scene_count(airport, protocol, receipt)
    checkpoint_path, checkpoint_receipt = _checkpoint(airport)
    # Test path resolution is intentionally below every authorization and hash gate.
    data_root = ROOT / protocol["data"]["root"]
    test_path = data_root / airport / "test"
    source = TrajectoryDataset(
        test_path.as_posix(),
        obs_len=16,
        obs_steps=1,
        pred_len=120,
        skip=5,
        pred_step=5,
        delim=protocol["data"]["delimiter"],
        cache_dir=ROOT / "dataset/_cache/partc_tartan_target_v4" / airport / "test",
    )
    dataset = CachedTrajectorySceneDataset(source)
    if len(dataset) != expected_scenes:
        raise RuntimeError(
            f"{airport} locked test scene count mismatch: "
            f"dataset={len(dataset)} scene_index={expected_scenes}"
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=aviation_collate,
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(True)
    model = EqMotionAviation(
        device=device,
        hidden_nf=int(protocol["model"]["hidden_nf"]),
        channels=int(protocol["model"]["channels"]),
        layers=int(protocol["model"]["layers"]),
        modes=int(protocol["grid"]["modes"]),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("airport") != airport or int(checkpoint.get("seed", -1)) != 42:
        raise RuntimeError("EqMotion checkpoint metadata mismatch")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    totals = {name: 0.0 for name in PUBLICATION_METRICS}
    actors = 0
    started = time.perf_counter()
    for batch in loader:
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        prediction = model(batch["history"], batch["num_valid"])
        prediction, truth = valid_actor_tensors(prediction, batch["future"], batch["valid"])
        displacement = torch.linalg.vector_norm(
            prediction.to(torch.float64) - truth.to(torch.float64)[:, None], dim=-1
        )
        ade = displacement.mean(dim=-1)
        minade = ade.min(dim=1).values
        minfde = displacement[..., -1].min(dim=1).values
        probability = torch.full((len(prediction), 5), 0.2, device=device, dtype=torch.float64)
        pairwise = torch.linalg.vector_norm(
            prediction.to(torch.float64)[:, :, None]
            - prediction.to(torch.float64)[:, None, :],
            dim=-1,
        ).mean(dim=-1)
        energy = (probability * ade).sum(dim=1) - 0.5 * torch.einsum(
            "bi,bij,bj->b", probability, pairwise, probability
        )
        values = {"minade": minade, "minfde": minfde, "energy_score": energy}
        for name in PUBLICATION_METRICS:
            totals[name] += float(values[name].sum().cpu())
        actors += len(prediction)
    elapsed = time.perf_counter() - started
    return {
        "format_version": 1,
        "experiment_id": "EqMotion_Tartan_target_only_locked_test_seed42",
        "airport": airport,
        "regime": "target_only",
        "seed": 42,
        "evidence_class": "locked_retrospective_test_single_pass",
        "scenes": len(dataset),
        "actors": actors,
        "metrics": {name: totals[name] / actors for name in PUBLICATION_METRICS},
        "efficiency": {
            "device": str(device),
            "elapsed_seconds": elapsed,
            "scenes_per_second": len(dataset) / max(elapsed, 1e-12),
            "actors_per_second": actors / max(elapsed, 1e-12),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        },
        "inputs": {
            **checkpoint_receipt,
            "evaluation_receipt": EVALUATION_RECEIPT.relative_to(ROOT).as_posix(),
            "evaluation_receipt_sha256": sha256(EVALUATION_RECEIPT),
            "evaluation_receipt_file_count": len(receipt["files"]),
            "scene_index_summary": SCENE_INDEX_SUMMARY.relative_to(ROOT).as_posix(),
            "scene_index_summary_sha256": sha256(SCENE_INDEX_SUMMARY),
            "expected_test_scenes": expected_scenes,
        },
        "integrity": {
            "test_constructed_after_gate": True,
            "trajectory_dataset_skip": 5,
            "scene_count_matches_frozen_index": True,
            "partial_test": False,
            "test_used_for_model_selection": False,
            "uniform_probability_measure": True,
            "learned_mode_ranking": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airport", choices=("KAGC", "KBTP"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--authorize-locked-test", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.authorize_locked_test:
        raise RuntimeError("locked test requires explicit --authorize-locked-test")
    result = run(args.airport, torch.device(args.device), args.batch_size)
    atomic_json(args.output.resolve(), result)
    print(json.dumps({"output": args.output.resolve().as_posix(), "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
