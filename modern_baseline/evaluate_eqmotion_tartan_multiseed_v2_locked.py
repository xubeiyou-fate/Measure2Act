"""Evaluate one new EqMotion seed after the five-seed locked-test receipt is frozen."""

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
from .run_eqmotion_tartan_target import PUBLICATION_METRICS, atomic_json
from .run_eqmotion_tartan_multiseed_v2 import (
    FORMAL_RESULT_ROOT,
    FORMAL_RUN_ROOT,
    NEW_SEEDS,
    PROTOCOL,
    ROOT,
    load_multiseed_protocol,
)


EVALUATION_RECEIPT = (
    ROOT
    / "artifacts/partc_two_dataset_20260812/modern_baseline"
    / "eqmotion_tartan_multiseed_v2_evaluation_receipt.json"
)
SCENE_INDEX_SUMMARY = (
    ROOT
    / "artifacts/partc_two_dataset_20260812/target_domain_scene_index_v3/summary.json"
)
LOCKED_RESULT_ROOT = FORMAL_RESULT_ROOT / "locked"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_output(airport: str, seed: int) -> Path:
    return LOCKED_RESULT_ROOT / f"{airport}_target_only_seed{seed}_locked_test_v2.json"


def verify_evaluation_receipt() -> dict[str, object]:
    receipt = json.loads(EVALUATION_RECEIPT.read_text(encoding="utf-8"))
    if receipt.get("seed42_locked_results_completed_before_freeze") is not True:
        raise RuntimeError("multi-seed receipt omits the inherited seed42 locked results")
    if receipt.get("new_seed_locked_test_model_inference_completed_before_freeze") is not False:
        raise RuntimeError("multi-seed receipt does not represent unopened new-seed locked tests")
    if tuple(receipt.get("registered_seeds", [])) != (42, *NEW_SEEDS):
        raise RuntimeError("multi-seed receipt registry mismatch")
    if tuple(receipt.get("new_locked_evaluation_seeds", [])) != NEW_SEEDS:
        raise RuntimeError("multi-seed receipt new-seed registry mismatch")
    if receipt.get("formal_checkpoint_count") != 10 or receipt.get("formal_result_count") != 10:
        raise RuntimeError("multi-seed receipt formal grid is incomplete")
    for relative, expected in receipt.get("files", {}).items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise RuntimeError(f"multi-seed evaluation receipt mismatch: {relative}")
    if receipt.get("protocol_sha256") != sha256(PROTOCOL):
        raise RuntimeError("multi-seed evaluation receipt protocol mismatch")
    return receipt


def expected_test_scene_count(
    airport: str,
    protocol: dict[str, object],
    receipt: dict[str, object],
    *,
    summary_path: Path = SCENE_INDEX_SUMMARY,
) -> int:
    relative = summary_path.resolve().relative_to(ROOT.resolve()).as_posix()
    bound = receipt.get("files", {}).get(relative)
    if bound is None:
        raise RuntimeError("multi-seed receipt does not bind the scene-index summary")
    if sha256(summary_path) != bound["sha256"]:
        raise RuntimeError("scene-index summary differs from multi-seed receipt")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("skip", -1)) != 5:
        raise RuntimeError("scene-index summary does not use the matched skip=5")
    if summary.get("data_manifest_sha256") != protocol["data"]["manifest_sha256"]:
        raise RuntimeError("scene-index summary data manifest differs from protocol")
    count = int(summary.get("datasets", {}).get(airport, {}).get("test", {}).get("scene_count", 0))
    if count < 1:
        raise RuntimeError(f"scene-index summary lacks a positive {airport} test count")
    return count


def checkpoint(airport: str, seed: int) -> tuple[Path, dict[str, object]]:
    result_path = FORMAL_RESULT_ROOT / f"{airport}_target_only_seed{seed}_formal.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result.get("formal") is not True
        or result.get("airport") != airport
        or result.get("regime") != "target_only"
        or int(result.get("seed", -1)) != seed
        or int(result.get("checkpoint", {}).get("epoch", -1)) != 20
        or result.get("integrity", {}).get("fixed_final_epoch") is not True
        or result.get("integrity", {}).get("development_checkpoint_selection") is not False
        or result.get("integrity", {}).get("locked_test_model_inference") is not False
    ):
        raise RuntimeError(f"EqMotion multi-seed formal result identity mismatch: {result_path}")
    expansion = result.get("multi_seed_expansion", {})
    if (
        expansion.get("new_training_seed") != seed
        or expansion.get("seed42_reused_without_rerun") is not True
        or expansion.get("protocol_sha256") != sha256(PROTOCOL)
    ):
        raise RuntimeError(f"EqMotion multi-seed provenance mismatch: {result_path}")
    checkpoint_path = FORMAL_RUN_ROOT / airport / f"seed{seed}_formal/last.pt"
    recorded_checkpoint = ROOT / result["checkpoint"]["path"]
    if recorded_checkpoint.resolve() != checkpoint_path.resolve():
        raise RuntimeError("EqMotion multi-seed checkpoint path mismatch")
    if sha256(checkpoint_path) != result["checkpoint"]["sha256"]:
        raise RuntimeError("EqMotion multi-seed checkpoint hash mismatch")
    return checkpoint_path, {
        "training_result": result_path.relative_to(ROOT).as_posix(),
        "training_result_sha256": sha256(result_path),
        "checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": sha256(checkpoint_path),
    }


@torch.inference_mode()
def run(
    airport: str,
    seed: int,
    device: torch.device,
    batch_size: int,
    *,
    authorized: bool = False,
) -> dict[str, object]:
    if not authorized:
        raise RuntimeError("locked test requires explicit authorization")
    if seed == 42:
        raise ValueError("seed42 locked result already exists and must not be rerun")
    if seed not in NEW_SEEDS:
        raise ValueError("seed is outside the four frozen expansion seeds")
    receipt = verify_evaluation_receipt()
    protocol = load_multiseed_protocol()
    if airport not in protocol["data"]["airports"]:
        raise ValueError("unregistered airport")
    expected_scenes = expected_test_scene_count(airport, protocol, receipt)
    checkpoint_path, checkpoint_receipt = checkpoint(airport, seed)

    # No test path is resolved or dataset constructed before all authorization,
    # receipt, identity, scene-count metadata, and checkpoint hash gates pass.
    test_path = ROOT / protocol["data"]["root"] / airport / "test"
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
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = EqMotionAviation(
        device=device,
        hidden_nf=int(protocol["model"]["hidden_nf"]),
        channels=int(protocol["model"]["channels"]),
        layers=int(protocol["model"]["layers"]),
        modes=int(protocol["grid"]["modes"]),
    ).to(device)
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if (
        saved.get("airport") != airport
        or int(saved.get("seed", -1)) != seed
        or int(saved.get("epoch", -1)) != 20
        or saved.get("protocol_sha256") != sha256(PROTOCOL)
        or saved.get("manifest_sha256") != protocol["data"]["manifest_sha256"]
    ):
        raise RuntimeError("EqMotion multi-seed checkpoint metadata mismatch")
    model.load_state_dict(saved["model_state_dict"], strict=True)
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
        prediction = prediction.to(torch.float64)
        truth = truth.to(torch.float64)
        displacement = torch.linalg.vector_norm(prediction - truth[:, None], dim=-1)
        ade = displacement.mean(dim=-1)
        minade = ade.min(dim=1).values
        minfde = displacement[..., -1].min(dim=1).values
        probability = torch.full((len(prediction), 5), 0.2, device=device, dtype=torch.float64)
        pairwise = torch.linalg.vector_norm(
            prediction[:, :, None] - prediction[:, None, :], dim=-1
        ).mean(dim=-1)
        energy = (probability * ade).sum(dim=1) - 0.5 * torch.einsum(
            "bi,bij,bj->b", probability, pairwise, probability
        )
        values = {"minade": minade, "minfde": minfde, "energy_score": energy}
        for name in PUBLICATION_METRICS:
            totals[name] += float(values[name].sum().cpu())
        actors += len(prediction)
    if not actors:
        raise RuntimeError("locked test produced no valid actors")
    elapsed = time.perf_counter() - started
    return {
        "format_version": 2,
        "experiment_id": f"EqMotion_Tartan_target_only_locked_test_seed{seed}",
        "airport": airport,
        "regime": "target_only",
        "seed": seed,
        "evidence_class": "locked_retrospective_test_single_pass",
        "scenes": len(dataset),
        "actors": actors,
        "metrics": {name: totals[name] / actors for name in PUBLICATION_METRICS},
        "efficiency": {
            "device": str(device),
            "elapsed_seconds": elapsed,
            "scenes_per_second": len(dataset) / max(elapsed, 1e-12),
            "actors_per_second": actors / max(elapsed, 1e-12),
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
        },
        "inputs": {
            **checkpoint_receipt,
            "evaluation_receipt": EVALUATION_RECEIPT.relative_to(ROOT).as_posix(),
            "evaluation_receipt_sha256": sha256(EVALUATION_RECEIPT),
            "evaluation_receipt_file_count": len(receipt["files"]),
            "scene_index_summary": SCENE_INDEX_SUMMARY.relative_to(ROOT).as_posix(),
            "scene_index_summary_sha256": sha256(SCENE_INDEX_SUMMARY),
            "expected_test_scenes": expected_scenes,
            "protocol": PROTOCOL.relative_to(ROOT).as_posix(),
            "protocol_sha256": sha256(PROTOCOL),
        },
        "integrity": {
            "test_constructed_after_all_gates": True,
            "trajectory_dataset_skip": 5,
            "scene_count_matches_frozen_index": True,
            "partial_test": False,
            "test_used_for_model_selection": False,
            "uniform_probability_measure": True,
            "learned_mode_ranking": False,
            "seed42_reused_without_rerun": True,
        },
        "claim_boundary": protocol["claim_boundary"],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airport", choices=("KAGC", "KBTP"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--authorize-locked-test", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.authorize_locked_test:
        raise RuntimeError("locked test requires explicit --authorize-locked-test")
    if args.output.resolve() != expected_output(args.airport, args.seed).resolve():
        raise ValueError(f"locked output path is frozen: {expected_output(args.airport, args.seed)}")
    if args.output.resolve().exists():
        raise FileExistsError(args.output.resolve())
    result = run(
        args.airport,
        args.seed,
        torch.device(args.device),
        args.batch_size,
        authorized=True,
    )
    atomic_json(args.output.resolve(), result)
    print(json.dumps({"output": args.output.resolve().as_posix(), "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
