"""Evaluate frozen ASCENT/MABPT checkpoints on identity-disjoint Tartan views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.edfa_ascent.relation import pack_scenes
from experiments.energy_predict_optimize.evaluation import compute_batch_metrics
from mabpt.evaluate_tartan_probability_ablation import (
    StreamingMetricStore,
    _shared_support_metrics,
    _update_metric_grid,
    _with_effective_modes,
    shared_support_probability_arms,
)
from mabpt.evaluate_tartan_retrain import _selected_checkpoint_triplet
from mabpt.operator import DEFAULT_ADE_SCALE, pairwise_trajectory_distance, support_cost
from mabpt.partc_seed_evaluate import _load_model_pair
from mabpt.train_tartan_retrain import sha256
from model.utils import TrajectoryDataset, seed_worker, seq_collate

from .train_awta_tartan import atomic_json


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("identity_disjoint_tartan_protocol_v1.json")
DATA_ROOT = ROOT / "artifacts/journal_extension_20260814/tartan_identity_disjoint_data_v1"
INDEX_ROOT = ROOT / "artifacts/journal_extension_20260814/tartan_identity_disjoint_index_v1"
RECEIPT = ROOT / "artifacts/journal_extension_20260814/tartan_identity_disjoint_evaluation_receipt_v1.json"
PARENT_PROTOCOL = ROOT / "mabpt/tartan_retrain_protocol_v1.json"
ARMS = ("ascent_native", "mabpt_selected_unweighted_energy_kl")


def verify_receipt() -> dict[str, Any]:
    if not RECEIPT.is_file():
        raise FileNotFoundError(RECEIPT)
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    if receipt.get("identity_disjoint_test_inference_completed_before_freeze") is not False:
        raise RuntimeError("identity-disjoint receipt does not precede test inference")
    for relative, expected in receipt["files"].items():
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise RuntimeError(f"identity-disjoint receipt mismatch: {relative}")
    return receipt


def dataset(airport: str, split: str) -> tuple[TrajectoryDataset, list[str], Path]:
    index_path = INDEX_ROOT / f"{airport}_{split}_scene_dates.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    value = TrajectoryDataset(
        (DATA_ROOT / airport / split).as_posix(),
        obs_len=16,
        obs_steps=1,
        pred_len=120,
        skip=5,
        pred_step=5,
        delim=" ",
        cache_dir=ROOT / "dataset/_cache/journal_identity_disjoint_v1" / airport / split,
    )
    dates = list(index["dates"])
    if len(value) != len(dates):
        raise RuntimeError("identity-disjoint dataset/index mismatch")
    return value, dates, index_path


@torch.inference_mode()
def run(
    *,
    airport: str,
    seed: int,
    split: str,
    device: torch.device,
    batch_size: int,
    workers: int,
    authorize_retrospective_test: bool,
) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    parent = json.loads(PARENT_PROTOCOL.read_text(encoding="utf-8"))
    if airport not in ("KAGC", "KBTP") or seed not in protocol["evaluation"]["seeds"]:
        raise ValueError("unregistered identity-disjoint cell")
    if split == "test":
        if not authorize_retrospective_test:
            raise RuntimeError("retrospective identity-disjoint test requires explicit authorization")
        receipt = verify_receipt()
    elif split == "development":
        receipt = None
    else:
        raise ValueError("split must be development or test")

    cohort, scene_dates, index_path = dataset(airport, split)
    loader = DataLoader(
        cohort,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=seq_collate,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        persistent_workers=workers > 0,
    )
    checkpoints = _selected_checkpoint_triplet(
        root=ROOT,
        protocol=parent,
        airport=airport,
        regime="target_only",
        seed=seed,
        formal=True,
    )
    source, target = _load_model_pair(
        source_checkpoint=ROOT / checkpoints["ascent"]["path"],
        target_checkpoint=ROOT / checkpoints["predicted_risk"]["path"],
        device=device,
        batch_size=batch_size,
    )
    states = {arm: StreamingMetricStore() for arm in ARMS}
    cursor = actors = batches = 0
    started = time.perf_counter()
    for data in loader:
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        truth = data["pred_traj"].transpose(1, 0).to(torch.float64)
        source_support, source_logits, _ = source(data)
        source_probability = source_logits.softmax(dim=1)
        target_support, target_energy, target_decision, auxiliary = target(data)
        target_native = auxiliary["decision_logits"].softmax(dim=1)
        cross = support_cost(source_support, target_support)
        pairwise = pairwise_trajectory_distance(target_support) / DEFAULT_ADE_SCALE
        probabilities, _ = shared_support_probability_arms(
            source_probability,
            target_native,
            target_energy,
            cross,
            auxiliary["centered_predicted_normalized_ade_risk"].to(torch.float64),
            pairwise,
        )
        selected = probabilities["gibbs_unweighted_energy_kl"]
        source_metrics = _with_effective_modes(
            compute_batch_metrics(
                source_support.to(torch.float64),
                source_probability.to(torch.float64),
                source_logits.argmax(dim=1),
                truth,
            ),
            source_probability,
        )
        target_metrics = _shared_support_metrics(
            target_support,
            {"mabpt_selected_unweighted_energy_kl": selected},
            target_decision,
            truth,
        )["mabpt_selected_unweighted_energy_kl"]
        packed = pack_scenes(data["adj"])
        dates = scene_dates[cursor : cursor + packed.scene_count]
        actor_dates = np.asarray(dates, dtype=object)[packed.inverse.detach().cpu().numpy()]
        _update_metric_grid(
            states,
            {"ascent_native": source_metrics, "mabpt_selected_unweighted_energy_kl": target_metrics},
            actor_dates,
        )
        cursor += packed.scene_count
        actors += int(source_support.shape[0])
        batches += 1
    if cursor != len(cohort):
        raise RuntimeError("identity-disjoint evaluation did not consume every scene")
    models = {name: state.summary() for name, state in states.items()}
    return {
        "format_version": 1,
        "experiment_id": "tartan_identity_disjoint_sensitivity_v1",
        "evidence_class": "development_sensitivity" if split == "development" else "retrospective_test_subgroup_sensitivity",
        "airport": airport,
        "seed": seed,
        "split": split,
        "scenes": len(cohort),
        "actors": actors,
        "models": models,
        "relative_gain": {
            metric: (models["ascent_native"]["overall"][metric] - models["mabpt_selected_unweighted_energy_kl"]["overall"][metric]) / models["ascent_native"]["overall"][metric]
            for metric in ("energy_score", "minade", "minfde", "top1_ade", "top1_fde")
        },
        "inputs": {
            "checkpoints": checkpoints,
            "data_manifest": DATA_ROOT.joinpath("manifest.json").relative_to(ROOT).as_posix(),
            "data_manifest_sha256": sha256(DATA_ROOT / "manifest.json"),
            "scene_index": index_path.relative_to(ROOT).as_posix(),
            "scene_index_sha256": sha256(index_path),
            "evaluation_receipt": RECEIPT.relative_to(ROOT).as_posix() if receipt else None,
            "evaluation_receipt_sha256": sha256(RECEIPT) if receipt else None,
        },
        "runtime": {"device": str(device), "batch_size": batch_size, "batches": batches, "elapsed_seconds": time.perf_counter() - started},
        "integrity": {"identity_disjoint": True, "date_order_preserved": True, "test_used_for_selection": False, "partial_cohort": False},
        "claim_boundary": protocol["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airport", choices=("KAGC", "KBTP"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=("development", "test"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--authorize-retrospective-test", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        airport=args.airport,
        seed=args.seed,
        split=args.split,
        device=torch.device(args.device),
        batch_size=args.batch_size,
        workers=args.workers,
        authorize_retrospective_test=args.authorize_retrospective_test,
    )
    atomic_json(args.output, result)
    print(json.dumps({"output": str(args.output), "energy_gain": result["relative_gain"]["energy_score"]}, indent=2))


if __name__ == "__main__":
    main()
