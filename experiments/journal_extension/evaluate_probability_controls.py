"""Evaluate matching, Energy-KL, and scalar calibration controls on Tartan."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from experiments.edfa_ascent.relation import pack_scenes
from mabpt.evaluate_tartan_probability_ablation import (
    StreamingMetricStore,
    _shared_support_metrics,
    _update_metric_grid,
)
from mabpt.evaluate_tartan_retrain import _dataset, _selected_checkpoint_triplet
from mabpt.operator import (
    DEFAULT_ADE_SCALE,
    energy_kl_projection,
    exact_gibbs_transport,
    hard_bijection_transport,
    pairwise_trajectory_distance,
    support_cost,
)
from mabpt.partc_seed_evaluate import _load_model_pair
from mabpt.train_tartan_retrain import _limited_indices, sha256
from model.utils import seed_worker, seq_collate

from .train_awta_tartan import atomic_json


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("probability_controls_protocol_v1.json")
PARENT_PROTOCOL = ROOT / "mabpt/tartan_retrain_protocol_v1.json"
RECEIPT = ROOT / "artifacts/journal_extension_20260814/probability_controls_receipt_v1.json"
TEMPERATURES = ((0.5, "0p50"), (0.75, "0p75"), (1.0, "1p00"), (1.5, "1p50"), (2.0, "2p00"))
GIBBS_TEMPERATURES = ((0.5, "0p50"), (1.0, "1p00"), (2.0, "2p00"))
WEIGHT_VALUES = ((0.5, "0p50"), (2.0, "2p00"))


def _arm_names() -> tuple[str, ...]:
    names = ["hard_unweighted_prior", "hard_unweighted_energy_kl", "gibbs_unweighted_prior", "gibbs_unweighted_energy_kl"]
    names.extend(f"gibbs_cost_temp_{label}_energy_kl" for _, label in GIBBS_TEMPERATURES)
    for factor in ("risk", "diversity", "kl"):
        names.extend(f"energy_kl_{factor}_{label}" for _, label in WEIGHT_VALUES)
    for prefix in ("target_native", "selected_mabpt"):
        names.extend(f"{prefix}_temp_{label}" for _, label in TEMPERATURES)
    return tuple(names)


ARMS = _arm_names()


def temperature_scale(probability: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply scalar temperature scaling to an already normalized distribution."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    probability = probability.to(torch.float64)
    if probability.ndim != 2 or not bool(torch.isfinite(probability).all()):
        raise ValueError("probability must be a finite [B,K] tensor")
    tiny = torch.finfo(probability.dtype).tiny
    return torch.softmax(probability.clamp_min(tiny).log() / temperature, dim=1)


def probability_control_arms(
    source_probability: torch.Tensor,
    target_native_probability: torch.Tensor,
    cross_cost: torch.Tensor,
    predicted_risk: torch.Tensor,
    pairwise: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Construct registered probability-only controls without future targets."""
    source_probability = source_probability.to(torch.float64)
    target_native_probability = target_native_probability.to(torch.float64)
    predicted_risk = predicted_risk.to(torch.float64)
    pairwise = pairwise.to(torch.float64)

    hard_prior = hard_bijection_transport(
        source_probability, cross_cost, mass_weighted=False
    )["transported"]
    gibbs_priors = {
        label: exact_gibbs_transport(
            source_probability, cross_cost / temperature, mass_weighted=False
        )["transported"]
        for temperature, label in GIBBS_TEMPERATURES
    }
    selected_prior = gibbs_priors["1p00"]

    def project(
        prior: torch.Tensor,
        *,
        risk_weight: float = 1.0,
        diversity_weight: float = 1.0,
        kl_weight: float = 1.0,
    ) -> torch.Tensor:
        return energy_kl_projection(
            prior,
            predicted_risk,
            pairwise,
            risk_weight=risk_weight,
            diversity_weight=diversity_weight,
            kl_weight=kl_weight,
        )[0]

    selected = project(selected_prior)
    arms = {
        "hard_unweighted_prior": hard_prior,
        "hard_unweighted_energy_kl": project(hard_prior),
        "gibbs_unweighted_prior": selected_prior,
        "gibbs_unweighted_energy_kl": selected,
    }
    arms.update(
        {
            f"gibbs_cost_temp_{label}_energy_kl": project(prior)
            for label, prior in gibbs_priors.items()
        }
    )
    for value, label in WEIGHT_VALUES:
        arms[f"energy_kl_risk_{label}"] = project(selected_prior, risk_weight=value)
        arms[f"energy_kl_diversity_{label}"] = project(selected_prior, diversity_weight=value)
        arms[f"energy_kl_kl_{label}"] = project(selected_prior, kl_weight=value)
    for temperature, label in TEMPERATURES:
        arms[f"target_native_temp_{label}"] = temperature_scale(
            target_native_probability, temperature
        )
        arms[f"selected_mabpt_temp_{label}"] = temperature_scale(selected, temperature)
    if set(arms) != set(ARMS):
        missing = sorted(set(ARMS) - set(arms))
        extra = sorted(set(arms) - set(ARMS))
        raise RuntimeError(f"probability-control arm mismatch: missing={missing}, extra={extra}")
    return {name: arms[name] for name in ARMS}


def verify_receipt() -> dict[str, Any]:
    if not RECEIPT.is_file():
        raise FileNotFoundError(RECEIPT)
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    if receipt.get("control_test_inference_completed_before_freeze") is not False:
        raise RuntimeError("control receipt does not precede test inference")
    for relative, expected in receipt["files"].items():
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size != int(expected["bytes"]) or sha256(path) != expected["sha256"]:
            raise RuntimeError(f"probability-control receipt mismatch: {relative}")
    return receipt


@torch.inference_mode()
def run(
    *,
    airport: str,
    seed: int,
    split: str,
    device: torch.device,
    workers: int,
    batch_size: int,
    max_scenes: int | None,
    authorize_retrospective_test: bool,
) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    parent = json.loads(PARENT_PROTOCOL.read_text(encoding="utf-8"))
    if airport not in protocol["data"]["airports"] or seed not in protocol["data"]["seeds"]:
        raise ValueError("unregistered probability-control cell")
    if split == "test":
        if not authorize_retrospective_test:
            raise RuntimeError("retrospective control test requires explicit authorization")
        if max_scenes is not None:
            raise RuntimeError("retrospective control test forbids partial evaluation")
        receipt = verify_receipt()
    elif split == "development":
        if authorize_retrospective_test:
            raise ValueError("test authorization is invalid for development")
        receipt = None
    else:
        raise ValueError("split must be development or test")

    cohort, scene_dates, index_path = _dataset(parent, airport, split)
    selected_indices = _limited_indices(len(cohort), max_scenes)
    evaluation = Subset(cohort, selected_indices)
    selected_dates = [scene_dates[index] for index in selected_indices]
    loader_options: dict[str, Any] = {
        "dataset": evaluation,
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": seq_collate,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }
    if workers > 0:
        loader_options.update({"persistent_workers": True, "prefetch_factor": 4})
    loader = DataLoader(**loader_options)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
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
        data = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in data.items()
        }
        truth = data["pred_traj"].transpose(1, 0).to(torch.float64)
        source_support, source_logits, _ = source(data)
        target_support, _, target_decision, auxiliary = target(data)
        cross = support_cost(source_support, target_support)
        pairwise = pairwise_trajectory_distance(target_support) / DEFAULT_ADE_SCALE
        probabilities = probability_control_arms(
            source_logits.softmax(dim=1),
            auxiliary["decision_logits"].softmax(dim=1),
            cross,
            auxiliary["centered_predicted_normalized_ade_risk"],
            pairwise,
        )
        metrics_by_arm = _shared_support_metrics(
            target_support, probabilities, target_decision, truth
        )
        packed = pack_scenes(data["adj"])
        batch_dates = selected_dates[cursor : cursor + packed.scene_count]
        if len(batch_dates) != packed.scene_count:
            raise RuntimeError("probability-control scene/date alignment failed")
        actor_dates = np.asarray(batch_dates, dtype=object)[
            packed.inverse.detach().cpu().numpy()
        ]
        _update_metric_grid(states, metrics_by_arm, actor_dates)
        cursor += packed.scene_count
        actors += int(source_support.shape[0])
        batches += 1
    if cursor != len(evaluation):
        raise RuntimeError("probability-control evaluation did not consume its cohort")
    models = {arm: state.summary() for arm, state in states.items()}
    reference = models[ARMS[0]]["overall"]
    geometry_difference = {
        metric: max(
            abs(float(models[arm]["overall"][metric]) - float(reference[metric]))
            for arm in ARMS
        )
        for metric in protocol["evaluation"]["geometry_metrics_expected_invariant"]
    }
    if any(value > 1e-12 for value in geometry_difference.values()):
        raise RuntimeError("probability-only controls changed fixed-support geometry")
    return {
        "format_version": 1,
        "experiment_id": "tartan_probability_controls_v1",
        "evidence_class": "development_sensitivity" if split == "development" else "retrospective_test_sensitivity",
        "airport": airport,
        "regime": "target_only",
        "seed": seed,
        "split": split,
        "scenes": len(evaluation),
        "actors": actors,
        "models": models,
        "geometry_invariance_max_absolute_difference": geometry_difference,
        "inputs": {
            "protocol": PROTOCOL.relative_to(ROOT).as_posix(),
            "protocol_sha256": sha256(PROTOCOL),
            "evaluator": Path(__file__).relative_to(ROOT).as_posix(),
            "evaluator_sha256": sha256(Path(__file__)),
            "checkpoints": checkpoints,
            "scene_date_index": index_path.relative_to(ROOT).as_posix(),
            "scene_date_index_sha256": sha256(index_path),
            "control_receipt": RECEIPT.relative_to(ROOT).as_posix() if receipt else None,
            "control_receipt_sha256": sha256(RECEIPT) if receipt else None,
        },
        "integrity": {
            "target_in_probability_forward": False,
            "shared_support": True,
            "all_registered_arms_reported": tuple(models) == ARMS,
            "test_used_for_selection": False,
            "partial_cohort": max_scenes is not None,
        },
        "runtime": {
            "device": str(device),
            "workers": workers,
            "batch_size": batch_size,
            "batches": batches,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "claim_boundary": protocol["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airport", choices=("KAGC", "KBTP"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=("development", "test"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--authorize-retrospective-test", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.smoke:
        if args.split != "development":
            parser.error("--smoke is development-only")
        args.max_scenes = args.max_scenes or 8
    result = run(
        airport=args.airport,
        seed=args.seed,
        split=args.split,
        device=torch.device(args.device),
        workers=args.workers,
        batch_size=args.batch_size,
        max_scenes=args.max_scenes,
        authorize_retrospective_test=args.authorize_retrospective_test,
    )
    atomic_json(args.output.resolve(), result)
    print(json.dumps({
        "output": args.output.resolve().as_posix(),
        "airport": result["airport"],
        "seed": result["seed"],
        "split": result["split"],
        "actors": result["actors"],
        "elapsed_seconds": result["runtime"]["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
