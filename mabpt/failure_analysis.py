"""E13 target-blind strata and failure-case index analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.edfa_ascent.relation import pack_scenes
from experiments.energy_predict_optimize.evaluation import compute_batch_metrics
from experiments.ascent_recomparison.common import fold_subsets, load_dataset, loader
from experiments.tpmo_ascent.protocol import load_protocol as load_legacy_data_protocol

from .evaluate import ROOT, _load_models
from .events import _training_actor_indices, atomic_json, sha256
from .freeze import verify as verify_legacy_freeze
from .operator import (
    DEFAULT_ADE_SCALE,
    energy_kl_projection,
    exact_gibbs_transport,
    pairwise_trajectory_distance,
    support_cost,
)


PROTOCOL = Path(__file__).with_name("e13_protocol.json")
METRICS = ("top1_ade", "top1_fde", "minade", "minfde", "energy_score")


def _phase_thresholds(dataset, training_data) -> dict[str, float]:
    indices = _training_actor_indices(dataset, training_data)
    history = dataset.obs_traj[indices]
    velocity = history[:, :, -1] - history[:, :, -2]
    horizontal = torch.linalg.vector_norm(velocity[:, :2], dim=1)
    vertical = velocity[:, 2].abs()
    return {
        "low_speed_km_per_second": float(torch.quantile(horizontal, 0.10)),
        "high_vertical_speed_km_per_second": float(torch.quantile(vertical, 0.90)),
    }


def _summary(arrays: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    if not mask.any():
        return {"actors": 0}
    return {
        "actors": int(mask.sum()),
        **{metric: float(arrays[metric][mask].mean()) for metric in METRICS},
    }


def _case_records(
    indices: np.ndarray,
    *,
    dates: np.ndarray,
    entropy: np.ndarray,
    mismatch: np.ndarray,
    horizontal_speed: np.ndarray,
    vertical_speed: np.ndarray,
    ascent: dict[str, np.ndarray],
    mabpt: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    return [
        {
            "actor_order": int(index),
            "date": str(dates[index]),
            "normalized_permutation_entropy": float(entropy[index]),
            "expected_support_cost": float(mismatch[index]),
            "observed_horizontal_speed_km_per_second": float(horizontal_speed[index]),
            "observed_absolute_vertical_speed_km_per_second": float(vertical_speed[index]),
            "ascent": {metric: float(ascent[metric][index]) for metric in METRICS},
            "mabpt": {metric: float(mabpt[metric][index]) for metric in METRICS},
        }
        for index in indices
    ]


@torch.inference_mode()
def run(
    *,
    fold: int,
    device: torch.device,
    workers: int,
    batch_size: int,
    max_validation_scenes: int | None,
) -> dict[str, object]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if fold not in protocol["folds"]:
        raise ValueError("fold is outside E13 protocol")
    if not verify_legacy_freeze()["ok"]:
        raise RuntimeError("legacy freeze verification failed")
    legacy = load_legacy_data_protocol()
    legacy.assert_boundaries()
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(True)
    started = time.time()
    dataset = load_dataset(legacy)
    training_data, validation_data, validation_dates = fold_subsets(
        legacy,
        dataset,
        fold,
        max_validation_scenes=max_validation_scenes,
    )
    phase_thresholds = _phase_thresholds(dataset, training_data)
    validation_loader = loader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        workers=workers,
        prefetch=4,
    )
    baseline, target_model, baseline_path, target_path = _load_models(fold, device)
    values = {
        "ascent": {metric: [] for metric in METRICS},
        "mabpt": {metric: [] for metric in METRICS},
    }
    entropy_values = []
    mismatch_values = []
    horizontal_values = []
    vertical_values = []
    date_values = []
    scene_cursor = 0
    for data in validation_loader:
        data = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in data.items()
        }
        target = data["pred_traj"].transpose(1, 0)
        source_support, source_logits, _ = baseline(data)
        source_probability = source_logits.softmax(dim=1)
        source_metrics = compute_batch_metrics(
            source_support, source_probability, source_logits.argmax(dim=1), target
        )
        target_support, _, target_decision, auxiliary = target_model(data)
        transport = exact_gibbs_transport(
            source_probability,
            support_cost(source_support, target_support),
            mass_weighted=True,
        )
        mabpt_probability = energy_kl_projection(
            transport["transported"],
            auxiliary["centered_predicted_normalized_ade_risk"].to(torch.float64),
            pairwise_trajectory_distance(target_support) / DEFAULT_ADE_SCALE,
        )[0]
        mabpt_metrics = compute_batch_metrics(
            target_support.to(torch.float64),
            mabpt_probability,
            target_decision,
            target.to(torch.float64),
        )
        for model, metrics in (("ascent", source_metrics), ("mabpt", mabpt_metrics)):
            for metric in METRICS:
                values[model][metric].append(metrics[metric].detach().cpu().numpy())
        entropy_values.append(
            transport["normalized_assignment_entropy"].detach().cpu().numpy()
        )
        mismatch_values.append(transport["expected_cost"].detach().cpu().numpy())
        velocity = data["obs_traj"][-1] - data["obs_traj"][-2]
        horizontal_values.append(
            torch.linalg.vector_norm(velocity[:, :2], dim=1).detach().cpu().numpy()
        )
        vertical_values.append(velocity[:, 2].abs().detach().cpu().numpy())
        packed = pack_scenes(data["adj"])
        dates = validation_dates[scene_cursor : scene_cursor + packed.scene_count]
        date_values.append(
            np.asarray(dates, dtype=object)[packed.inverse.detach().cpu().numpy()]
        )
        scene_cursor += packed.scene_count
    arrays = {
        model: {metric: np.concatenate(chunks) for metric, chunks in metrics.items()}
        for model, metrics in values.items()
    }
    entropy = np.concatenate(entropy_values)
    mismatch = np.concatenate(mismatch_values)
    horizontal = np.concatenate(horizontal_values)
    vertical = np.concatenate(vertical_values)
    dates = np.concatenate(date_values)
    entropy_q = np.quantile(entropy, [0.25, 0.75])
    mismatch_q = np.quantile(mismatch, [0.25, 0.75])
    masks = {
        "low_permutation_entropy": entropy <= entropy_q[0],
        "high_permutation_entropy": entropy >= entropy_q[1],
        "low_support_mismatch": mismatch <= mismatch_q[0],
        "high_support_mismatch": mismatch >= mismatch_q[1],
        "rare_low_speed": horizontal <= phase_thresholds["low_speed_km_per_second"],
        "rare_vertical_maneuver": vertical
        >= phase_thresholds["high_vertical_speed_km_per_second"],
        "nominal_phase": (
            (horizontal > phase_thresholds["low_speed_km_per_second"])
            & (vertical < phase_thresholds["high_vertical_speed_km_per_second"])
        ),
    }
    strata = {}
    for name, mask in masks.items():
        strata[name] = {
            model: _summary(arrays[model], mask) for model in ("ascent", "mabpt")
        }
        strata[name]["relative_gain_mabpt_vs_ascent"] = (
            {
                metric: (
                    strata[name]["ascent"][metric] - strata[name]["mabpt"][metric]
                )
                / strata[name]["ascent"][metric]
                for metric in METRICS
            }
            if strata[name]["ascent"]["actors"]
            else None
        )
    count = int(protocol["case_selection"]["count_per_type"])
    selected = {
        "highest_permutation_entropy": np.argsort(-entropy, kind="stable")[:count],
        "lowest_permutation_entropy": np.argsort(entropy, kind="stable")[:count],
        "highest_support_mismatch": np.argsort(-mismatch, kind="stable")[:count],
        "lowest_observed_speed": np.argsort(horizontal, kind="stable")[:count],
        "highest_observed_vertical_speed": np.argsort(-vertical, kind="stable")[:count],
    }
    cases = {
        name: _case_records(
            indices,
            dates=dates,
            entropy=entropy,
            mismatch=mismatch,
            horizontal_speed=horizontal,
            vertical_speed=vertical,
            ascent=arrays["ascent"],
            mabpt=arrays["mabpt"],
        )
        for name, indices in selected.items()
    }
    energy_gain = arrays["ascent"]["energy_score"] - arrays["mabpt"]["energy_score"]
    return {
        "format_version": 1,
        "model": "MABPT",
        "experiment_id": "E13",
        "evidence_class": "legacy_development_only",
        "protocol_sha256": sha256(PROTOCOL),
        "fold": fold,
        "actors": int(len(entropy)),
        "target_free_thresholds": {
            "permutation_entropy_q25_q75": entropy_q.tolist(),
            "support_mismatch_q25_q75": mismatch_q.tolist(),
            **phase_thresholds,
        },
        "strata": strata,
        "target_blind_selected_cases": cases,
        "diagnostics": {
            "correlation_entropy_with_energy_gain": float(
                np.corrcoef(entropy, energy_gain)[0, 1]
            ),
            "correlation_mismatch_with_energy_gain": float(
                np.corrcoef(mismatch, energy_gain)[0, 1]
            ),
        },
        "integrity": {
            "future_or_metric_used_for_case_selection": False,
            "target_in_probability_forward": False,
            "success_only_display": False,
            "gate_or_residual_used": False,
        },
        "inputs": {
            "baseline_checkpoint": baseline_path,
            "target_checkpoint": target_path,
        },
        "runtime": {
            "device": str(device),
            "elapsed_seconds": time.time() - started,
            "peak_allocated_gpu_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=(1, 2))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-validation-scenes", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.smoke and args.max_validation_scenes is None:
        args.max_validation_scenes = 64
    result = run(
        fold=args.fold,
        device=torch.device(args.device),
        workers=args.workers,
        batch_size=args.batch_size,
        max_validation_scenes=args.max_validation_scenes,
    )
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "fold": result["fold"],
                "actors": result["actors"],
                "strata": {
                    name: value["relative_gain_mabpt_vs_ascent"]
                    for name, value in result["strata"].items()
                },
                "elapsed_seconds": result["runtime"]["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
