"""E10 paired corruption and operating-stratum evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.edfa_ascent.relation import pack_scenes
from experiments.metric_exact.evaluation import target_tail_threshold
from experiments.energy_predict_optimize.evaluation import compute_batch_metrics
from experiments.ascent_recomparison.common import fold_subsets, load_dataset, loader
from experiments.tpmo_ascent.protocol import load_protocol as load_legacy_data_protocol

from .evaluate import ArmAccumulator, ROOT, _load_models
from .freeze import verify as verify_legacy_freeze
from .operator import (
    DEFAULT_ADE_SCALE,
    energy_kl_projection,
    exact_gibbs_transport,
    pairwise_trajectory_distance,
    support_cost,
)


PROTOCOL = Path(__file__).with_name("e10_protocol.json")


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


def _recompute_relative(observations: torch.Tensor) -> torch.Tensor:
    relative = torch.zeros_like(observations)
    relative[1:] = observations[1:] - observations[:-1]
    return relative


def perturb(
    data: dict[str, object],
    condition: str,
    specification: dict[str, object],
    *,
    generator: torch.Generator,
) -> dict[str, object]:
    result = dict(data)
    observations = data["obs_traj"].clone()
    kind = specification.get("kind")
    if kind == "ADS_B_dropout":
        rate = float(specification["rate"])
        mask = torch.rand(
            observations.shape[:2],
            generator=generator,
            device=observations.device,
        ) < rate
        mask[-1] = False
        for step in range(1, observations.shape[0]):
            observations[step] = torch.where(
                mask[step, :, None], observations[step - 1], observations[step]
            )
    elif kind == "position_noise":
        horizontal = float(specification["horizontal_sigma_km"])
        vertical = float(specification["vertical_sigma_km"])
        noise = torch.randn(
            observations.shape,
            generator=generator,
            device=observations.device,
            dtype=observations.dtype,
        )
        noise[..., :2] *= horizontal
        noise[..., 2] *= vertical
        observations += noise
    elif kind == "history_truncation":
        retained = int(specification["retained_steps"])
        if retained < 2 or retained > observations.shape[0]:
            raise ValueError(f"invalid retained history for {condition}")
        observations[:-retained] = observations[-retained]
    elif kind is not None:
        raise ValueError(f"unknown E10 condition kind: {kind}")
    result["obs_traj"] = observations
    result["obs_traj_rel"] = _recompute_relative(observations)
    return result


def _mabpt_forward(baseline, target_model, data):
    source_support, source_logits, _ = baseline(data)
    source_probabilities = source_logits.softmax(dim=1)
    source_decision = source_logits.argmax(dim=1)
    source_metrics = compute_batch_metrics(
        source_support,
        source_probabilities,
        source_decision,
        data["pred_traj"].transpose(1, 0),
    )
    target_support, _, target_decision, auxiliary = target_model(data)
    cross = support_cost(source_support, target_support)
    transported = exact_gibbs_transport(
        source_probabilities, cross, mass_weighted=True
    )["transported"]
    risk = auxiliary["centered_predicted_normalized_ade_risk"].to(torch.float64)
    pairwise = pairwise_trajectory_distance(target_support) / DEFAULT_ADE_SCALE
    probabilities = energy_kl_projection(transported, risk, pairwise)[0]
    target_metrics = compute_batch_metrics(
        target_support.to(torch.float64),
        probabilities,
        target_decision,
        data["pred_traj"].transpose(1, 0).to(torch.float64),
    )
    return source_probabilities, source_metrics, probabilities, target_metrics


def _training_range_quartiles(dataset, training_subset) -> list[float]:
    scene_sizes = torch.tensor(
        [end - start for start, end in dataset.seq_start_end], dtype=torch.long
    )
    actor_scene = torch.repeat_interleave(
        torch.arange(len(scene_sizes), dtype=torch.long), scene_sizes
    )
    selected_scenes = torch.zeros(len(scene_sizes), dtype=torch.bool)
    selected_scenes[torch.as_tensor(training_subset.indices, dtype=torch.long)] = True
    actor_mask = selected_scenes[actor_scene]
    current = dataset.obs_traj[:, :2, -1]
    ranges = torch.linalg.vector_norm(current[actor_mask], dim=1)
    return [float(value) for value in torch.quantile(ranges, torch.tensor([0.25, 0.5, 0.75]))]


def _masked_metrics(metrics: dict[str, torch.Tensor], mask: torch.Tensor):
    return {name: value[mask] for name, value in metrics.items()}


@torch.inference_mode()
def run(
    *,
    fold: int,
    device: torch.device,
    workers: int,
    batch_size: int,
    max_validation_scenes: int | None,
) -> dict[str, object]:
    specification = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if fold not in specification["folds"]:
        raise ValueError("fold is outside the frozen E10 protocol")
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
    range_quartiles = _training_range_quartiles(dataset, training_data)
    validation_loader = loader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        workers=workers,
        prefetch=4,
    )
    baseline, target_model, baseline_path, target_path = _load_models(fold, device)
    condition_states = {
        condition: {"ascent": ArmAccumulator(), "mabpt": ArmAccumulator()}
        for condition in specification["conditions"]
    }
    stratum_states = {
        "distance_q1": {"ascent": ArmAccumulator(), "mabpt": ArmAccumulator()},
        "distance_q2": {"ascent": ArmAccumulator(), "mabpt": ArmAccumulator()},
        "distance_q3": {"ascent": ArmAccumulator(), "mabpt": ArmAccumulator()},
        "distance_q4": {"ascent": ArmAccumulator(), "mabpt": ArmAccumulator()},
        "density_1": {"ascent": ArmAccumulator(), "mabpt": ArmAccumulator()},
        "density_2_3": {"ascent": ArmAccumulator(), "mabpt": ArmAccumulator()},
        "density_4_plus": {"ascent": ArmAccumulator(), "mabpt": ArmAccumulator()},
    }
    generators = {
        condition: torch.Generator(device=device).manual_seed(
            int(specification["noise_seed"]) + 100 * fold + index
        )
        for index, condition in enumerate(specification["conditions"])
    }
    scene_cursor = 0
    for raw_data in validation_loader:
        raw_data = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in raw_data.items()
        }
        packed = pack_scenes(raw_data["adj"])
        dates = validation_dates[scene_cursor : scene_cursor + packed.scene_count]
        actor_dates = np.asarray(dates, dtype=object)[
            packed.inverse.detach().cpu().numpy()
        ]
        target = raw_data["pred_traj"].transpose(1, 0)
        endpoint_travel = torch.linalg.vector_norm(
            target[:, -1] - raw_data["obs_traj"][-1], dim=-1
        )
        tail = endpoint_travel >= target_tail_threshold(dataset)
        clean_outputs = None
        for condition, condition_spec in specification["conditions"].items():
            data = perturb(
                raw_data,
                condition,
                condition_spec,
                generator=generators[condition],
            )
            outputs = _mabpt_forward(baseline, target_model, data)
            source_probabilities, source_metrics, probabilities, target_metrics = outputs
            condition_states[condition]["ascent"].update(
                source_metrics, source_probabilities, actor_dates, tail
            )
            condition_states[condition]["mabpt"].update(
                target_metrics, probabilities, actor_dates, tail
            )
            if condition == "clean":
                clean_outputs = outputs
        if clean_outputs is None:
            raise RuntimeError("E10 clean condition was not evaluated")
        source_probabilities, source_metrics, probabilities, target_metrics = clean_outputs
        ranges = torch.linalg.vector_norm(raw_data["obs_traj"][-1, :, :2], dim=1)
        q1, q2, q3 = range_quartiles
        distance_masks = {
            "distance_q1": ranges <= q1,
            "distance_q2": (ranges > q1) & (ranges <= q2),
            "distance_q3": (ranges > q2) & (ranges <= q3),
            "distance_q4": ranges > q3,
        }
        density = packed.counts[packed.inverse]
        density_masks = {
            "density_1": density == 1,
            "density_2_3": (density >= 2) & (density <= 3),
            "density_4_plus": density >= 4,
        }
        for name, mask in {**distance_masks, **density_masks}.items():
            if not bool(mask.any()):
                continue
            masked_dates = actor_dates[mask.detach().cpu().numpy()]
            stratum_states[name]["ascent"].update(
                _masked_metrics(source_metrics, mask),
                source_probabilities[mask],
                masked_dates,
                tail[mask],
            )
            stratum_states[name]["mabpt"].update(
                _masked_metrics(target_metrics, mask),
                probabilities[mask],
                masked_dates,
                tail[mask],
            )
        scene_cursor += packed.scene_count
    conditions = {
        condition: {model: state.summary() for model, state in models.items()}
        for condition, models in condition_states.items()
    }
    strata = {
        name: {model: state.summary() for model, state in models.items()}
        for name, models in stratum_states.items()
        if all(state.count for state in models.values())
    }
    for condition, models in conditions.items():
        models["relative_gain_mabpt_vs_ascent"] = {
            metric: (models["ascent"][metric] - models["mabpt"][metric])
            / models["ascent"][metric]
            for metric in ("top1_ade", "top1_fde", "energy_score", "nll", "brier")
        }
        models["degradation_vs_clean"] = {
            model: {
                metric: (models[model][metric] - conditions["clean"][model][metric])
                / conditions["clean"][model][metric]
                for metric in ("top1_ade", "top1_fde", "energy_score", "nll", "brier")
            }
            for model in ("ascent", "mabpt")
        }
    return {
        "format_version": 1,
        "model": "MABPT",
        "experiment_id": "E10",
        "evidence_class": "legacy_development_only",
        "protocol_sha256": _sha256(PROTOCOL),
        "fold": fold,
        "validation_scenes": len(validation_dates),
        "training_only_range_quartiles_km": range_quartiles,
        "conditions": conditions,
        "strata": strata,
        "integrity": {
            "target_in_probability_forward": False,
            "same_corruption_for_both_models": True,
            "condition_specific_tuning": False,
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
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.smoke and args.max_validation_scenes is None:
        args.max_validation_scenes = 8
    result = run(
        fold=args.fold,
        device=torch.device(args.device),
        workers=args.workers,
        batch_size=args.batch_size,
        max_validation_scenes=args.max_validation_scenes,
    )
    if args.output is None:
        suffix = "smoke" if args.smoke else "formal"
        args.output = ROOT / "artifacts/mabpt" / f"e10_fold{args.fold}_{suffix}_v1.json"
    _atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "fold": result["fold"],
                "actors": result["conditions"]["clean"]["mabpt"]["actors"],
                "elapsed_seconds": result["runtime"]["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
