"""Evaluate registered robustness conditions for one MABPT-ASCENT seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import Subset

from experiments.edfa_ascent.relation import pack_scenes
from experiments.metric_exact.evaluation import target_tail_threshold
from experiments.energy_predict_optimize.evaluation import compute_batch_metrics

from .evaluate import ArmAccumulator, _atomic_json, _sha256
from .partc_design import PROTOCOL as PARTC_PROTOCOL, load_protocol
from .partc_seed_evaluate import (
    ROOT,
    _data,
    _default_source,
    _default_target,
    _load_model_pair,
    _loader,
    _model_outputs,
)
from .robustness import (
    PROTOCOL as ROBUSTNESS_PROTOCOL,
    _masked_metrics,
    _training_range_quartiles,
    perturb,
)


ARTIFACT_ROOT = ROOT / "artifacts/mabpt_partc_20260811"


def _relative(models: dict[str, dict[str, object]]) -> dict[str, float]:
    return {
        metric: (models["original_ascent"][metric] - models["mabpt_ascent"][metric])
        / models["original_ascent"][metric]
        for metric in ("top1_ade", "top1_fde", "energy_score", "nll", "brier")
    }


@torch.inference_mode()
def run(
    *,
    seed: int,
    device: torch.device,
    workers: int,
    batch_size: int,
    max_train_scenes: int | None,
    max_development_scenes: int | None,
    source_checkpoint: Path | None,
    target_checkpoint: Path | None,
) -> dict[str, object]:
    protocol = load_protocol()
    if seed not in [int(value) for value in protocol["fixed_seeds"]]:
        raise RuntimeError("seed lies outside the frozen Part C registry")
    specification = json.loads(ROBUSTNESS_PROTOCOL.read_text(encoding="utf-8"))
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    train, train_indices, development, dates, date_path = _data(
        max_train_scenes, max_development_scenes
    )
    training_subset = Subset(train, train_indices)
    range_quartiles = _training_range_quartiles(train, training_subset)
    development_loader = _loader(
        development, batch_size=batch_size, workers=workers
    )
    source_path = (source_checkpoint or _default_source(seed)).resolve()
    target_path = (target_checkpoint or _default_target(seed)).resolve()
    source, target_model = _load_model_pair(
        source_checkpoint=source_path,
        target_checkpoint=target_path,
        device=device,
        batch_size=batch_size,
    )
    model_names = ("original_ascent", "mabpt_ascent")
    condition_states = {
        condition: {model: ArmAccumulator() for model in model_names}
        for condition in specification["conditions"]
    }
    stratum_states = {
        name: {model: ArmAccumulator() for model in model_names}
        for name in (
            "distance_q1",
            "distance_q2",
            "distance_q3",
            "distance_q4",
            "density_1",
            "density_2_3",
            "density_4_plus",
        )
    }
    generator_seeds = {
        condition: int(specification["noise_seed"]) + 1000 * seed + index
        for index, condition in enumerate(specification["conditions"])
    }
    generators = {
        condition: torch.Generator(device=device).manual_seed(generator_seed)
        for condition, generator_seed in generator_seeds.items()
    }
    tail_threshold = target_tail_threshold(train)
    scene_cursor = 0
    for raw_data in development_loader:
        raw_data = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in raw_data.items()
        }
        packed = pack_scenes(raw_data["adj"])
        batch_dates = dates[scene_cursor : scene_cursor + packed.scene_count]
        if len(batch_dates) != packed.scene_count:
            raise RuntimeError("scene-date alignment failed")
        actor_dates = np.asarray(batch_dates, dtype=object)[
            packed.inverse.detach().cpu().numpy()
        ]
        truth = raw_data["pred_traj"].transpose(1, 0)
        endpoint_travel = torch.linalg.vector_norm(
            truth[:, -1] - raw_data["obs_traj"][-1], dim=-1
        )
        tail = endpoint_travel >= tail_threshold
        clean = None
        for condition, condition_specification in specification["conditions"].items():
            data = perturb(
                raw_data,
                condition,
                condition_specification,
                generator=generators[condition],
            )
            outputs = _model_outputs(source, target_model, data)
            metrics = {
                model: compute_batch_metrics(
                    values["support"].to(torch.float64),
                    values["probability"].to(torch.float64),
                    values["decision"],
                    truth.to(torch.float64),
                )
                for model, values in outputs.items()
            }
            for model, values in outputs.items():
                condition_states[condition][model].update(
                    metrics[model], values["probability"], actor_dates, tail
                )
            if condition == "clean":
                clean = outputs, metrics
        if clean is None:
            raise RuntimeError("registered clean condition was not evaluated")
        outputs, metrics = clean
        ranges = torch.linalg.vector_norm(raw_data["obs_traj"][-1, :, :2], dim=1)
        q1, q2, q3 = range_quartiles
        density = packed.counts[packed.inverse]
        masks = {
            "distance_q1": ranges <= q1,
            "distance_q2": (ranges > q1) & (ranges <= q2),
            "distance_q3": (ranges > q2) & (ranges <= q3),
            "distance_q4": ranges > q3,
            "density_1": density == 1,
            "density_2_3": (density >= 2) & (density <= 3),
            "density_4_plus": density >= 4,
        }
        for name, mask in masks.items():
            if not bool(mask.any()):
                continue
            masked_dates = actor_dates[mask.detach().cpu().numpy()]
            for model, values in outputs.items():
                stratum_states[name][model].update(
                    _masked_metrics(metrics[model], mask),
                    values["probability"][mask],
                    masked_dates,
                    tail[mask],
                )
        scene_cursor += packed.scene_count
    if scene_cursor != len(dates):
        raise RuntimeError("robustness evaluation did not consume all development dates")
    conditions = {
        condition: {model: state.summary() for model, state in models.items()}
        for condition, models in condition_states.items()
    }
    strata = {
        name: {model: state.summary() for model, state in models.items()}
        for name, models in stratum_states.items()
        if all(state.count for state in models.values())
    }
    for models in (*conditions.values(), *strata.values()):
        models["relative_gain_mabpt_vs_ascent"] = _relative(models)
    for condition, models in conditions.items():
        models["degradation_vs_clean"] = {
            model: {
                metric: (models[model][metric] - conditions["clean"][model][metric])
                / conditions["clean"][model][metric]
                for metric in ("top1_ade", "top1_fde", "energy_score", "nll", "brier")
            }
            for model in model_names
        }
    return {
        "format_version": 1,
        "model": "MABPT-ASCENT",
        "experiment_id": "five_seed_robustness",
        "evidence_class": "development_only",
        "seed": seed,
        "partc_protocol_sha256": _sha256(PARTC_PROTOCOL),
        "robustness_protocol_sha256": _sha256(ROBUSTNESS_PROTOCOL),
        "training_scenes_for_range_quartiles": len(train_indices),
        "development_scenes": len(development),
        "development_dates_sha256": _sha256(date_path),
        "training_only_range_quartiles_km": range_quartiles,
        "condition_generator_seeds": generator_seeds,
        "conditions": conditions,
        "strata": strata,
        "inputs": {
            "source_checkpoint": str(source_path.relative_to(ROOT)),
            "source_checkpoint_sha256": _sha256(source_path),
            "target_checkpoint": str(target_path.relative_to(ROOT)),
            "target_checkpoint_sha256": _sha256(target_path),
        },
        "integrity": {
            "train_and_development_only": True,
            "historical_locked_test_used": False,
            "same_corruption_for_both_models": True,
            "target_in_probability_forward": False,
            "condition_specific_tuning": False,
        },
        "runtime": {
            "device": str(device),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_allocated_gpu_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-train-scenes", type=int)
    parser.add_argument("--max-development-scenes", type=int)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--target-checkpoint", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.smoke:
        args.max_train_scenes = args.max_train_scenes or 256
        args.max_development_scenes = args.max_development_scenes or 8
    result = run(
        seed=args.seed,
        device=torch.device(args.device),
        workers=args.workers,
        batch_size=args.batch_size,
        max_train_scenes=args.max_train_scenes,
        max_development_scenes=args.max_development_scenes,
        source_checkpoint=args.source_checkpoint,
        target_checkpoint=args.target_checkpoint,
    )
    if args.output is None:
        suffix = "smoke" if args.smoke else "formal"
        args.output = ARTIFACT_ROOT / f"seed{args.seed}_robustness_{suffix}_v1.json"
    _atomic_json(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "seed": args.seed,
                "conditions": len(result["conditions"]),
                "actors": result["conditions"]["clean"]["mabpt_ascent"]["actors"],
                "elapsed_seconds": result["runtime"]["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
