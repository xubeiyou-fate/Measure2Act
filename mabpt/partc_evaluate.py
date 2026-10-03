"""Evaluate Part C fusion and physical experiments for MABPT-ASCENT."""

from __future__ import annotations

import argparse
from collections import defaultdict
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

from .evaluate import ArmAccumulator, _atomic_json, _load_models, _sha256
from .freeze import verify as verify_legacy_freeze
from .operator import DEFAULT_ADE_SCALE, pairwise_trajectory_distance, support_cost
from .partc_factorial import design_matrix, factorial_probability_arms, full_model_arm
from .partc_design import PROTOCOL as PARTC_PROTOCOL
from .physical import fit_training_envelope, kinematic_features, physical_summary


ROOT = Path(__file__).resolve().parents[1]


def _actor_mask(dataset, scene_indices: list[int]) -> torch.Tensor:
    ranges = torch.as_tensor(dataset.seq_start_end, dtype=torch.long)
    if ranges.ndim != 2 or ranges.shape != (len(dataset), 2):
        raise RuntimeError("unexpected scene-to-actor index shape")
    if int(ranges[0, 0]) != 0 or not torch.equal(ranges[1:, 0], ranges[:-1, 1]):
        raise RuntimeError("scene-to-actor indices are not contiguous")
    scene_selected = torch.zeros(len(dataset), dtype=torch.bool)
    scene_selected[torch.as_tensor(scene_indices, dtype=torch.long)] = True
    sizes = ranges[:, 1] - ranges[:, 0]
    mask = torch.repeat_interleave(scene_selected, sizes)
    if mask.numel() != dataset.obs_traj.shape[0]:
        raise RuntimeError("scene-to-actor expansion is misaligned")
    return mask


def _training_envelope(dataset, scene_indices: list[int]) -> dict[str, object]:
    mask = _actor_mask(dataset, scene_indices)
    positions = dataset.pred_traj[mask].permute(0, 2, 1)
    initial = dataset.obs_traj[mask, :, -1]
    features = kinematic_features(
        positions,
        initial_position=initial,
        stride_seconds=5.0,
    )
    envelope = fit_training_envelope(features)
    return {
        "actors": int(mask.sum()),
        "scene_count": len(scene_indices),
        "stride_seconds": 5.0,
        "envelope": envelope,
    }


class PhysicalAccumulator:
    def __init__(self) -> None:
        self.actors = 0
        self.sums: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

    def update(self, summary: dict[str, dict[str, float]], actors: int) -> None:
        self.actors += actors
        for feature, values in summary.items():
            for metric, value in values.items():
                self.sums[feature][metric] += value * actors

    def summary(self) -> dict[str, object]:
        if self.actors == 0:
            raise RuntimeError("empty physical accumulator")
        return {
            "actors": self.actors,
            "features": {
                feature: {
                    metric: value / self.actors
                    for metric, value in metrics.items()
                }
                for feature, metrics in self.sums.items()
            },
        }


@torch.inference_mode()
def run(
    *,
    fold: int,
    device: torch.device,
    workers: int,
    batch_size: int,
    max_train_scenes: int | None,
    max_validation_scenes: int | None,
) -> dict[str, object]:
    if fold not in (1, 2):
        raise ValueError("current frozen checkpoint pairs exist only for folds 1 and 2")
    freeze = verify_legacy_freeze()
    if not freeze["ok"]:
        raise RuntimeError("legacy input freeze verification failed")
    protocol = load_legacy_data_protocol()
    protocol.assert_boundaries()
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    dataset = load_dataset(protocol)
    train_data, validation_data, validation_dates = fold_subsets(
        protocol,
        dataset,
        fold,
        max_train_scenes=max_train_scenes,
        max_validation_scenes=max_validation_scenes,
    )
    physical_envelope = _training_envelope(dataset, list(train_data.indices))
    validation_loader = loader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        workers=workers,
        prefetch=4,
    )
    source_model, target_model, source_path, target_path = _load_models(fold, device)
    arm_states = {arm["arm"]: ArmAccumulator() for arm in design_matrix()}
    source_state = ArmAccumulator()
    physical_states = {
        "original_ascent": PhysicalAccumulator(),
        "mabpt_ascent": PhysicalAccumulator(),
    }
    scene_cursor = 0
    full_arm = full_model_arm()
    for data in validation_loader:
        data = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in data.items()
        }
        truth = data["pred_traj"].transpose(1, 0)
        source_support, source_logits, _ = source_model(data)
        source_probabilities = source_logits.softmax(dim=1)
        source_decision = source_logits.argmax(dim=1)
        source_metrics = compute_batch_metrics(
            source_support,
            source_probabilities,
            source_decision,
            truth,
        )
        target_support, _, target_decision, auxiliary = target_model(data)
        cross_cost = support_cost(source_support, target_support)
        pairwise = pairwise_trajectory_distance(target_support) / DEFAULT_ADE_SCALE
        predicted_risk = auxiliary["centered_predicted_normalized_ade_risk"].to(
            torch.float64
        )
        arms, _ = factorial_probability_arms(
            source_probabilities,
            cross_cost,
            predicted_risk,
            pairwise,
        )

        packed = pack_scenes(data["adj"])
        dates = validation_dates[scene_cursor : scene_cursor + packed.scene_count]
        if len(dates) != packed.scene_count:
            raise RuntimeError("scene-date alignment failed")
        actor_dates = np.asarray(dates, dtype=object)[
            packed.inverse.detach().cpu().numpy()
        ]
        endpoint_travel = torch.linalg.vector_norm(
            truth[:, -1] - data["obs_traj"][-1], dim=-1
        )
        tail = endpoint_travel >= target_tail_threshold(dataset)
        source_state.update(source_metrics, source_probabilities, actor_dates, tail)
        target_support_fp64 = target_support.to(torch.float64)
        truth_fp64 = truth.to(torch.float64)
        for arm, probabilities in arms.items():
            metrics = compute_batch_metrics(
                target_support_fp64,
                probabilities,
                target_decision,
                truth_fp64,
            )
            arm_states[arm].update(metrics, probabilities, actor_dates, tail)

        initial = data["obs_traj"][-1]
        source_physical = kinematic_features(
            source_support, initial_position=initial, stride_seconds=5.0
        )
        target_physical = kinematic_features(
            target_support, initial_position=initial, stride_seconds=5.0
        )
        envelope = physical_envelope["envelope"]
        actors = int(source_support.shape[0])
        physical_states["original_ascent"].update(
            physical_summary(
                source_physical,
                envelope,
                mode_probabilities=source_probabilities,
            ),
            actors,
        )
        physical_states["mabpt_ascent"].update(
            physical_summary(
                target_physical,
                envelope,
                mode_probabilities=arms[full_arm],
            ),
            actors,
        )
        scene_cursor += packed.scene_count

    if scene_cursor != len(validation_dates):
        raise RuntimeError("evaluation did not consume the complete validation cohort")
    arm_summaries = {name: state.summary() for name, state in arm_states.items()}
    return {
        "format_version": 1,
        "model": "MABPT-ASCENT",
        "experiment_ids": ["factorial_fusion", "physical_plausibility"],
        "evidence_class": "retrospective_development_only",
        "partc_protocol": str(PARTC_PROTOCOL.relative_to(ROOT)),
        "partc_protocol_sha256": _sha256(PARTC_PROTOCOL),
        "fold": fold,
        "training_scenes_for_physical_envelope": len(train_data),
        "validation_scenes": len(validation_data),
        "original_ascent": source_state.summary(),
        "factorial_arms": arm_summaries,
        "full_model_arm": full_arm,
        "physical_envelope": physical_envelope,
        "physical": {
            name: state.summary() for name, state in physical_states.items()
        },
        "inputs": {
            "source_checkpoint": source_path,
            "target_checkpoint": target_path,
        },
        "integrity": {
            "legacy_freeze_verified": True,
            "target_in_probability_forward": False,
            "physical_envelope_uses_training_scenes_only": True,
            "all_factorial_arms_reported": len(arm_summaries) == 8,
        },
        "runtime": {
            "device": str(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
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
    parser.add_argument("--fold", type=int, choices=(1, 2), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-train-scenes", type=int)
    parser.add_argument("--max-validation-scenes", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.smoke:
        if args.max_train_scenes is None:
            args.max_train_scenes = 256
        if args.max_validation_scenes is None:
            args.max_validation_scenes = 8
    result = run(
        fold=args.fold,
        device=torch.device(args.device),
        workers=args.workers,
        batch_size=args.batch_size,
        max_train_scenes=args.max_train_scenes,
        max_validation_scenes=args.max_validation_scenes,
    )
    if args.output is None:
        suffix = "smoke" if args.smoke else "formal"
        args.output = (
            ROOT
            / "artifacts/mabpt_partc_20260811"
            / f"factorial_physical_fold{args.fold}_{suffix}_v1.json"
        )
    _atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "fold": result["fold"],
                "actors": result["factorial_arms"][result["full_model_arm"]]["actors"],
                "elapsed_seconds": result["runtime"]["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
