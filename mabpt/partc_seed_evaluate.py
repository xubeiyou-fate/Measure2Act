"""Evaluate one fixed-seed MABPT-ASCENT model on the development cohort."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from experiments.edfa_ascent.relation import pack_scenes
from experiments.metric_exact.evaluation import target_tail_threshold
from experiments.metric_exact.model import build_model as build_ascent_model
from experiments.energy_predict_optimize.evaluation import compute_batch_metrics
from experiments.energy_predict_optimize.model import build_model as build_target_model
from model.utils import TrajectoryDataset, seed_worker, seq_collate

from .calibration import (
    CalibrationAccumulator,
    continuous_mixture_nll,
    event_probabilities,
    kernel_event_probabilities,
)
from .evaluate import ArmAccumulator, _atomic_json, _sha256
from .events import event_labels, event_membership
from .operator import DEFAULT_ADE_SCALE, pairwise_trajectory_distance, support_cost
from .partc_design import PROTOCOL, load_protocol
from .partc_factorial import factorial_probability_arms, full_model_arm
from .partc_evaluate import PhysicalAccumulator, _training_envelope
from .physical import kinematic_features, physical_summary
from .traffic import PairAccumulator, pair_conflict_probabilities


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/mabpt_partc_20260811"
EVENT_DEFINITION = (
    ROOT / "artifacts/mabpt_partc_20260811/event_definition_train_v1.json"
)
CALIBRATION_PROTOCOL = Path(__file__).with_name("e11_protocol_v2.json")
CONFLICT_PROTOCOL = Path(__file__).with_name("e12_protocol.json")


def _indices(length: int, maximum: int | None) -> list[int]:
    if maximum is None or maximum >= length:
        return list(range(length))
    if maximum < 1:
        raise ValueError("scene limit must be positive")
    return (
        torch.linspace(0, length - 1, steps=maximum)
        .round()
        .long()
        .unique()
        .tolist()
    )


def _data(max_train_scenes: int | None, max_development_scenes: int | None):
    protocol = load_protocol()
    train = TrajectoryDataset(
        (ROOT / protocol["data"]["train"]).as_posix(),
        obs_len=16,
        obs_steps=1,
        pred_len=120,
        pred_step=5,
        delim=" ",
    )
    development = TrajectoryDataset(
        (ROOT / protocol["data"]["development"]).as_posix(),
        obs_len=16,
        obs_steps=1,
        pred_len=120,
        pred_step=5,
        delim=" ",
    )
    date_path = ROOT / "artifacts/c97_ikd_ascent/dev_scene_dates.json"
    dates = json.loads(date_path.read_text(encoding="utf-8"))["dates"]
    if len(dates) != len(development):
        raise RuntimeError("development scene-date index mismatch")
    train_indices = _indices(len(train), max_train_scenes)
    development_indices = _indices(len(development), max_development_scenes)
    return (
        train,
        train_indices,
        Subset(development, development_indices),
        [dates[index] for index in development_indices],
        date_path,
    )


def _loader(dataset, *, batch_size: int, workers: int) -> DataLoader:
    options = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": seq_collate,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
    }
    if workers > 0:
        options.update({"persistent_workers": True, "prefetch_factor": 4})
    return DataLoader(**options)


def _default_source(seed: int) -> Path:
    return (
        ROOT
        / "runs/metric_exact"
        / f"P3_B0_signed_coupled_all_train_seed{seed}_formal"
        / "last.pt"
    )


def _default_target(seed: int) -> Path:
    return (
        RUN_ROOT
        / f"mabpt_ascent_predicted_risk_seed{seed}_formal"
        / "epoch_020.pt"
    )


def _load_model_pair(
    *,
    source_checkpoint: Path,
    target_checkpoint: Path,
    device: torch.device,
    batch_size: int,
):
    for path in (source_checkpoint, target_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    source = build_ascent_model("B0_signed_coupled", batch_size=batch_size).to(device)
    source_state = torch.load(
        source_checkpoint, map_location=device, weights_only=False
    )
    source.load_state_dict(source_state["model_state_dict"])
    target = build_target_model(batch_size=batch_size).to(device)
    target_state = torch.load(
        target_checkpoint, map_location=device, weights_only=False
    )
    if target_state.get("stage") != "predicted_risk":
        raise RuntimeError("target checkpoint is not a predicted-risk stage checkpoint")
    target.load_state_dict(target_state["model_state_dict"])
    return source.eval(), target.eval()


def _model_outputs(source, target_model, data):
    source_support, source_logits, _ = source(data)
    source_probabilities = source_logits.softmax(dim=1)
    target_support, _, target_decision, auxiliary = target_model(data)
    arms, _ = factorial_probability_arms(
        source_probabilities,
        support_cost(source_support, target_support),
        auxiliary["centered_predicted_normalized_ade_risk"].to(torch.float64),
        pairwise_trajectory_distance(target_support) / DEFAULT_ADE_SCALE,
    )
    return {
        "original_ascent": {
            "support": source_support,
            "probability": source_probabilities,
            "decision": source_logits.argmax(dim=1),
        },
        "mabpt_ascent": {
            "support": target_support,
            "probability": arms[full_model_arm()],
            "decision": target_decision,
        },
    }


def _conflict_result(outputs, truth, adjacency, event):
    return {
        model: pair_conflict_probabilities(
            values["support"],
            values["probability"],
            truth,
            adjacency,
            horizontal_threshold=float(event["horizontal_threshold_km"]),
            vertical_threshold=float(event["vertical_threshold_km"]),
            horizontal_scale=float(event["horizontal_kernel_scale_km"]),
            vertical_scale=float(event["vertical_kernel_scale_km"]),
        )
        for model, values in outputs.items()
    }


def _fit_conflict_thresholds(states: dict[str, PairAccumulator]):
    thresholds = {}
    diagnostics = {}
    for model, state in states.items():
        probability, _, label, _ = state.arrays()
        negatives = probability[~label]
        if not len(negatives):
            raise RuntimeError("training split contains no negative conflict pairs")
        threshold = float(np.quantile(negatives, 0.95, method="higher"))
        thresholds[model] = threshold
        diagnostics[model] = {
            "pairs": int(len(label)),
            "positive_pairs": int(label.sum()),
            "negative_pairs": int((~label).sum()),
            "threshold": threshold,
            "training_fpr": float((negatives >= threshold).mean()),
        }
    return thresholds, diagnostics


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
    fixed_seeds = [int(value) for value in protocol["fixed_seeds"]]
    if seed not in fixed_seeds:
        raise RuntimeError("seed lies outside the frozen Part C registry")
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    source_path = (source_checkpoint or _default_source(seed)).resolve()
    target_path = (target_checkpoint or _default_target(seed)).resolve()
    started = time.perf_counter()
    train, train_indices, development, dates, date_path = _data(
        max_train_scenes, max_development_scenes
    )
    envelope = _training_envelope(train, train_indices)
    training_loader = _loader(
        Subset(train, train_indices), batch_size=batch_size, workers=workers
    )
    development_loader = _loader(
        development, batch_size=batch_size, workers=workers
    )
    source, target_model = _load_model_pair(
        source_checkpoint=source_path,
        target_checkpoint=target_path,
        device=device,
        batch_size=batch_size,
    )
    definition = json.loads(EVENT_DEFINITION.read_text(encoding="utf-8"))
    calibration_protocol = json.loads(CALIBRATION_PROTOCOL.read_text(encoding="utf-8"))
    conflict_protocol = json.loads(CONFLICT_PROTOCOL.read_text(encoding="utf-8"))
    if definition["partc_protocol_sha256"] != _sha256(PROTOCOL):
        raise RuntimeError("complete-training event definition protocol mismatch")
    conflict_training_states = {
        model: PairAccumulator() for model in ("original_ascent", "mabpt_ascent")
    }
    threshold_fit_started = time.perf_counter()
    for data in training_loader:
        data = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in data.items()
        }
        truth = data["pred_traj"].transpose(1, 0)
        outputs = _model_outputs(source, target_model, data)
        conflict_results = _conflict_result(
            outputs,
            truth,
            data["adj"],
            conflict_protocol["event"],
        )
        for model, values in conflict_results.items():
            conflict_training_states[model].update(values, None)
    conflict_thresholds, conflict_training_diagnostics = _fit_conflict_thresholds(
        conflict_training_states
    )
    threshold_fit_seconds = time.perf_counter() - threshold_fit_started
    del conflict_training_states, training_loader
    source_state = ArmAccumulator()
    mabpt_state = ArmAccumulator()
    physical_states = {
        "original_ascent": PhysicalAccumulator(),
        "mabpt_ascent": PhysicalAccumulator(),
    }
    calibration_states = {
        model: CalibrationAccumulator()
        for model in ("original_ascent", "mabpt_ascent")
    }
    conflict_states = {
        model: PairAccumulator() for model in ("original_ascent", "mabpt_ascent")
    }
    variance = torch.tensor(
        definition["constant_velocity_residual_variance_km2"],
        dtype=torch.float64,
        device=device,
    )
    tail_threshold = target_tail_threshold(train)
    scene_cursor = 0
    batches = 0
    inference_seconds = 0.0
    for data in development_loader:
        data = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in data.items()
        }
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        batch_started = time.perf_counter()
        truth = data["pred_traj"].transpose(1, 0)
        outputs = _model_outputs(source, target_model, data)
        source_support = outputs["original_ascent"]["support"]
        source_probabilities = outputs["original_ascent"]["probability"]
        source_decision = outputs["original_ascent"]["decision"]
        target_support = outputs["mabpt_ascent"]["support"]
        mabpt_probabilities = outputs["mabpt_ascent"]["probability"]
        target_decision = outputs["mabpt_ascent"]["decision"]
        source_metrics = compute_batch_metrics(
            source_support, source_probabilities, source_decision, truth
        )
        mabpt_metrics = compute_batch_metrics(
            target_support.to(torch.float64),
            mabpt_probabilities,
            target_decision,
            truth.to(torch.float64),
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter() - batch_started

        packed = pack_scenes(data["adj"])
        batch_dates = dates[scene_cursor : scene_cursor + packed.scene_count]
        if len(batch_dates) != packed.scene_count:
            raise RuntimeError("scene-date alignment failed")
        actor_dates = np.asarray(batch_dates, dtype=object)[
            packed.inverse.detach().cpu().numpy()
        ]
        endpoint_travel = torch.linalg.vector_norm(
            truth[:, -1] - data["obs_traj"][-1], dim=-1
        )
        tail = endpoint_travel >= tail_threshold
        source_state.update(source_metrics, source_probabilities, actor_dates, tail)
        mabpt_state.update(mabpt_metrics, mabpt_probabilities, actor_dates, tail)

        observations = data["obs_traj"].transpose(1, 0)
        event_truth = event_labels(
            observations,
            truth,
            turn_threshold=float(definition["turn_threshold_radians"]),
            altitude_threshold=float(definition["altitude_threshold_km"]),
        )
        metric_by_model = {
            "original_ascent": source_metrics,
            "mabpt_ascent": mabpt_metrics,
        }
        for model, values in outputs.items():
            mode_event = event_labels(
                observations,
                values["support"],
                turn_threshold=float(definition["turn_threshold_radians"]),
                altitude_threshold=float(definition["altitude_threshold_km"]),
            )
            hard_probability = event_probabilities(
                values["probability"].to(torch.float64), mode_event
            )
            rows = torch.arange(event_truth.shape[0], device=event_truth.device)
            membership = event_membership(
                observations,
                values["support"],
                turn_threshold=float(definition["turn_threshold_radians"]),
                altitude_threshold=float(definition["altitude_threshold_km"]),
            )
            pairwise = pairwise_trajectory_distance(values["support"])
            calibration_states[model].update(
                event_probability=kernel_event_probabilities(
                    values["probability"], membership
                ),
                event_label=event_truth,
                mixture_nll=continuous_mixture_nll(
                    values["support"], values["probability"], truth, variance
                ),
                trajectory_probability=values["probability"],
                metrics=metric_by_model[model],
                pairwise=pairwise,
                actor_dates=actor_dates,
                duplicate_threshold=float(
                    calibration_protocol["duplicate_threshold_mean_distance_km"]
                ),
                hard_zero_event=hard_probability[rows, event_truth] == 0,
            )

        conflict_results = _conflict_result(
            outputs,
            truth,
            data["adj"],
            conflict_protocol["event"],
        )
        for model, values in conflict_results.items():
            pair_dates = np.asarray(batch_dates, dtype=object)[
                values["pair_scene"].detach().cpu().numpy()
            ]
            conflict_states[model].update(values, pair_dates)

        initial = data["obs_traj"][-1]
        physical_states["original_ascent"].update(
            physical_summary(
                kinematic_features(
                    source_support, initial_position=initial, stride_seconds=5.0
                ),
                envelope["envelope"],
                mode_probabilities=source_probabilities,
            ),
            int(source_support.shape[0]),
        )
        physical_states["mabpt_ascent"].update(
            physical_summary(
                kinematic_features(
                    target_support, initial_position=initial, stride_seconds=5.0
                ),
                envelope["envelope"],
                mode_probabilities=mabpt_probabilities,
            ),
            int(target_support.shape[0]),
        )
        scene_cursor += packed.scene_count
        batches += 1
    if scene_cursor != len(dates):
        raise RuntimeError("evaluation did not consume the complete development cohort")
    source_summary = source_state.summary()
    mabpt_summary = mabpt_state.summary()
    if source_summary["actors"] != mabpt_summary["actors"]:
        raise RuntimeError("paired model actor counts differ")
    calibration = {
        model: state.summary(
            calibration_bins=int(calibration_protocol["calibration_bins"]),
            reliability_bins=int(calibration_protocol["reliability_bins"]),
        )
        for model, state in calibration_states.items()
    }
    calibration["relative_gain_mabpt_vs_ascent"] = {
        metric: (calibration["original_ascent"][metric] - calibration["mabpt_ascent"][metric])
        / calibration["original_ascent"][metric]
        for metric in ("event_nll", "event_brier", "mixture_nll", "energy_score")
    }
    conflict = {
        model: state.summary(alert_threshold=conflict_thresholds[model])
        for model, state in conflict_states.items()
    }
    conflict["relative_gain_mabpt_vs_ascent"] = {
        "brier": (conflict["original_ascent"]["brier"] - conflict["mabpt_ascent"]["brier"])
        / conflict["original_ascent"]["brier"],
        "nll": (conflict["original_ascent"]["nll"] - conflict["mabpt_ascent"]["nll"])
        / conflict["original_ascent"]["nll"],
        "auprc": (conflict["mabpt_ascent"]["auprc"] - conflict["original_ascent"]["auprc"])
        / conflict["original_ascent"]["auprc"],
    }
    return {
        "format_version": 1,
        "model": "MABPT-ASCENT",
        "experiment_ids": [
            "main_development_evaluation",
            "calibration",
            "physical_plausibility",
            "runtime_memory",
            "fixed_event_calibration",
            "conflict_risk_operations",
        ],
        "evidence_class": "development_only",
        "seed": seed,
        "partc_protocol": str(PROTOCOL.relative_to(ROOT)),
        "partc_protocol_sha256": _sha256(PROTOCOL),
        "training_scenes_for_physical_envelope": len(train_indices),
        "development_scenes": len(development),
        "development_dates_sha256": _sha256(date_path),
        "models": {
            "original_ascent": source_summary,
            "mabpt_ascent": mabpt_summary,
        },
        "physical_envelope": envelope,
        "physical": {
            name: state.summary() for name, state in physical_states.items()
        },
        "fixed_event_calibration": calibration,
        "conflict_risk": {
            "training_thresholds": conflict_training_diagnostics,
            "development": conflict,
            "regulatory_claim": False,
        },
        "inputs": {
            "source_checkpoint": str(source_path.relative_to(ROOT)),
            "source_checkpoint_sha256": _sha256(source_path),
            "target_checkpoint": str(target_path.relative_to(ROOT)),
            "target_checkpoint_sha256": _sha256(target_path),
        },
        "integrity": {
            "train_and_development_only": True,
            "historical_locked_test_used": False,
            "target_in_probability_forward": False,
            "matched_seed": True,
            "training_only_physical_envelope": True,
            "event_definition_fit_on_training_only": True,
            "conflict_thresholds_fit_on_training_only": True,
            "post_hoc_calibration": False,
        },
        "runtime": {
            "device": str(device),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "batches": batches,
            "inference_seconds": inference_seconds,
            "conflict_threshold_fit_seconds": threshold_fit_seconds,
            "actors_per_second": source_summary["actors"]
            / max(inference_seconds, 1e-12),
            "total_elapsed_seconds": time.perf_counter() - started,
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
        source_checkpoint=(
            args.source_checkpoint.resolve() if args.source_checkpoint else None
        ),
        target_checkpoint=(
            args.target_checkpoint.resolve() if args.target_checkpoint else None
        ),
    )
    if args.output is None:
        suffix = "smoke" if args.smoke else "formal"
        args.output = (
            ROOT
            / "artifacts/mabpt_partc_20260811"
            / f"seed{args.seed}_development_{suffix}_v1.json"
        )
    _atomic_json(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "seed": args.seed,
                "actors": result["models"]["mabpt_ascent"]["actors"],
                "elapsed_seconds": result["runtime"]["total_elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
