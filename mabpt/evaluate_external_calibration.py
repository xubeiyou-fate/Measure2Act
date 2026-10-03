"""Apply the frozen E11 event definition to registered zero-shot views."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from experiments.energy_predict_optimize.evaluation import compute_batch_metrics
from model.utils import TrajectoryDataset, seed_worker, seq_collate

from .calibration import (
    CalibrationAccumulator,
    continuous_mixture_nll,
    event_probabilities,
    kernel_event_probabilities,
)
from .evaluate import _load_models
from .events import event_labels, event_membership
from .freeze import verify as verify_legacy_freeze
from .operator import (
    DEFAULT_ADE_SCALE,
    energy_kl_projection,
    exact_gibbs_transport,
    pairwise_trajectory_distance,
    support_cost,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("e11_external_protocol.json")


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


def _limit(dataset, maximum: int | None):
    if maximum is None or maximum >= len(dataset):
        return dataset
    index = torch.linspace(0, len(dataset) - 1, maximum).round().long().unique().tolist()
    return Subset(dataset, index)


@torch.inference_mode()
def run(
    *,
    dataset_name: str,
    device: torch.device,
    workers: int,
    batch_size: int,
    max_scenes: int | None,
) -> dict[str, object]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    model_protocol_path = ROOT / protocol["model_protocol"]["path"]
    base_protocol_path = ROOT / protocol["base_protocol"]["path"]
    definition_path = ROOT / protocol["event_definition"]["path"]
    for path, expected in (
        (model_protocol_path, protocol["model_protocol"]["sha256"]),
        (base_protocol_path, protocol["base_protocol"]["sha256"]),
        (definition_path, protocol["event_definition"]["sha256"]),
    ):
        if _sha256(path) != expected:
            raise RuntimeError(f"E11 external input hash mismatch: {path}")
    model_protocol = json.loads(model_protocol_path.read_text(encoding="utf-8"))
    base_protocol = json.loads(base_protocol_path.read_text(encoding="utf-8"))
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    if dataset_name not in protocol["datasets"]:
        raise ValueError("dataset is outside the frozen external E11 registry")
    if not verify_legacy_freeze()["ok"]:
        raise RuntimeError("legacy C165 freeze verification failed")
    specification = model_protocol["datasets"][dataset_name]
    dataset = TrajectoryDataset(
        (ROOT / specification["path"]).as_posix(),
        obs_len=16,
        obs_steps=1,
        pred_len=120,
        pred_step=5,
        delim=specification["delimiter"],
    )
    evaluation_dataset = _limit(dataset, max_scenes)
    data_loader = DataLoader(
        evaluation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=seq_collate,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    source, target_model, source_path, target_path = _load_models(1, device)
    states = {"ascent": CalibrationAccumulator(), "mabpt": CalibrationAccumulator()}
    variance = torch.tensor(
        definition["constant_velocity_residual_variance_km2"],
        dtype=torch.float64,
        device=device,
    )
    turn_threshold = float(definition["turn_threshold_radians"])
    altitude_threshold = float(definition["altitude_threshold_km"])
    duplicate_threshold = float(base_protocol["duplicate_threshold_mean_distance_km"])
    started = time.perf_counter()
    for data in data_loader:
        data = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in data.items()
        }
        observations = data["obs_traj"].transpose(1, 0)
        target = data["pred_traj"].transpose(1, 0)
        source_support, source_logits, _ = source(data)
        source_probability = source_logits.softmax(dim=1)
        source_metrics = compute_batch_metrics(
            source_support, source_probability, source_logits.argmax(dim=1), target
        )
        target_support, _, target_decision, auxiliary = target_model(data)
        transported = exact_gibbs_transport(
            source_probability,
            support_cost(source_support, target_support),
            mass_weighted=True,
        )["transported"]
        mabpt_probability = energy_kl_projection(
            transported,
            auxiliary["centered_predicted_normalized_ade_risk"].to(torch.float64),
            pairwise_trajectory_distance(target_support) / DEFAULT_ADE_SCALE,
        )[0]
        mabpt_metrics = compute_batch_metrics(
            target_support.to(torch.float64),
            mabpt_probability,
            target_decision,
            target.to(torch.float64),
        )
        truth = event_labels(
            observations,
            target,
            turn_threshold=turn_threshold,
            altitude_threshold=altitude_threshold,
        )
        actor_dates = np.full(truth.shape[0], dataset_name, dtype=object)
        for name, predictions, probabilities, metrics in (
            ("ascent", source_support, source_probability, source_metrics),
            ("mabpt", target_support, mabpt_probability, mabpt_metrics),
        ):
            hard_events = event_labels(
                observations,
                predictions,
                turn_threshold=turn_threshold,
                altitude_threshold=altitude_threshold,
            )
            hard_probability = event_probabilities(
                probabilities.to(torch.float64), hard_events
            )
            rows = torch.arange(truth.shape[0], device=truth.device)
            membership = event_membership(
                observations,
                predictions,
                turn_threshold=turn_threshold,
                altitude_threshold=altitude_threshold,
            )
            event_probability = kernel_event_probabilities(probabilities, membership)
            pairwise = pairwise_trajectory_distance(predictions)
            states[name].update(
                event_probability=event_probability,
                event_label=truth,
                mixture_nll=continuous_mixture_nll(
                    predictions, probabilities, target, variance
                ),
                trajectory_probability=probabilities,
                metrics=metrics,
                pairwise=pairwise,
                actor_dates=actor_dates,
                duplicate_threshold=duplicate_threshold,
                hard_zero_event=hard_probability[rows, truth] == 0,
            )
    summaries = {
        name: state.summary(
            calibration_bins=int(base_protocol["calibration_bins"]),
            reliability_bins=int(base_protocol["reliability_bins"]),
        )
        for name, state in states.items()
    }
    gains = {
        metric: (summaries["ascent"][metric] - summaries["mabpt"][metric])
        / summaries["ascent"][metric]
        for metric in ("event_nll", "event_brier", "mixture_nll", "energy_score")
    }
    return {
        "format_version": 1,
        "model": "MABPT",
        "experiment_id": "E11_external",
        "dataset": dataset_name,
        "dataset_class": specification["class"],
        "scenes": len(evaluation_dataset),
        "protocol_sha256": _sha256(PROTOCOL),
        "definition": protocol["event_definition"],
        "results": {**summaries, "relative_gain_mabpt_vs_ascent": gains},
        "inputs": {"source_checkpoint": source_path, "target_checkpoint": target_path},
        "integrity": {
            "external_fitting": False,
            "target_in_probability_forward": False,
            "residual_or_gate_used": False,
            "fresh_confirmatory_test": False,
        },
        "runtime": {
            "device": str(device),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
        },
        "claim_boundary": protocol["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.smoke and args.max_scenes is None:
        args.max_scenes = 8
    result = run(
        dataset_name=args.dataset,
        device=torch.device(args.device),
        workers=args.workers,
        batch_size=args.batch_size,
        max_scenes=args.max_scenes,
    )
    if args.output is None:
        suffix = "smoke" if args.smoke else "formal"
        args.output = ROOT / "artifacts/mabpt" / f"e11_{args.dataset}_{suffix}_v1.json"
    _atomic_json(args.output, result)
    print(json.dumps({
        "output": str(args.output),
        "dataset": args.dataset,
        "actors": result["results"]["mabpt"]["actors"],
        "relative_gain": result["results"]["relative_gain_mabpt_vs_ascent"],
    }, indent=2))


if __name__ == "__main__":
    main()
