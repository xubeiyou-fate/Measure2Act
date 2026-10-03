"""Zero-shot MABPT evaluation on registered official and external views."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset

from experiments.energy_predict_optimize.evaluation import (
    RankingMetricAccumulator,
    compute_batch_metrics,
)
from model.utils import TrajectoryDataset, seed_worker, seq_collate

from .evaluate import _load_models
from .freeze import verify as verify_legacy_freeze
from .operator import (
    DEFAULT_ADE_SCALE,
    energy_kl_projection,
    exact_gibbs_transport,
    pairwise_trajectory_distance,
    support_cost,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("e1_external_protocol.json")
ARMS = (
    "constant_velocity",
    "ascent_native",
    "target_native_logits",
    "target_energy_probabilities",
    "mabpt",
)


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


def _limited_dataset(dataset: TrajectoryDataset, maximum: int | None):
    if maximum is None or maximum >= len(dataset):
        return dataset
    indices = (
        torch.linspace(0, len(dataset) - 1, maximum)
        .round()
        .long()
        .unique()
        .tolist()
    )
    return Subset(dataset, indices)


def _constant_velocity(
    data: dict[str, torch.Tensor],
    *,
    prediction_stride_seconds: int = 5,
    forecast_horizon_seconds: int = 120,
) -> torch.Tensor:
    observation = data["obs_traj"]
    velocity = observation[-1] - observation[-2]
    seconds = torch.arange(
        prediction_stride_seconds,
        forecast_horizon_seconds + 1,
        prediction_stride_seconds,
        device=observation.device,
        dtype=observation.dtype,
    )
    return (observation[-1, :, None] + velocity[:, None] * seconds[None, :, None])[:, None]


def _mabpt_probability(
    source_support: torch.Tensor,
    source_probability: torch.Tensor,
    target_support: torch.Tensor,
    predicted_risk: torch.Tensor,
) -> torch.Tensor:
    transported = exact_gibbs_transport(
        source_probability.to(torch.float64),
        support_cost(source_support, target_support),
        mass_weighted=True,
    )["transported"]
    pairwise = pairwise_trajectory_distance(target_support) / DEFAULT_ADE_SCALE
    return energy_kl_projection(
        transported,
        predicted_risk.to(torch.float64),
        pairwise,
    )[0]


@torch.inference_mode()
def run(
    *,
    dataset_name: str,
    device: torch.device,
    workers: int,
    batch_size: int,
    max_scenes: int | None,
    protocol_path: Path = PROTOCOL,
) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if dataset_name not in protocol["datasets"]:
        raise ValueError(f"dataset is outside the frozen E1 registry: {dataset_name}")
    if not verify_legacy_freeze()["ok"]:
        raise RuntimeError("legacy C165 freeze verification failed")
    manifest = ROOT / protocol["external_view_manifest"]["path"]
    if _sha256(manifest) != protocol["external_view_manifest"]["sha256"]:
        raise RuntimeError("external view manifest hash mismatch")
    specification = protocol["datasets"][dataset_name]
    evaluation = protocol.get("evaluation", {})
    observation_steps = int(evaluation.get("observation_steps", 16))
    forecast_horizon = int(evaluation.get("forecast_horizon_seconds", 120))
    prediction_stride = int(evaluation.get("prediction_stride_seconds", 5))
    native_stride = int(evaluation.get("mabpt_native_prediction_stride_seconds", 5))
    if forecast_horizon != 120 or prediction_stride % native_stride:
        raise ValueError("external E1 metric grid is incompatible with native MABPT support")
    support_indices = evaluation.get("mabpt_fixed_output_indices_zero_based")
    if support_indices is None:
        support_indices = list(
            range(prediction_stride // native_stride - 1, 24, prediction_stride // native_stride)
        )
    support_indices = [int(index) for index in support_indices]
    expected_targets = forecast_horizon // prediction_stride
    if len(support_indices) != expected_targets or support_indices[-1] >= 24:
        raise ValueError("external E1 fixed MABPT output indices are invalid")
    dataset = TrajectoryDataset(
        (ROOT / specification["path"]).as_posix(),
        obs_len=observation_steps,
        obs_steps=1,
        pred_len=forecast_horizon,
        pred_step=prediction_stride,
        delim=specification["delimiter"],
    )
    evaluation_dataset = _limited_dataset(dataset, max_scenes)
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
    torch.use_deterministic_algorithms(True)
    source, target_model, source_path, target_path = _load_models(1, device)
    states = {arm: RankingMetricAccumulator() for arm in ARMS}
    started = time.perf_counter()
    for data in data_loader:
        data = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in data.items()
        }
        truth = data["pred_traj"].transpose(1, 0).to(torch.float64)
        source_support, source_logits, _ = source(data)
        source_probability = source_logits.softmax(dim=1)
        target_support, target_energy, target_decision, auxiliary = target_model(data)
        target_native = auxiliary["decision_logits"].softmax(dim=1)
        mabpt_probability = _mabpt_probability(
            source_support,
            source_probability,
            target_support,
            auxiliary["centered_predicted_normalized_ade_risk"],
        )
        output_index = torch.as_tensor(support_indices, device=device)
        source_metric_support = source_support.index_select(2, output_index)
        target_metric_support = target_support.index_select(2, output_index)
        cv_support = _constant_velocity(
            data,
            prediction_stride_seconds=prediction_stride,
            forecast_horizon_seconds=forecast_horizon,
        ).to(torch.float64)
        cv_probability = torch.ones(
            (truth.shape[0], 1), device=device, dtype=torch.float64
        )
        measures = {
            "constant_velocity": (cv_support, cv_probability, torch.zeros(truth.shape[0], device=device, dtype=torch.long)),
            "ascent_native": (source_metric_support, source_probability, source_logits.argmax(dim=1)),
            "target_native_logits": (target_metric_support, target_native, target_decision),
            "target_energy_probabilities": (target_metric_support, target_energy, target_decision),
            "mabpt": (target_metric_support, mabpt_probability, target_decision),
        }
        for arm, (support, probability, decision) in measures.items():
            states[arm].update(
                compute_batch_metrics(
                    support.to(torch.float64),
                    probability.to(torch.float64),
                    decision,
                    truth,
                )
            )
    return {
        "format_version": 1,
        "model": "MABPT",
        "model_version": "1.0.0",
        "experiment_id": "E1",
        "dataset": dataset_name,
        "dataset_class": specification["class"],
        "scenes": len(evaluation_dataset),
        "arms": {arm: states[arm].summarize() for arm in ARMS},
        "protocol_sha256": _sha256(protocol_path),
        "evaluation": {
            "observation_steps": observation_steps,
            "forecast_horizon_seconds": forecast_horizon,
            "prediction_stride_seconds": prediction_stride,
            "mabpt_native_prediction_stride_seconds": native_stride,
            "mabpt_fixed_output_indices_zero_based": support_indices,
        },
        "inputs": {
            "source_checkpoint": source_path,
            "target_checkpoint": target_path,
            "external_view_manifest": protocol["external_view_manifest"],
        },
        "integrity": {
            "zero_shot": True,
            "external_finetuning_or_calibration": False,
            "target_in_probability_forward": False,
            "residual_or_gate_used": False,
            "temperature_or_weight_search_used": False,
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
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    if args.smoke and args.max_scenes is None:
        args.max_scenes = 8
    result = run(
        dataset_name=args.dataset,
        device=torch.device(args.device),
        workers=args.workers,
        batch_size=args.batch_size,
        max_scenes=args.max_scenes,
        protocol_path=args.protocol.resolve(),
    )
    if args.output is None:
        suffix = "smoke" if args.smoke else "formal"
        args.output = ROOT / "artifacts/mabpt" / f"e1_{args.dataset}_{suffix}_v1.json"
    _atomic_json(args.output, result)
    print(json.dumps({
        "output": str(args.output),
        "dataset": args.dataset,
        "actors": result["arms"]["mabpt"]["agents"],
        "energy": {arm: result["arms"][arm]["energy_score"] for arm in ARMS},
    }, indent=2))


if __name__ == "__main__":
    main()
