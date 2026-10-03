"""Paired evaluation of registered MABPT experiments E6-E8."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from experiments.edfa_ascent.relation import pack_scenes
from experiments.metric_exact.evaluation import target_tail_threshold
from experiments.energy_predict_optimize.evaluation import compute_batch_metrics
from experiments.ascent_recomparison.common import fold_subsets, load_dataset, loader
from experiments.tpmo_ascent.protocol import load_protocol as load_data_protocol

from .controls import build_control_model
from .ensemble_controls import (
    exact_weighted_kmedoids_compression,
    hungarian_aligned_average,
    union_measure,
)
from .evaluate import ArmAccumulator, _load_models
from .freeze import verify as verify_legacy_freeze
from .operator import (
    DEFAULT_ADE_SCALE,
    energy_kl_projection,
    exact_gibbs_transport,
    pairwise_trajectory_distance,
    support_cost,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("e6_e8_protocol.json")
ARMS = (
    "ascent_seed42",
    "ascent_seed7_epoch20",
    "e6_union10",
    "e6_compressed5",
    "e6_hungarian_average5",
    "e7_widened_ascent",
    "e7_shared_encoder_dual_decoder10",
    "e8_equal_update_union10",
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


def _control_checkpoint(variant: str, fold: int, epoch: int) -> Path:
    return (
        ROOT
        / "runs/mabpt"
        / f"{variant}_fold{fold}_seed7_formal"
        / f"epoch{epoch}.pt"
    )


def _load_control(
    variant: str, fold: int, epoch: int, device: torch.device
) -> tuple[torch.nn.Module, Path, dict[str, object]]:
    path = _control_checkpoint(variant, fold, epoch)
    if not path.is_file():
        raise FileNotFoundError(f"missing registered E6-E8 checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint["protocol_sha256"] != _sha256(PROTOCOL):
        raise RuntimeError(f"control checkpoint protocol mismatch: {path}")
    if int(checkpoint["epoch"]) != epoch:
        raise RuntimeError(f"control checkpoint epoch mismatch: {path}")
    model = build_control_model(variant, batch_size=512).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval(), path, checkpoint


def _mabpt_probabilities(
    source_support: torch.Tensor,
    source_probability: torch.Tensor,
    target_support: torch.Tensor,
    predicted_risk: torch.Tensor,
) -> torch.Tensor:
    prior = exact_gibbs_transport(
        source_probability.to(torch.float64),
        support_cost(source_support, target_support),
        mass_weighted=True,
    )["transported"]
    pairwise = pairwise_trajectory_distance(target_support) / DEFAULT_ADE_SCALE
    return energy_kl_projection(
        prior,
        predicted_risk.to(torch.float64),
        pairwise,
    )[0]


@torch.inference_mode()
def run(
    *,
    fold: int,
    device: torch.device,
    workers: int,
    batch_size: int,
    max_validation_scenes: int | None,
) -> dict[str, object]:
    if fold not in (1, 2):
        raise ValueError("E6-E8 are frozen to folds 1 and 2")
    if not verify_legacy_freeze()["ok"]:
        raise RuntimeError("legacy C165 freeze verification failed")
    data_protocol = load_data_protocol()
    data_protocol.assert_boundaries()
    dataset = load_dataset(data_protocol)
    _, validation_data, validation_dates = fold_subsets(
        data_protocol,
        dataset,
        fold,
        max_validation_scenes=max_validation_scenes,
    )
    validation_loader = loader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        workers=workers,
        prefetch=4,
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(True)
    source_one, target_model, source_path, target_path = _load_models(fold, device)
    source_two20, source_two20_path, checkpoint20 = _load_control(
        "b0_extended", fold, 20, device
    )
    source_two25, source_two25_path, checkpoint25 = _load_control(
        "b0_extended", fold, 25, device
    )
    widened, widened_path, _ = _load_control(
        "widened_ascent", fold, 20, device
    )
    shared, shared_path, _ = _load_control(
        "shared_encoder_dual_decoder", fold, 20, device
    )

    states = {arm: ArmAccumulator() for arm in ARMS}
    tail_threshold = target_tail_threshold(dataset)
    scene_cursor = 0
    compression_cost = 0.0
    compression_actors = 0
    started = time.perf_counter()
    for data in validation_loader:
        data = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in data.items()
        }
        target = data["pred_traj"].transpose(1, 0)
        support_one, logits_one, _ = source_one(data)
        probability_one = logits_one.softmax(dim=1)
        support_two20, logits_two20, _ = source_two20(data)
        probability_two20 = logits_two20.softmax(dim=1)
        support_two25, logits_two25, _ = source_two25(data)
        probability_two25 = logits_two25.softmax(dim=1)
        target_support, _, target_decision, target_aux = target_model(data)
        mabpt_probability = _mabpt_probabilities(
            support_one,
            probability_one,
            target_support,
            target_aux["centered_predicted_normalized_ade_risk"],
        )

        union20_support, union20_probability = union_measure(
            support_one, probability_one, support_two20, probability_two20
        )
        compressed_support, compressed_probability, compressed_aux = (
            exact_weighted_kmedoids_compression(
                union20_support, union20_probability, output_modes=5
            )
        )
        aligned_support, aligned_probability, _ = hungarian_aligned_average(
            support_one, probability_one, support_two20, probability_two20
        )
        union25_support, union25_probability = union_measure(
            support_one, probability_one, support_two25, probability_two25
        )
        widened_support, widened_logits, _ = widened(data)
        widened_probability = widened_logits.softmax(dim=1)
        shared_support, shared_logits, _ = shared(data)
        shared_probability = shared_logits.softmax(dim=1)

        measures = {
            "ascent_seed42": (support_one, probability_one, logits_one.argmax(dim=1)),
            "ascent_seed7_epoch20": (
                support_two20, probability_two20, logits_two20.argmax(dim=1)
            ),
            "e6_union10": (
                union20_support, union20_probability, union20_probability.argmax(dim=1)
            ),
            "e6_compressed5": (
                compressed_support,
                compressed_probability,
                compressed_probability.argmax(dim=1),
            ),
            "e6_hungarian_average5": (
                aligned_support, aligned_probability, aligned_probability.argmax(dim=1)
            ),
            "e7_widened_ascent": (
                widened_support, widened_probability, widened_logits.argmax(dim=1)
            ),
            "e7_shared_encoder_dual_decoder10": (
                shared_support, shared_probability, shared_logits.argmax(dim=1)
            ),
            "e8_equal_update_union10": (
                union25_support, union25_probability, union25_probability.argmax(dim=1)
            ),
            "mabpt": (target_support, mabpt_probability, target_decision),
        }
        packed = pack_scenes(data["adj"])
        dates = validation_dates[scene_cursor : scene_cursor + packed.scene_count]
        actor_dates = np.asarray(dates, dtype=object)[
            packed.inverse.detach().cpu().numpy()
        ]
        tail = torch.linalg.vector_norm(
            target[:, -1] - data["obs_traj"][-1], dim=-1
        ) >= tail_threshold
        for arm, (support, probability, decision) in measures.items():
            metrics = compute_batch_metrics(
                support.to(torch.float64),
                probability.to(torch.float64),
                decision,
                target.to(torch.float64),
            )
            states[arm].update(metrics, probability, actor_dates, tail)
        compression_cost += float(
            compressed_aux["weighted_reconstruction_cost"].sum().cpu()
        )
        compression_actors += int(target.shape[0])
        scene_cursor += packed.scene_count
    if scene_cursor != len(validation_dates):
        raise RuntimeError("E6-E8 scene/date alignment failed")

    train_scenes = len(fold_subsets(data_protocol, dataset, fold)[0])
    b256 = math.ceil(train_scenes / 256)
    b1024 = math.ceil(train_scenes / 1024)
    mabpt_updates = 40 * b256 + 20 * b1024
    equal_control_updates = 20 * b256 + int(checkpoint25["training_updates"])
    if mabpt_updates != equal_control_updates:
        raise RuntimeError(
            f"E8 update budget mismatch: {mabpt_updates} != {equal_control_updates}"
        )
    return {
        "format_version": 1,
        "model": "MABPT",
        "model_version": "1.0.0",
        "experiment_ids": ["E6", "E7", "E8"],
        "evidence_class": "retrospective_train_date_blocked_controls",
        "fold": fold,
        "protocol_sha256": _sha256(PROTOCOL),
        "validation_scenes": len(validation_dates),
        "arms": {arm: states[arm].summary() for arm in ARMS},
        "diagnostics": {
            "mean_compression_reconstruction_cost": compression_cost / compression_actors,
            "mabpt_training_updates": mabpt_updates,
            "equal_compute_control_training_updates": equal_control_updates,
            "ordinary_ensemble_training_updates": 40 * b256,
        },
        "inputs": {
            "ascent_seed42": source_path,
            "mabpt_target": target_path,
            "ascent_seed7_epoch20": source_two20_path.relative_to(ROOT).as_posix(),
            "ascent_seed7_epoch25": source_two25_path.relative_to(ROOT).as_posix(),
            "widened_ascent": widened_path.relative_to(ROOT).as_posix(),
            "shared_encoder_dual_decoder": shared_path.relative_to(ROOT).as_posix(),
        },
        "integrity": {
            "legacy_freeze_verified": True,
            "target_in_ensemble_operator": False,
            "residual_or_gate_used": False,
            "temperature_or_checkpoint_search_used": False,
            "equal_update_count_exact": True,
            "locked_test_used": False,
        },
        "runtime": {
            "device": str(device),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=(1, 2), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-validation-scenes", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
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
        args.output = ROOT / "artifacts/mabpt" / f"e6_e8_fold{args.fold}_{suffix}.json"
    _atomic_json(args.output, result)
    print(json.dumps({
        "output": str(args.output),
        "fold": args.fold,
        "actors": result["arms"]["mabpt"]["actors"],
        "energy": {arm: result["arms"][arm]["energy_score"] for arm in ARMS},
    }, indent=2))


if __name__ == "__main__":
    main()
