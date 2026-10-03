"""Evaluate trained native-cardinality MABPT models for experiment E9."""

from __future__ import annotations

import argparse
from collections import defaultdict
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
from experiments.ascent_recomparison.protocol import load_protocol as load_data_protocol

from .evaluate import ArmAccumulator
from .evaluate import _load_models as _load_frozen_k5_models
from .freeze import verify as verify_legacy_freeze
from .operator import (
    DEFAULT_ADE_SCALE,
    energy_kl_projection,
    exact_gibbs_transport,
    pairwise_trajectory_distance,
    sinkhorn_transport,
    support_cost,
    top_m_gibbs_transport,
)
from .scalable import ScalableEnergyAscent
from .train_e9 import PROTOCOL_PATH, ROOT, build_model


TOP_M = {3: (6,), 5: (8, 32), 7: (32, 128, 512)}


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


def _checkpoint(stage: str, modes: int, fold: int) -> Path:
    return (
        ROOT
        / "runs/mabpt/e9"
        / f"K{modes}_{stage}_fold{fold}_seed42_formal"
        / "epoch20.pt"
    )


def _load_models(
    modes: int, fold: int, device: torch.device, batch_size: int
) -> tuple[torch.nn.Module, ScalableEnergyAscent, dict[str, str]]:
    if modes == 5:
        freeze_result = verify_legacy_freeze()
        if not freeze_result["ok"]:
            raise RuntimeError("frozen K=5 MABPT checkpoint verification failed")
        source, energy, source_path, energy_path = _load_frozen_k5_models(
            fold, device
        )
        return source, energy, {
            "source": source_path,
            "energy": energy_path,
            "source_sha256": _sha256(ROOT / source_path),
            "energy_sha256": _sha256(ROOT / energy_path),
        }
    source_path = _checkpoint("source", modes, fold)
    energy_path = _checkpoint("energy", modes, fold)
    for path in (source_path, energy_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing formal E9 checkpoint: {path}")
    source_checkpoint = torch.load(source_path, map_location=device, weights_only=False)
    energy_checkpoint = torch.load(energy_path, map_location=device, weights_only=False)
    protocol_hash = _sha256(PROTOCOL_PATH)
    for path, checkpoint in (
        (source_path, source_checkpoint),
        (energy_path, energy_checkpoint),
    ):
        if checkpoint.get("protocol_sha256") != protocol_hash:
            raise RuntimeError(f"E9 checkpoint protocol mismatch: {path}")
        if int(checkpoint.get("epoch", -1)) != 20:
            raise RuntimeError(f"E9 checkpoint is not fixed epoch 20: {path}")
        if int(checkpoint.get("modes", -1)) != modes:
            raise RuntimeError(f"E9 checkpoint cardinality mismatch: {path}")
    source = build_model("source", modes, batch_size).to(device)
    source.load_state_dict(source_checkpoint["model_state_dict"])
    energy = ScalableEnergyAscent(modes, batch_size=batch_size).to(device)
    energy.load_state_dict(energy_checkpoint["model_state_dict"])
    return source.eval(), energy.eval(), {
        "source": source_path.relative_to(ROOT).as_posix(),
        "energy": energy_path.relative_to(ROOT).as_posix(),
        "source_sha256": _sha256(source_path),
        "energy_sha256": _sha256(energy_path),
    }


def e9_probability_arms(
    source_probability: torch.Tensor,
    cross_cost: torch.Tensor,
    predicted_risk: torch.Tensor,
    pairwise: torch.Tensor,
    *,
    modes: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return frozen E9 correspondence arms on one target support."""
    exact = exact_gibbs_transport(
        source_probability.to(torch.float64), cross_cost, mass_weighted=True
    )
    priors = {"mabpt_exact": exact["transported"]}
    diagnostics: dict[str, torch.Tensor] = {
        "assignment_entropy": exact["assignment_entropy"],
        "normalized_assignment_entropy": exact["normalized_assignment_entropy"],
    }
    for top_m in TOP_M[modes]:
        result = top_m_gibbs_transport(
            source_probability.to(torch.float64),
            cross_cost,
            top_m=top_m,
            mass_weighted=True,
        )
        name = f"mabpt_top{top_m}"
        priors[name] = result["transported"]
        diagnostics[f"top{top_m}_retained_mass"] = result[
            "retained_posterior_mass"
        ]
        diagnostics[f"top{top_m}_prior_l1"] = (
            result["transported"] - exact["transported"]
        ).abs().sum(dim=1)
    sinkhorn = sinkhorn_transport(source_probability.to(torch.float64), cross_cost)
    priors["mabpt_sinkhorn"] = sinkhorn["transported"]
    diagnostics["sinkhorn_prior_l1"] = (
        sinkhorn["transported"] - exact["transported"]
    ).abs().sum(dim=1)
    diagnostics["sinkhorn_row_error"] = sinkhorn["row_error"]
    diagnostics["sinkhorn_column_error"] = sinkhorn["column_error"]

    arms = {}
    for name, prior in priors.items():
        arms[name] = energy_kl_projection(prior, predicted_risk, pairwise)[0]
        if name != "mabpt_exact":
            diagnostics[f"{name}_projected_l1"] = (
                arms[name] - arms["mabpt_exact"]
            ).abs().sum(dim=1)
    return arms, diagnostics


@torch.inference_mode()
def run(
    *,
    modes: int,
    fold: int,
    device: torch.device,
    workers: int,
    batch_size: int,
    max_validation_scenes: int | None,
) -> dict[str, object]:
    if modes not in (3, 5, 7) or fold not in (1, 2):
        raise ValueError("formal E9 evaluation is frozen to K in {3,5,7}, fold in {1,2}")
    protocol = load_data_protocol()
    protocol.assert_boundaries()
    dataset = load_dataset(protocol)
    _, validation_data, validation_dates = fold_subsets(
        protocol,
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
    source, energy, checkpoints = _load_models(modes, fold, device, batch_size)
    probability_names = (
        "mabpt_exact",
        *(f"mabpt_top{top_m}" for top_m in TOP_M[modes]),
        "mabpt_sinkhorn",
    )
    states = {
        name: ArmAccumulator()
        for name in ("ascent_native", "target_energy_native", *probability_names)
    }
    diagnostic_sums = defaultdict(float)
    diagnostic_count = 0
    operator_seconds = defaultdict(float)
    operator_actors = 0
    scene_cursor = 0
    started = time.perf_counter()
    threshold = target_tail_threshold(dataset)
    for data in validation_loader:
        data = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in data.items()
        }
        target = data["pred_traj"].transpose(1, 0)
        source_support, source_logits, _ = source(data)
        source_probability = source_logits.softmax(dim=1)
        target_support, target_probability, target_decision, auxiliary = energy(data)
        cross = support_cost(source_support, target_support)
        pairwise = pairwise_trajectory_distance(target_support) / DEFAULT_ADE_SCALE
        predicted_risk = auxiliary[
            "centered_predicted_normalized_ade_risk"
        ].to(torch.float64)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        operator_started = time.perf_counter()
        probability_arms, diagnostics = e9_probability_arms(
            source_probability,
            cross,
            predicted_risk,
            pairwise,
            modes=modes,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        operator_seconds["all_registered_correspondence_and_projection"] += (
            time.perf_counter() - operator_started
        )
        operator_actors += int(target.shape[0])

        packed = pack_scenes(data["adj"])
        dates = validation_dates[scene_cursor : scene_cursor + packed.scene_count]
        if len(dates) != packed.scene_count:
            raise RuntimeError("E9 scene/date alignment failed")
        actor_dates = np.asarray(dates, dtype=object)[
            packed.inverse.detach().cpu().numpy()
        ]
        tail = torch.linalg.vector_norm(
            target[:, -1] - data["obs_traj"][-1], dim=-1
        ) >= threshold
        measures = {
            "ascent_native": (
                source_support,
                source_probability,
                source_logits.argmax(dim=1),
            ),
            "target_energy_native": (
                target_support,
                target_probability,
                target_decision,
            ),
            **{
                name: (target_support, probability, target_decision)
                for name, probability in probability_arms.items()
            },
        }
        for name, (support, probability, decision) in measures.items():
            metrics = compute_batch_metrics(
                support.to(torch.float64),
                probability.to(torch.float64),
                decision,
                target.to(torch.float64),
            )
            states[name].update(metrics, probability, actor_dates, tail)
        for name, value in diagnostics.items():
            diagnostic_sums[name] += float(value.sum().cpu())
        diagnostic_count += int(target.shape[0])
        scene_cursor += packed.scene_count
    if scene_cursor != len(validation_dates):
        raise RuntimeError("E9 did not consume exactly the registered validation cohort")
    summaries = {name: state.summary() for name, state in states.items()}
    for name in probability_names[1:]:
        for metric in ("top1_ade", "top1_fde", "minade", "minfde"):
            if not math.isclose(
                summaries[name][metric], summaries["mabpt_exact"][metric],
                rel_tol=0.0, abs_tol=1e-12,
            ):
                raise RuntimeError(f"E9 probability arm changed geometry: {name}/{metric}")
    return {
        "format_version": 1,
        "model": "MABPT",
        "model_version": "1.0.0",
        "experiment_id": "E9",
        "evidence_class": "retrospective_train_date_native_cardinality",
        "modes": modes,
        "fold": fold,
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "validation_scenes": len(validation_dates),
        "arms": summaries,
        "diagnostics": {
            name: value / diagnostic_count
            for name, value in diagnostic_sums.items()
        },
        "runtime": {
            "device": str(device),
            "elapsed_seconds": time.perf_counter() - started,
            "operator_seconds": dict(operator_seconds),
            "operator_microseconds_per_actor": {
                name: seconds * 1e6 / operator_actors
                for name, seconds in operator_seconds.items()
            },
            "peak_allocated_gpu_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda" else 0
            ),
        },
        "inputs": checkpoints,
        "integrity": {
            "target_in_probability_forward": False,
            "native_cardinality": True,
            "top_m_backend": "enumerate_then_truncate",
            "residual_or_gate_used": False,
            "temperature_or_weight_search_used": False,
            "validation_selection_used": False,
            "locked_test_used": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", type=int, choices=(3, 5, 7), required=True)
    parser.add_argument("--fold", type=int, choices=(1, 2), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-validation-scenes", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch_size is None:
        args.batch_size = {3: 512, 5: 1024, 7: 32}[args.modes]
    if args.smoke and args.max_validation_scenes is None:
        args.max_validation_scenes = 8
    result = run(
        modes=args.modes,
        fold=args.fold,
        device=torch.device(args.device),
        workers=args.workers,
        batch_size=args.batch_size,
        max_validation_scenes=args.max_validation_scenes,
    )
    if args.output is None:
        suffix = "smoke" if args.smoke else "formal"
        args.output = (
            ROOT / "artifacts/mabpt" / f"e9_K{args.modes}_fold{args.fold}_{suffix}.json"
        )
    _atomic_json(args.output, result)
    print(json.dumps({
        "output": str(args.output),
        "actors": result["arms"]["mabpt_exact"]["actors"],
        "energy": {
            name: arm["energy_score"] for name, arm in result["arms"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
