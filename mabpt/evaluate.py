"""Paired legacy-development evaluation for MABPT experiments E2-E5.

This evaluator is standalone with respect to the MABPT algorithm: it does not
import or monkey-patch any C162/C165 operator. Frozen legacy modules are used
only to reconstruct the matched data cohort and neural checkpoint architectures.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.edfa_ascent.relation import pack_scenes
from experiments.metric_exact.evaluation import target_tail_threshold
from experiments.metric_exact.model import build_model as build_ascent_model
from experiments.energy_predict_optimize.evaluation import compute_batch_metrics
from experiments.energy_predict_optimize.model import build_model as build_energy_model
from experiments.ascent_recomparison.common import fold_subsets, load_dataset, loader
from experiments.tpmo_ascent.protocol import load_protocol as load_legacy_data_protocol

from .freeze import verify as verify_legacy_freeze
from .operator import (
    DEFAULT_ADE_SCALE,
    energy_kl_projection,
    exact_gibbs_transport,
    hard_bijection_transport,
    pairwise_trajectory_distance,
    row_softmax_transport,
    sinkhorn_transport,
    support_cost,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("protocol.json")
CORE_METRICS = (
    "top1_ade",
    "top1_fde",
    "minade",
    "minfde",
    "energy_score",
    "nll",
    "brier",
)
TARGET_ARMS = (
    "target_native_logits",
    "target_energy_single_support",
    "identity_full_projection",
    "ordinary_hungarian_full_projection",
    "mass_hungarian_full_projection",
    "row_softmax_full_projection",
    "sinkhorn_full_projection",
    "unweighted_gibbs_full_projection",
    "mabpt_u_only",
    "mabpt_risk_kl",
    "mabpt_diversity_kl",
    "uniform_energy_kl",
    "mabpt_no_kl",
    "mabpt",
)
ARMS = ("ascent_native", *TARGET_ARMS)


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


def _ece_sums(
    confidence: np.ndarray, correct: np.ndarray, bins: int = 15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    counts = np.zeros(bins, dtype=np.int64)
    confidence_sum = np.zeros(bins, dtype=np.float64)
    correct_sum = np.zeros(bins, dtype=np.float64)
    for index in range(bins):
        upper = (
            confidence <= edges[index + 1]
            if index == bins - 1
            else confidence < edges[index + 1]
        )
        mask = (confidence >= edges[index]) & upper
        counts[index] = int(mask.sum())
        confidence_sum[index] = float(confidence[mask].sum())
        correct_sum[index] = float(correct[mask].sum())
    return counts, confidence_sum, correct_sum


class ArmAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.sums = defaultdict(float)
        self.date_counts = defaultdict(int)
        self.date_sums = defaultdict(lambda: defaultdict(float))
        self.ece_counts = np.zeros(15, dtype=np.int64)
        self.ece_confidence = np.zeros(15, dtype=np.float64)
        self.ece_correct = np.zeros(15, dtype=np.float64)
        self.minfde_values: list[np.ndarray] = []
        self.tail_sum = 0.0
        self.tail_count = 0
        self.effective_modes_sum = 0.0

    def update(
        self,
        metrics: dict[str, torch.Tensor],
        probabilities: torch.Tensor,
        actor_dates: np.ndarray,
        tail: torch.Tensor,
    ) -> None:
        arrays = {
            name: metrics[name].detach().cpu().numpy()
            for name in (*CORE_METRICS, "minfde", "oracle_ade_mode")
        }
        probability_array = probabilities.detach().to(torch.float64).cpu().numpy()
        count = int(arrays["top1_ade"].shape[0])
        self.count += count
        for metric in CORE_METRICS:
            self.sums[metric] += float(arrays[metric].sum())
        minfde = arrays["minfde"].astype(np.float64, copy=False)
        self.minfde_values.append(minfde)
        tail_array = tail.detach().cpu().numpy().astype(bool)
        self.tail_sum += float(minfde[tail_array].sum())
        self.tail_count += int(tail_array.sum())
        tiny = np.finfo(np.float64).tiny
        entropy = -(probability_array * np.log(np.maximum(probability_array, tiny))).sum(1)
        self.effective_modes_sum += float(np.exp(entropy).sum())
        decision = probability_array.argmax(axis=1)
        confidence = probability_array.max(axis=1)
        correct = decision == arrays["oracle_ade_mode"]
        counts, confidence_sum, correct_sum = _ece_sums(confidence, correct)
        self.ece_counts += counts
        self.ece_confidence += confidence_sum
        self.ece_correct += correct_sum
        dates = np.asarray(actor_dates, dtype=object)
        for date in sorted(set(dates.tolist())):
            mask = dates == date
            date_count = int(mask.sum())
            self.date_counts[str(date)] += date_count
            for metric in CORE_METRICS:
                self.date_sums[str(date)][metric] += float(arrays[metric][mask].sum())

    def summary(self) -> dict[str, object]:
        if not self.count:
            raise RuntimeError("empty MABPT accumulator")
        ece = 0.0
        reliability = []
        for count, confidence, correct in zip(
            self.ece_counts, self.ece_confidence, self.ece_correct, strict=True
        ):
            mean_confidence = confidence / count if count else None
            accuracy = correct / count if count else None
            if count:
                ece += (count / self.count) * abs(accuracy - mean_confidence)
            reliability.append(
                {
                    "count": int(count),
                    "mean_confidence": mean_confidence,
                    "accuracy": accuracy,
                }
            )
        minfde = np.concatenate(self.minfde_values)
        return {
            "actors": self.count,
            **{metric: self.sums[metric] / self.count for metric in CORE_METRICS},
            "ece_argmax": ece,
            "effective_modes": self.effective_modes_sum / self.count,
            "minfde_p95": float(np.quantile(minfde, 0.95)),
            "tail_minfde": self.tail_sum / max(self.tail_count, 1),
            "tail_samples": self.tail_count,
            "reliability": reliability,
            "date_metrics": {
                date: {
                    "actors": self.date_counts[date],
                    **{
                        metric: self.date_sums[date][metric] / self.date_counts[date]
                        for metric in CORE_METRICS
                    },
                }
                for date in sorted(self.date_counts)
            },
        }


def _load_models(fold: int, device: torch.device):
    manifest = json.loads(
        Path(__file__).with_name("frozen_c165_manifest.json").read_text(encoding="utf-8")
    )["files"]
    baseline_relative = (
        f"runs/experiments/metric_exact/P2_B0_signed_coupled_fold{fold}_seed42_formal/last.pt"
    )
    target_relative = (
        "runs/experiments/ascent_recomparison/"
        f"E1_energy_predict_optimize_fold{fold}_seed42_formal/last.pt"
    )
    for relative in (baseline_relative, target_relative):
        if _sha256(ROOT / relative) != manifest[relative]:
            raise RuntimeError(f"frozen checkpoint hash mismatch: {relative}")
    baseline = build_ascent_model("B0_signed_coupled", batch_size=2048).to(device)
    baseline.load_state_dict(
        torch.load(ROOT / baseline_relative, map_location=device, weights_only=False)[
            "model_state_dict"
        ]
    )
    target = build_energy_model(batch_size=2048).to(device)
    target.load_state_dict(
        torch.load(ROOT / target_relative, map_location=device, weights_only=False)[
            "model_state_dict"
        ]
    )
    return baseline.eval(), target.eval(), baseline_relative, target_relative


def _projection(
    prior: torch.Tensor,
    risk: torch.Tensor,
    pairwise: torch.Tensor,
    *,
    risk_weight: float = 1.0,
    diversity_weight: float = 1.0,
    kl_weight: float = 1.0,
) -> torch.Tensor:
    return energy_kl_projection(
        prior,
        risk,
        pairwise,
        risk_weight=risk_weight,
        diversity_weight=diversity_weight,
        kl_weight=kl_weight,
    )[0]


def probability_arms(
    source_probabilities: torch.Tensor,
    target_native_probabilities: torch.Tensor,
    target_energy_probabilities: torch.Tensor,
    cross_cost: torch.Tensor,
    predicted_risk: torch.Tensor,
    pairwise: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Construct registered E2-E5 arms from one target-free paired forward."""
    source_probabilities = source_probabilities.to(torch.float64)
    target_native_probabilities = target_native_probabilities.to(torch.float64)
    target_energy_probabilities = target_energy_probabilities.to(torch.float64)
    ordinary_hard = hard_bijection_transport(
        source_probabilities, cross_cost, mass_weighted=False
    )
    mass_hard = hard_bijection_transport(
        source_probabilities, cross_cost, mass_weighted=True
    )
    row = row_softmax_transport(source_probabilities, cross_cost)
    sinkhorn = sinkhorn_transport(source_probabilities, cross_cost)
    unweighted = exact_gibbs_transport(
        source_probabilities, cross_cost, mass_weighted=False
    )
    mass = exact_gibbs_transport(source_probabilities, cross_cost, mass_weighted=True)
    identity = source_probabilities
    uniform = torch.full_like(source_probabilities, 1.0 / source_probabilities.shape[1])
    arms = {
        "target_native_logits": target_native_probabilities,
        "target_energy_single_support": target_energy_probabilities,
        "identity_full_projection": _projection(identity, predicted_risk, pairwise),
        "ordinary_hungarian_full_projection": _projection(
            ordinary_hard["transported"], predicted_risk, pairwise
        ),
        "mass_hungarian_full_projection": _projection(
            mass_hard["transported"], predicted_risk, pairwise
        ),
        "row_softmax_full_projection": _projection(
            row["transported"], predicted_risk, pairwise
        ),
        "sinkhorn_full_projection": _projection(
            sinkhorn["transported"], predicted_risk, pairwise
        ),
        "unweighted_gibbs_full_projection": _projection(
            unweighted["transported"], predicted_risk, pairwise
        ),
        "mabpt_u_only": mass["transported"],
        "mabpt_risk_kl": _projection(
            mass["transported"], predicted_risk, pairwise, diversity_weight=0.0
        ),
        "mabpt_diversity_kl": _projection(
            mass["transported"], predicted_risk, pairwise, risk_weight=0.0
        ),
        "uniform_energy_kl": _projection(uniform, predicted_risk, pairwise),
        "mabpt_no_kl": _projection(
            mass["transported"], predicted_risk, pairwise, kl_weight=0.0
        ),
        "mabpt": _projection(mass["transported"], predicted_risk, pairwise),
    }
    diagnostics = {
        "mabpt_assignment_entropy": mass["assignment_entropy"],
        "mabpt_normalized_assignment_entropy": mass[
            "normalized_assignment_entropy"
        ],
        "mabpt_expected_assignment_cost": mass["expected_cost"],
        "sinkhorn_row_error": sinkhorn["row_error"],
        "sinkhorn_column_error": sinkhorn["column_error"],
    }
    return arms, diagnostics


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
        raise ValueError("legacy development evaluation is frozen to folds 1 and 2")
    freeze_result = verify_legacy_freeze()
    if not freeze_result["ok"]:
        raise RuntimeError("legacy C165 freeze verification failed")
    protocol = load_legacy_data_protocol()
    protocol.assert_boundaries()
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(True)
    started = time.time()
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
    baseline, target_model, baseline_path, target_path = _load_models(fold, device)
    states = {arm: ArmAccumulator() for arm in ARMS}
    diagnostic_sums = defaultdict(float)
    diagnostic_count = 0
    scene_cursor = 0
    geometry_checks = 0
    for data in validation_loader:
        data = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in data.items()
        }
        target = data["pred_traj"].transpose(1, 0)
        source_support, source_logits, _ = baseline(data)
        source_probabilities = source_logits.softmax(dim=1)
        source_decision = source_logits.argmax(dim=1)
        source_metrics = compute_batch_metrics(
            source_support, source_probabilities, source_decision, target
        )
        target_support, target_energy, target_decision, auxiliary = target_model(data)
        target_native = auxiliary["decision_logits"].softmax(dim=1)
        target_pairwise_raw = pairwise_trajectory_distance(target_support)
        target_pairwise = target_pairwise_raw / DEFAULT_ADE_SCALE
        cross = support_cost(source_support, target_support)
        predicted_risk = auxiliary["centered_predicted_normalized_ade_risk"].to(
            torch.float64
        )
        arms, diagnostics = probability_arms(
            source_probabilities,
            target_native,
            target_energy,
            cross,
            predicted_risk,
            target_pairwise,
        )
        packed = pack_scenes(data["adj"])
        dates = validation_dates[scene_cursor : scene_cursor + packed.scene_count]
        if len(dates) != packed.scene_count:
            raise RuntimeError("MABPT scene-date alignment failed")
        actor_dates = np.asarray(dates, dtype=object)[
            packed.inverse.detach().cpu().numpy()
        ]
        endpoint_travel = torch.linalg.vector_norm(
            target[:, -1] - data["obs_traj"][-1], dim=-1
        )
        tail = endpoint_travel >= target_tail_threshold(dataset)
        states["ascent_native"].update(
            source_metrics, source_probabilities, actor_dates, tail
        )
        target_support_fp64 = target_support.to(torch.float64)
        target_fp64 = target.to(torch.float64)
        for arm, probabilities in arms.items():
            metrics = compute_batch_metrics(
                target_support_fp64,
                probabilities,
                target_decision,
                target_fp64,
            )
            states[arm].update(metrics, probabilities, actor_dates, tail)
        for name, values in diagnostics.items():
            diagnostic_sums[name] += float(values.sum().cpu())
        diagnostic_count += int(source_probabilities.shape[0])
        geometry_checks += 1
        scene_cursor += packed.scene_count
    if scene_cursor != len(validation_dates):
        raise RuntimeError("MABPT did not consume exactly the validation cohort")
    summaries = {arm: states[arm].summary() for arm in ARMS}
    for arm in TARGET_ARMS[1:]:
        for metric in ("top1_ade", "top1_fde", "minade", "minfde"):
            if summaries[arm][metric] != summaries["target_native_logits"][metric]:
                raise RuntimeError(f"probability arm changed target geometry: {arm}/{metric}")
    elapsed = time.time() - started
    return {
        "format_version": 1,
        "model": "MABPT",
        "model_version": "1.0.0",
        "experiment_ids": ["E2", "E3", "E4", "E5"],
        "evidence_class": "legacy_development_only",
        "protocol_sha256": _sha256(PROTOCOL),
        "fold": fold,
        "validation_scenes": len(validation_dates),
        "arms": summaries,
        "diagnostics": {
            name: value / diagnostic_count for name, value in diagnostic_sums.items()
        },
        "integrity": {
            "legacy_freeze_verified": True,
            "target_in_probability_forward": False,
            "geometry_batch_identity_checks": geometry_checks,
            "gate_or_residual_used": False,
            "temperature_or_weight_search_used": False,
            "arm_selection_used": False,
        },
        "inputs": {
            "baseline_checkpoint": baseline_path,
            "target_checkpoint": target_path,
        },
        "runtime": {
            "device": str(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "elapsed_seconds": elapsed,
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
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-validation-scenes", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
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
        args.output = ROOT / "artifacts/mabpt" / f"e2_e5_fold{args.fold}_{suffix}.json"
    _atomic_json(args.output, result)
    if args.quiet:
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "fold": result["fold"],
                    "actors": result["arms"]["mabpt"]["actors"],
                    "mabpt": {
                        metric: result["arms"]["mabpt"][metric]
                        for metric in ("energy_score", "nll", "brier")
                    },
                    "elapsed_seconds": result["runtime"]["elapsed_seconds"],
                },
                indent=2,
            )
        )
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
