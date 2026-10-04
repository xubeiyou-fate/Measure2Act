"""Audit ASCENT probability transfer onto frozen EqMotion K=5 supports.

This sidecar leaves the original forecasting workspace read-only.  It answers a
narrow E2-style question: if an external generator supplies only K=5
trajectories, can the source measure be transported onto that support
without future-target leakage?

Important boundary: EqMotion does not expose MABPT's learned risk features, so
this is a support-correspondence audit, not a full learned Energy-KL projection.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("ASCENT_FULL_ROOT", DEFAULT_ROOT)).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.energy_predict_optimize.evaluation import compute_batch_metrics  # noqa: E402
from mabpt.evaluate_tartan_retrain import (  # noqa: E402
    AIRPORTS,
    _authorize_split,
    _formal_test_gate,
    _selected_checkpoint_triplet,
    _verify_freeze_receipt,
)
from mabpt.operator import (  # noqa: E402
    exact_gibbs_transport,
    hard_bijection_transport,
    row_softmax_transport,
    sinkhorn_transport,
    support_cost,
)
from mabpt.partc_seed_evaluate import _load_model_pair  # noqa: E402
from mabpt.train_tartan_retrain import _dataset, _limited_indices  # noqa: E402
from model.utils import seed_worker, seq_collate  # noqa: E402
from modern_baseline.eqmotion_aviation import EqMotionAviation  # noqa: E402
from modern_baseline.evaluate_eqmotion_tartan_locked import (  # noqa: E402
    EVALUATION_RECEIPT as EQMOTION_EVALUATION_RECEIPT,
    SCENE_INDEX_SUMMARY as EQMOTION_SCENE_INDEX_SUMMARY,
    expected_test_scene_count as expected_seed42_test_scene_count,
    verify_evaluation_receipt as verify_seed42_evaluation_receipt,
)
from modern_baseline.evaluate_eqmotion_tartan_multiseed_v2_locked import (  # noqa: E402
    expected_test_scene_count as expected_multiseed_test_scene_count,
    verify_evaluation_receipt as verify_multiseed_evaluation_receipt,
)
from modern_baseline.run_eqmotion_tartan_multiseed_v2 import (  # noqa: E402
    NEW_SEEDS,
    PROTOCOL as EQMOTION_MULTI_PROTOCOL,
    load_multiseed_protocol,
)
from modern_baseline.run_eqmotion_tartan_target import (  # noqa: E402
    PROTOCOL as EQMOTION_SEED42_PROTOCOL,
    load_protocol as load_eqmotion_seed42_protocol,
)


MABPT_PROTOCOL = ROOT / "mabpt/tartan_retrain_protocol_v1.json"
MABPT_FREEZE_RECEIPT = (
    ROOT / "artifacts/partc_two_dataset_20260812/tartan_retrain_frozen_receipt_v1.json"
)
REGIME = "target_only"
SEEDS = (42, *NEW_SEEDS)
SHARED_SUPPORT_ARMS = (
    "eqmotion_uniform",
    "hard_unweighted_prior",
    "hard_mass_aware_prior",
    "row_softmax_prior",
    "sinkhorn_prior",
    "gibbs_unweighted_prior",
    "gibbs_mass_aware_prior",
)
REFERENCE_ARMS = ("source_ascent_native",)
METRIC_KEYS = (
    "top1_ade",
    "top1_fde",
    "minade",
    "minfde",
    "energy_score",
    "nll",
    "brier",
    "oracle_ade_rank",
    "oracle_fde_rank",
    "oracle_ade_rank1",
    "oracle_fde_rank1",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class StreamingSummary:
    def __init__(self, bins: int = 15) -> None:
        self.count = 0
        self.sums: dict[str, float] = defaultdict(float)
        self.bin_counts = np.zeros(bins, dtype=np.int64)
        self.bin_confidence = np.zeros(bins, dtype=np.float64)
        self.bin_correct = np.zeros(bins, dtype=np.float64)

    def update(self, arrays: dict[str, np.ndarray], mask: np.ndarray | None = None) -> None:
        if mask is None:
            mask = np.ones(arrays["top1_ade"].shape[0], dtype=bool)
        count = int(mask.sum())
        self.count += count
        for metric in METRIC_KEYS:
            self.sums[metric] += float(arrays[metric][mask].sum())
        confidence = arrays["confidence"][mask]
        correct = arrays["oracle_ade_rank1"][mask]
        edges = np.linspace(0.0, 1.0, len(self.bin_counts) + 1)
        for index in range(len(self.bin_counts)):
            upper = (
                confidence <= edges[index + 1]
                if index == len(self.bin_counts) - 1
                else confidence < edges[index + 1]
            )
            selected = (confidence >= edges[index]) & upper
            self.bin_counts[index] += int(selected.sum())
            self.bin_confidence[index] += float(confidence[selected].sum())
            self.bin_correct[index] += float(correct[selected].sum())

    def summary(self) -> dict[str, object]:
        if not self.count:
            raise RuntimeError("empty EqMotion support-transfer summary")
        ece = 0.0
        for count, confidence, correct in zip(
            self.bin_counts, self.bin_confidence, self.bin_correct, strict=True
        ):
            if count:
                ece += (count / self.count) * abs(correct / count - confidence / count)
        return {
            "agents": self.count,
            **{metric: self.sums[metric] / self.count for metric in METRIC_KEYS},
            "ece": float(ece),
        }


class MetricStore:
    def __init__(self) -> None:
        self.overall = StreamingSummary()
        self.per_date: dict[str, StreamingSummary] = {}

    def update(self, metrics: dict[str, torch.Tensor], actor_dates: np.ndarray) -> None:
        arrays = {
            name: value.detach().to(torch.float64).cpu().numpy()
            for name, value in metrics.items()
            if name in (*METRIC_KEYS, "confidence")
        }
        self.overall.update(arrays)
        for date in sorted(set(actor_dates.tolist())):
            mask = actor_dates == date
            self.per_date.setdefault(str(date), StreamingSummary()).update(arrays, mask)

    def summary(self) -> dict[str, object]:
        return {
            "overall": self.overall.summary(),
            "per_date": {
                date: state.summary() for date, state in sorted(self.per_date.items())
            },
        }


def normalized(probability: torch.Tensor) -> torch.Tensor:
    probability = probability.to(torch.float64).clamp_min(torch.finfo(torch.float64).tiny)
    return probability / probability.sum(dim=1, keepdim=True)


def eqmotion_result_path(airport: str, seed: int) -> Path:
    if seed == 42:
        return (
            ROOT
            / "artifacts/partc_two_dataset_20260812/modern_baseline"
            / f"eqmotion_tartan_{airport}_target_only_seed42_formal.json"
        )
    return (
        ROOT
        / "artifacts/partc_two_dataset_20260812/modern_baseline/"
        "eqmotion_tartan_multiseed_v2"
        / f"{airport}_target_only_seed{seed}_formal.json"
    )


def eqmotion_checkpoint_receipt(airport: str, seed: int) -> dict[str, object]:
    path = eqmotion_result_path(airport, seed)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("formal") is not True
        or payload.get("airport") != airport
        or payload.get("regime") != REGIME
        or int(payload.get("seed", -1)) != seed
        or payload.get("integrity", {}).get("locked_test_model_inference") is not False
    ):
        raise RuntimeError(f"EqMotion formal result identity mismatch: {path}")
    checkpoint = ROOT / payload["checkpoint"]["path"]
    if sha256(checkpoint) != payload["checkpoint"]["sha256"]:
        raise RuntimeError(f"EqMotion checkpoint hash mismatch: {checkpoint}")
    return {
        "training_result": path.relative_to(ROOT).as_posix(),
        "training_result_sha256": sha256(path),
        "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_epoch": int(payload["checkpoint"]["epoch"]),
        "checkpoint_bytes": checkpoint.stat().st_size,
    }


def eqmotion_protocol_and_test_gate(
    airport: str,
    seed: int,
    split: str,
) -> tuple[dict[str, object], dict[str, object] | None]:
    if seed == 42:
        protocol = load_eqmotion_seed42_protocol(EQMOTION_SEED42_PROTOCOL)
    else:
        protocol = load_multiseed_protocol()
    if split != "test":
        return protocol, None
    if seed == 42:
        receipt = verify_seed42_evaluation_receipt()
        expected = expected_seed42_test_scene_count(airport, protocol, receipt)
        receipt_path = EQMOTION_EVALUATION_RECEIPT
        protocol_path = EQMOTION_SEED42_PROTOCOL
    else:
        receipt = verify_multiseed_evaluation_receipt()
        expected = expected_multiseed_test_scene_count(airport, protocol, receipt)
        receipt_path = EQMOTION_EVALUATION_RECEIPT
        protocol_path = EQMOTION_MULTI_PROTOCOL
    return protocol, {
        "passed": True,
        "evaluation_receipt": receipt_path.relative_to(ROOT).as_posix(),
        "evaluation_receipt_sha256": sha256(receipt_path),
        "scene_index_summary": EQMOTION_SCENE_INDEX_SUMMARY.relative_to(ROOT).as_posix(),
        "scene_index_summary_sha256": sha256(EQMOTION_SCENE_INDEX_SUMMARY),
        "expected_test_scenes": expected,
        "protocol": protocol_path.relative_to(ROOT).as_posix(),
        "protocol_sha256": sha256(protocol_path),
    }


def build_eqmotion_batch(
    data: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    obs = data["obs_traj"].transpose(1, 0).contiguous()
    future = data["pred_traj"].transpose(1, 0).contiguous()
    adj = data["adj"].to(torch.long)
    if adj.numel() != obs.shape[0]:
        raise RuntimeError("scene adjacency length does not match actors")
    scene_count = int(adj.max().item()) + 1 if adj.numel() else 0
    counts = torch.bincount(adj, minlength=scene_count).to(torch.long)
    max_agents = int(counts.max().item()) if counts.numel() else 0
    history = obs.new_zeros((scene_count, max_agents, 16, 3))
    target = future.new_zeros((scene_count, max_agents, 24, 3))
    valid = torch.zeros((scene_count, max_agents), device=obs.device, dtype=torch.bool)
    centers: list[torch.Tensor] = []
    for scene in range(scene_count):
        actor_index = (adj == scene).nonzero(as_tuple=False).flatten()
        count = int(actor_index.numel())
        if not count:
            raise RuntimeError("empty scene in batch")
        scene_history = obs[actor_index]
        scene_future = future[actor_index]
        center = scene_history[:, -1].mean(dim=0, keepdim=True)
        history[scene, :count] = scene_history - center[:, None]
        target[scene, :count] = scene_future - center[:, None]
        valid[scene, :count] = True
        centers.append(center.expand(count, -1))
    actor_centers = torch.cat(centers, dim=0)
    return history, target, valid, counts, actor_centers


def probability_arms(
    source_probability: torch.Tensor,
    cross_cost: torch.Tensor,
) -> dict[str, torch.Tensor]:
    source_probability = normalized(source_probability)
    uniform = torch.full_like(source_probability, 1.0 / source_probability.shape[1])
    hard_unweighted = hard_bijection_transport(
        source_probability, cross_cost, mass_weighted=False
    )["transported"]
    hard_mass_aware = hard_bijection_transport(
        source_probability, cross_cost, mass_weighted=True
    )["transported"]
    row_softmax = row_softmax_transport(source_probability, cross_cost)["transported"]
    sinkhorn = sinkhorn_transport(source_probability, cross_cost)["transported"]
    gibbs_unweighted = exact_gibbs_transport(
        source_probability, cross_cost, mass_weighted=False
    )["transported"]
    gibbs_mass_aware = exact_gibbs_transport(
        source_probability, cross_cost, mass_weighted=True
    )["transported"]
    arms = {
        "eqmotion_uniform": uniform,
        "hard_unweighted_prior": hard_unweighted,
        "hard_mass_aware_prior": hard_mass_aware,
        "row_softmax_prior": row_softmax,
        "sinkhorn_prior": sinkhorn,
        "gibbs_unweighted_prior": gibbs_unweighted,
        "gibbs_mass_aware_prior": gibbs_mass_aware,
    }
    if set(arms) != set(SHARED_SUPPORT_ARMS):
        raise RuntimeError("support-transfer arm registry mismatch")
    return arms


@torch.inference_mode()
def run(
    *,
    airport: str,
    seed: int,
    split: str,
    device: torch.device,
    workers: int,
    batch_size: int,
    max_scenes: int | None,
    authorize_locked_test: bool,
) -> dict[str, object]:
    if airport not in AIRPORTS:
        raise ValueError("airport lies outside the frozen Tartan registry")
    if seed not in SEEDS:
        raise ValueError("seed lies outside the frozen EqMotion/MABPT seed registry")
    parent = json.loads(MABPT_PROTOCOL.read_text(encoding="utf-8"))
    test_gate = _authorize_split(
        split=split,
        authorize_locked_test=authorize_locked_test,
        max_scenes=max_scenes,
        formal_gate=lambda: _formal_test_gate(
            root=ROOT,
            protocol=parent,
            receipt_path=MABPT_FREEZE_RECEIPT,
        ),
    )
    parent_freeze = _verify_freeze_receipt(root=ROOT, receipt_path=MABPT_FREEZE_RECEIPT)
    eqmotion_protocol, eqmotion_test_gate = eqmotion_protocol_and_test_gate(
        airport, seed, split
    )
    eqmotion_checkpoint = eqmotion_checkpoint_receipt(airport, seed)
    checkpoints = _selected_checkpoint_triplet(
        root=ROOT,
        protocol=parent,
        airport=airport,
        regime=REGIME,
        seed=seed,
        formal=True,
    )

    dataset, all_dates, index_path = _dataset(parent, airport, split)
    if split == "test" and eqmotion_test_gate is not None:
        expected = int(eqmotion_test_gate["expected_test_scenes"])
        if len(dataset) != expected:
            raise RuntimeError(
                f"{airport}/{split} scene-count mismatch: dataset={len(dataset)} "
                f"eqmotion_expected={expected}"
            )
    selected_indices = _limited_indices(len(dataset), max_scenes)
    selected_dates = [all_dates[index] for index in selected_indices]
    evaluation = Subset(dataset, selected_indices)
    loader_options: dict[str, Any] = {
        "dataset": evaluation,
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": seq_collate,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }
    if workers > 0:
        loader_options.update({"persistent_workers": True, "prefetch_factor": 4})
    loader = DataLoader(**loader_options)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)

    source, _ = _load_model_pair(
        source_checkpoint=ROOT / checkpoints["ascent"]["path"],
        target_checkpoint=ROOT / checkpoints["predicted_risk"]["path"],
        device=device,
        batch_size=batch_size,
    )
    eqmotion = EqMotionAviation(
        device=device,
        hidden_nf=int(eqmotion_protocol["model"]["hidden_nf"]),
        channels=int(eqmotion_protocol["model"]["channels"]),
        layers=int(eqmotion_protocol["model"]["layers"]),
        modes=int(eqmotion_protocol["grid"]["modes"]),
    ).to(device)
    saved = torch.load(
        ROOT / eqmotion_checkpoint["checkpoint"],
        map_location=device,
        weights_only=False,
    )
    if (
        saved.get("airport") != airport
        or int(saved.get("seed", -1)) != seed
        or int(saved.get("epoch", -1)) != int(eqmotion_checkpoint["checkpoint_epoch"])
    ):
        raise RuntimeError("EqMotion checkpoint metadata mismatch")
    eqmotion.load_state_dict(saved["model_state_dict"], strict=True)
    eqmotion.eval()
    source.eval()

    states = {arm: MetricStore() for arm in (*REFERENCE_ARMS, *SHARED_SUPPORT_ARMS)}
    diagnostic_sum: dict[str, float] = defaultdict(float)
    diagnostic_count = 0
    coordinate_alignment_max_abs = 0.0
    probability_sum_error_max = 0.0
    geometry_reference: dict[str, float] | None = None
    scene_cursor = 0
    actors = 0
    batches = 0
    inference_seconds = 0.0
    started = time.perf_counter()
    for data in loader:
        data = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in data.items()
        }
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        batch_started = time.perf_counter()
        truth = data["pred_traj"].transpose(1, 0).to(torch.float64)
        history, eq_truth_centered, valid, num_valid, actor_centers = build_eqmotion_batch(data)
        eq_prediction_centered = eqmotion(history, num_valid)
        eq_support_centered = eq_prediction_centered[valid].to(torch.float64)
        eq_truth_centered_flat = eq_truth_centered[valid].to(torch.float64)
        actor_centers = actor_centers.to(torch.float64)
        eq_support = eq_support_centered + actor_centers[:, None, None, :]
        eq_truth = eq_truth_centered_flat + actor_centers[:, None, :]
        coordinate_alignment_max_abs = max(
            coordinate_alignment_max_abs,
            float((eq_truth - truth).abs().amax().detach().cpu()),
        )
        if coordinate_alignment_max_abs > 1e-5:
            raise RuntimeError(
                "ASCENT and EqMotion truth tensors are not in the same actor/order frame"
            )
        source_support, source_logits, _ = source(data)
        source_probability = source_logits.softmax(dim=1).to(torch.float64)
        cross = support_cost(source_support.to(torch.float64), eq_support)
        arms = probability_arms(source_probability, cross)
        shared_metrics = {}
        for name, probability in arms.items():
            probability = normalized(probability)
            probability_sum_error_max = max(
                probability_sum_error_max,
                float((probability.sum(dim=1) - 1.0).abs().amax().detach().cpu()),
            )
            shared_metrics[name] = compute_batch_metrics(
                eq_support,
                probability,
                probability.argmax(dim=1),
                truth,
            )
        source_metrics = compute_batch_metrics(
            source_support.to(torch.float64),
            source_probability,
            source_probability.argmax(dim=1),
            truth,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter() - batch_started

        local_scene_count = int(num_valid.shape[0])
        batch_dates = selected_dates[scene_cursor : scene_cursor + local_scene_count]
        if len(batch_dates) != local_scene_count:
            raise RuntimeError("scene/date alignment failed")
        actor_dates = np.asarray(batch_dates, dtype=object)[
            data["adj"].detach().cpu().numpy()
        ]
        states["source_ascent_native"].update(source_metrics, actor_dates)
        for arm, metrics in shared_metrics.items():
            states[arm].update(metrics, actor_dates)

        uniform = arms["eqmotion_uniform"]
        for arm, probability in arms.items():
            diagnostic_sum[f"{arm}_entropy"] += float(
                (
                    -probability
                    * probability.clamp_min(torch.finfo(probability.dtype).tiny).log()
                )
                .sum(dim=1)
                .sum()
                .detach()
                .cpu()
            )
            diagnostic_sum[f"{arm}_l1_from_uniform"] += float(
                (probability - uniform).abs().sum(dim=1).sum().detach().cpu()
            )
        diagnostic_sum["cross_cost_min"] += float(
            cross.amin(dim=(1, 2)).sum().detach().cpu()
        )
        diagnostic_sum["cross_cost_mean"] += float(
            cross.mean(dim=(1, 2)).sum().detach().cpu()
        )
        diagnostic_count += int(eq_support.shape[0])
        actors += int(eq_support.shape[0])
        scene_cursor += local_scene_count
        batches += 1
    if scene_cursor != len(evaluation):
        raise RuntimeError("evaluation did not consume the cohort exactly once")

    models = {arm: state.summary() for arm, state in states.items()}
    uniform_reference = models["eqmotion_uniform"]["overall"]
    geometry_invariance = {
        metric: max(
            abs(float(models[arm]["overall"][metric]) - float(uniform_reference[metric]))
            for arm in SHARED_SUPPORT_ARMS
        )
        for metric in ("minade", "minfde")
    }
    geometry_reference = {
        metric: float(uniform_reference[metric]) for metric in ("minade", "minfde")
    }
    if any(value > 1e-12 for value in geometry_invariance.values()):
        raise RuntimeError("EqMotion shared-support geometry unexpectedly differs")
    return {
        "format_version": 1,
        "experiment_id": "EqMotion_support_probability_transfer_v1",
        "evidence_class": (
            "development_only_external_support_audit"
            if split == "development"
            else "locked_retrospective_test_external_support_audit"
        ),
        "airport": airport,
        "regime": REGIME,
        "seed": seed,
        "split": split,
        "scenes": len(evaluation),
        "actors": actors,
        "models": models,
        "shared_support_arms": list(SHARED_SUPPORT_ARMS),
        "reference_arms": list(REFERENCE_ARMS),
        "geometry_reference": geometry_reference,
        "geometry_invariance_max_absolute_difference": geometry_invariance,
        "diagnostics": {
            name: value / max(diagnostic_count, 1)
            for name, value in sorted(diagnostic_sum.items())
        },
        "inputs": {
            "sidecar_evaluator": {
                "path": Path(__file__).resolve().as_posix(),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "mabpt_protocol": {
                "path": MABPT_PROTOCOL.relative_to(ROOT).as_posix(),
                "sha256": sha256(MABPT_PROTOCOL),
            },
            "mabpt_parent_freeze_receipt": parent_freeze,
            "mabpt_test_gate": test_gate,
            "mabpt_checkpoints": checkpoints,
            "eqmotion_checkpoint": eqmotion_checkpoint,
            "eqmotion_test_gate": eqmotion_test_gate,
            "scene_date_index": index_path.relative_to(ROOT).as_posix(),
            "scene_date_index_sha256": sha256(index_path),
        },
        "integrity": {
            "target_in_probability_forward": False,
            "external_generator_support": "EqMotion",
            "same_actor_order_verified": coordinate_alignment_max_abs <= 1e-5,
            "coordinate_alignment_max_absolute_difference": coordinate_alignment_max_abs,
            "probability_sum_error_max": probability_sum_error_max,
            "eqmotion_support_k": 5,
            "ascent_source_k": 5,
            "shared_support_geometry_exactly_invariant": True,
            "learned_energy_kl_projection_applied": False,
            "risk_head_not_defined_for_external_support": True,
            "locked_test_used": split == "test",
            "partial_locked_test": split == "test" and max_scenes is not None,
        },
        "runtime": {
            "device": str(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "workers": workers,
            "batch_size_scenes": batch_size,
            "batches": batches,
            "inference_seconds": inference_seconds,
            "actors_per_second": actors / max(inference_seconds, 1e-12),
            "total_elapsed_seconds": time.perf_counter() - started,
            "peak_allocated_gpu_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        },
        "claim_boundaries": [
            "This is an external-support correspondence audit, not a full MABPT run on EqMotion.",
            "EqMotion checkpoints expose K=5 trajectories but no MABPT mode features or learned risk head.",
            "Therefore the audit tests target-free probability transport onto external support; learned Energy-KL projection is deliberately not applied.",
            "Uniform EqMotion is the geometry baseline; all shared-support arms must keep minADE/minFDE invariant.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airport", choices=AIRPORTS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--split", choices=("development", "test"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--authorize-locked-test", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.smoke:
        if args.split != "development":
            parser.error("--smoke is development-only")
        args.max_scenes = args.max_scenes or 8
    result = run(
        airport=args.airport,
        seed=args.seed,
        split=args.split,
        device=torch.device(args.device),
        workers=args.workers,
        batch_size=args.batch_size,
        max_scenes=args.max_scenes,
        authorize_locked_test=args.authorize_locked_test,
    )
    atomic_json(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "output": args.output.resolve().as_posix(),
                "airport": result["airport"],
                "seed": result["seed"],
                "split": result["split"],
                "actors": result["actors"],
                "uniform_energy": result["models"]["eqmotion_uniform"]["overall"][
                    "energy_score"
                ],
                "gibbs_unweighted_energy": result["models"][
                    "gibbs_unweighted_prior"
                ]["overall"]["energy_score"],
                "elapsed_seconds": result["runtime"]["total_elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
