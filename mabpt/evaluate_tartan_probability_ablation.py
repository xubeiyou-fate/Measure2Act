"""Evaluate frozen Tartan checkpoints under a shared-support probability ablation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from experiments.edfa_ascent.relation import pack_scenes
from experiments.energy_predict_optimize.evaluation import compute_batch_metrics
from model.utils import seed_worker, seq_collate

from .evaluate_tartan_retrain import (
    AIRPORTS,
    REGIMES,
    _authorize_split,
    _dataset,
    _formal_test_gate,
    _selected_checkpoint_triplet,
    _verify_freeze_receipt,
)
from .operator import (
    DEFAULT_ADE_SCALE,
    energy_kl_projection,
    exact_gibbs_transport,
    pairwise_trajectory_distance,
    sinkhorn_transport,
    support_cost,
)
from .partc_seed_evaluate import _load_model_pair
from .train_tartan_retrain import _limited_indices


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("tartan_probability_ablation_protocol_v1.json")
PARENT_PROTOCOL = Path(__file__).with_name("tartan_retrain_protocol_v1.json")
PARENT_FREEZE = ROOT / "artifacts/partc_two_dataset_20260812/tartan_retrain_frozen_receipt_v1.json"
SELECTION_RECEIPT = ROOT / "artifacts/partc_two_dataset_20260812/probability_ablation_v1/development_selection_receipt_final_v1.json"

SHARED_SUPPORT_ARMS = (
    "target_native",
    "target_energy",
    "uniform_energy_kl",
    "sinkhorn_energy_kl",
    "gibbs_unweighted_prior",
    "gibbs_mass_aware_prior",
    "gibbs_unweighted_energy_kl",
    "gibbs_mass_aware_energy_kl",
)
ARMS = ("ascent_native", *SHARED_SUPPORT_ARMS, "source_target_native_union10")
METRICS = (
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
    "effective_modes",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"probability ablation refuses to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _verify_selection_receipt(
    *, root: Path = ROOT, receipt_path: Path = SELECTION_RECEIPT
) -> dict[str, object]:
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if payload.get("selection_split") != "development":
        raise RuntimeError("probability-ablation selection was not development-only")
    if payload.get("new_ablation_test_outputs_completed_before_freeze") is not False:
        raise RuntimeError("selection receipt does not precede the new test analysis")
    for relative, expected in payload.get("files", {}).items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(expected["bytes"]) or _sha256(path) != expected["sha256"]:
            raise RuntimeError(f"probability-ablation selection receipt mismatch: {relative}")
    return {
        "path": receipt_path.relative_to(root).as_posix(),
        "sha256": _sha256(receipt_path),
        "selected_arm": payload["selected_arm"],
        "algorithm_identity": payload["algorithm_identity"],
    }


class _StreamingSummary:
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
        for metric in METRICS:
            self.sums[metric] += float(arrays[metric][mask].sum())
        confidence = arrays["confidence"][mask]
        correct = arrays["oracle_ade_rank1"][mask]
        edges = np.linspace(0.0, 1.0, len(self.bin_counts) + 1)
        for index in range(len(self.bin_counts)):
            upper = confidence <= edges[index + 1] if index == len(self.bin_counts) - 1 else confidence < edges[index + 1]
            selected = (confidence >= edges[index]) & upper
            self.bin_counts[index] += int(selected.sum())
            self.bin_confidence[index] += float(confidence[selected].sum())
            self.bin_correct[index] += float(correct[selected].sum())

    def summary(self) -> dict[str, object]:
        if not self.count:
            raise RuntimeError("empty probability-ablation metric summary")
        ece = 0.0
        for count, confidence, correct in zip(
            self.bin_counts, self.bin_confidence, self.bin_correct, strict=True
        ):
            if count:
                ece += (count / self.count) * abs(correct / count - confidence / count)
        return {
            "agents": self.count,
            **{metric: self.sums[metric] / self.count for metric in METRICS},
            "ece": float(ece),
        }


class StreamingMetricStore:
    def __init__(self) -> None:
        self.overall = _StreamingSummary()
        self.per_date: dict[str, _StreamingSummary] = {}

    def update(self, metrics: dict[str, torch.Tensor], actor_dates: np.ndarray) -> None:
        arrays = {
            name: value.detach().to(torch.float64).cpu().numpy()
            for name, value in metrics.items()
        }
        self.update_arrays(arrays, actor_dates)

    def update_arrays(
        self, arrays: dict[str, np.ndarray], actor_dates: np.ndarray
    ) -> None:
        self.overall.update(arrays)
        for date in sorted(set(actor_dates.tolist())):
            mask = actor_dates == date
            self.per_date.setdefault(str(date), _StreamingSummary()).update(arrays, mask)

    def summary(self) -> dict[str, object]:
        return {
            "overall": self.overall.summary(),
            "per_date": {
                date: state.summary() for date, state in sorted(self.per_date.items())
            },
        }


def _update_metric_grid(
    states: dict[str, StreamingMetricStore],
    metrics_by_arm: dict[str, dict[str, torch.Tensor]],
    actor_dates: np.ndarray,
) -> None:
    """Transfer one tensor per metric instead of one tensor per arm and metric."""
    arm_names = list(states)
    if set(metrics_by_arm) != set(arm_names):
        raise RuntimeError("metric grid does not match the registered arms")
    metric_names = (*METRICS, "confidence")
    if any(
        metric not in metrics_by_arm[arm]
        for arm in arm_names
        for metric in metric_names
    ):
        raise RuntimeError("metric grid is missing a registered summary field")
    arrays = {
        metric: torch.stack(
            [metrics_by_arm[arm][metric] for arm in arm_names], dim=0
        ).detach().to(torch.float64).cpu().numpy()
        for metric in metric_names
    }
    for arm_index, arm in enumerate(arm_names):
        states[arm].update_arrays(
            {metric: values[arm_index] for metric, values in arrays.items()},
            actor_dates,
        )


def shared_support_probability_arms(
    source_probability: torch.Tensor,
    target_native_probability: torch.Tensor,
    target_energy_probability: torch.Tensor,
    cross_cost: torch.Tensor,
    predicted_risk: torch.Tensor,
    pairwise: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Construct all registered same-support arms without a future target."""
    source_probability = source_probability.to(torch.float64)
    target_native_probability = target_native_probability.to(torch.float64)
    target_energy_probability = target_energy_probability.to(torch.float64)
    unweighted = exact_gibbs_transport(
        source_probability, cross_cost, mass_weighted=False
    )
    mass_aware = exact_gibbs_transport(
        source_probability, cross_cost, mass_weighted=True
    )
    sinkhorn = sinkhorn_transport(source_probability, cross_cost)
    uniform = torch.full_like(source_probability, 1.0 / source_probability.shape[1])
    prior_names = (
        "uniform_energy_kl",
        "sinkhorn_energy_kl",
        "gibbs_unweighted_energy_kl",
        "gibbs_mass_aware_energy_kl",
    )
    priors = torch.stack(
        (
            uniform,
            sinkhorn["transported"],
            unweighted["transported"],
            mass_aware["transported"],
        ),
        dim=1,
    )
    batch, prior_count, modes = priors.shape
    projected = energy_kl_projection(
        priors.reshape(batch * prior_count, modes),
        predicted_risk[:, None].expand(-1, prior_count, -1).reshape(
            batch * prior_count, modes
        ),
        pairwise[:, None].expand(-1, prior_count, -1, -1).reshape(
            batch * prior_count, modes, modes
        ),
    )[0].reshape(batch, prior_count, modes)
    projection = {
        name: projected[:, index] for index, name in enumerate(prior_names)
    }
    arms = {
        "target_native": target_native_probability,
        "target_energy": target_energy_probability,
        "uniform_energy_kl": projection["uniform_energy_kl"],
        "sinkhorn_energy_kl": projection["sinkhorn_energy_kl"],
        "gibbs_unweighted_prior": unweighted["transported"],
        "gibbs_mass_aware_prior": mass_aware["transported"],
        "gibbs_unweighted_energy_kl": projection[
            "gibbs_unweighted_energy_kl"
        ],
        "gibbs_mass_aware_energy_kl": projection[
            "gibbs_mass_aware_energy_kl"
        ],
    }
    diagnostics = {
        "projected_mass_vs_unweighted_l1": (
            arms["gibbs_mass_aware_energy_kl"]
            - arms["gibbs_unweighted_energy_kl"]
        ).abs().sum(dim=1),
        "prior_mass_vs_unweighted_l1": (
            mass_aware["transported"] - unweighted["transported"]
        ).abs().sum(dim=1),
        "mass_assignment_entropy": mass_aware["assignment_entropy"],
        "unweighted_assignment_entropy": unweighted["assignment_entropy"],
        "sinkhorn_row_error": sinkhorn["row_error"],
        "sinkhorn_column_error": sinkhorn["column_error"],
    }
    return arms, diagnostics


def _shared_support_metrics(
    support: torch.Tensor,
    probabilities: dict[str, torch.Tensor],
    decision: torch.Tensor,
    truth: torch.Tensor,
) -> dict[str, dict[str, torch.Tensor]]:
    """Reuse target geometry while matching compute_batch_metrics semantics."""
    support = support.to(torch.float64)
    truth = truth.to(torch.float64)
    displacement = torch.linalg.vector_norm(support - truth[:, None], dim=-1)
    ade = displacement.mean(dim=-1)
    fde = displacement[..., -1]
    pairwise = torch.linalg.vector_norm(
        support[:, :, None] - support[:, None, :], dim=-1
    ).mean(dim=-1)
    batch = torch.arange(support.shape[0], device=support.device)
    oracle_ade = ade.argmin(dim=1)
    oracle_fde = fde.argmin(dim=1)
    common = {
        "top1_ade": ade[batch, decision],
        "top1_fde": fde[batch, decision],
        "minade": ade.min(dim=1).values,
        "minfde": fde.min(dim=1).values,
        "oracle_ade_rank1": (decision == oracle_ade).to(torch.float64),
        "oracle_fde_rank1": (decision == oracle_fde).to(torch.float64),
    }
    one_hot = F.one_hot(oracle_ade, num_classes=support.shape[1]).to(torch.float64)
    names = list(probabilities)
    probability = torch.stack(
        [probabilities[name].to(torch.float64) for name in names], dim=0
    )
    order = probability.argsort(dim=2, descending=True, stable=True)
    entropy = -(
        probability
        * probability.clamp_min(torch.finfo(probability.dtype).tiny).log()
    ).sum(dim=2)
    probability_at_oracle = probability.gather(
        2, oracle_ade[None, :, None].expand(len(names), -1, -1)
    ).squeeze(2)
    confidence = probability.gather(
        2, decision[None, :, None].expand(len(names), -1, -1)
    ).squeeze(2)
    energy = torch.einsum("abk,bk->ab", probability, ade) - 0.5 * torch.einsum(
        "abk,bkl,abl->ab", probability, pairwise, probability
    )
    brier = torch.square(probability - one_hot[None]).sum(dim=2)
    ade_rank = (
        order == oracle_ade[None, :, None]
    ).to(torch.int64).argmax(dim=2) + 1
    fde_rank = (
        order == oracle_fde[None, :, None]
    ).to(torch.int64).argmax(dim=2) + 1
    result = {}
    for arm_index, name in enumerate(names):
        result[name] = {
            **common,
            "energy_score": energy[arm_index],
            "nll": -probability_at_oracle[arm_index].clamp_min(
                torch.finfo(probability.dtype).tiny
            ).log(),
            "brier": brier[arm_index],
            "oracle_ade_rank": ade_rank[arm_index],
            "oracle_fde_rank": fde_rank[arm_index],
            "confidence": confidence[arm_index],
            "effective_modes": entropy[arm_index].exp(),
        }
    return result


def _with_effective_modes(
    metrics: dict[str, torch.Tensor], probability: torch.Tensor
) -> dict[str, torch.Tensor]:
    probability = probability.to(torch.float64)
    entropy = -(probability * probability.clamp_min(torch.finfo(probability.dtype).tiny).log()).sum(dim=1)
    return {**metrics, "effective_modes": entropy.exp()}


@torch.inference_mode()
def run(
    *,
    airport: str,
    regime: str,
    seed: int,
    split: str,
    device: torch.device,
    workers: int,
    batch_size: int,
    max_scenes: int | None,
    authorize_locked_test: bool = False,
    root: Path = ROOT,
    preloaded_development: tuple[object, list[str], Path] | None = None,
) -> dict[str, object]:
    protocol = json.loads((root / PROTOCOL.relative_to(ROOT)).read_text(encoding="utf-8"))
    parent_protocol_path = root / PARENT_PROTOCOL.relative_to(ROOT)
    parent = json.loads(parent_protocol_path.read_text(encoding="utf-8"))
    if airport not in AIRPORTS or airport not in protocol["data"]["airports"]:
        raise ValueError("airport lies outside the probability-ablation registry")
    if regime not in REGIMES or regime not in protocol["data"]["regimes"]:
        raise ValueError("regime lies outside the probability-ablation registry")
    if seed not in [int(value) for value in protocol["data"]["seeds"]]:
        raise ValueError("seed lies outside the probability-ablation registry")

    parent_receipt = root / PARENT_FREEZE.relative_to(ROOT)
    test_gate = _authorize_split(
        split=split,
        authorize_locked_test=authorize_locked_test,
        max_scenes=max_scenes,
        formal_gate=lambda: {
            "parent": _formal_test_gate(
                root=root, protocol=parent, receipt_path=parent_receipt
            ),
            "selection": _verify_selection_receipt(
                root=root, receipt_path=root / SELECTION_RECEIPT.relative_to(ROOT)
            ),
        },
    )
    parent_freeze = _verify_freeze_receipt(root=root, receipt_path=parent_receipt)
    checkpoints = _selected_checkpoint_triplet(
        root=root,
        protocol=parent,
        airport=airport,
        regime=regime,
        seed=seed,
        formal=True,
    )

    if preloaded_development is not None:
        if split != "development":
            raise RuntimeError("preloaded data is permitted only for development")
        dataset, all_dates, index_path = preloaded_development
    else:
        dataset, all_dates, index_path = _dataset(parent, airport, split)
    selected_indices = _limited_indices(len(dataset), max_scenes)
    selected_dates = [all_dates[index] for index in selected_indices]
    evaluation = Subset(dataset, selected_indices)
    options: dict[str, Any] = {
        "dataset": evaluation,
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": seq_collate,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }
    if workers > 0:
        options.update({"persistent_workers": True, "prefetch_factor": 4})
    loader = DataLoader(**options)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    source, target = _load_model_pair(
        source_checkpoint=root / checkpoints["ascent"]["path"],
        target_checkpoint=root / checkpoints["predicted_risk"]["path"],
        device=device,
        batch_size=batch_size,
    )

    states = {arm: StreamingMetricStore() for arm in ARMS}
    diagnostic_sum: dict[str, float] = defaultdict(float)
    diagnostic_count = 0
    cursor = 0
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
        inference_started = time.perf_counter()
        truth = data["pred_traj"].transpose(1, 0).to(torch.float64)
        source_support, source_logits, _ = source(data)
        source_probability = source_logits.softmax(dim=1)
        target_support, target_energy, target_decision, auxiliary = target(data)
        target_native = auxiliary["decision_logits"].softmax(dim=1)
        cross = support_cost(source_support, target_support)
        pairwise = pairwise_trajectory_distance(target_support) / DEFAULT_ADE_SCALE
        predicted_risk = auxiliary[
            "centered_predicted_normalized_ade_risk"
        ].to(torch.float64)
        target_probabilities, diagnostics = shared_support_probability_arms(
            source_probability,
            target_native,
            target_energy,
            cross,
            predicted_risk,
            pairwise,
        )
        shared_metrics = _shared_support_metrics(
            target_support, target_probabilities, target_decision, truth
        )
        source_metrics = _with_effective_modes(
            compute_batch_metrics(
                source_support.to(torch.float64),
                source_probability.to(torch.float64),
                source_logits.argmax(dim=1),
                truth,
            ),
            source_probability,
        )
        union_support = torch.cat((source_support, target_support), dim=1)
        union_probability = torch.cat((source_probability, target_native), dim=1) * 0.5
        union_metrics = _with_effective_modes(
            compute_batch_metrics(
                union_support.to(torch.float64),
                union_probability.to(torch.float64),
                union_probability.argmax(dim=1),
                truth,
            ),
            union_probability,
        )
        metrics_by_arm = {
            "ascent_native": source_metrics,
            **shared_metrics,
            "source_target_native_union10": union_metrics,
        }
        packed = pack_scenes(data["adj"])
        batch_dates = selected_dates[cursor : cursor + packed.scene_count]
        if len(batch_dates) != packed.scene_count:
            raise RuntimeError("probability-ablation scene/date alignment failed")
        actor_dates = np.asarray(batch_dates, dtype=object)[
            packed.inverse.detach().cpu().numpy()
        ]
        _update_metric_grid(states, metrics_by_arm, actor_dates)
        for name, values in diagnostics.items():
            diagnostic_sum[name] += float(values.sum().detach().cpu())
        diagnostic_count += int(source_support.shape[0])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter() - inference_started
        cursor += packed.scene_count
        actors += int(source_support.shape[0])
        batches += 1
    if cursor != len(evaluation):
        raise RuntimeError("probability ablation did not consume the cohort exactly once")
    models = {arm: state.summary() for arm, state in states.items()}
    reference = models["target_native"]["overall"]
    geometry_invariance = {
        metric: max(
            abs(float(models[arm]["overall"][metric]) - float(reference[metric]))
            for arm in SHARED_SUPPORT_ARMS
        )
        for metric in ("top1_ade", "top1_fde", "minade", "minfde")
    }
    if any(value > 1e-12 for value in geometry_invariance.values()):
        raise RuntimeError("shared-support geometry unexpectedly differs across probability arms")
    return {
        "format_version": 1,
        "experiment_id": "Tartan_probability_ablation_v1",
        "evidence_class": (
            "development_only_operator_selection"
            if split == "development"
            else "locked_retrospective_test_post_main_analysis"
        ),
        "airport": airport,
        "regime": regime,
        "seed": seed,
        "split": split,
        "scenes": len(evaluation),
        "actors": actors,
        "models": models,
        "shared_support_arms": list(SHARED_SUPPORT_ARMS),
        "capacity_control_arm": "source_target_native_union10",
        "diagnostics": {
            name: value / max(diagnostic_count, 1)
            for name, value in sorted(diagnostic_sum.items())
        },
        "geometry_invariance_max_absolute_difference": geometry_invariance,
        "inputs": {
            "evaluator": {
                "path": Path(__file__).relative_to(ROOT).as_posix(),
                "sha256": _sha256(Path(__file__)),
            },
            "protocol": {
                "path": PROTOCOL.relative_to(ROOT).as_posix(),
                "sha256": _sha256(root / PROTOCOL.relative_to(ROOT)),
            },
            "evaluator": {
                "path": Path(__file__).resolve().relative_to(root).as_posix(),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "parent_protocol": {
                "path": PARENT_PROTOCOL.relative_to(ROOT).as_posix(),
                "sha256": _sha256(parent_protocol_path),
            },
            "parent_freeze_receipt": parent_freeze,
            "selection_receipt": test_gate["selection"] if test_gate else None,
            "checkpoints": checkpoints,
            "scene_date_index": index_path.relative_to(root).as_posix(),
            "scene_date_index_sha256": _sha256(index_path),
        },
        "integrity": {
            "selection_split": split == "development",
            "locked_test_used": split == "test",
            "partial_locked_test": False,
            "target_in_probability_forward": False,
            "all_registered_arms_reported": set(models) == set(ARMS),
            "shared_support_geometry_exactly_invariant": True,
            "union10_excluded_from_component_attribution": True,
        },
        "runtime": {
            "device": str(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "workers": workers,
            "batch_size": batch_size,
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
        "claim_boundaries": protocol["claim_boundaries"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airport", choices=AIRPORTS, required=True)
    parser.add_argument("--regime", choices=REGIMES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=("development", "test"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
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
        regime=args.regime,
        seed=args.seed,
        split=args.split,
        device=torch.device(args.device),
        workers=args.workers,
        batch_size=args.batch_size,
        max_scenes=args.max_scenes,
        authorize_locked_test=args.authorize_locked_test,
    )
    _atomic_json(args.output.resolve(), result)
    print(json.dumps({
        "output": args.output.resolve().as_posix(),
        "airport": result["airport"],
        "regime": result["regime"],
        "seed": result["seed"],
        "split": result["split"],
        "actors": result["actors"],
        "mass_energy": result["models"]["gibbs_mass_aware_energy_kl"]["overall"]["energy_score"],
        "unweighted_energy": result["models"]["gibbs_unweighted_energy_kl"]["overall"]["energy_score"],
        "elapsed_seconds": result["runtime"]["total_elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
