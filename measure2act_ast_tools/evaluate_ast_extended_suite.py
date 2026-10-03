"""Extended AST review experiments for Measure2Act.

This sidecar evaluator keeps the frozen ASCENT workspace read-only.  It runs
one source/target forward pass per airport-regime-seed-split cell, then derives
the extra AST-review diagnostics requested after the core closure:

E1 wrong-source proxies, E3 full-path score, E4 fixed physical-event
calibration, E6 operator sensitivity, E7 risk-head diagnostics, E8 physical
envelope summaries, and E9 end-to-end runtime.
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
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("ASCENT_FULL_ROOT", DEFAULT_ROOT)).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.edfa_ascent.relation import pack_scenes  # noqa: E402
from mabpt.evaluate_tartan_probability_ablation import (  # noqa: E402
    METRICS,
    PARENT_FREEZE,
    PARENT_PROTOCOL,
    PROTOCOL,
    SELECTION_RECEIPT,
    _json_safe,
    _sha256,
    _shared_support_metrics,
    _verify_selection_receipt,
)
from mabpt.evaluate_tartan_retrain import (  # noqa: E402
    AIRPORTS,
    REGIMES,
    _authorize_split,
    _dataset,
    _formal_test_gate,
    _selected_checkpoint_triplet,
    _verify_freeze_receipt,
)
from mabpt.operator import (  # noqa: E402
    DEFAULT_ADE_SCALE,
    energy_kl_projection,
    exact_gibbs_transport,
    pairwise_trajectory_distance,
    support_cost,
)
from mabpt.partc_seed_evaluate import _load_model_pair  # noqa: E402
from mabpt.physical import FEATURES, kinematic_features  # noqa: E402
from mabpt.train_tartan_retrain import _limited_indices  # noqa: E402
from model.utils import seed_worker, seq_collate  # noqa: E402


SELECTED_ARM = "selected_mabpt"
ARMS = (
    "target_native",
    SELECTED_ARM,
    "wrong_source_actor_shift_gibbs_energy_kl",
    "temp0p5_gibbs_energy_kl",
    "temp2_gibbs_energy_kl",
    "risk0_energy_kl",
    "risk0p5_energy_kl",
    "risk2_energy_kl",
    "kl0p5_energy_kl",
    "kl2_energy_kl",
    "diversity0_energy_kl",
    "diversity2_energy_kl",
)
SCALAR_METRICS = tuple(
    dict.fromkeys(
        (
            *METRICS,
            "energy_score_full_path",
            "endpoint_grid_nll",
            "endpoint_grid_brier",
            "endpoint_grid_confidence",
            "endpoint_grid_correct",
            "endpoint_radial_nll",
            "endpoint_radial_brier",
            "endpoint_radial_confidence",
            "endpoint_radial_correct",
            "effective_modes",
        )
    )
)
PHYSICAL_ENVELOPE_SOURCE = Path(
    "$LOCAL_WORKSPACE/_paper_migration_build_20260906_v5/"
    "MABPT_ASCENT_Paper_Migration_20260906_v5/03_核心实验结果/"
    "mabpt_partc_20260811/seed42_development_formal_v1.json"
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def normalized(probability: torch.Tensor) -> torch.Tensor:
    probability = probability.to(torch.float64).clamp_min(torch.finfo(torch.float64).tiny)
    return probability / probability.sum(dim=1, keepdim=True)


def project(
    prior: torch.Tensor,
    predicted_risk: torch.Tensor,
    pairwise: torch.Tensor,
    *,
    risk_weight: float = 1.0,
    diversity_weight: float = 1.0,
    kl_weight: float = 1.0,
) -> torch.Tensor:
    return energy_kl_projection(
        prior,
        predicted_risk,
        pairwise,
        risk_weight=risk_weight,
        diversity_weight=diversity_weight,
        kl_weight=kl_weight,
        backtracking_steps=32,
        tolerance=1e-12,
    )[0]


def gibbs_prior(
    probability: torch.Tensor,
    cost: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return exact_gibbs_transport(
        probability,
        cost.to(torch.float64) / temperature,
        mass_weighted=False,
    )["transported"]


def probability_arms(
    source_support: torch.Tensor,
    source_probability: torch.Tensor,
    target_support: torch.Tensor,
    target_native_probability: torch.Tensor,
    predicted_risk: torch.Tensor,
    pairwise: torch.Tensor,
) -> dict[str, torch.Tensor]:
    source_probability = normalized(source_probability)
    target_native_probability = normalized(target_native_probability)
    cross = support_cost(source_support, target_support)
    base_prior = gibbs_prior(source_probability, cross, temperature=1.0)
    wrong_cross = support_cost(source_support.roll(1, dims=0), target_support)
    wrong_prior = gibbs_prior(
        source_probability.roll(1, dims=0),
        wrong_cross,
        temperature=1.0,
    )
    priors = {
        "selected_mabpt": base_prior,
        "wrong_source_actor_shift_gibbs_energy_kl": wrong_prior,
        "temp0p5_gibbs_energy_kl": gibbs_prior(source_probability, cross, temperature=0.5),
        "temp2_gibbs_energy_kl": gibbs_prior(source_probability, cross, temperature=2.0),
        "risk0_energy_kl": base_prior,
        "risk0p5_energy_kl": base_prior,
        "risk2_energy_kl": base_prior,
        "kl0p5_energy_kl": base_prior,
        "kl2_energy_kl": base_prior,
        "diversity0_energy_kl": base_prior,
        "diversity2_energy_kl": base_prior,
    }
    arms: dict[str, torch.Tensor] = {"target_native": target_native_probability}
    for name, prior in priors.items():
        kwargs: dict[str, float] = {}
        if name == "risk0_energy_kl":
            kwargs["risk_weight"] = 0.0
        elif name == "risk0p5_energy_kl":
            kwargs["risk_weight"] = 0.5
        elif name == "risk2_energy_kl":
            kwargs["risk_weight"] = 2.0
        elif name == "kl0p5_energy_kl":
            kwargs["kl_weight"] = 0.5
        elif name == "kl2_energy_kl":
            kwargs["kl_weight"] = 2.0
        elif name == "diversity0_energy_kl":
            kwargs["diversity_weight"] = 0.0
        elif name == "diversity2_energy_kl":
            kwargs["diversity_weight"] = 2.0
        arms[name] = project(prior, predicted_risk, pairwise, **kwargs)
    if set(arms) != set(ARMS):
        raise RuntimeError("extended arm registry mismatch")
    return {name: arms[name] for name in ARMS}


def full_path_energy(
    support: torch.Tensor,
    probabilities: dict[str, torch.Tensor],
    truth: torch.Tensor,
) -> dict[str, torch.Tensor]:
    support = support.to(torch.float64)
    truth = truth.to(torch.float64)
    steps = support.shape[2]
    flat_error = (support - truth[:, None]).reshape(support.shape[0], support.shape[1], -1)
    distance = torch.linalg.vector_norm(flat_error, dim=-1) / math.sqrt(steps)
    flat_pair = (support[:, :, None] - support[:, None]).reshape(
        support.shape[0], support.shape[1], support.shape[1], -1
    )
    pairwise = torch.linalg.vector_norm(flat_pair, dim=-1) / math.sqrt(steps)
    return {
        name: (probability.to(torch.float64) * distance).sum(dim=1)
        - 0.5 * torch.einsum(
            "bi,bij,bj->b",
            probability.to(torch.float64),
            pairwise,
            probability.to(torch.float64),
        )
        for name, probability in probabilities.items()
    }


def endpoint_grid_label(endpoint: torch.Tensor) -> torch.Tensor:
    x_bin = torch.bucketize(endpoint[:, 0], torch.tensor([-0.5, 0.5], device=endpoint.device))
    y_bin = torch.bucketize(endpoint[:, 1], torch.tensor([-0.5, 0.5], device=endpoint.device))
    return x_bin * 3 + y_bin


def endpoint_event_metrics(
    support: torch.Tensor,
    probabilities: dict[str, torch.Tensor],
    truth: torch.Tensor,
) -> dict[str, dict[str, torch.Tensor]]:
    endpoint = support[:, :, -1].to(torch.float64)
    truth_endpoint = truth[:, -1].to(torch.float64)
    target_grid = endpoint_grid_label(truth_endpoint)
    candidate_grid = endpoint_grid_label(endpoint.reshape(-1, 3)).reshape(endpoint.shape[:2])
    radial_edges = torch.tensor([0.5, 1.0], dtype=torch.float64, device=endpoint.device)
    target_radial = torch.bucketize(
        torch.linalg.vector_norm(truth_endpoint[:, :2], dim=1),
        radial_edges,
    )
    candidate_radial = torch.bucketize(
        torch.linalg.vector_norm(endpoint[..., :2], dim=2),
        radial_edges,
    )
    result: dict[str, dict[str, torch.Tensor]] = {}
    for name, probability in probabilities.items():
        probability = probability.to(torch.float64)
        grid_mass = torch.zeros((support.shape[0], 9), dtype=torch.float64, device=support.device)
        grid_mass.scatter_add_(1, candidate_grid, probability)
        radial_mass = torch.zeros((support.shape[0], 3), dtype=torch.float64, device=support.device)
        radial_mass.scatter_add_(1, candidate_radial, probability)
        grid_true = grid_mass.gather(1, target_grid[:, None]).squeeze(1)
        radial_true = radial_mass.gather(1, target_radial[:, None]).squeeze(1)
        grid_pred = grid_mass.argmax(dim=1)
        radial_pred = radial_mass.argmax(dim=1)
        grid_one_hot = F.one_hot(target_grid, num_classes=9).to(torch.float64)
        radial_one_hot = F.one_hot(target_radial, num_classes=3).to(torch.float64)
        result[name] = {
            "endpoint_grid_nll": -grid_true.clamp_min(torch.finfo(torch.float64).tiny).log(),
            "endpoint_grid_brier": torch.square(grid_mass - grid_one_hot).sum(dim=1),
            "endpoint_grid_confidence": grid_mass.max(dim=1).values,
            "endpoint_grid_correct": (grid_pred == target_grid).to(torch.float64),
            "endpoint_radial_nll": -radial_true.clamp_min(torch.finfo(torch.float64).tiny).log(),
            "endpoint_radial_brier": torch.square(radial_mass - radial_one_hot).sum(dim=1),
            "endpoint_radial_confidence": radial_mass.max(dim=1).values,
            "endpoint_radial_correct": (radial_pred == target_radial).to(torch.float64),
        }
    return result


def effective_modes(probability: torch.Tensor) -> torch.Tensor:
    probability = normalized(probability)
    entropy = -(probability * probability.log()).sum(dim=1)
    return entropy.exp()


class ScalarSummary:
    def __init__(self, bins: int = 15) -> None:
        self.count = 0
        self.sums: dict[str, float] = defaultdict(float)
        self.bin_counts: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(bins, dtype=np.int64))
        self.bin_conf: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(bins, dtype=np.float64))
        self.bin_corr: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(bins, dtype=np.float64))

    def update(self, arrays: dict[str, np.ndarray], mask: np.ndarray | None = None) -> None:
        if mask is None:
            mask = np.ones(arrays["energy_score"].shape[0], dtype=bool)
        count = int(mask.sum())
        self.count += count
        for metric in SCALAR_METRICS:
            if metric in arrays:
                self.sums[metric] += float(arrays[metric][mask].sum())
        self._update_ece("support_index", arrays["confidence"][mask], arrays["oracle_ade_rank1"][mask])
        self._update_ece("endpoint_grid", arrays["endpoint_grid_confidence"][mask], arrays["endpoint_grid_correct"][mask])
        self._update_ece("endpoint_radial", arrays["endpoint_radial_confidence"][mask], arrays["endpoint_radial_correct"][mask])

    def _update_ece(self, name: str, confidence: np.ndarray, correct: np.ndarray) -> None:
        edges = np.linspace(0.0, 1.0, len(self.bin_counts[name]) + 1)
        for index in range(len(self.bin_counts[name])):
            upper = confidence <= edges[index + 1] if index == len(self.bin_counts[name]) - 1 else confidence < edges[index + 1]
            selected = (confidence >= edges[index]) & upper
            self.bin_counts[name][index] += int(selected.sum())
            self.bin_conf[name][index] += float(confidence[selected].sum())
            self.bin_corr[name][index] += float(correct[selected].sum())

    def summary(self) -> dict[str, float | int]:
        if not self.count:
            raise RuntimeError("empty scalar summary")
        result: dict[str, float | int] = {"agents": self.count}
        for metric, value in sorted(self.sums.items()):
            result[metric] = value / self.count
        for name in sorted(self.bin_counts):
            ece = 0.0
            for count, conf, corr in zip(self.bin_counts[name], self.bin_conf[name], self.bin_corr[name], strict=True):
                if count:
                    ece += (count / self.count) * abs(corr / count - conf / count)
            result[f"{name}_ece"] = float(ece)
        return result


class ArmStore:
    def __init__(self) -> None:
        self.overall = ScalarSummary()
        self.per_date: dict[str, ScalarSummary] = {}

    def update(self, metrics: dict[str, torch.Tensor], dates: np.ndarray) -> None:
        arrays = {
            name: value.detach().to(torch.float64).cpu().numpy()
            for name, value in metrics.items()
        }
        self.overall.update(arrays)
        for date in sorted(set(dates.tolist())):
            mask = dates == date
            self.per_date.setdefault(str(date), ScalarSummary()).update(arrays, mask)

    def summary(self) -> dict[str, Any]:
        return {
            "overall": self.overall.summary(),
            "per_date": {date: state.summary() for date, state in sorted(self.per_date.items())},
        }


class RunningMoments:
    def __init__(self) -> None:
        self.count = 0
        self.weight = 0.0
        self.sums: dict[str, float] = defaultdict(float)

    def add(self, *, weight: int, **values: float) -> None:
        self.count += 1
        self.weight += float(weight)
        for key, value in values.items():
            self.sums[key] += float(value) * float(weight)

    def summary(self) -> dict[str, float | int]:
        return {
            "batches": self.count,
            "actors": int(self.weight),
            **{key: value / max(self.weight, 1.0) for key, value in sorted(self.sums.items())},
        }


def rankdata(values: torch.Tensor) -> torch.Tensor:
    order = values.argsort(dim=1, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float64)
    base = torch.arange(values.shape[1], dtype=torch.float64, device=values.device)
    ranks.scatter_(1, order, base[None].expand_as(ranks))
    return ranks


def risk_diagnostics(
    target_support: torch.Tensor,
    truth: torch.Tensor,
    predicted_risk: torch.Tensor,
) -> dict[str, float]:
    displacement = torch.linalg.vector_norm(target_support.to(torch.float64) - truth[:, None].to(torch.float64), dim=-1)
    ade = displacement.mean(dim=-1) / DEFAULT_ADE_SCALE
    true_centered = ade - ade.mean(dim=1, keepdim=True)
    pred = predicted_risk.to(torch.float64)
    error = pred - true_centered
    pred_centered = pred - pred.mean(dim=1, keepdim=True)
    true_centered2 = true_centered - true_centered.mean(dim=1, keepdim=True)
    pearson_num = (pred_centered * true_centered2).sum(dim=1)
    pearson_den = torch.sqrt((pred_centered.square().sum(dim=1) * true_centered2.square().sum(dim=1)).clamp_min(torch.finfo(torch.float64).tiny))
    pred_rank = rankdata(pred)
    true_rank = rankdata(true_centered)
    pred_rank = pred_rank - pred_rank.mean(dim=1, keepdim=True)
    true_rank = true_rank - true_rank.mean(dim=1, keepdim=True)
    spearman = (pred_rank * true_rank).sum(dim=1) / torch.sqrt(
        (pred_rank.square().sum(dim=1) * true_rank.square().sum(dim=1)).clamp_min(torch.finfo(torch.float64).tiny)
    )
    batch = torch.arange(target_support.shape[0], device=target_support.device)
    best_pred = pred.argmin(dim=1)
    worst_pred = pred.argmax(dim=1)
    return {
        "mse": float(error.square().mean().detach().cpu()),
        "mae": float(error.abs().mean().detach().cpu()),
        "pearson": float((pearson_num / pearson_den).mean().detach().cpu()),
        "spearman": float(spearman.mean().detach().cpu()),
        "true_risk_predicted_best": float(true_centered[batch, best_pred].mean().detach().cpu()),
        "true_risk_predicted_worst": float(true_centered[batch, worst_pred].mean().detach().cpu()),
        "best_minus_worst_true_risk": float((true_centered[batch, best_pred] - true_centered[batch, worst_pred]).mean().detach().cpu()),
    }


def load_physical_envelope() -> dict[str, Any]:
    payload = json.loads(PHYSICAL_ENVELOPE_SOURCE.read_text(encoding="utf-8"))
    return payload["physical_envelope"]["envelope"]


def physical_batch_metrics(
    support: torch.Tensor,
    probability: torch.Tensor,
    decision: torch.Tensor,
    initial: torch.Tensor,
    envelope: dict[str, Any],
) -> dict[str, torch.Tensor]:
    features = kinematic_features(support, initial_position=initial, stride_seconds=5.0)
    probability = normalized(probability)
    batch = support.shape[0]
    decision_weight = F.one_hot(decision, num_classes=support.shape[1]).to(torch.float64)
    uniform = torch.full_like(probability, 1.0 / support.shape[1])
    result: dict[str, torch.Tensor] = {}
    weights = {
        "probability_weighted": probability,
        "selected": decision_weight,
        "equal_candidate": uniform,
    }
    for feature in FEATURES:
        values = features[feature].to(torch.float64)
        lower = float(envelope[feature]["lower"])
        upper = float(envelope[feature]["upper"])
        outside_by_mode = ((values < lower) | (values > upper)).to(torch.float64).mean(dim=2)
        mean_by_mode = values.mean(dim=2)
        for weight_name, weight in weights.items():
            result[f"physical_{weight_name}_{feature}_outside"] = (outside_by_mode * weight).sum(dim=1)
            result[f"physical_{weight_name}_{feature}_mean"] = (mean_by_mode * weight).sum(dim=1)
    if not result:
        raise RuntimeError("physical metric registry unexpectedly empty")
    for key, value in result.items():
        if value.shape != (batch,):
            raise RuntimeError(f"physical metric shape mismatch: {key}")
    return result


class PhysicalStore:
    def __init__(self) -> None:
        self.count = 0
        self.sums: dict[str, float] = defaultdict(float)

    def update(self, metrics: dict[str, torch.Tensor]) -> None:
        first = next(iter(metrics.values()))
        self.count += int(first.shape[0])
        for name, values in metrics.items():
            self.sums[name] += float(values.detach().to(torch.float64).sum().cpu())

    def summary(self) -> dict[str, float | int]:
        return {"agents": self.count, **{key: value / max(self.count, 1) for key, value in sorted(self.sums.items())}}


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
    authorize_locked_test: bool,
) -> dict[str, Any]:
    protocol = json.loads((ROOT / PROTOCOL.relative_to(ROOT)).read_text(encoding="utf-8"))
    parent_path = ROOT / PARENT_PROTOCOL.relative_to(ROOT)
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if airport not in AIRPORTS or airport not in protocol["data"]["airports"]:
        raise ValueError("unregistered airport")
    if regime not in REGIMES or regime not in protocol["data"]["regimes"]:
        raise ValueError("unregistered regime")
    if seed not in [int(value) for value in protocol["data"]["seeds"]]:
        raise ValueError("unregistered seed")
    parent_receipt = ROOT / PARENT_FREEZE.relative_to(ROOT)
    test_gate = _authorize_split(
        split=split,
        authorize_locked_test=authorize_locked_test,
        max_scenes=max_scenes,
        formal_gate=lambda: {
            "parent": _formal_test_gate(root=ROOT, protocol=parent, receipt_path=parent_receipt),
            "selection": _verify_selection_receipt(
                root=ROOT, receipt_path=ROOT / SELECTION_RECEIPT.relative_to(ROOT)
            ),
        },
    )
    parent_freeze = _verify_freeze_receipt(root=ROOT, receipt_path=parent_receipt)
    checkpoints = _selected_checkpoint_triplet(
        root=ROOT, protocol=parent, airport=airport, regime=regime, seed=seed, formal=True
    )
    dataset, all_dates, index_path = _dataset(parent, airport, split)
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
    source, target = _load_model_pair(
        source_checkpoint=ROOT / checkpoints["ascent"]["path"],
        target_checkpoint=ROOT / checkpoints["predicted_risk"]["path"],
        device=device,
        batch_size=batch_size,
    )
    envelope = load_physical_envelope()
    stores = {arm: ArmStore() for arm in ARMS}
    risk_store = RunningMoments()
    physical_store = PhysicalStore()
    probability_sum_error = defaultdict(float)
    cursor = actors = batches = 0
    inference_seconds = 0.0
    started = time.perf_counter()
    for data in loader:
        data = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in data.items()
        }
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tic = time.perf_counter()
        truth = data["pred_traj"].transpose(1, 0).to(torch.float64)
        source_support, source_logits, _ = source(data)
        source_probability = source_logits.softmax(dim=1)
        target_support, _, target_decision, auxiliary = target(data)
        target_native = auxiliary["decision_logits"].softmax(dim=1)
        pairwise = pairwise_trajectory_distance(target_support) / DEFAULT_ADE_SCALE
        predicted_risk = auxiliary["centered_predicted_normalized_ade_risk"].to(torch.float64)
        probabilities = probability_arms(
            source_support,
            source_probability,
            target_support,
            target_native,
            predicted_risk,
            pairwise,
        )
        standard = _shared_support_metrics(target_support, probabilities, target_decision, truth)
        full_path = full_path_energy(target_support, probabilities, truth)
        events = endpoint_event_metrics(target_support, probabilities, truth)
        packed = pack_scenes(data["adj"])
        batch_dates = selected_dates[cursor : cursor + packed.scene_count]
        if len(batch_dates) != packed.scene_count:
            raise RuntimeError("scene/date alignment failed")
        actor_dates = np.asarray(batch_dates, dtype=object)[packed.inverse.detach().cpu().numpy()]
        for arm in ARMS:
            merged = {
                **standard[arm],
                **events[arm],
                "energy_score_full_path": full_path[arm],
                "effective_modes": effective_modes(probabilities[arm]),
            }
            stores[arm].update(merged, actor_dates)
            probability_sum_error[arm] = max(
                probability_sum_error[arm],
                float((probabilities[arm].sum(dim=1) - 1.0).abs().max().detach().cpu()),
            )
        risk_store.add(
            weight=int(target_support.shape[0]),
            **risk_diagnostics(target_support, truth, predicted_risk),
        )
        physical_store.update(
            physical_batch_metrics(
                target_support,
                probabilities[SELECTED_ARM],
                target_decision,
                data["obs_traj"][-1].to(torch.float64),
                envelope,
            )
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter() - tic
        cursor += packed.scene_count
        actors += int(target_support.shape[0])
        batches += 1
    if cursor != len(evaluation):
        raise RuntimeError("extended evaluator did not consume cohort exactly once")
    models = {arm: stores[arm].summary() for arm in ARMS}
    reference = models["target_native"]["overall"]
    geometry_invariance = {
        metric: max(
            abs(float(models[arm]["overall"][metric]) - float(reference[metric]))
            for arm in ARMS
        )
        for metric in ("top1_ade", "top1_fde", "minade", "minfde")
    }
    return {
        "format_version": 1,
        "experiment_id": "measure2act_ast_extended_suite_v1",
        "airport": airport,
        "regime": regime,
        "seed": seed,
        "split": split,
        "scenes": len(evaluation),
        "actors": actors,
        "models": models,
        "arms": list(ARMS),
        "risk_head_diagnostics": risk_store.summary(),
        "physical_envelope_source": PHYSICAL_ENVELOPE_SOURCE.as_posix(),
        "physical_envelope": envelope,
        "physical_selected_mabpt": physical_store.summary(),
        "probability_sum_max_abs_error": dict(probability_sum_error),
        "geometry_invariance_max_absolute_difference": geometry_invariance,
        "inputs": {
            "root": ROOT.as_posix(),
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "parent_protocol": {
                "path": PARENT_PROTOCOL.relative_to(ROOT).as_posix(),
                "sha256": _sha256(parent_path),
            },
            "probability_ablation_protocol": {
                "path": PROTOCOL.relative_to(ROOT).as_posix(),
                "sha256": _sha256(ROOT / PROTOCOL.relative_to(ROOT)),
            },
            "parent_freeze_receipt": parent_freeze,
            "selection_receipt": test_gate["selection"] if test_gate else None,
            "checkpoints": checkpoints,
            "scene_date_index": index_path.relative_to(ROOT).as_posix(),
            "scene_date_index_sha256": _sha256(index_path),
        },
        "integrity": {
            "target_in_probability_forward": False,
            "all_registered_arms_reported": set(models) == set(ARMS),
            "shared_support_geometry_invariant": True,
            "locked_test_used": split == "test",
            "partial_locked_test": False,
            "output_refuses_overwrite": True,
            "wrong_source_is_actor_shift_proxy_not_new_airport": True,
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
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
        },
        "claim_boundary": (
            "Local retrospective AST extended diagnostics. Wrong-source control is an "
            "actor-shift proxy because no unused third-airport source cohort is present."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airport", choices=AIRPORTS, required=True)
    parser.add_argument("--regime", choices=REGIMES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=("development", "test"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=6)
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
    atomic_json(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "output": args.output.resolve().as_posix(),
                "airport": result["airport"],
                "regime": result["regime"],
                "seed": result["seed"],
                "split": result["split"],
                "actors": result["actors"],
                "selected_energy": result["models"][SELECTED_ARM]["overall"]["energy_score"],
                "selected_full_path": result["models"][SELECTED_ARM]["overall"]["energy_score_full_path"],
                "elapsed_seconds": result["runtime"]["total_elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
