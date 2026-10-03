"""Fixed-support negative controls for Measure2Act/AST closure.

This sidecar evaluator deliberately lives outside the frozen ASCENT workspace.
It reuses the registered Tartan checkpoints and target support, then evaluates
probability-only negative controls without changing candidate trajectories.
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
    StreamingMetricStore,
    _json_safe,
    _sha256,
    _shared_support_metrics,
    _update_metric_grid,
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
    hard_bijection_transport,
    pairwise_trajectory_distance,
    support_cost,
)
from mabpt.partc_seed_evaluate import _load_model_pair  # noqa: E402
from mabpt.train_tartan_retrain import _limited_indices  # noqa: E402
from model.utils import seed_worker, seq_collate  # noqa: E402


NEGATIVE_ARMS = (
    "target_native",
    "selected_mabpt",
    "hard_unweighted_energy_kl",
    "uniform_energy_kl",
    "index_copy_energy_kl",
    "cyclic_index_energy_kl",
    "reverse_index_energy_kl",
    "actor_shift_source_energy_kl",
    "risk_only_softmax",
    "entropy_matched_target_native",
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


def entropy(probability: torch.Tensor) -> torch.Tensor:
    probability = normalized(probability)
    return -(probability * probability.log()).sum(dim=1)


def temperature_scale(probability: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    probability = normalized(probability)
    logits = probability.log() / temperature[:, None]
    return torch.softmax(logits, dim=1)


def entropy_match_target(
    probability: torch.Tensor,
    target_entropy: torch.Tensor,
    *,
    iterations: int = 32,
) -> torch.Tensor:
    """Per-row scalar temperature that matches the selected MABPT entropy."""
    probability = normalized(probability)
    target_entropy = target_entropy.to(torch.float64).clamp(0.0, math.log(probability.shape[1]))
    low = torch.full((probability.shape[0],), 0.02, dtype=torch.float64, device=probability.device)
    high = torch.full((probability.shape[0],), 50.0, dtype=torch.float64, device=probability.device)
    for _ in range(iterations):
        mid = (low + high) * 0.5
        mid_entropy = entropy(temperature_scale(probability, mid))
        low = torch.where(mid_entropy < target_entropy, mid, low)
        high = torch.where(mid_entropy >= target_entropy, mid, high)
    return temperature_scale(probability, (low + high) * 0.5)


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


def control_arms(
    source_probability: torch.Tensor,
    target_native_probability: torch.Tensor,
    cross_cost: torch.Tensor,
    predicted_risk: torch.Tensor,
    pairwise: torch.Tensor,
) -> dict[str, torch.Tensor]:
    source_probability = normalized(source_probability)
    target_native_probability = normalized(target_native_probability)
    predicted_risk = predicted_risk.to(torch.float64)
    pairwise = pairwise.to(torch.float64)
    modes = source_probability.shape[1]
    uniform = torch.full_like(source_probability, 1.0 / modes)
    hard_prior = hard_bijection_transport(
        source_probability, cross_cost, mass_weighted=False
    )["transported"]
    gibbs_prior = exact_gibbs_transport(
        source_probability, cross_cost, mass_weighted=False
    )["transported"]
    selected = project(gibbs_prior, predicted_risk, pairwise)
    arms = {
        "target_native": target_native_probability,
        "selected_mabpt": selected,
        "hard_unweighted_energy_kl": project(hard_prior, predicted_risk, pairwise),
        "uniform_energy_kl": project(uniform, predicted_risk, pairwise),
        "index_copy_energy_kl": project(source_probability, predicted_risk, pairwise),
        "cyclic_index_energy_kl": project(source_probability.roll(1, dims=1), predicted_risk, pairwise),
        "reverse_index_energy_kl": project(source_probability.flip(dims=(1,)), predicted_risk, pairwise),
        "actor_shift_source_energy_kl": project(source_probability.roll(1, dims=0), predicted_risk, pairwise),
        "risk_only_softmax": torch.softmax(-predicted_risk, dim=1),
        "entropy_matched_target_native": entropy_match_target(
            target_native_probability, entropy(selected)
        ),
    }
    if set(arms) != set(NEGATIVE_ARMS):
        raise RuntimeError("negative-control arm registry mismatch")
    return {name: arms[name] for name in NEGATIVE_ARMS}


def batch_support_digest(support: torch.Tensor) -> list[str]:
    support_np = support.detach().to(torch.float32).cpu().numpy()
    return [hashlib.sha256(row.tobytes()).hexdigest() for row in support_np]


def batch_probability_digest(probabilities: dict[str, torch.Tensor]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for name, probability in probabilities.items():
        array = probability.detach().to(torch.float32).cpu().numpy()
        result[name] = [hashlib.sha256(row.tobytes()).hexdigest() for row in array]
    return result


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
    audit_sample_limit: int,
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

    states = {arm: StreamingMetricStore() for arm in NEGATIVE_ARMS}
    cursor = actors = batches = 0
    inference_seconds = 0.0
    started = time.perf_counter()
    probability_sum_max_abs_error = defaultdict(float)
    probability_minimum = defaultdict(lambda: float("inf"))
    probability_maximum = defaultdict(float)
    audit_rows: list[dict[str, Any]] = []

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
        cross = support_cost(source_support, target_support)
        pairwise = pairwise_trajectory_distance(target_support) / DEFAULT_ADE_SCALE
        predicted_risk = auxiliary["centered_predicted_normalized_ade_risk"].to(torch.float64)
        probabilities = control_arms(
            source_probability, target_native, cross, predicted_risk, pairwise
        )
        metrics = _shared_support_metrics(target_support, probabilities, target_decision, truth)
        packed = pack_scenes(data["adj"])
        batch_dates = selected_dates[cursor : cursor + packed.scene_count]
        if len(batch_dates) != packed.scene_count:
            raise RuntimeError("scene/date alignment failed")
        actor_dates = np.asarray(batch_dates, dtype=object)[packed.inverse.detach().cpu().numpy()]
        _update_metric_grid(states, metrics, actor_dates)

        for name, probability in probabilities.items():
            sums = probability.sum(dim=1)
            probability_sum_max_abs_error[name] = max(
                probability_sum_max_abs_error[name],
                float((sums - 1.0).abs().max().detach().cpu()),
            )
            probability_minimum[name] = min(
                probability_minimum[name], float(probability.min().detach().cpu())
            )
            probability_maximum[name] = max(
                probability_maximum[name], float(probability.max().detach().cpu())
            )
        if len(audit_rows) < audit_sample_limit:
            support_hashes = batch_support_digest(target_support)
            probability_hashes = batch_probability_digest(probabilities)
            agent_ids = list(data.get("agent_id", []))
            start_indices = list(data.get("start_idx", []))
            remaining = audit_sample_limit - len(audit_rows)
            for local_index in range(min(remaining, target_support.shape[0])):
                audit_rows.append(
                    {
                        "date": str(actor_dates[local_index]),
                        "agent_id": int(agent_ids[local_index]) if agent_ids else None,
                        "start_idx": int(start_indices[local_index]) if start_indices else None,
                        "support_sha256": support_hashes[local_index],
                        "probability_sha256": {
                            name: hashes[local_index]
                            for name, hashes in probability_hashes.items()
                        },
                    }
                )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter() - tic
        cursor += packed.scene_count
        actors += int(target_support.shape[0])
        batches += 1

    if cursor != len(evaluation):
        raise RuntimeError("negative-control evaluator did not consume cohort exactly once")
    models = {arm: state.summary() for arm, state in states.items()}
    reference = models["target_native"]["overall"]
    geometry_invariance = {
        metric: max(
            abs(float(models[arm]["overall"][metric]) - float(reference[metric]))
            for arm in NEGATIVE_ARMS
        )
        for metric in ("top1_ade", "top1_fde", "minade", "minfde")
    }
    if any(value > 1e-12 for value in geometry_invariance.values()):
        raise RuntimeError("negative-control arms changed target support geometry")
    selected = models["selected_mabpt"]["overall"]
    comparisons = {}
    for arm in NEGATIVE_ARMS:
        if arm == "selected_mabpt":
            continue
        comparisons[f"selected_mabpt_vs_{arm}"] = {
            metric: float(models[arm]["overall"][metric] - selected[metric])
            for metric in ("energy_score", "nll", "brier", "ece", "effective_modes")
        }
    return {
        "format_version": 1,
        "experiment_id": "measure2act_fixed_t_negative_controls_v1",
        "airport": airport,
        "regime": regime,
        "seed": seed,
        "split": split,
        "scenes": len(evaluation),
        "actors": actors,
        "models": models,
        "comparisons": comparisons,
        "negative_control_arms": list(NEGATIVE_ARMS),
        "geometry_invariance_max_absolute_difference": geometry_invariance,
        "support_audit": {
            "target_support_reused_by_all_arms": True,
            "support_hash_algorithm": "sha256(float32 target_support row bytes)",
            "sample_rows": audit_rows,
            "sample_row_count": len(audit_rows),
        },
        "probability_audit": {
            "sum_max_abs_error": dict(probability_sum_max_abs_error),
            "minimum_probability": dict(probability_minimum),
            "maximum_probability": dict(probability_maximum),
        },
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
            "all_registered_arms_reported": set(models) == set(NEGATIVE_ARMS),
            "shared_support_geometry_exactly_invariant": True,
            "output_refuses_overwrite": True,
            "locked_test_used": split == "test",
            "partial_locked_test": False,
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
            "Local retrospective fixed-support negative controls; probability-only "
            "arms cannot establish trajectory geometry superiority."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airport", choices=AIRPORTS, required=True)
    parser.add_argument("--regime", choices=REGIMES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=("development", "test"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--authorize-locked-test", action="store_true")
    parser.add_argument("--audit-sample-limit", type=int, default=32)
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
        audit_sample_limit=args.audit_sample_limit,
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
                "selected_energy": result["models"]["selected_mabpt"]["overall"]["energy_score"],
                "target_native_energy": result["models"]["target_native"]["overall"]["energy_score"],
                "elapsed_seconds": result["runtime"]["total_elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
