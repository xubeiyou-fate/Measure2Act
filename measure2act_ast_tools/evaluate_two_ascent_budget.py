"""Evaluate a total-budget two-ASCENT ensemble control.

This sidecar keeps the frozen ASCENT workspace read-only.  For each registered
airport/regime/seed/split cell, it compares the selected MABPT probability
forecast against a fixed two-ASCENT capacity control:

* primary ASCENT seed = the cell seed;
* buddy ASCENT seed = the next seed in the registered seed cycle;
* support = concatenated primary and buddy ASCENT K=5 supports, hence K=10;
* probability = half primary ASCENT softmax mass and half buddy ASCENT softmax
  mass, with no test-tuned weights.

The control intentionally has a similar neural-parameter budget to MABPT and
more candidate atoms than MABPT.  It is a capacity/generalization control, not a
fixed-support attribution arm.
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
TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from experiments.metric_exact.model import build_model as build_ascent_model  # noqa: E402
from experiments.edfa_ascent.relation import pack_scenes  # noqa: E402
from evaluate_ast_extended_suite import (  # noqa: E402
    ArmStore,
    effective_modes,
    endpoint_event_metrics,
    full_path_energy,
)
from mabpt.evaluate_tartan_probability_ablation import (  # noqa: E402
    PARENT_FREEZE,
    PARENT_PROTOCOL,
    _json_safe,
    _shared_support_metrics,
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
from mabpt.train_tartan_retrain import _limited_indices  # noqa: E402
from model.utils import seed_worker, seq_collate  # noqa: E402


SEEDS = (42, 7, 123, 2024, 2026)
BUDDY_SEED = {seed: SEEDS[(index + 1) % len(SEEDS)] for index, seed in enumerate(SEEDS)}
ARMS = (
    "selected_mabpt",
    "target_native",
    "ascent_native",
    "buddy_ascent_native",
    "two_ascent_union10",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def parameter_count(model: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def load_source_model(checkpoint: Path, *, device: torch.device, batch_size: int) -> torch.nn.Module:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = build_ascent_model("B0_signed_coupled", batch_size=batch_size).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    return model.eval()


def project_selected(
    source_support: torch.Tensor,
    source_probability: torch.Tensor,
    target_support: torch.Tensor,
    predicted_risk: torch.Tensor,
) -> torch.Tensor:
    cross = support_cost(source_support, target_support)
    prior = exact_gibbs_transport(
        normalized(source_probability),
        cross,
        mass_weighted=False,
    )["transported"]
    pairwise = pairwise_trajectory_distance(target_support) / DEFAULT_ADE_SCALE
    return energy_kl_projection(
        prior,
        predicted_risk.to(torch.float64),
        pairwise,
        backtracking_steps=32,
        tolerance=1e-12,
    )[0]


def merge_metrics(
    support: torch.Tensor,
    probability: torch.Tensor,
    decision: torch.Tensor,
    truth: torch.Tensor,
    *,
    arm: str,
) -> dict[str, torch.Tensor]:
    probabilities = {arm: normalized(probability)}
    shared = _shared_support_metrics(support, probabilities, decision, truth)[arm]
    path_energy = full_path_energy(support, probabilities, truth)[arm]
    events = endpoint_event_metrics(support, probabilities, truth)[arm]
    return {
        **shared,
        **events,
        "energy_score_full_path": path_energy,
        "effective_modes": effective_modes(probabilities[arm]),
    }


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
    parent_path = ROOT / PARENT_PROTOCOL.relative_to(ROOT)
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if airport not in AIRPORTS:
        raise ValueError("unregistered airport")
    if regime not in REGIMES:
        raise ValueError("unregistered regime")
    if seed not in SEEDS:
        raise ValueError("unregistered seed")

    buddy_seed = BUDDY_SEED[seed]
    parent_receipt = ROOT / PARENT_FREEZE.relative_to(ROOT)
    test_gate = _authorize_split(
        split=split,
        authorize_locked_test=authorize_locked_test,
        max_scenes=max_scenes,
        formal_gate=lambda: {
            "parent": _formal_test_gate(root=ROOT, protocol=parent, receipt_path=parent_receipt),
        },
    )
    parent_freeze = _verify_freeze_receipt(root=ROOT, receipt_path=parent_receipt)
    primary_checkpoints = _selected_checkpoint_triplet(
        root=ROOT, protocol=parent, airport=airport, regime=regime, seed=seed, formal=True
    )
    buddy_checkpoints = _selected_checkpoint_triplet(
        root=ROOT, protocol=parent, airport=airport, regime=regime, seed=buddy_seed, formal=True
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
        source_checkpoint=ROOT / primary_checkpoints["ascent"]["path"],
        target_checkpoint=ROOT / primary_checkpoints["predicted_risk"]["path"],
        device=device,
        batch_size=batch_size,
    )
    buddy_source = load_source_model(
        ROOT / buddy_checkpoints["ascent"]["path"],
        device=device,
        batch_size=batch_size,
    )
    counts = {
        "single_ascent": parameter_count(source),
        "target_predicted_risk_branch": parameter_count(target),
        "two_ascent_union10": {
            "total": parameter_count(source)["total"] + parameter_count(buddy_source)["total"],
            "trainable": parameter_count(source)["trainable"] + parameter_count(buddy_source)["trainable"],
        },
        "mabpt_loaded_neural": {
            "total": parameter_count(source)["total"] + parameter_count(target)["total"],
            "trainable": parameter_count(source)["trainable"] + parameter_count(target)["trainable"],
        },
    }

    stores = {arm: ArmStore() for arm in ARMS}
    probability_sum_error = defaultdict(float)
    probability_minimum = defaultdict(lambda: float("inf"))
    probability_maximum = defaultdict(float)
    support_modes = {}
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
        buddy_support, buddy_logits, _ = buddy_source(data)
        buddy_probability = buddy_logits.softmax(dim=1)
        target_support, _, target_decision, auxiliary = target(data)
        target_native = auxiliary["decision_logits"].softmax(dim=1)
        predicted_risk = auxiliary["centered_predicted_normalized_ade_risk"].to(torch.float64)
        selected_mabpt = project_selected(
            source_support,
            source_probability,
            target_support,
            predicted_risk,
        )

        union_support = torch.cat((source_support, buddy_support), dim=1)
        union_probability = normalized(
            torch.cat((0.5 * source_probability, 0.5 * buddy_probability), dim=1)
        )
        probabilities = {
            "selected_mabpt": selected_mabpt,
            "target_native": target_native,
            "ascent_native": source_probability,
            "buddy_ascent_native": buddy_probability,
            "two_ascent_union10": union_probability,
        }
        supports = {
            "selected_mabpt": target_support,
            "target_native": target_support,
            "ascent_native": source_support,
            "buddy_ascent_native": buddy_support,
            "two_ascent_union10": union_support,
        }
        decisions = {
            "selected_mabpt": target_decision,
            "target_native": target_decision,
            "ascent_native": source_logits.argmax(dim=1),
            "buddy_ascent_native": buddy_logits.argmax(dim=1),
            "two_ascent_union10": union_probability.argmax(dim=1),
        }
        packed = pack_scenes(data["adj"])
        batch_dates = selected_dates[cursor : cursor + packed.scene_count]
        if len(batch_dates) != packed.scene_count:
            raise RuntimeError("scene/date alignment failed")
        actor_dates = np.asarray(batch_dates, dtype=object)[packed.inverse.detach().cpu().numpy()]

        for arm in ARMS:
            metrics = merge_metrics(
                supports[arm],
                probabilities[arm],
                decisions[arm],
                truth,
                arm=arm,
            )
            stores[arm].update(metrics, actor_dates)
            support_modes[arm] = int(supports[arm].shape[1])
            sums = normalized(probabilities[arm]).sum(dim=1)
            probability_sum_error[arm] = max(
                probability_sum_error[arm],
                float((sums - 1.0).abs().max().detach().cpu()),
            )
            probability_minimum[arm] = min(
                probability_minimum[arm],
                float(normalized(probabilities[arm]).min().detach().cpu()),
            )
            probability_maximum[arm] = max(
                probability_maximum[arm],
                float(normalized(probabilities[arm]).max().detach().cpu()),
            )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter() - tic
        cursor += packed.scene_count
        actors += int(target_support.shape[0])
        batches += 1

    if cursor != len(evaluation):
        raise RuntimeError("two-ASCENT budget evaluator did not consume cohort exactly once")

    models = {arm: stores[arm].summary() for arm in ARMS}
    return {
        "format_version": 1,
        "experiment_id": "measure2act_total_budget_two_ascent_v1",
        "airport": airport,
        "regime": regime,
        "seed": seed,
        "buddy_seed": buddy_seed,
        "split": split,
        "scenes": len(evaluation),
        "actors": actors,
        "arms": list(ARMS),
        "support_modes": support_modes,
        "models": models,
        "probability_sum_max_abs_error": dict(probability_sum_error),
        "probability_minimum": dict(probability_minimum),
        "probability_maximum": dict(probability_maximum),
        "parameter_counts": counts,
        "inputs": {
            "root": ROOT.as_posix(),
            "script": Path(__file__).resolve().as_posix(),
            "script_sha256": sha256(Path(__file__).resolve()),
            "parent_protocol": {
                "path": PARENT_PROTOCOL.relative_to(ROOT).as_posix(),
                "sha256": sha256(parent_path),
            },
            "parent_freeze_receipt": parent_freeze,
            "formal_test_gate": test_gate,
            "primary_checkpoints": primary_checkpoints,
            "buddy_checkpoints": buddy_checkpoints,
            "scene_date_index": index_path.relative_to(ROOT).as_posix(),
            "scene_date_index_sha256": sha256(index_path),
        },
        "integrity": {
            "buddy_seed_rule": "next seed in fixed registered cycle 42->7->123->2024->2026->42",
            "test_used_for_pairing_or_weight_selection": False,
            "two_ascent_mixture_weights": [0.5, 0.5],
            "capacity_control_not_fixed_support_attribution": True,
            "locked_test_used": split == "test",
            "output_refuses_overwrite": True,
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
            "Retrospective total-budget capacity control. The two-ASCENT arm has "
            "K=10 support and fixed 0.5/0.5 model mass, so it is not a fixed-support "
            "probability-attribution experiment."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airport", choices=AIRPORTS, required=True)
    parser.add_argument("--regime", choices=REGIMES, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
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
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "actors": result["actors"],
                "two_ascent_energy": result["models"]["two_ascent_union10"]["overall"][
                    "energy_score"
                ],
                "mabpt_energy": result["models"]["selected_mabpt"]["overall"][
                    "energy_score"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
