"""Evaluate C165 mass-aware transport using the frozen C162 metric harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import experiments.tpmo_ascent.evaluate as c162_evaluate

from .operator import mass_aware_transported_prior
from .protocol import atomic_json, load_protocol, sha256


ROOT = Path(__file__).resolve().parents[1]


def _tpmo_reference(fold: int) -> dict[str, object]:
    path = ROOT / "artifacts/tpmo_ascent" / f"fold{fold}_formal.json"
    if not path.exists():
        raise RuntimeError(f"C165 requires the frozen C162 formal control: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("smoke") or int(payload.get("fold", -1)) != fold:
        raise RuntimeError("C165 C162 control artifact is not a formal matching fold")
    return payload["arms"]["tpmo"]


def run(
    *,
    fold: int,
    device: torch.device,
    workers: int,
    batch_size: int,
    smoke: bool,
    max_validation_scenes: int | None,
) -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_boundaries()
    if fold not in [int(value) for value in protocol.payload["evaluation"]["folds"]]:
        raise RuntimeError("C165 fold is outside frozen evaluation folds")

    # The C162 evaluator is the immutable metric/data/checkpoint harness. Only
    # its exact finite-support assignment operator is replaced for this run.
    original_transport = c162_evaluate.transported_prior
    c162_evaluate.transported_prior = mass_aware_transported_prior
    try:
        result = c162_evaluate.run(
            fold=fold,
            device=device,
            workers=workers,
            batch_size=batch_size,
            smoke=smoke,
            max_validation_scenes=max_validation_scenes,
        )
    finally:
        c162_evaluate.transported_prior = original_transport

    candidate_arms = result["arms"]
    result["arms"] = {
        "baseline_native": candidate_arms["baseline_native"],
        "c161_native": candidate_arms["c161_native"],
        "mabpt_identity_prior": candidate_arms["identity_prior"],
        "mabpt_hard_transport": candidate_arms["hard_transport"],
        "mabpt_soft_transport": candidate_arms["soft_transport"],
        "mabpt": candidate_arms["tpmo"],
    }
    if smoke:
        result["arms"]["tpmo"] = result["arms"]["mabpt"]
    else:
        result["arms"]["tpmo"] = _tpmo_reference(fold)
    result["format_version"] = 1
    result["cycle"] = "C165_MASS_AWARE_BAYESIAN_PERMUTATION_TRANSPORT"
    result["protocol_sha256"] = sha256(protocol.path)
    result["base_protocol_sha256"] = str(protocol.payload["base_protocol_sha256"])
    result["seed"] = int(protocol.payload["evaluation"]["seed"])
    result["integrity"]["target_in_validation_probability_forward"] = result["integrity"].pop(
        "target_in_probability_forward"
    )
    result["integrity"]["mass_aware_assignment_cost"] = True
    result["integrity"]["c162_tpmo_control_frozen"] = not smoke
    result["integrity"]["temperature_used"] = False
    result["integrity"]["gate_or_residual_used"] = False
    result["inputs"]["c162_tpmo_control"] = str(
        ROOT / "artifacts/tpmo_ascent" / f"fold{fold}_formal.json"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=(1, 2))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-validation-scenes", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.smoke:
        args.max_validation_scenes = args.max_validation_scenes or 8
    result = run(
        fold=args.fold,
        device=torch.device(args.device),
        workers=args.workers,
        batch_size=args.batch_size,
        smoke=args.smoke,
        max_validation_scenes=args.max_validation_scenes,
    )
    if args.output is None:
        suffix = "smoke" if args.smoke else "formal"
        args.output = ROOT / "artifacts/mabpt_ascent" / f"fold{args.fold}_{suffix}.json"
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
