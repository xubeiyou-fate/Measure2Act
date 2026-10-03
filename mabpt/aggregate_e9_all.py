"""Combine the three registered E9 cardinalities and operator benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .aggregate import _atomic_json, _sha256
from .evaluate import ROOT
from .evaluate_e9 import PROTOCOL_PATH, TOP_M


MODES = (3, 5, 7)
SCALING_PROTOCOL_PATH = ROOT / "mabpt/protocol.json"


def aggregate(summary_paths: dict[int, Path], scaling_path: Path) -> dict[str, object]:
    training_protocol_hash = _sha256(PROTOCOL_PATH)
    scaling_protocol_hash = _sha256(SCALING_PROTOCOL_PATH)
    scaling = json.loads(scaling_path.read_text(encoding="utf-8"))
    if scaling["protocol_sha256"] != scaling_protocol_hash:
        raise RuntimeError("E9 scaling protocol hash mismatch")
    cardinalities: dict[str, object] = {}
    inputs = [{
        "path": scaling_path.relative_to(ROOT).as_posix(),
        "sha256": _sha256(scaling_path),
    }]
    for modes in MODES:
        path = summary_paths[modes]
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["protocol_sha256"] != training_protocol_hash:
            raise RuntimeError(f"E9 K={modes} protocol hash mismatch")
        if int(payload["modes"]) != modes:
            raise RuntimeError(f"E9 K={modes} summary cardinality mismatch")
        expected_arms = {
            "ascent_native",
            "target_energy_native",
            "mabpt_exact",
            *(f"mabpt_top{top_m}" for top_m in TOP_M[modes]),
            "mabpt_sinkhorn",
        }
        if set(payload["arms"]) != expected_arms:
            raise RuntimeError(f"E9 K={modes} arm registry mismatch")
        exact_energy = float(payload["arms"]["mabpt_exact"]["energy_score"])
        approximations = {
            arm: {
                "energy_score": float(values["energy_score"]),
                "relative_energy_degradation_vs_exact": (
                    float(values["energy_score"]) - exact_energy
                ) / exact_energy,
            }
            for arm, values in payload["arms"].items()
            if arm.startswith("mabpt_") and arm != "mabpt_exact"
        }
        cardinalities[str(modes)] = {
            "actors": int(payload["arms"]["mabpt_exact"]["actors"]),
            "energy_score": {
                "ascent_native": float(
                    payload["arms"]["ascent_native"]["energy_score"]
                ),
                "target_energy_native": float(
                    payload["arms"]["target_energy_native"]["energy_score"]
                ),
                "mabpt_exact": exact_energy,
            },
            "relative_energy_gain_exact_vs_ascent": float(
                payload["comparisons"]["mabpt_exact"]
                ["relative_gain_ascent_minus_arm"]["energy_score"]
            ),
            "approximations": approximations,
            "data_diagnostics": payload["diagnostics"],
            "operator_benchmark": scaling["modes"][str(modes)],
        }
        inputs.append({
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(path),
        })
    return {
        "format_version": 1,
        "model": "MABPT",
        "model_version": "1.0.0",
        "experiment_id": "E9",
        "evidence_class": "retrospective_native_cardinality_plus_synthetic_operator_benchmark",
        "protocols": {
            "trained_cardinality_sha256": training_protocol_hash,
            "operator_scaling_sha256": scaling_protocol_hash,
        },
        "cardinalities": cardinalities,
        "inputs": inputs,
        "selection_performed": False,
        "limitations": [
            "Top-M enumerates K! assignments before truncation and is not a scalable Murty backend.",
            "Operator latency is an isolated synthetic benchmark; full evaluation runtime is retained in the per-fold receipts.",
            "The historically opened train-date folds do not constitute fresh confirmation.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/mabpt/e9_summary_v1.json",
    )
    parser.add_argument(
        "--scaling", type=Path,
        default=ROOT / "artifacts/mabpt/e9_scaling_v1.json",
    )
    args = parser.parse_args()
    summaries = {
        modes: ROOT / f"artifacts/mabpt/e9_K{modes}_summary_v1.json"
        for modes in MODES
    }
    result = aggregate(summaries, args.scaling.resolve())
    _atomic_json(args.output.resolve(), result)
    print(json.dumps({
        "output": str(args.output),
        "energy": {
            modes: values["energy_score"]
            for modes, values in result["cardinalities"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
