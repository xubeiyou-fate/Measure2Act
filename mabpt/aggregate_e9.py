"""Aggregate native-cardinality E9 results without selecting an approximation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .aggregate import _atomic_json, _combine, _paired_date_bootstrap, _sha256
from .evaluate import CORE_METRICS, ROOT
from .evaluate_e9 import PROTOCOL_PATH


def aggregate(paths: tuple[Path, Path]) -> dict[str, object]:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if [int(payload["fold"]) for payload in payloads] != [1, 2]:
        raise RuntimeError("E9 aggregation requires ordered folds 1 and 2")
    modes = int(payloads[0]["modes"])
    if modes not in (3, 5, 7) or any(int(payload["modes"]) != modes for payload in payloads):
        raise RuntimeError("E9 aggregation requires one registered cardinality")
    protocol_hash = _sha256(PROTOCOL_PATH)
    if any(payload["protocol_sha256"] != protocol_hash for payload in payloads):
        raise RuntimeError("E9 protocol hash mismatch")
    arm_names = tuple(payloads[0]["arms"])
    if any(tuple(payload["arms"]) != arm_names for payload in payloads):
        raise RuntimeError("E9 fold arm sets differ")
    arms = {
        arm: _combine([payload["arms"][arm] for payload in payloads])
        for arm in arm_names
    }
    comparisons = {}
    for index, arm in enumerate(name for name in arm_names if name.startswith("mabpt_")):
        comparisons[arm] = {
            "relative_gain_ascent_minus_arm": {
                metric: (arms["ascent_native"][metric] - arms[arm][metric])
                / arms["ascent_native"][metric]
                for metric in CORE_METRICS
            },
            "paired_date_bootstrap": {
                metric: _paired_date_bootstrap(
                    arms["ascent_native"]["date_metrics"],
                    arms[arm]["date_metrics"],
                    metric,
                    replicates=10000,
                    seed=16900 + 10 * index + metric_index,
                )
                for metric_index, metric in enumerate(
                    ("top1_ade", "top1_fde", "energy_score", "nll", "brier")
                )
            },
        }
    actors = sum(int(payload["arms"]["mabpt_exact"]["actors"]) for payload in payloads)
    diagnostics = {
        key: sum(
            float(payload["diagnostics"][key])
            * int(payload["arms"]["mabpt_exact"]["actors"])
            for payload in payloads
        ) / actors
        for key in payloads[0]["diagnostics"]
    }
    return {
        "format_version": 1,
        "model": "MABPT",
        "model_version": "1.0.0",
        "experiment_id": "E9",
        "evidence_class": "retrospective_train_date_native_cardinality",
        "modes": modes,
        "protocol_sha256": protocol_hash,
        "inputs": [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha256(path)}
            for path in paths
        ],
        "arms": arms,
        "comparisons": comparisons,
        "diagnostics": diagnostics,
        "selection_performed": False,
        "claim_boundary": "Retrospective native-cardinality study; top-M enumerates before truncation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", type=int, choices=(3, 5, 7), required=True)
    parser.add_argument("--fold1", type=Path)
    parser.add_argument("--fold2", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.fold1 is None:
        args.fold1 = ROOT / f"artifacts/mabpt/e9_K{args.modes}_fold1_formal_v1.json"
    if args.fold2 is None:
        args.fold2 = ROOT / f"artifacts/mabpt/e9_K{args.modes}_fold2_formal_v1.json"
    if args.output is None:
        args.output = ROOT / f"artifacts/mabpt/e9_K{args.modes}_summary_v1.json"
    result = aggregate((args.fold1.resolve(), args.fold2.resolve()))
    _atomic_json(args.output.resolve(), result)
    print(json.dumps({
        "output": str(args.output),
        "modes": args.modes,
        "actors": result["arms"]["mabpt_exact"]["actors"],
        "energy": {
            arm: values["energy_score"] for arm, values in result["arms"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
