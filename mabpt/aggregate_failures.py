"""Aggregate E13 target-blind strata while retaining every selected case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .aggregate import _atomic_json, _sha256
from .failure_analysis import METRICS, PROTOCOL, ROOT


def aggregate(paths: tuple[Path, Path]) -> dict[str, object]:
    folds = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if [fold["fold"] for fold in folds] != [1, 2]:
        raise RuntimeError("E13 requires ordered folds 1 and 2")
    protocol_sha = _sha256(PROTOCOL)
    if any(fold["protocol_sha256"] != protocol_sha for fold in folds):
        raise RuntimeError("E13 protocol hash mismatch")
    strata = {}
    for name in folds[0]["strata"]:
        actors = sum(fold["strata"][name]["ascent"]["actors"] for fold in folds)
        strata[name] = {
            model: {
                "actors": actors,
                **{
                    metric: sum(
                        fold["strata"][name][model][metric]
                        * fold["strata"][name][model]["actors"]
                        for fold in folds
                    )
                    / actors
                    for metric in METRICS
                },
            }
            for model in ("ascent", "mabpt")
        }
        strata[name]["relative_gain_mabpt_vs_ascent"] = {
            metric: (strata[name]["ascent"][metric] - strata[name]["mabpt"][metric])
            / strata[name]["ascent"][metric]
            for metric in METRICS
        }
    return {
        "format_version": 1,
        "model": "MABPT",
        "experiment_id": "E13",
        "evidence_class": "legacy_development_only",
        "protocol_sha256": protocol_sha,
        "inputs": [
            {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)} for path in paths
        ],
        "strata": strata,
        "fold_target_free_thresholds": {
            str(fold["fold"]): fold["target_free_thresholds"] for fold in folds
        },
        "target_blind_selected_cases": {
            str(fold["fold"]): fold["target_blind_selected_cases"] for fold in folds
        },
        "fold_diagnostics": {
            str(fold["fold"]): fold["diagnostics"] for fold in folds
        },
        "future_or_metric_used_for_case_selection": False,
        "selection_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold1",
        type=Path,
        default=ROOT / "artifacts/mabpt/e13_fold1_formal_v1.json",
    )
    parser.add_argument(
        "--fold2",
        type=Path,
        default=ROOT / "artifacts/mabpt/e13_fold2_formal_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/mabpt/e13_summary_v1.json",
    )
    args = parser.parse_args()
    result = aggregate((args.fold1.resolve(), args.fold2.resolve()))
    _atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "strata": {
                    name: value["relative_gain_mabpt_vs_ascent"]
                    for name, value in result["strata"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
