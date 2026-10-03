"""Aggregate the two registered E10 folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .aggregate import _atomic_json, _combine, _sha256
from .robustness import PROTOCOL, ROOT


def aggregate(paths: tuple[Path, Path]) -> dict[str, object]:
    folds = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if [fold["fold"] for fold in folds] != [1, 2]:
        raise RuntimeError("E10 requires ordered folds 1 and 2")
    expected_protocol = _sha256(PROTOCOL)
    if any(fold["protocol_sha256"] != expected_protocol for fold in folds):
        raise RuntimeError("E10 protocol hash mismatch")
    conditions = {}
    for condition in folds[0]["conditions"]:
        conditions[condition] = {
            model: _combine(
                [fold["conditions"][condition][model] for fold in folds]
            )
            for model in ("ascent", "mabpt")
        }
        conditions[condition]["relative_gain_mabpt_vs_ascent"] = {
            metric: (
                conditions[condition]["ascent"][metric]
                - conditions[condition]["mabpt"][metric]
            )
            / conditions[condition]["ascent"][metric]
            for metric in ("top1_ade", "top1_fde", "energy_score", "nll", "brier")
        }
    clean = conditions["clean"]
    for condition, models in conditions.items():
        models["degradation_vs_clean"] = {
            model: {
                metric: (models[model][metric] - clean[model][metric])
                / clean[model][metric]
                for metric in ("top1_ade", "top1_fde", "energy_score", "nll", "brier")
            }
            for model in ("ascent", "mabpt")
        }
    common_strata = sorted(set(folds[0]["strata"]) & set(folds[1]["strata"]))
    strata = {}
    for stratum in common_strata:
        strata[stratum] = {
            model: _combine([fold["strata"][stratum][model] for fold in folds])
            for model in ("ascent", "mabpt")
        }
        strata[stratum]["relative_gain_mabpt_vs_ascent"] = {
            metric: (strata[stratum]["ascent"][metric] - strata[stratum]["mabpt"][metric])
            / strata[stratum]["ascent"][metric]
            for metric in ("top1_ade", "top1_fde", "energy_score", "nll", "brier")
        }
    return {
        "format_version": 1,
        "model": "MABPT",
        "experiment_id": "E10",
        "evidence_class": "legacy_development_only",
        "protocol_sha256": expected_protocol,
        "inputs": [
            {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)} for path in paths
        ],
        "conditions": conditions,
        "strata": strata,
        "fold_training_range_quartiles_km": {
            str(fold["fold"]): fold["training_only_range_quartiles_km"] for fold in folds
        },
        "selection_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold1",
        type=Path,
        default=ROOT / "artifacts/mabpt/e10_fold1_formal_v1.json",
    )
    parser.add_argument(
        "--fold2",
        type=Path,
        default=ROOT / "artifacts/mabpt/e10_fold2_formal_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/mabpt/e10_summary_v1.json",
    )
    args = parser.parse_args()
    result = aggregate((args.fold1.resolve(), args.fold2.resolve()))
    _atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "actors": result["conditions"]["clean"]["mabpt"]["actors"],
                "conditions": {
                    name: values["relative_gain_mabpt_vs_ascent"]
                    for name, values in result["conditions"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
