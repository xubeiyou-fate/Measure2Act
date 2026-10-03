"""Aggregate five-seed MABPT-ASCENT robustness and operating strata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .aggregate_partc_seeds import hierarchical_paired_bootstrap
from .evaluate import CORE_METRICS, _atomic_json, _sha256
from .partc_design import PROTOCOL, load_protocol


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/mabpt_partc_20260811"
SCALARS = (*CORE_METRICS, "ece_argmax", "effective_modes", "tail_minfde")


def _scalar_aggregate(
    payloads: dict[int, dict[str, object]],
    group: str,
    name: str,
    model: str,
) -> dict[str, object]:
    result = {}
    for metric in SCALARS:
        values = np.asarray(
            [payloads[seed][group][name][model][metric] for seed in sorted(payloads)],
            dtype=np.float64,
        )
        result[metric] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "values": values.tolist(),
        }
    return result


def _group_summary(
    payloads: dict[int, dict[str, object]],
    group: str,
    name: str,
    *,
    bootstrap_seed: int,
    replicates: int,
) -> dict[str, object]:
    control = {
        seed: payloads[seed][group][name]["original_ascent"]["date_metrics"]
        for seed in payloads
    }
    candidate = {
        seed: payloads[seed][group][name]["mabpt_ascent"]["date_metrics"]
        for seed in payloads
    }
    return {
        "aggregates": {
            model: _scalar_aggregate(payloads, group, name, model)
            for model in ("original_ascent", "mabpt_ascent")
        },
        "paired_hierarchical_bootstrap": {
            metric: hierarchical_paired_bootstrap(
                control,
                candidate,
                metric,
                replicates=replicates,
                random_seed=bootstrap_seed + index,
            )
            for index, metric in enumerate(("energy_score", "top1_fde"))
        },
    }


def aggregate(payloads: dict[int, dict[str, object]]) -> dict[str, object]:
    protocol = load_protocol()
    seeds = [int(value) for value in protocol["fixed_seeds"]]
    if sorted(payloads) != sorted(seeds):
        raise RuntimeError("all five frozen Part C seeds are required")
    protocol_hash = _sha256(PROTOCOL)
    for seed, payload in payloads.items():
        if payload.get("model") != "MABPT-ASCENT" or payload.get("seed") != seed:
            raise RuntimeError("robustness paper-model identity mismatch")
        if payload.get("partc_protocol_sha256") != protocol_hash:
            raise RuntimeError("Part C protocol hash mismatch")
        if payload.get("evidence_class") != "development_only":
            raise RuntimeError("unexpected robustness evidence class")
        if payload.get("integrity", {}).get("historical_locked_test_used") is not False:
            raise RuntimeError("historical locked test entered robustness evidence")
    condition_names = list(payloads[seeds[0]]["conditions"])
    stratum_names = list(payloads[seeds[0]]["strata"])
    replicates = int(protocol["analysis"]["bootstrap_replicates"])
    return {
        "format_version": 1,
        "model": "MABPT-ASCENT",
        "experiment_id": "five_seed_robustness",
        "evidence_class": "development_only",
        "partc_protocol_sha256": protocol_hash,
        "seeds": seeds,
        "conditions": {
            name: _group_summary(
                payloads,
                "conditions",
                name,
                bootstrap_seed=20261400 + 10 * index,
                replicates=replicates,
            )
            for index, name in enumerate(condition_names)
        },
        "strata": {
            name: _group_summary(
                payloads,
                "strata",
                name,
                bootstrap_seed=20261600 + 10 * index,
                replicates=replicates,
            )
            for index, name in enumerate(stratum_names)
        },
        "integrity": {
            "matched_five_seeds": True,
            "same_corruption_for_both_models": True,
            "date_nested_within_seed_bootstrap": True,
            "condition_specific_tuning": False,
            "historical_locked_test_used": False,
        },
        "claim_boundary": "Five-seed development robustness evidence; not fresh confirmation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_ROOT / "five_seed_robustness_summary_v1.json",
    )
    args = parser.parse_args()
    protocol = load_protocol()
    paths = {
        int(seed): ARTIFACT_ROOT / f"seed{seed}_robustness_formal_v1.json"
        for seed in protocol["fixed_seeds"]
    }
    payloads = {
        seed: json.loads(path.read_text(encoding="utf-8"))
        for seed, path in paths.items()
    }
    result = aggregate(payloads)
    result["inputs"] = [
        {"path": str(paths[seed].relative_to(ROOT)), "sha256": _sha256(paths[seed])}
        for seed in result["seeds"]
    ]
    _atomic_json(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "seeds": result["seeds"],
                "conditions": len(result["conditions"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
