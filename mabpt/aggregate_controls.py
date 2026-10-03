"""Aggregate the paired date-blocked E6-E8 capacity controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .aggregate import _atomic_json, _combine, _paired_date_bootstrap, _sha256
from .evaluate import CORE_METRICS, ROOT
from .evaluate_controls import ARMS, PROTOCOL


COMPARISONS = {
    "E6_union10": "e6_union10",
    "E6_compressed5": "e6_compressed5",
    "E6_hungarian_average5": "e6_hungarian_average5",
    "E7_widened_ascent": "e7_widened_ascent",
    "E7_shared_encoder_dual_decoder10": "e7_shared_encoder_dual_decoder10",
    "E8_equal_update_union10": "e8_equal_update_union10",
}


def aggregate(paths: tuple[Path, Path]) -> dict[str, object]:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if [int(payload["fold"]) for payload in payloads] != [1, 2]:
        raise RuntimeError("E6-E8 aggregation requires ordered folds 1 and 2")
    protocol_hash = _sha256(PROTOCOL)
    if any(payload["protocol_sha256"] != protocol_hash for payload in payloads):
        raise RuntimeError("E6-E8 protocol hash mismatch")
    arms = {
        arm: _combine([payload["arms"][arm] for payload in payloads])
        for arm in ARMS
    }
    comparisons = {}
    for comparison_index, (label, control) in enumerate(COMPARISONS.items()):
        comparisons[label] = {
            "control": control,
            "relative_gain_control_minus_mabpt": {
                metric: (arms[control][metric] - arms["mabpt"][metric])
                / arms[control][metric]
                for metric in (*CORE_METRICS, "ece_argmax")
            },
            "paired_date_bootstrap": {
                metric: _paired_date_bootstrap(
                    arms[control]["date_metrics"],
                    arms["mabpt"]["date_metrics"],
                    metric,
                    replicates=10000,
                    seed=16800 + 10 * comparison_index + metric_index,
                )
                for metric_index, metric in enumerate(
                    ("top1_ade", "top1_fde", "energy_score", "nll", "brier")
                )
            },
        }
    diagnostics = {
        key: sum(
            float(payload["diagnostics"][key])
            * int(payload["arms"]["mabpt"]["actors"])
            for payload in payloads
        ) / sum(int(payload["arms"]["mabpt"]["actors"]) for payload in payloads)
        for key in payloads[0]["diagnostics"]
        if key == "mean_compression_reconstruction_cost"
    }
    diagnostics.update({
        key: [int(payload["diagnostics"][key]) for payload in payloads]
        for key in (
            "mabpt_training_updates",
            "equal_compute_control_training_updates",
            "ordinary_ensemble_training_updates",
        )
    })
    return {
        "format_version": 1,
        "model": "MABPT",
        "model_version": "1.0.0",
        "experiment_ids": ["E6", "E7", "E8"],
        "evidence_class": "retrospective_train_date_blocked_controls",
        "protocol_sha256": protocol_hash,
        "inputs": [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha256(path)}
            for path in paths
        ],
        "arms": arms,
        "comparisons": comparisons,
        "diagnostics": diagnostics,
        "integrity": {
            "paired_folds": True,
            "date_cluster_bootstrap": True,
            "equal_update_count_exact_each_fold": all(
                int(payload["diagnostics"]["mabpt_training_updates"])
                == int(payload["diagnostics"]["equal_compute_control_training_updates"])
                for payload in payloads
            ),
            "arm_or_checkpoint_selection": False,
        },
        "claim_boundary": "Retrospective development controls; not fresh confirmation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold1", type=Path,
        default=ROOT / "artifacts/mabpt/e6_e8_fold1_formal_v1.json",
    )
    parser.add_argument(
        "--fold2", type=Path,
        default=ROOT / "artifacts/mabpt/e6_e8_fold2_formal_v1.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/mabpt/e6_e8_summary_v1.json",
    )
    args = parser.parse_args()
    result = aggregate((args.fold1.resolve(), args.fold2.resolve()))
    _atomic_json(args.output.resolve(), result)
    print(json.dumps({
        "output": str(args.output),
        "actors": result["arms"]["mabpt"]["actors"],
        "energy": {
            arm: result["arms"][arm]["energy_score"] for arm in ARMS
        },
    }, indent=2))


if __name__ == "__main__":
    main()
