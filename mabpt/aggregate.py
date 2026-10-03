"""Aggregate paired MABPT development ablations without selecting an arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .evaluate import ARMS, CORE_METRICS, PROTOCOL, ROOT


SCALAR_METRICS = (*CORE_METRICS, "ece_argmax", "effective_modes", "minfde_p95", "tail_minfde")
COMPARISONS = {
    "E2_ascent_native": "ascent_native",
    "E2_target_native": "target_native_logits",
    "E2_single_support_energy": "target_energy_single_support",
    "E3_identity": "identity_full_projection",
    "E3_ordinary_hungarian": "ordinary_hungarian_full_projection",
    "E3_mass_hungarian": "mass_hungarian_full_projection",
    "E4_row_softmax": "row_softmax_full_projection",
    "E4_sinkhorn": "sinkhorn_full_projection",
    "E4_unweighted_gibbs": "unweighted_gibbs_full_projection",
    "E5_u_only": "mabpt_u_only",
    "E5_risk_kl": "mabpt_risk_kl",
    "E5_diversity_kl": "mabpt_diversity_kl",
    "E5_uniform_energy_kl": "uniform_energy_kl",
    "E5_no_kl": "mabpt_no_kl",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"MABPT refuses to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _combine(summaries: list[dict[str, object]]) -> dict[str, object]:
    actors = sum(int(summary["actors"]) for summary in summaries)
    tail_samples = sum(int(summary["tail_samples"]) for summary in summaries)
    combined = {
        "actors": actors,
        **{
            metric: sum(float(summary[metric]) * int(summary["actors"]) for summary in summaries)
            / actors
            for metric in SCALAR_METRICS
            if metric not in {"minfde_p95", "tail_minfde"}
        },
        "tail_minfde": sum(
            float(summary["tail_minfde"]) * int(summary["tail_samples"])
            for summary in summaries
        )
        / max(tail_samples, 1),
        "tail_samples": tail_samples,
    }
    dates = {}
    for summary in summaries:
        for date, metrics in summary["date_metrics"].items():
            if date in dates:
                raise RuntimeError(f"date {date} appears in more than one fold")
            dates[date] = metrics
    combined["date_metrics"] = {date: dates[date] for date in sorted(dates)}
    return combined


def _paired_date_bootstrap(
    control: dict[str, object],
    candidate: dict[str, object],
    metric: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    dates = sorted(set(control) & set(candidate))
    if dates != sorted(control) or dates != sorted(candidate):
        raise RuntimeError("paired date sets differ")
    weights = np.asarray([control[date]["actors"] for date in dates], dtype=np.float64)
    if not np.array_equal(
        weights, np.asarray([candidate[date]["actors"] for date in dates], dtype=np.float64)
    ):
        raise RuntimeError("paired date actor counts differ")
    effects = np.asarray(
        [control[date][metric] - candidate[date][metric] for date in dates],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(dates), size=(replicates, len(dates)))
    sampled_weights = weights[draws]
    estimates = (effects[draws] * sampled_weights).sum(axis=1) / sampled_weights.sum(axis=1)
    observed = float((effects * weights).sum() / weights.sum())
    return {
        "absolute_gain_control_minus_mabpt": observed,
        "ci95": [float(value) for value in np.quantile(estimates, [0.025, 0.975])],
        "dates": len(dates),
        "replicates": replicates,
    }


def aggregate(paths: tuple[Path, Path]) -> dict[str, object]:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if [payload["fold"] for payload in payloads] != [1, 2]:
        raise RuntimeError("MABPT aggregation requires ordered folds 1 and 2")
    if any(payload["protocol_sha256"] != _sha256(PROTOCOL) for payload in payloads):
        raise RuntimeError("MABPT protocol hash mismatch")
    arms = {
        arm: _combine([payload["arms"][arm] for payload in payloads]) for arm in ARMS
    }
    comparisons = {}
    for comparison_index, (name, control_name) in enumerate(COMPARISONS.items()):
        comparisons[name] = {
            "control": control_name,
            "relative_gain_control_minus_mabpt": {
                metric: (arms[control_name][metric] - arms["mabpt"][metric])
                / arms[control_name][metric]
                for metric in ("energy_score", "nll", "brier")
            },
            "paired_date_bootstrap": {
                metric: _paired_date_bootstrap(
                    arms[control_name]["date_metrics"],
                    arms["mabpt"]["date_metrics"],
                    metric,
                    replicates=10000,
                    seed=16500 + 10 * comparison_index + metric_index,
                )
                for metric_index, metric in enumerate(("energy_score", "nll", "brier"))
            },
        }
    legacy = json.loads(
        (ROOT / "artifacts/experiments/mabpt_ascent/final_summary_corrected.json").read_text(
            encoding="utf-8"
        )
    )
    replay = {
        metric: arms["mabpt"][metric] - legacy["aggregate"]["mabpt"][metric]
        for metric in ("energy_score", "nll", "brier")
    }
    if abs(replay["energy_score"]) > 5e-8 or any(
        abs(replay[metric]) > 5e-10 for metric in ("nll", "brier")
    ):
        raise RuntimeError(f"standalone MABPT replay mismatch: {replay}")
    if not all(
        math.isfinite(float(arms[arm][metric]))
        for arm in ARMS
        for metric in ("energy_score", "nll", "brier")
    ):
        raise RuntimeError("non-finite MABPT ablation result")
    return {
        "format_version": 1,
        "model": "MABPT",
        "experiment_ids": ["E2", "E3", "E4", "E5"],
        "evidence_class": "legacy_development_only",
        "protocol_sha256": _sha256(PROTOCOL),
        "inputs": [
            {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)} for path in paths
        ],
        "arms": arms,
        "comparisons": comparisons,
        "replay_delta_vs_frozen_c165": replay,
        "selection_performed": False,
        "claim_boundary": "Retrospective development evidence; not confirmatory or external.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold1",
        type=Path,
        default=ROOT / "artifacts/mabpt/e2_e5_fold1_formal_v1.json",
    )
    parser.add_argument(
        "--fold2",
        type=Path,
        default=ROOT / "artifacts/mabpt/e2_e5_fold2_formal_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/mabpt/e2_e5_summary_v1.json",
    )
    args = parser.parse_args()
    result = aggregate((args.fold1.resolve(), args.fold2.resolve()))
    _atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "actors": result["arms"]["mabpt"]["actors"],
                "mabpt": {
                    metric: result["arms"]["mabpt"][metric]
                    for metric in ("energy_score", "nll", "brier")
                },
                "replay_delta": result["replay_delta_vs_frozen_c165"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
