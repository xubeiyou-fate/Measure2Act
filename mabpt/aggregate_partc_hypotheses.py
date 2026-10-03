"""Combine retrospective H1-H4 development tests with Holm correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate import _atomic_json, _sha256
from .partc_design import PROTOCOL, load_protocol


ROOT = Path(__file__).resolve().parents[1]
PARTC = ROOT / "artifacts/mabpt_partc_20260811"
LEGACY = ROOT / "artifacts/mabpt"


def paired_date_bootstrap(
    control: dict[str, object],
    candidate: dict[str, object],
    metric: str,
    count_key: str,
    *,
    replicates: int,
    random_seed: int,
) -> dict[str, object]:
    dates = sorted(control)
    if dates != sorted(candidate):
        raise RuntimeError("paired date sets differ")
    counts = np.asarray(
        [int(control[date][count_key]) for date in dates], dtype=np.float64
    )
    candidate_counts = np.asarray(
        [int(candidate[date][count_key]) for date in dates], dtype=np.float64
    )
    if not np.array_equal(counts, candidate_counts):
        raise RuntimeError("paired date counts differ")
    effects = np.asarray(
        [float(control[date][metric]) - float(candidate[date][metric]) for date in dates],
        dtype=np.float64,
    )
    rng = np.random.default_rng(random_seed)
    draws = rng.integers(0, len(dates), size=(replicates, len(dates)))
    samples = (effects[draws] * counts[draws]).sum(1) / counts[draws].sum(1)
    return {
        "effect_definition": "original_ASCENT_minus_MABPT_ASCENT; positive favors MABPT-ASCENT",
        "absolute_gain": float((effects * counts).sum() / counts.sum()),
        "ci95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
        "raw_one_sided_p": (int((samples <= 0).sum()) + 1) / (replicates + 1),
        "dates": dates,
        "replicates": replicates,
    }


def holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.get)
    adjusted: dict[str, float] = {}
    previous = 0.0
    hypotheses = len(ordered)
    for rank, name in enumerate(ordered):
        value = min(1.0, (hypotheses - rank) * raw[name])
        previous = max(previous, value)
        adjusted[name] = previous
    return {name: adjusted[name] for name in raw}


def aggregate() -> dict[str, object]:
    load_protocol()
    seed_path = PARTC / "five_seed_development_summary_v1.json"
    seeds = json.loads(seed_path.read_text(encoding="utf-8"))
    tests = {
        "H1_energy": seeds["primary_development_tests"]["H1_energy"],
        "H2_top1_fde": seeds["primary_development_tests"][
            "H2_top1_fde_superiority_component"
        ],
        "H3_fixed_event_brier": seeds["primary_development_tests"][
            "H3_fixed_event_brier"
        ],
        "H4_conflict_brier": seeds["primary_development_tests"][
            "H4_conflict_brier"
        ],
    }
    raw = {name: float(result["raw_one_sided_p"]) for name, result in tests.items()}
    adjusted = holm_adjust(raw)
    for name, result in tests.items():
        result["holm_adjusted_p"] = adjusted[name]
        result["holm_reject_at_0_05"] = adjusted[name] <= 0.05
    return {
        "format_version": 1,
        "model": "MABPT-ASCENT",
        "evidence_class": "retrospective_development_only",
        "partc_protocol_sha256": _sha256(PROTOCOL),
        "tests": tests,
        "multiplicity": "Holm correction across the four superiority components H1-H4",
        "composite_hypothesis_status": {
            "H1": "evaluated_on_five_seed_development",
            "H2": seeds["primary_development_tests"][
                "H2_minfde_noninferiority_component"
            ],
            "H3": seeds["primary_development_tests"][
                "H3_fixed_event_nll_companion"
            ],
            "H4": seeds["primary_development_tests"][
                "H4_recall_at_fixed_fpr_companion"
            ],
        },
        "inputs": [
            {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
            for path in (seed_path,)
        ],
        "claim_boundary": (
            "Holm results are retrospective development diagnostics. They are not "
            "confirmatory p values, and no noninferiority margin was chosen after results."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=PARTC / "hypothesis_summary_v1.json"
    )
    args = parser.parse_args()
    result = aggregate()
    _atomic_json(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "tests": {
                    name: values["holm_adjusted_p"]
                    for name, values in result["tests"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
