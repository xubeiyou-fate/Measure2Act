"""Generate the frozen Part C experiment layout for MABPT-ASCENT."""

from __future__ import annotations

import argparse
from itertools import product
import hashlib
import json
from pathlib import Path
import random
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("partc_protocol.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def factorial_runs(*, seed: int) -> list[dict[str, object]]:
    """Return every fusion combination in a seeded randomized run order."""
    combinations = [
        {
            "correspondence": correspondence,
            "assignment_cost_mass": assignment_mass,
            "projection": projection,
        }
        for correspondence, assignment_mass, projection in product(
            ("hard_bijection", "exact_Gibbs"),
            ("uniform", "source_predicted_mass"),
            ("transported_prior", "Energy_KL"),
        )
    ]
    random.Random(seed).shuffle(combinations)
    return [
        {"run_order": index, "arm_id": f"F1-{index:02d}", **combination}
        for index, combination in enumerate(combinations, start=1)
    ]


def blocked_schedule(protocol: dict[str, Any]) -> list[dict[str, object]]:
    """Randomize experiment order within each matched-seed block."""
    families = [
        "target_support_training",
        "target_risk_training",
        "main_development_evaluation",
        "factorial_fusion",
        "physical_plausibility",
        "calibration",
        "robustness",
        "conflict_operations",
        "runtime_memory",
        "failure_analysis",
    ]
    schedule: list[dict[str, object]] = []
    schedule_seed = int(protocol["training"]["run_order_seed"])
    for block_index, seed in enumerate(protocol["fixed_seeds"], start=1):
        order = list(families)
        random.Random(schedule_seed + int(seed)).shuffle(order)
        schedule.extend(
            {
                "block": block_index,
                "seed": int(seed),
                "within_block_order": index,
                "experiment": experiment,
            }
            for index, experiment in enumerate(order, start=1)
        )
    return schedule


def build_design() -> dict[str, object]:
    protocol = load_protocol()
    return {
        "format_version": 1,
        "model": protocol["paper_model"]["name"],
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": sha256(PROTOCOL),
        "evidence_boundary": protocol["evidence_boundary"],
        "experimental_unit": protocol["data"]["independent_unit"],
        "factorial_runs": factorial_runs(
            seed=int(protocol["training"]["run_order_seed"])
        ),
        "blocked_schedule": blocked_schedule(protocol),
        "future_confirmatory_release_gates": [
            "all five source checkpoints verified",
            "all five target-support checkpoints complete",
            "all five target-risk checkpoints complete",
            "all baseline and ablation smoke tests pass",
            "all development result schemas validate",
            "noninferiority margins signed off before opening",
            "confirmatory cohort has never been used by any earlier route or model",
        ],
    }


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    design = build_design()
    if args.output is None:
        print(json.dumps(design, indent=2))
        return
    atomic_json(args.output.resolve(), design)
    print(json.dumps({"output": str(args.output), "runs": len(design["blocked_schedule"])}, indent=2))


if __name__ == "__main__":
    main()
