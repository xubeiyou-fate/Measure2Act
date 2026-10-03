"""Aggregate the preregistered five-seed C99 development comparison."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

from .gates import compare
from .protocol import load_protocol, sha256


METRIC_NAMES = ("minade", "minfde", "minfde_p95", "energy_score", "tail_minfde")


def _load(root: Path, variant: str, seed: int) -> dict[str, object]:
    path = root / f"{variant}_seed{seed}_formal" / "training_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("complete") is not True or payload.get("formal") is not True:
        raise RuntimeError(f"incomplete C99 summary: {path}")
    return payload


def _aggregate(metrics: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name in METRIC_NAMES:
        values = [float(item[name]) for item in metrics]
        result[name] = mean(values)
        result[f"{name}_std"] = stdev(values) if len(values) > 1 else 0.0
    fractions = [
        mean(float(item["winner_distribution"]["fractions"][mode]) for item in metrics)
        for mode in range(5)
    ]
    total = sum(fractions)
    fractions = [value / total for value in fractions]
    entropy = -sum(value * math.log(max(value, 1e-12)) for value in fractions)
    result["winner_distribution"] = {
        "fractions": fractions,
        "effective_modes": math.exp(entropy),
    }
    result["minimum_winner_fraction"] = min(fractions)
    return result


def run(output: Path | None = None) -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_development_sealed()
    seed_gate_path = protocol.repository_root / "artifacts/experiments/dive_ascent/seed42_gate.json"
    if not seed_gate_path.is_file():
        raise RuntimeError("C99 development summary requires seed42_gate.json")
    seed_gate = json.loads(seed_gate_path.read_text(encoding="utf-8"))
    if seed_gate.get("replication_authorized") is not True:
        raise RuntimeError("C99 seed-42 gate did not authorize replication")
    run_root = protocol.repository_root / str(protocol.payload["run_root"])
    seeds = [int(seed) for seed in protocol.payload["development"]["seeds"]]
    variants = list(protocol.payload["development"]["primary_variants"])
    per_seed = {}
    by_variant = {variant: [] for variant in variants}
    positive = 0
    for seed in seeds:
        per_seed[str(seed)] = {}
        for variant in variants:
            summary = _load(run_root, variant, seed)
            if summary["protocol_sha256"] != sha256(protocol.path):
                raise RuntimeError("C99 replication summary protocol mismatch")
            metrics = summary["development_metrics"]["overall"]
            per_seed[str(seed)][variant] = metrics
            by_variant[variant].append(metrics)
        if (
            float(per_seed[str(seed)][variants[1]]["minfde"])
            < float(per_seed[str(seed)][variants[0]]["minfde"])
        ):
            positive += 1
    aggregates = {variant: _aggregate(values) for variant, values in by_variant.items()}
    aggregate_gate = compare(aggregates[variants[0]], aggregates[variants[1]], protocol.payload["gates"])
    seed_consistency = positive >= int(
        protocol.payload["development"]["positive_minfde_seed_count_minimum"]
    )
    passed = bool(aggregate_gate["passed"] and seed_consistency)
    result = {
        "format_version": 1,
        "cycle": protocol.payload["cycle"],
        "protocol_sha256": sha256(protocol.path),
        "manifest_sha256": sha256(protocol.manifest_path),
        "seeds": seeds,
        "per_seed": per_seed,
        "aggregates": aggregates,
        "aggregate_gate": aggregate_gate,
        "positive_minfde_seed_count": positive,
        "seed_consistency_passed": seed_consistency,
        "passed": passed,
        "locked_test_authorized": passed,
        "locked_test_used": False,
    }
    output = output or protocol.repository_root / "artifacts/experiments/dive_ascent/development_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({
        "passed": result["passed"],
        "locked_test_authorized": result["locked_test_authorized"],
        "positive_minfde_seed_count": result["positive_minfde_seed_count"],
        "aggregate_gains": result["aggregate_gate"]["gains"],
    }, indent=2))


if __name__ == "__main__":
    main()
