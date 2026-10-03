"""Select development-only scalar temperatures and freeze control test inputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from mabpt.evaluate_tartan_retrain import _selected_checkpoint_triplet
from mabpt.train_tartan_retrain import sha256

from .evaluate_probability_controls import PROTOCOL, ROOT, TEMPERATURES
from .train_awta_tartan import atomic_json


DEFAULT_DEVELOPMENT_ROOT = ROOT / "artifacts/journal_extension_20260814/probability_controls/development"
DEFAULT_TEST_ROOT = ROOT / "artifacts/journal_extension_20260814/probability_controls/test"
DEFAULT_OUTPUT = ROOT / "artifacts/journal_extension_20260814/probability_controls_receipt_v1.json"
PARENT_PROTOCOL = ROOT / "mabpt/tartan_retrain_protocol_v1.json"
EVALUATOR = Path(__file__).with_name("evaluate_probability_controls.py")
SEEDS = (42, 7, 123, 2024, 2026)
AIRPORTS = ("KAGC", "KBTP")


def _expected_path(root: Path, airport: str, seed: int) -> Path:
    return root / f"{airport}_seed{seed}_development_v1.json"


def _select_temperature(payloads: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    candidates = []
    for temperature, label in TEMPERATURES:
        arm = f"{prefix}_temp_{label}"
        values = [float(payload["models"][arm]["overall"]["nll"]) for payload in payloads]
        candidates.append({
            "temperature": temperature,
            "arm": arm,
            "equal_cell_mean_nll": sum(values) / len(values),
            "cell_nll": values,
        })
    selected = min(candidates, key=lambda item: (item["equal_cell_mean_nll"], abs(math.log(item["temperature"]))))
    return {"selected_temperature": selected["temperature"], "selected_arm": selected["arm"], "candidates": candidates}


def freeze(development_root: Path, test_root: Path, output: Path) -> dict[str, Any]:
    existing_test = sorted(test_root.glob("*.json")) if test_root.exists() else []
    if existing_test:
        raise RuntimeError("probability-control test outputs exist before freeze")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    parent = json.loads(PARENT_PROTOCOL.read_text(encoding="utf-8"))
    paths = [_expected_path(development_root, airport, seed) for airport in AIRPORTS for seed in SEEDS]
    payloads = []
    for path, (airport, seed) in zip(paths, ((airport, seed) for airport in AIRPORTS for seed in SEEDS), strict=True):
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("airport") != airport or int(payload.get("seed", -1)) != seed or payload.get("split") != "development":
            raise RuntimeError(f"development control identity mismatch: {path}")
        if payload.get("inputs", {}).get("protocol_sha256") != sha256(PROTOCOL):
            raise RuntimeError(f"development control protocol mismatch: {path}")
        payloads.append(payload)

    checkpoint_paths = []
    for airport in AIRPORTS:
        for seed in SEEDS:
            checkpoints = _selected_checkpoint_triplet(
                root=ROOT,
                protocol=parent,
                airport=airport,
                regime="target_only",
                seed=seed,
                formal=True,
            )
            checkpoint_paths.extend(ROOT / checkpoints[name]["path"] for name in ("ascent", "predicted_risk"))
    frozen_paths = [PROTOCOL, EVALUATOR, Path(__file__), PARENT_PROTOCOL, *paths, *checkpoint_paths]
    unique_paths = sorted(set(frozen_paths), key=lambda value: value.as_posix())
    result = {
        "format_version": 1,
        "experiment_id": "tartan_probability_controls_freeze_v1",
        "control_test_inference_completed_before_freeze": False,
        "selection_split": "development",
        "selection": {
            "target_native": _select_temperature(payloads, "target_native"),
            "gibbs_unweighted_energy_kl": _select_temperature(payloads, "selected_mabpt"),
            "criterion": protocol["selection"]["criterion"],
        },
        "development_cells": [{"airport": payload["airport"], "seed": payload["seed"], "path": path.relative_to(ROOT).as_posix()} for payload, path in zip(payloads, paths, strict=True)],
        "files": {
            path.relative_to(ROOT).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in unique_paths
        },
        "integrity": {
            "complete_development_grid": len(payloads) == len(AIRPORTS) * len(SEEDS),
            "test_outputs_present_before_freeze": False,
            "operator_reselected": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-root", type=Path, default=DEFAULT_DEVELOPMENT_ROOT)
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = freeze(args.development_root.resolve(), args.test_root.resolve(), args.output.resolve())
    print(json.dumps({"output": args.output.resolve().as_posix(), "selection": result["selection"]}, indent=2))


if __name__ == "__main__":
    main()
