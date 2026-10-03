"""Summarize registered Tartan 10/25/100-percent development diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("tartan_retrain_protocol_v1.json")
RUN_ROOT = ROOT / "runs/partc_tartan_retrain_20260812"
V2_RUN_ROOT = ROOT / "runs/partc_tartan_data_efficiency_nested_v2"
AMENDMENT = Path(__file__).with_name("tartan_data_efficiency_amendment_v2.json")
FRACTIONS = (0.1, 0.25, 1.0)
AIRPORTS = ("KAGC", "KBTP")
REGIMES = ("target_only", "full_finetune")
STAGES = ("ascent", "predicted_risk")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _summary(airport: str, regime: str, fraction: float, stage: str) -> tuple[Path, dict]:
    root = V2_RUN_ROOT if fraction == 0.1 else RUN_ROOT
    path = (
        root
        / airport
        / regime
        / f"p{int(fraction * 100):03d}"
        / f"{stage}_seed42_formal"
        / "training_summary.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = (airport, regime, stage, 42, fraction)
    actual = (
        payload["airport"],
        payload["regime"],
        payload["stage"],
        int(payload["seed"]),
        float(payload["fraction"]),
    )
    if actual != expected or payload.get("integrity", {}).get("locked_test_used") is not False:
        raise RuntimeError(f"data-efficiency summary identity mismatch: {path}")
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    expected_dates = set(
        amendment["dates"][airport][f"p{int(fraction * 100):03d}"]
        if fraction < 1.0
        else payload["train_dates"]
    )
    if set(payload["train_dates"]) != expected_dates:
        raise RuntimeError(f"data-efficiency dates differ from amendment: {path}")
    if fraction == 0.1:
        receipt_path = path.with_name("amendment_receipt_v2.json")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("amendment_sha256") != sha256(AMENDMENT)
            or receipt.get("training_summary_sha256") != sha256(path)
            or receipt.get("locked_test_accessed") is not False
        ):
            raise RuntimeError(f"invalid nested-p10 amendment receipt: {receipt_path}")
    return path, payload


def aggregate() -> dict[str, object]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    result = {}
    inputs = []
    for airport in AIRPORTS:
        result[airport] = {}
        for regime in REGIMES:
            result[airport][regime] = {}
            date_sets = {}
            for fraction in FRACTIONS:
                key = f"p{int(fraction * 100):03d}"
                result[airport][regime][key] = {}
                for stage in STAGES:
                    path, payload = _summary(airport, regime, fraction, stage)
                    metrics = payload["development_metrics"]["overall"]
                    result[airport][regime][key][stage] = {
                        "train_scenes": payload["train_scenes"],
                        "train_date_count": len(payload["train_dates"]),
                        "minade": metrics["minade"],
                        "minfde": metrics["minfde"],
                        "energy_score": metrics["energy_score"],
                        "nll": metrics["nll"],
                        "brier": metrics["brier"],
                        "ece": metrics["ece"],
                        "tail_minfde": metrics["tail_minfde"],
                    }
                    inputs.append(
                        {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}
                    )
                    if stage == "predicted_risk":
                        date_sets[fraction] = set(payload["train_dates"])
            if not date_sets[0.1].issubset(date_sets[0.25]) or not date_sets[0.25].issubset(date_sets[1.0]):
                raise RuntimeError(f"data-efficiency date subsets are not nested: {airport}/{regime}")
            result[airport][regime]["nested_date_sets"] = True
    return {
        "format_version": 1,
        "experiment_id": "Tartan_target_domain_data_efficiency_seed42_development",
        "fractions": list(FRACTIONS),
        "seed": 42,
        "results": result,
        "inputs": inputs,
        "protocol": {"path": PROTOCOL.relative_to(ROOT).as_posix(), "sha256": sha256(PROTOCOL)},
        "amendment": {
            "path": AMENDMENT.relative_to(ROOT).as_posix(),
            "sha256": sha256(AMENDMENT),
        },
        "integrity": {
            "development_only": True,
            "locked_test_accessed": False,
            "calendar_date_nested_subsets": True,
            "fixed_final_epoch": True,
            "single_seed_exploratory": True,
            "superseded_non_nested_p10_excluded": True,
        },
        "claim_boundary": (
            "Exploratory seed-42 development evidence for target-data efficiency; "
            "not a five-seed confirmatory comparison and not a locked-test result."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = aggregate()
    atomic_json(args.output.resolve(), payload)
    print(json.dumps({"output": args.output.resolve().as_posix()}, indent=2))


if __name__ == "__main__":
    main()
