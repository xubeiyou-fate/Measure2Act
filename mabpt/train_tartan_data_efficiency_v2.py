"""Run the frozen nested-p10 amendment through the unchanged Tartan trainer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import train_tartan_retrain as upstream


ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = Path(__file__).with_name("tartan_data_efficiency_amendment_v2.json")
RUN_ROOT = ROOT / "runs/partc_tartan_data_efficiency_nested_v2"
STATUS_ROOT = ROOT / "artifacts/partc_two_dataset_20260812/data_efficiency_nested_v2_status"


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
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def verify_amendment() -> dict[str, object]:
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    frozen = amendment["frozen_inputs"]
    checks = {
        ROOT / "mabpt/tartan_retrain_protocol_v1.json": frozen[
            "tartan_retrain_protocol_v1_sha256"
        ],
        ROOT
        / "artifacts/partc_two_dataset_20260812/tartan_retrain_frozen_receipt_v1.json": frozen[
            "tartan_retrain_frozen_receipt_v1_sha256"
        ],
        ROOT / "mabpt/train_tartan_retrain.py": frozen["original_train_runner_sha256"],
    }
    for path, expected in checks.items():
        if sha256(path) != expected:
            raise RuntimeError(f"nested-p10 frozen input mismatch: {path}")
    return amendment


def run(args: argparse.Namespace) -> dict[str, object]:
    amendment = verify_amendment()
    if args.fraction != 0.1 or args.seed != 42:
        raise ValueError("v2 amendment is registered only for p10 seed42")
    expected_dates = set(amendment["dates"][args.airport]["p010"])

    def frozen_dates(values: list[str], fraction: float) -> set[str]:
        if fraction != 0.1:
            raise RuntimeError("nested-p10 selector received an unregistered fraction")
        available = set(values)
        if not expected_dates.issubset(available):
            raise RuntimeError("amendment dates are absent from the registered training split")
        return expected_dates

    upstream.RUN_ROOT = RUN_ROOT
    upstream.STATUS_ROOT = STATUS_ROOT
    upstream._fraction_dates = frozen_dates
    result = upstream.train(args)
    if set(result["train_dates"]) != expected_dates:
        raise RuntimeError("completed v2 run used dates outside the amendment")
    checkpoint = ROOT / result["checkpoint"]
    summary_path = checkpoint.with_name("training_summary.json")
    receipt_path = checkpoint.with_name("amendment_receipt_v2.json")
    receipt = {
        "schema_version": 2,
        "amendment": AMENDMENT.relative_to(ROOT).as_posix(),
        "amendment_sha256": sha256(AMENDMENT),
        "upstream_runner": "mabpt/train_tartan_retrain.py",
        "upstream_runner_sha256": sha256(ROOT / "mabpt/train_tartan_retrain.py"),
        "airport": args.airport,
        "regime": args.regime,
        "stage": args.stage,
        "seed": args.seed,
        "fraction": args.fraction,
        "selected_train_dates": sorted(expected_dates),
        "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": sha256(checkpoint),
        "training_summary": summary_path.relative_to(ROOT).as_posix(),
        "training_summary_sha256": sha256(summary_path),
        "locked_test_accessed": False,
        "supersedes": (
            f"runs/partc_tartan_retrain_20260812/{args.airport}/{args.regime}/p010/"
            f"{args.stage}_seed42_formal"
        ),
    }
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        if existing != receipt:
            raise RuntimeError("existing v2 amendment receipt differs")
    else:
        atomic_json(receipt_path, receipt)
    return {**result, "amendment_receipt": receipt_path.relative_to(ROOT).as_posix()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airport", choices=upstream.AIRPORTS, required=True)
    parser.add_argument("--regime", choices=upstream.REGIMES, required=True)
    parser.add_argument("--stage", choices=upstream.STAGES, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-train-scenes", type=int)
    parser.add_argument("--max-dev-scenes", type=int)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    formal = upstream._protocol()["training"]["batch_sizes"][args.stage]
    args.batch_size = args.batch_size or int(formal[0])
    args.eval_batch_size = args.eval_batch_size or int(formal[1])
    result = run(args)
    print(
        json.dumps(
            {
                "checkpoint": result["checkpoint"],
                "amendment_receipt": result["amendment_receipt"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
