"""Run the four new EqMotion seeds under the frozen Tartan multi-seed protocol."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from . import run_eqmotion_tartan_target as base


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("eqmotion_tartan_multiseed_protocol_v2.json")
NEW_SEEDS = (7, 123, 2024, 2026)
FORMAL_RUN_ROOT = ROOT / "runs/partc_two_dataset_20260812/eqmotion_tartan_multiseed_v2"
FORMAL_RESULT_ROOT = (
    ROOT
    / "artifacts/partc_two_dataset_20260812/modern_baseline/eqmotion_tartan_multiseed_v2"
)
SMOKE_RUN_ROOT = ROOT / "runs/partc_two_dataset_20260812/eqmotion_tartan_multiseed_v2_smoke"
SMOKE_RESULT_ROOT = (
    ROOT
    / "artifacts/partc_two_dataset_20260812/modern_baseline/eqmotion_tartan_multiseed_v2_smoke"
)


def load_multiseed_protocol() -> dict[str, Any]:
    protocol = base.load_protocol(PROTOCOL)
    if tuple(protocol["training"]["registered_seeds"]) != (42, *NEW_SEEDS):
        raise RuntimeError("multi-seed registry differs from the frozen five-seed order")
    if tuple(protocol["training"]["new_training_seeds"]) != NEW_SEEDS:
        raise RuntimeError("new EqMotion seed registry mismatch")
    inherited = json.loads(
        (ROOT / protocol["inherits"]["base_protocol"]).read_text(encoding="utf-8")
    )
    for section in ("official_source", "adapter", "grid", "model"):
        if protocol[section] != inherited[section]:
            raise RuntimeError(f"multi-seed protocol changed inherited {section}")
    for key in (
        "root",
        "manifest",
        "manifest_sha256",
        "frozen_receipt",
        "frozen_receipt_sha256",
        "airports",
        "allowed_splits",
        "forbidden_splits",
        "split_unit",
        "split_rule",
        "delimiter",
    ):
        if protocol["data"].get(key) != inherited["data"].get(key):
            raise RuntimeError(f"multi-seed protocol changed inherited data.{key}")
    for key in (
        "regime",
        "epochs",
        "batch_size",
        "evaluation_batch_size",
        "optimizer",
        "learning_rate",
        "milestones",
        "gamma",
        "loss",
        "gradient_clip_norm",
        "checkpoint_policy",
        "development_evaluation",
        "deterministic_algorithms",
        "tf32",
        "automatic_mixed_precision",
    ):
        if protocol["training"].get(key) != inherited["training"].get(key):
            raise RuntimeError(f"multi-seed protocol changed inherited training.{key}")
    return protocol


def verify_multiseed_sources(protocol: dict[str, Any]) -> None:
    checks = {
        ROOT / protocol["inherits"]["base_protocol"]: protocol["inherits"][
            "base_protocol_sha256"
        ],
        ROOT / protocol["inherits"]["skip5_amendment"]: protocol["inherits"][
            "skip5_amendment_sha256"
        ],
        ROOT / protocol["inherits"]["skip5_receipt"]: protocol["inherits"][
            "skip5_receipt_sha256"
        ],
        ROOT / protocol["data"]["manifest"]: protocol["data"]["manifest_sha256"],
        ROOT / protocol["data"]["frozen_receipt"]: protocol["data"][
            "frozen_receipt_sha256"
        ],
        ROOT / protocol["data"]["scene_index_summary"]: protocol["data"][
            "scene_index_summary_sha256"
        ],
        ROOT / protocol["adapter"]["path"]: protocol["adapter"]["sha256"],
        ROOT / protocol["official_source"]["model_file"]: protocol["official_source"][
            "model_file_sha256"
        ],
    }
    for airport, inherited in protocol["seed42_inheritance"].items():
        checks[ROOT / inherited["result"]] = inherited["result_sha256"]
        checks[ROOT / inherited["checkpoint"]] = inherited["checkpoint_sha256"]
        checks[ROOT / inherited["locked_result"]] = inherited["locked_result_sha256"]
        result = json.loads((ROOT / inherited["result"]).read_text(encoding="utf-8"))
        if (
            result.get("formal") is not True
            or result.get("airport") != airport
            or result.get("regime") != "target_only"
            or int(result.get("seed", -1)) != 42
            or result.get("integrity", {}).get("locked_test_model_inference") is not False
        ):
            raise RuntimeError(f"inherited seed42 result identity mismatch: {airport}")
        locked = json.loads((ROOT / inherited["locked_result"]).read_text(encoding="utf-8"))
        if (
            locked.get("airport") != airport
            or locked.get("regime") != "target_only"
            or int(locked.get("seed", -1)) != 42
            or locked.get("evidence_class") != "locked_retrospective_test_single_pass"
            or locked.get("integrity", {}).get("partial_test") is not False
        ):
            raise RuntimeError(f"inherited seed42 locked result identity mismatch: {airport}")
    for path, expected in checks.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if base.sha256(path) != expected:
            raise RuntimeError(f"frozen multi-seed source hash mismatch: {path}")


def expected_formal_paths(airport: str, seed: int) -> tuple[Path, Path]:
    return (
        FORMAL_RUN_ROOT / airport / f"seed{seed}_formal",
        FORMAL_RESULT_ROOT / f"{airport}_target_only_seed{seed}_formal.json",
    )


def validate_paths(args) -> None:
    run_dir = args.run_dir.resolve()
    output = args.output.resolve()
    if args.smoke:
        if not run_dir.is_relative_to(SMOKE_RUN_ROOT.resolve()):
            raise ValueError(f"smoke run must be below {SMOKE_RUN_ROOT}")
        if not output.is_relative_to(SMOKE_RESULT_ROOT.resolve()):
            raise ValueError(f"smoke output must be below {SMOKE_RESULT_ROOT}")
        return
    expected_run, expected_output = expected_formal_paths(args.airport, args.seed)
    if run_dir != expected_run.resolve() or output != expected_output.resolve():
        raise ValueError(
            "formal multi-seed paths are frozen: "
            f"run_dir={expected_run.relative_to(ROOT).as_posix()} "
            f"output={expected_output.relative_to(ROOT).as_posix()}"
        )


def main(argv: list[str] | None = None) -> None:
    forwarded = list(sys.argv[1:] if argv is None else argv)
    args = base.build_parser().parse_args(forwarded)
    protocol = load_multiseed_protocol()
    if args.seed == 42:
        raise ValueError("seed42 is inherited and must not be rerun")
    if args.seed not in NEW_SEEDS:
        raise ValueError("seed is outside the four frozen expansion seeds")
    validate_paths(args)

    original_protocol = base.PROTOCOL
    original_verify = base.verify_formal_sources
    original_atomic = base.atomic_json

    def write_result(path: Path, payload: dict[str, Any]) -> None:
        payload["command"] = [
            sys.executable,
            "-m",
            "modern_baseline.run_eqmotion_tartan_multiseed_v2",
            *forwarded,
        ]
        payload["multi_seed_expansion"] = {
            "protocol": PROTOCOL.relative_to(ROOT).as_posix(),
            "protocol_sha256": base.sha256(PROTOCOL),
            "registered_seeds": protocol["training"]["registered_seeds"],
            "seed42_reused_without_rerun": True,
            "new_training_seed": args.seed,
            "new_output_namespace": True,
        }
        original_atomic(path, payload)

    try:
        base.PROTOCOL = PROTOCOL
        base.verify_formal_sources = verify_multiseed_sources
        base.atomic_json = write_result
        base.main(forwarded)
    finally:
        base.PROTOCOL = original_protocol
        base.verify_formal_sources = original_verify
        base.atomic_json = original_atomic


if __name__ == "__main__":
    main()
