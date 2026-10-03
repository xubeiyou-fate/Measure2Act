"""Independent integrity review of C99 artifacts and authorization boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .model import C99_VARIANTS
from .protocol import load_protocol, sha256


def run(output: Path | None = None) -> dict[str, object]:
    protocol = load_protocol()
    root = protocol.repository_root
    expected_protocol = sha256(protocol.path)
    expected_manifest = sha256(protocol.manifest_path)
    preflight_path = root / "artifacts/experiments/dive_ascent/preflight.json"
    gradient_path = root / "artifacts/experiments/dive_ascent/gradient_audit.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    gradient = json.loads(gradient_path.read_text(encoding="utf-8"))
    formal = {}
    for variant in C99_VARIANTS:
        run_dir = root / str(protocol.payload["run_root"]) / f"{variant}_seed42_formal"
        summary_path = run_dir / "training_summary.json"
        checkpoint_path = run_dir / "last.pt"
        entry = {
            "summary_exists": summary_path.is_file(),
            "checkpoint_exists": checkpoint_path.is_file(),
        }
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            entry.update(
                complete=summary.get("complete") is True,
                formal=summary.get("formal") is True,
                fixed_epoch=summary.get("fixed_final_epoch"),
                protocol_match=summary.get("protocol_sha256") == expected_protocol,
                manifest_match=summary.get("manifest_sha256") == expected_manifest,
                locked_test_unused=summary.get("locked_test_used") is False,
            )
        if checkpoint_path.is_file():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            entry["checkpoint_epoch"] = checkpoint.get("epoch")
            entry["checkpoint_protocol_match"] = checkpoint.get("protocol_sha256") == expected_protocol
            entry["checkpoint_manifest_match"] = checkpoint.get("manifest_sha256") == expected_manifest
            entry["forbidden_flags_false"] = all(
                checkpoint.get(name) is False
                for name in (
                    "trajectory_residual",
                    "learned_gate",
                    "token_codebook",
                    "future_autoregression",
                    "post_generation_selector",
                )
            )
            if variant == "a5_dive":
                entry["split_history"] = checkpoint.get("split_history", [])
        formal[variant] = entry
    seed_gate_path = root / "artifacts/experiments/dive_ascent/seed42_gate.json"
    seed_gate = (
        json.loads(seed_gate_path.read_text(encoding="utf-8"))
        if seed_gate_path.is_file()
        else None
    )
    development_gate_path = root / "artifacts/experiments/dive_ascent/development_gate.json"
    development_gate = (
        json.loads(development_gate_path.read_text(encoding="utf-8"))
        if development_gate_path.is_file()
        else None
    )
    receipt_path = root / str(protocol.payload["locked_test_policy"]["receipt"])
    all_formal_complete = all(
        entry.get("complete") is True
        and entry.get("formal") is True
        and entry.get("fixed_epoch") == int(protocol.payload["training"]["epochs"])
        and entry.get("protocol_match") is True
        and entry.get("manifest_match") is True
        and entry.get("locked_test_unused") is True
        and entry.get("checkpoint_exists") is True
        and entry.get("checkpoint_epoch") == int(protocol.payload["training"]["epochs"])
        and entry.get("checkpoint_protocol_match") is True
        and entry.get("checkpoint_manifest_match") is True
        and entry.get("forbidden_flags_false") is True
        for entry in formal.values()
    )
    unauthorized_replication = False
    if seed_gate is not None and seed_gate.get("replication_authorized") is not True:
        for seed in set(protocol.payload["development"]["seeds"]) - {42}:
            for variant in protocol.payload["development"]["primary_variants"]:
                if (root / str(protocol.payload["run_root"]) / f"{variant}_seed{seed}_formal").exists():
                    unauthorized_replication = True
    unauthorized_locked = receipt_path.exists() and (
        development_gate is None or development_gate.get("locked_test_authorized") is not True
    )
    checks = {
        "preflight_protocol_match": preflight["protocol_sha256"] == expected_protocol,
        "preflight_manifest_match": preflight["manifest_sha256"] == expected_manifest,
        "preflight_passed": preflight["passed"] is True,
        "gradient_audit_protocol_match": gradient["protocol_sha256"] == expected_protocol,
        "gradient_audit_complete": gradient["processed_batches"]
        == int(protocol.payload["gradient_audit"]["batches"]),
        "gradient_mechanism_not_relabelled": gradient["mechanism_supported"] is False,
        "formal_screen_complete": all_formal_complete,
        "no_unauthorized_replication": not unauthorized_replication,
        "no_unauthorized_locked_test": not unauthorized_locked,
    }
    result = {
        "format_version": 1,
        "cycle": protocol.payload["cycle"],
        "protocol_sha256": expected_protocol,
        "manifest_sha256": expected_manifest,
        "formal": formal,
        "seed42_gate": seed_gate,
        "development_gate": development_gate,
        "locked_test_receipt_exists": receipt_path.exists(),
        "checks": checks,
        "passed": all(checks.values()),
    }
    output = output or root / "artifacts/experiments/dive_ascent/integrity_review.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"passed": result["passed"], "checks": result["checks"]}, indent=2))


if __name__ == "__main__":
    main()
