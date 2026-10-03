"""Repeat the representative C127 smoke run and prove deterministic outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .protocol import load_protocol, sha256


ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "P1_B6_dual_oracle_fold0_seed42_smoke"
SUMMARY = ROOT / "runs/metric_exact" / RUN_NAME / "training_summary.json"


def deterministic_view(summary: dict[str, object]) -> dict[str, object]:
    history = []
    for record in summary["history"]:
        history.append(
            {
                name: value
                for name, value in record.items()
                if name != "elapsed_seconds"
            }
        )
    return {
        "history": history,
        "validation_metrics": summary["validation_metrics"],
        "parameter_count": summary["parameter_count"],
        "train_scenes": summary["train_scenes"],
        "validation_scenes": summary["validation_scenes"],
    }


def run(output: Path, device: str) -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_boundaries()
    if not SUMMARY.is_file():
        raise RuntimeError("run the representative B6 smoke once before determinism check")
    reference = json.loads(SUMMARY.read_text(encoding="utf-8"))
    command = [
        sys.executable,
        "-m",
        "experiments.metric_exact.train",
        "--phase",
        "P1",
        "--variant",
        "B6_dual_oracle",
        "--fold",
        "0",
        "--seed",
        "42",
        "--device",
        device,
        "--epochs",
        "1",
        "--batch-size",
        "64",
        "--eval-batch-size",
        "128",
        "--num-workers",
        "2",
        "--max-train-scenes",
        "512",
        "--max-validation-scenes",
        "256",
    ]
    completed = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)
    repeated = json.loads(SUMMARY.read_text(encoding="utf-8"))
    reference_view = deterministic_view(reference)
    repeated_view = deterministic_view(repeated)
    checks = {
        "same_epoch_diagnostics": reference_view["history"]
        == repeated_view["history"],
        "same_complete_validation_metrics": reference_view["validation_metrics"]
        == repeated_view["validation_metrics"],
        "same_parameter_and_sample_contract": all(
            reference_view[name] == repeated_view[name]
            for name in ("parameter_count", "train_scenes", "validation_scenes")
        ),
        "locked_test_unused": reference["locked_test_used"] is False
        and repeated["locked_test_used"] is False,
    }
    result = {
        "format_version": 1,
        "cycle": protocol.payload["cycle"],
        "protocol_sha256": sha256(protocol.path),
        "run": RUN_NAME,
        "command": command,
        "checks": checks,
        "passed": all(checks.values()),
        "reference_overall": reference["validation_metrics"]["overall"],
        "repeated_overall": repeated["validation_metrics"]["overall"],
        "repeat_stdout": completed.stdout.strip().splitlines(),
        "locked_test_used": False,
    }
    if not result["passed"]:
        raise RuntimeError(f"C127 deterministic smoke failed: {checks}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/experiments/metric_exact/deterministic_smoke.json",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.device), indent=2))


if __name__ == "__main__":
    main()
