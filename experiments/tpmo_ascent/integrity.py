"""Independent integrity checks for the C162 formal artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .protocol import atomic_json, load_protocol, sha256


def audit(summary_path: Path, *, output: Path | None = None) -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_boundaries()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = sha256(protocol.path)
    if summary.get("protocol_sha256") != expected:
        raise RuntimeError("C162 final summary protocol hash mismatch")
    checks = {
        "summary_protocol_matches": True,
        "formal_folds_only": summary.get("folds") == [1, 2],
        "development_unused": summary.get("development_used") is False,
        "locked_test_unused": summary.get("locked_test_used") is False,
        "all_fold_probability_forwards_target_free": all(
            payload["integrity"]["target_in_probability_forward"] is False
            for payload in summary["fold_results"].values()
        ),
        "all_fold_physical_violations_zero": all(
            int(payload["integrity"]["physical_violation_count"]) == 0
            for payload in summary["fold_results"].values()
        ),
        "all_fold_geometry_checks_present": all(
            int(payload["integrity"]["geometry_batch_identity_checks"]) > 0
            for payload in summary["fold_results"].values()
        ),
    }
    result = {
        "format_version": 1,
        "cycle": "C162_TRANSPORTED_PRIOR_MEASURE_OPTIMIZATION",
        "protocol_sha256": expected,
        "summary_path": str(summary_path),
        "summary_sha256": sha256(summary_path),
        "checks": checks,
        "passed": all(checks.values()),
        "gates_reproduced": summary.get("gates", {}),
    }
    if output is not None:
        atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = audit(args.summary, output=args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
