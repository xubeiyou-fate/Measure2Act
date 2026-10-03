"""Generate the C99 terminal report from immutable artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .protocol import load_protocol


def _percent(value: float) -> str:
    return f"{100.0 * value:+.2f}%"


def _decision_sentence(seed_gate: dict[str, object] | None) -> str:
    if seed_gate is None:
        return "Seed-42 training is incomplete; no performance decision is available."
    if seed_gate["screen_passed"]:
        return "The frozen seed-42 gate passed and preregistered five-seed replication is authorized."
    return "The frozen seed-42 gate failed. Replication and locked-test evaluation are prohibited."


def _structure_checks(seed_gate: dict[str, object]) -> dict[str, bool]:
    primary = seed_gate["primary"]
    assert isinstance(primary, dict)
    checks = {
        f"primary:{name}": bool(passed)
        for name, passed in primary["gates"].items()
    }
    checks.update(
        {
            f"causal:{name}": bool(passed)
            for name, passed in seed_gate["causal_increment_checks"].items()
        }
    )
    checks["dac:curriculum_complete"] = bool(seed_gate["dac_curriculum_complete"])
    return checks


def run(report: Path | None = None) -> str:
    protocol = load_protocol()
    root = protocol.repository_root
    artifacts = root / "artifacts/dive_ascent"
    preflight = json.loads((artifacts / "preflight.json").read_text(encoding="utf-8"))
    gradient = json.loads((artifacts / "gradient_audit.json").read_text(encoding="utf-8"))
    seed_gate = (
        json.loads((artifacts / "seed42_gate.json").read_text(encoding="utf-8"))
        if (artifacts / "seed42_gate.json").is_file()
        else None
    )
    integrity = (
        json.loads((artifacts / "integrity_review.json").read_text(encoding="utf-8"))
        if (artifacts / "integrity_review.json").is_file()
        else None
    )
    lines = [
        "# C99 DIVE-ASCENT Terminal Report",
        "",
        "## Decision",
        "",
    ]
    lines.append(_decision_sentence(seed_gate))
    lines.extend(
        [
            "",
            "## Data and implementation",
            "",
            f"- Full preflight: {'PASS' if preflight['passed'] else 'FAIL'}.",
            f"- Train/dev scenes: {preflight['loaded']['train_scenes']:,}/{preflight['loaded']['dev_scenes']:,}.",
            f"- A3/A4 parameter ratio: {preflight['a3_to_a4_parameter_ratio']:.6f}.",
            "- Residuals, learned gates/routers, tokens/codebooks, autoregression, and post-selection: disabled.",
            "- Locked test: sealed unless a later five-seed development gate explicitly authorizes one event.",
            "",
            "## Gradient audit",
            "",
            f"The 64-batch shared-context score/regression gradient-norm ratio was {gradient['overall']['shared_context']['median_classification_to_regression_ratio']:.2f}x, while the negative-cosine fraction was {100.0 * gradient['overall']['shared_context']['negative_cosine_fraction']:.2f}%. The frozen conflict mechanism gate passed {gradient['passing_date_blocks']}/3 date blocks and is therefore {'supported' if gradient['mechanism_supported'] else 'rejected'}.",
        ]
    )
    if seed_gate is not None:
        lines.extend(["", "## Seed-42 results", ""])
        lines.append("| Variant | minADE | minFDE | p95 FDE | Energy | tail minFDE | effective modes |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for variant, metrics in seed_gate["metrics"].items():
            lines.append(
                f"| {variant} | {metrics['minade']:.6f} | {metrics['minfde']:.6f} | {metrics['minfde_p95']:.6f} | {metrics['energy_score']:.6f} | {metrics['tail_minfde']:.6f} | {metrics['winner_distribution']['effective_modes']:.3f} |"
            )
        gains = seed_gate["primary"]["gains"]
        lines.extend(
            [
                "",
                "A5 versus A0: "
                + ", ".join(
                    f"{name} {_percent(value)}" for name, value in gains.items()
                )
                + ".",
                "",
                "Structural gates: "
                + ", ".join(
                    f"{name}={'PASS' if passed else 'FAIL'}"
                    for name, passed in _structure_checks(seed_gate).items()
                )
                + ".",
            ]
        )
    if integrity is not None:
        lines.extend(
            [
                "",
                "## Integrity review",
                "",
                f"Independent artifact review: {'PASS' if integrity['passed'] else 'FAIL'}.",
            ]
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "C12 development has been reused across prior cycles. Seed-42 results are screening evidence only. No paper claim may use the locked-test partition unless the frozen five-seed gate authorizes and records its single evaluation event.",
            "",
        ]
    )
    document = "\n".join(lines)
    report = report or root / "docs/dive_ascent_terminal_report_20260731.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(document, encoding="utf-8")
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    print(run(args.report))


if __name__ == "__main__":
    main()
