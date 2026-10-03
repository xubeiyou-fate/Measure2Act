"""Summarize existing capacity, architecture, and generalization evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("ASCENT_FULL_ROOT", DEFAULT_ROOT)).resolve()


def load(rel: str) -> dict[str, Any]:
    path = ROOT / rel
    return json.loads(path.read_text(encoding="utf-8"))


def metric_pack(block: dict[str, Any], metrics=("energy_score", "minade", "minfde")) -> dict[str, Any]:
    return {
        metric: {
            "absolute_effect": block["metrics"][metric]["absolute_effect"],
            "ci95": block["metrics"][metric].get("ci95"),
            "holm_adjusted_p": block["metrics"][metric].get("holm_adjusted_p"),
            "supported": block["metrics"][metric].get("registered_superiority_supported"),
        }
        for metric in metrics
        if metric in block.get("metrics", {})
    }


def e17_capacity() -> dict[str, Any]:
    summary = load("artifacts/journal_e14_e20_20260817/e17_e18/e17_e18_summary_v1.json")
    e17 = summary["e17"]
    return {
        "source": "artifacts/journal_e14_e20_20260817/e17_e18/e17_e18_summary_v1.json",
        "controls": {
            name: {
                "all_primary_supported": block.get("all_registered_primary_metrics_supported"),
                "metrics": metric_pack(block),
            }
            for name, block in e17.items()
            if name in (
                "original_ascent",
                "widened_ascent",
                "shared_encoder_dual_decoder10",
                "equal_update_union10",
                "awta",
            )
        },
        "claim_boundary": summary.get("claim_boundaries"),
    }


def e18_component() -> dict[str, Any]:
    summary = load("artifacts/journal_e14_e20_20260817/e17_e18/e17_e18_summary_v1.json")
    wanted = (
        "geometry_plus_decision_C133_to_plus_risk_C134",
        "plus_risk_C134_to_plus_unweighted_gibbs",
        "plus_unweighted_gibbs_to_plus_energy_kl_final",
    )
    result = {}
    for name in wanted:
        block = summary["e18"].get(name)
        if not block:
            continue
        result[name] = {
            metric: {
                "absolute_effect": block[metric]["absolute_effect"],
                "ci95": block[metric].get("ci95"),
                "holm_adjusted_p": block[metric].get("holm_adjusted_p"),
            }
            for metric in ("energy_score", "nll", "brier", "ece", "minade", "minfde")
            if metric in block
        }
    return {
        "source": "artifacts/journal_e14_e20_20260817/e17_e18/e17_e18_summary_v1.json",
        "stage_effects": result,
    }


def e15_independent_gru() -> dict[str, Any]:
    summary = load("artifacts/journal_e14_e20_20260817/e15/e15_summary_v1.json")
    return {
        "source": "artifacts/journal_e14_e20_20260817/e15/e15_summary_v1.json",
        "model": summary.get("model"),
        "identity": summary.get("identity"),
        "paired_vs_mabpt": summary.get("paired_vs_mabpt"),
        "claim_boundary": summary.get("claim_boundary"),
    }


def summarize_cells(summary: dict[str, Any], root_key: str, comparators: tuple[str, ...]) -> dict[str, Any]:
    cells = summary[root_key] if root_key else summary["cells"]
    result: dict[str, Any] = {}
    for cell_name, cell in cells.items():
        metrics = cell.get("metrics", {})
        result[cell_name] = {}
        for metric_name, metric_payload in metrics.items():
            result[cell_name][metric_name] = {
                name: metric_payload.get(name)
                for name in comparators
                if name in metric_payload
            }
    return result


def eqmotion_awta_identity() -> dict[str, Any]:
    eqmotion = load("artifacts/journal_extension_20260814/eqmotion_five_seed_summary_v1.json")
    awta = load("artifacts/journal_extension_20260814/awta_summary_v1.json")
    identity = load("artifacts/journal_extension_20260814/tartan_identity_disjoint_summary_v1.json")
    return {
        "eqmotion": {
            "source": "artifacts/journal_extension_20260814/eqmotion_five_seed_summary_v1.json",
            "cells": summarize_cells(
                eqmotion,
                "",
                ("mabpt_selected_improvement_vs_eqmotion", "ascent_improvement_vs_eqmotion"),
            ),
            "claim_boundary": eqmotion.get("claim_boundary"),
        },
        "awta": {
            "source": "artifacts/journal_extension_20260814/awta_summary_v1.json",
            "cells": summarize_cells(
                awta,
                "tartan",
                ("mabpt_selected_vs_awta", "awta_vs_ascent"),
            ),
            "claim_boundary": awta.get("claim_boundary"),
        },
        "identity_disjoint": {
            "source": "artifacts/journal_extension_20260814/tartan_identity_disjoint_summary_v1.json",
            "cells": identity.get("cells"),
            "claim_boundary": identity.get("claim_boundary"),
        },
    }


def e19_rolling() -> dict[str, Any]:
    summary = load("artifacts/journal_e14_e20_20260817/e19/e19_summary_v1.json")
    compact = {}
    for domain, origins in summary.get("domain_origin_results", {}).items():
        compact[domain] = {}
        for origin, payload in origins.items():
            compact[domain][origin] = {
                "selected_arm": payload.get("selected_arm"),
                "selected_temperature": payload.get("selected_temperature"),
                "effect_original_minus_mabpt": payload.get("effect_original_minus_mabpt"),
                "sign": payload.get("sign"),
            }
    return {
        "source": "artifacts/journal_e14_e20_20260817/e19/e19_summary_v1.json",
        "domain_origin_results": compact,
        "claim_boundaries": summary.get("claim_boundaries"),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    lines = [
        "# Capacity And Generalization Summary",
        "",
        "## Capacity Controls",
    ]
    for name, block in payload["capacity_controls"]["controls"].items():
        energy = block["metrics"].get("energy_score", {})
        lines.append(
            f"- `{name}`: energy control-minus-MABPT `{energy.get('absolute_effect')}`, "
            f"CI `{energy.get('ci95')}`, supported `{energy.get('supported')}`."
        )
    lines.extend(["", "## Independent Probability Baseline"])
    for metric, block in payload["independent_gru"]["paired_vs_mabpt"].items():
        lines.append(f"- `{metric}`: effect `{block.get('absolute_effect')}`, CI `{block.get('ci95')}`.")
    lines.extend(["", "## Existing Generalization / Baseline Artifacts"])
    lines.append("- EqMotion, aWTA, identity-disjoint, and E19 rolling summaries were found and compacted in the JSON output.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "format_version": 1,
        "experiment_id": "measure2act_capacity_generalization_summary_v1",
        "root": ROOT.as_posix(),
        "capacity_controls": e17_capacity(),
        "internal_stage_effects": e18_component(),
        "independent_gru": e15_independent_gru(),
        "matched_and_modern_baselines": eqmotion_awta_identity(),
        "rolling_generalization": e19_rolling(),
        "interpretation": {
            "double_network_not_main_innovation": True,
            "main_supported_claim": (
                "Existing capacity and matched-objective controls do not explain "
                "the probability-transfer/operator gains as merely added network capacity."
            ),
            "caveat": "All listed evidence is local/retrospective unless the source summary states otherwise.",
        },
    }
    atomic_json(args.output_json.resolve(), result)
    write_markdown(args.output_md.resolve(), result)
    print(json.dumps({"output": args.output_json.resolve().as_posix()}, indent=2))


if __name__ == "__main__":
    main()
