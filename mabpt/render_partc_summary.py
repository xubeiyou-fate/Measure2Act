"""Render manuscript summary panels from completed MABPT-ASCENT experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluate import ROOT, _sha256


LEGACY = ROOT / "artifacts/mabpt"
PARTC = ROOT / "artifacts/mabpt_partc_20260811"
OUTPUT_ROOT = ROOT / "artifacts/mabpt_partc_20260811/figures"


def _load(name: str) -> dict[str, object]:
    return json.loads((LEGACY / name).read_text(encoding="utf-8"))


def render(output: Path) -> dict[str, object]:
    robustness_path = PARTC / "five_seed_robustness_summary_v1.json"
    seed_summary_path = PARTC / "five_seed_development_summary_v1.json"
    robustness = json.loads(robustness_path.read_text(encoding="utf-8"))
    seed_summary = json.loads(seed_summary_path.read_text(encoding="utf-8"))
    scaling = _load("e9_summary_v1.json")
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.7), constrained_layout=True)

    conditions = list(robustness["conditions"])
    condition_labels = {
        "clean": "Clean",
        "dropout_10": "Drop 10%",
        "dropout_30": "Drop 30%",
        "dropout_50": "Drop 50%",
        "noise_10m": "Noise 10 m",
        "noise_30m": "Noise 30 m",
        "noise_50m": "Noise 50 m",
        "history_4": "History 4",
        "history_8": "History 8",
        "history_12": "History 12",
    }
    gains = [
        100
        * float(
            robustness["conditions"][condition]["paired_hierarchical_bootstrap"][
                "energy_score"
            ]["relative_gain"]
        )
        for condition in conditions
    ]
    x = np.arange(len(conditions))
    axes[0, 0].bar(x, gains, color="#3a7d44", width=0.72)
    axes[0, 0].set_xticks(
        x, [condition_labels[condition] for condition in conditions], rotation=36, ha="right"
    )
    axes[0, 0].set_ylabel("Energy improvement vs ASCENT (%)")
    axes[0, 0].set_title("a) Registered robustness conditions", loc="left")
    axes[0, 0].grid(axis="y", alpha=0.22, linewidth=0.6)

    calibration_axis = axes[0, 1]
    for model, label, color, marker in (
        ("original_ascent", "ASCENT", "#4c78a8", "o"),
        ("mabpt_ascent", "MABPT-ASCENT", "#d1495b", "s"),
    ):
        bins = seed_summary["fixed_event_calibration"][
            "pooled_seed_reliability_diagram"
        ][model]
        confidence = [value["mean_confidence"] for value in bins if value["count"]]
        accuracy = [value["accuracy"] for value in bins if value["count"]]
        calibration_axis.plot(
            confidence,
            accuracy,
            marker=marker,
            markersize=4.2,
            linewidth=1.7,
            color=color,
            label=label,
        )
    calibration_axis.plot([0, 1], [0, 1], color="#555555", linestyle=":", label="Ideal")
    calibration_axis.set_xlim(0.15, 1.0)
    calibration_axis.set_ylim(0.15, 1.0)
    calibration_axis.set_xlabel("Predicted event confidence")
    calibration_axis.set_ylabel("Observed event frequency")
    calibration_axis.set_title("b) Fixed-event reliability", loc="left")
    calibration_axis.legend(frameon=False, fontsize=8.5)
    calibration_axis.grid(alpha=0.22, linewidth=0.6)

    modes = [3, 5, 7]
    ascent_energy = [
        scaling["cardinalities"][str(mode)]["energy_score"]["ascent_native"]
        for mode in modes
    ]
    mabpt_energy = [
        scaling["cardinalities"][str(mode)]["energy_score"]["mabpt_exact"]
        for mode in modes
    ]
    latency = [
        scaling["cardinalities"][str(mode)]["operator_benchmark"]["exact"][
            "median_actor_microseconds"
        ]
        for mode in modes
    ]
    scale_axis = axes[1, 0]
    scale_axis.plot(
        modes, ascent_energy, color="#4c78a8", marker="o", linewidth=2, label="ASCENT Energy"
    )
    scale_axis.plot(
        modes,
        mabpt_energy,
        color="#d1495b",
        marker="s",
        linewidth=2,
        label="MABPT-ASCENT Energy",
    )
    scale_axis.set_xticks(modes)
    scale_axis.set_xlabel("Number of modes K")
    scale_axis.set_ylabel("Energy Score (km)")
    latency_axis = scale_axis.twinx()
    latency_axis.plot(
        modes,
        latency,
        color="#7a5195",
        marker="^",
        linestyle="--",
        linewidth=1.7,
        label="Exact latency (us/actor, log axis)",
    )
    latency_axis.set_yscale("log")
    lines = scale_axis.lines + latency_axis.lines
    scale_axis.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=8.3)
    scale_axis.set_title("c) Finite-measure cardinality scaling", loc="left")
    scale_axis.grid(alpha=0.22, linewidth=0.6)

    conflict = seed_summary["conflict_risk"]["aggregates"]
    ascent = conflict["original_ascent"]
    mabpt = conflict["mabpt_ascent"]
    operational = {
        "Brier": 100
        * (ascent["brier"]["mean"] - mabpt["brier"]["mean"])
        / ascent["brier"]["mean"],
        "NLL": 100
        * (ascent["nll"]["mean"] - mabpt["nll"]["mean"])
        / ascent["nll"]["mean"],
        "AUPRC": 100
        * (mabpt["auprc"]["mean"] - ascent["auprc"]["mean"])
        / ascent["auprc"]["mean"],
        "Recall": 100
        * (
            mabpt["recall_at_fixed_fpr"]["mean"]
            - ascent["recall_at_fixed_fpr"]["mean"]
        )
        / ascent["recall_at_fixed_fpr"]["mean"],
        "Lead time": 100
        * (
            mabpt["mean_warning_lead_seconds"]["mean"]
            - ascent["mean_warning_lead_seconds"]["mean"]
        )
        / ascent["mean_warning_lead_seconds"]["mean"],
    }
    labels = list(operational)
    values = list(operational.values())
    colors = ["#3a7d44" if value >= 0 else "#b23a48" for value in values]
    axes[1, 1].barh(labels, values, color=colors, height=0.62)
    axes[1, 1].axvline(0, color="#333333", linewidth=0.8)
    axes[1, 1].set_xlabel("Improvement vs ASCENT (%)")
    axes[1, 1].set_xlim(min(-0.5, min(values) - 0.5), max(0.5, max(values) + 0.7))
    axes[1, 1].set_title("d) Near-conflict surrogate diagnostics", loc="left")
    axes[1, 1].grid(axis="x", alpha=0.22, linewidth=0.6)
    for index, value in enumerate(values):
        axes[1, 1].text(
            value + 0.10,
            index,
            f"{value:+.2f}%",
            ha="left",
            va="center",
            fontsize=8.5,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=260, facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)
    inputs = [robustness_path, seed_summary_path, LEGACY / "e9_summary_v1.json"]
    return {
        "format_version": 1,
        "model": "MABPT-ASCENT",
        "evidence_class": "five_seed_development_plus_retrospective_scaling",
        "panels": ["robustness", "fixed_event_reliability", "scaling", "conflict_surrogate"],
        "inputs": [
            {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
            for path in inputs
        ],
        "outputs": {
            "png": str(output.relative_to(ROOT)),
            "png_sha256": _sha256(output),
            "pdf": str(output.with_suffix(".pdf").relative_to(ROOT)),
            "pdf_sha256": _sha256(output.with_suffix(".pdf")),
        },
        "limitations": [
            "The conflict event is a non-regulatory surrogate.",
            "Exact K-mode scaling enumerates K! assignments and is not a large-K runtime claim.",
            "Fixed-event ECE is reported independently from Brier and NLL and is not assumed to improve.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_ROOT / "partc_secondary_results.png"
    )
    args = parser.parse_args()
    result = render(args.output.resolve())
    receipt = args.output.with_suffix(".json")
    receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": result["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
