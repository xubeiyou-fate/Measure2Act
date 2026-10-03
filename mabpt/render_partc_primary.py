"""Render five-seed and factorial primary panels for MABPT-ASCENT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluate import ROOT, _sha256


PARTC = ROOT / "artifacts/mabpt_partc_20260811"
OUTPUT_ROOT = PARTC / "figures"


def render(output: Path) -> dict[str, object]:
    seed_path = PARTC / "five_seed_development_summary_v1.json"
    factorial_path = PARTC / "factorial_physical_summary_v1.json"
    seeds = json.loads(seed_path.read_text(encoding="utf-8"))
    factorial = json.loads(factorial_path.read_text(encoding="utf-8"))
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 8.0), constrained_layout=True)
    seed_values = seeds["seeds"]
    colors = {"original_ascent": "#4c78a8", "mabpt_ascent": "#d1495b"}
    labels = {"original_ascent": "ASCENT", "mabpt_ascent": "MABPT-ASCENT"}

    for axis, metric, title, ylabel in (
        (axes[0, 0], "energy_score", "a) Five-seed Energy", "Energy Score (km)"),
        (axes[0, 1], "top1_fde", "b) Five-seed Top-1 FDE", "Top-1 FDE (km)"),
    ):
        values = {
            model: seeds["aggregates"][model][metric]["values"]
            for model in ("original_ascent", "mabpt_ascent")
        }
        x = np.arange(len(seed_values))
        for position in x:
            axis.plot(
                [position - 0.10, position + 0.10],
                [values["original_ascent"][position], values["mabpt_ascent"][position]],
                color="#999999",
                linewidth=1.0,
                zorder=1,
            )
        for offset, model, marker in (
            (-0.10, "original_ascent", "o"),
            (0.10, "mabpt_ascent", "s"),
        ):
            axis.scatter(
                x + offset,
                values[model],
                color=colors[model],
                marker=marker,
                s=42,
                label=labels[model],
                zorder=2,
            )
        axis.set_xticks(x, [str(seed) for seed in seed_values])
        axis.set_xlabel("Matched random seed")
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left")
        axis.grid(axis="y", alpha=0.22, linewidth=0.6)
        axis.legend(frameon=False, fontsize=8.5)

    effect_order = (
        "correspondence",
        "assignment_cost_mass",
        "projection",
        "correspondence_x_assignment_cost_mass",
        "correspondence_x_projection",
        "assignment_cost_mass_x_projection",
        "correspondence_x_assignment_cost_mass_x_projection",
    )
    effect_labels = {
        "correspondence": "Exact Gibbs",
        "assignment_cost_mass": "Predicted mass",
        "projection": "Energy-KL",
        "correspondence_x_assignment_cost_mass": "Gibbs x mass",
        "correspondence_x_projection": "Gibbs x Energy-KL",
        "assignment_cost_mass_x_projection": "Mass x Energy-KL",
        "correspondence_x_assignment_cost_mass_x_projection": "Three-way interaction",
    }
    effects = factorial["factorial_effects"]["energy_score"]
    estimates = np.asarray([effects[name]["estimate"] for name in effect_order])
    intervals = np.asarray([effects[name]["ci95"] for name in effect_order])
    y = np.arange(len(effect_order))
    axes[1, 0].errorbar(
        estimates,
        y,
        xerr=np.vstack((estimates - intervals[:, 0], intervals[:, 1] - estimates)),
        fmt="o",
        color="#7a5195",
        ecolor="#7a5195",
        capsize=3,
        markersize=5,
    )
    axes[1, 0].axvline(0, color="#444444", linewidth=0.8)
    axes[1, 0].set_yticks(y, [effect_labels[name] for name in effect_order])
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel("Energy effect: high minus low (km)")
    axes[1, 0].set_title("c) Full 2x2x2 module effects", loc="left")
    axes[1, 0].grid(axis="x", alpha=0.22, linewidth=0.6)

    features = list(factorial["physical"]["original_ascent"]["features"])
    feature_labels = {
        "horizontal_speed_km_per_second": "Horizontal speed",
        "absolute_vertical_speed_km_per_second": "Vertical speed",
        "horizontal_acceleration_km_per_second2": "Horizontal accel.",
        "vertical_acceleration_km_per_second2": "Vertical accel.",
        "absolute_turn_rate_radians_per_second": "Turn rate",
    }
    x = np.arange(len(features))
    width = 0.36
    for offset, model in ((-width / 2, "original_ascent"), (width / 2, "mabpt_ascent")):
        rates = [
            100
            * factorial["physical"][model]["features"][feature][
                "outside_training_envelope_rate"
            ]
            for feature in features
        ]
        axes[1, 1].bar(
            x + offset,
            rates,
            width,
            color=colors[model],
            label=labels[model],
        )
    axes[1, 1].set_xticks(
        x, [feature_labels[feature] for feature in features], rotation=32, ha="right"
    )
    axes[1, 1].set_ylabel("Outside training envelope (%)")
    axes[1, 1].set_title("d) Probability-weighted physical support", loc="left")
    axes[1, 1].grid(axis="y", alpha=0.22, linewidth=0.6)
    axes[1, 1].legend(frameon=False, fontsize=8.5)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=260, facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)
    return {
        "format_version": 1,
        "model": "MABPT-ASCENT",
        "evidence_class": "development_and_retrospective_development",
        "panels": ["five_seed_energy", "five_seed_top1_fde", "factorial", "physical"],
        "inputs": [
            {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
            for path in (seed_path, factorial_path)
        ],
        "outputs": {
            "png": str(output.relative_to(ROOT)),
            "png_sha256": _sha256(output),
            "pdf": str(output.with_suffix(".pdf").relative_to(ROOT)),
            "pdf_sha256": _sha256(output.with_suffix(".pdf")),
        },
        "claim_boundary": "Development evidence; no fresh confirmatory cohort was available.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_ROOT / "partc_primary_results.png"
    )
    args = parser.parse_args()
    result = render(args.output.resolve())
    args.output.with_suffix(".json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": result["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
