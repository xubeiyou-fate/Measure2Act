"""Render target-blind E13 qualitative cases for MABPT-ASCENT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Subset

from experiments.ascent_recomparison.common import fold_subsets, load_dataset, loader
from experiments.tpmo_ascent.protocol import load_protocol as load_legacy_data_protocol

from .evaluate import ROOT, _load_models, _sha256
from .operator import DEFAULT_ADE_SCALE, pairwise_trajectory_distance, support_cost
from .partc_factorial import factorial_probability_arms, full_model_arm


ARTIFACT_ROOT = ROOT / "artifacts/mabpt"
FIGURE_ROOT = ROOT / "artifacts/mabpt_partc_20260811/figures"
CASE_TYPES = (
    "highest_permutation_entropy",
    "highest_support_mismatch",
    "lowest_observed_speed",
    "highest_observed_vertical_speed",
)
CASE_LABELS = {
    "highest_permutation_entropy": "High transport entropy",
    "highest_support_mismatch": "High support mismatch",
    "lowest_observed_speed": "Low observed speed",
    "highest_observed_vertical_speed": "High vertical speed",
}


def _selected_cases(fold: int) -> list[tuple[str, dict[str, object]]]:
    path = ARTIFACT_ROOT / f"e13_fold{fold}_formal_v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = []
    used: set[int] = set()
    for case_type in CASE_TYPES:
        for case in payload["target_blind_selected_cases"][case_type]:
            actor_order = int(case["actor_order"])
            if actor_order not in used:
                selected.append((case_type, case))
                used.add(actor_order)
                break
        else:
            raise RuntimeError(f"no distinct target-blind case for {case_type}")
    return selected


def _locate_actor(dataset, scene_indices: list[int], actor_order: int) -> tuple[int, int]:
    cursor = 0
    for scene_index in scene_indices:
        start, end = [int(value) for value in dataset.seq_start_end[scene_index]]
        actors = end - start
        if actor_order < cursor + actors:
            return scene_index, actor_order - cursor
        cursor += actors
    raise IndexError(f"actor order {actor_order} lies outside validation subset")


@torch.inference_mode()
def _case_forward(
    *,
    dataset,
    scene_index: int,
    actor_within_scene: int,
    source_model,
    target_model,
    device: torch.device,
) -> dict[str, np.ndarray | int | float]:
    one_loader = loader(
        Subset(dataset, [scene_index]),
        batch_size=1,
        shuffle=False,
        workers=0,
        prefetch=2,
    )
    data = next(iter(one_loader))
    data = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in data.items()
    }
    truth = data["pred_traj"].transpose(1, 0)
    source_support, source_logits, _ = source_model(data)
    source_probability = source_logits.softmax(dim=1)
    target_support, _, target_decision, auxiliary = target_model(data)
    arms, _ = factorial_probability_arms(
        source_probability,
        support_cost(source_support, target_support),
        auxiliary["centered_predicted_normalized_ade_risk"].to(torch.float64),
        pairwise_trajectory_distance(target_support) / DEFAULT_ADE_SCALE,
    )
    probabilities = arms[full_model_arm()]
    row = actor_within_scene
    return {
        "observation": data["obs_traj"][:, row].detach().cpu().numpy(),
        "truth": truth[row].detach().cpu().numpy(),
        "source_support": source_support[row].detach().cpu().numpy(),
        "source_top1": int(source_logits[row].argmax().cpu()),
        "target_support": target_support[row].detach().cpu().numpy(),
        "target_top1": int(target_decision[row].cpu()),
        "probabilities": probabilities[row].detach().cpu().numpy(),
    }


def _plot_fold(
    *,
    fold: int,
    cases: list[tuple[str, dict[str, object], dict[str, object]]],
    output: Path,
) -> None:
    figure, axes = plt.subplots(
        len(cases),
        2,
        figsize=(11.2, 12.0),
        gridspec_kw={"width_ratios": [1.15, 1]},
        constrained_layout=True,
    )
    for row, (case_type, record, values) in enumerate(cases):
        horizontal, altitude = axes[row]
        observation = np.asarray(values["observation"])
        truth = np.asarray(values["truth"])
        source_support = np.asarray(values["source_support"])
        target_support = np.asarray(values["target_support"])
        probabilities = np.asarray(values["probabilities"])
        source_top1 = int(values["source_top1"])
        target_top1 = int(values["target_top1"])
        origin = observation[-1]
        observation = observation - origin
        truth = truth - origin
        source_support = source_support - origin
        target_support = target_support - origin

        horizontal.plot(
            observation[:, 0], observation[:, 1], color="#4c78a8", linewidth=2.0,
            label="Observed history",
        )
        horizontal.plot(
            truth[:, 0], truth[:, 1], color="#111111", linewidth=2.2,
            label="Ground truth",
        )
        for mode in range(target_support.shape[0]):
            horizontal.plot(
                target_support[mode, :, 0],
                target_support[mode, :, 1],
                color="#f2a541",
                linewidth=0.8 + 2.2 * probabilities[mode],
                alpha=0.20 + 0.65 * probabilities[mode] / max(probabilities.max(), 1e-12),
                label="MABPT finite-measure support" if mode == 0 else None,
            )
        horizontal.plot(
            source_support[source_top1, :, 0],
            source_support[source_top1, :, 1],
            color="#4c78a8",
            linestyle="--",
            linewidth=1.8,
            label="ASCENT Top-1",
        )
        horizontal.plot(
            target_support[target_top1, :, 0],
            target_support[target_top1, :, 1],
            color="#d1495b",
            linewidth=2.0,
            label="MABPT-ASCENT Top-1",
        )
        horizontal.scatter([0], [0], color="#111111", s=18, zorder=5)
        horizontal.set_aspect("equal", adjustable="datalim")
        horizontal.set_xlabel("East-west displacement (km)")
        horizontal.set_ylabel("North-south displacement (km)")
        horizontal.grid(alpha=0.22, linewidth=0.6)
        horizontal.set_title(
            f"{chr(97 + row)}) {CASE_LABELS[case_type]} | {record['date']}",
            loc="left",
            fontsize=10.5,
        )

        seconds = np.arange(1, truth.shape[0] + 1) * 5
        altitude.plot(
            seconds, truth[:, 2], color="#111111", linewidth=2.2,
            label="Ground truth",
        )
        for mode in range(target_support.shape[0]):
            altitude.plot(
                seconds,
                target_support[mode, :, 2],
                color="#f2a541",
                linewidth=0.8 + 2.2 * probabilities[mode],
                alpha=0.20 + 0.65 * probabilities[mode] / max(probabilities.max(), 1e-12),
            )
        altitude.plot(
            seconds,
            source_support[source_top1, :, 2],
            color="#4c78a8",
            linestyle="--",
            linewidth=1.8,
            label="ASCENT Top-1",
        )
        altitude.plot(
            seconds,
            target_support[target_top1, :, 2],
            color="#d1495b",
            linewidth=2.0,
            label="MABPT-ASCENT Top-1",
        )
        altitude.set_xlabel("Forecast horizon (s)")
        altitude.set_ylabel("Relative altitude (km)")
        altitude.grid(alpha=0.22, linewidth=0.6)
        altitude.text(
            0.99,
            0.04,
            "p = " + ", ".join(f"{value:.2f}" for value in probabilities),
            transform=altitude.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#444444",
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncols=5, frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)


def run(*, fold: int, device: torch.device, output: Path) -> dict[str, object]:
    if fold not in (1, 2):
        raise ValueError("E13 qualitative rendering is frozen to folds 1 and 2")
    legacy = load_legacy_data_protocol()
    legacy.assert_boundaries()
    dataset = load_dataset(legacy)
    _, validation, validation_dates = fold_subsets(legacy, dataset, fold)
    scene_indices = list(validation.indices)
    source_model, target_model, source_path, target_path = _load_models(fold, device)
    rendered = []
    for case_type, record in _selected_cases(fold):
        scene_index, actor_within_scene = _locate_actor(
            dataset, scene_indices, int(record["actor_order"])
        )
        scene_position = scene_indices.index(scene_index)
        if validation_dates[scene_position] != record["date"]:
            raise RuntimeError("selected-case date does not match validation index")
        values = _case_forward(
            dataset=dataset,
            scene_index=scene_index,
            actor_within_scene=actor_within_scene,
            source_model=source_model,
            target_model=target_model,
            device=device,
        )
        rendered.append((case_type, record, values))
    _plot_fold(fold=fold, cases=rendered, output=output)
    receipt = {
        "format_version": 1,
        "model": "MABPT-ASCENT",
        "experiment_id": "prespecified_qualitative_cases",
        "evidence_class": "retrospective_development_only",
        "fold": fold,
        "selection": "first distinct actor in each frozen target-blind E13 case list",
        "future_or_metric_used_for_case_selection": False,
        "cases": [
            {"case_type": case_type, **record}
            for case_type, record, _values in rendered
        ],
        "inputs": {
            "e13": str(
                (ARTIFACT_ROOT / f"e13_fold{fold}_formal_v1.json").relative_to(ROOT)
            ),
            "source_checkpoint": source_path,
            "target_checkpoint": target_path,
        },
        "outputs": {
            "png": str(output.relative_to(ROOT)),
            "pdf": str(output.with_suffix(".pdf").relative_to(ROOT)),
            "png_sha256": _sha256(output),
            "pdf_sha256": _sha256(output.with_suffix(".pdf")),
        },
    }
    receipt_path = output.with_suffix(".json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=(1, 2), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or FIGURE_ROOT / f"e13_target_blind_cases_fold{args.fold}.png"
    result = run(fold=args.fold, device=torch.device(args.device), output=output.resolve())
    print(json.dumps({"output": result["outputs"], "fold": args.fold}, indent=2))


if __name__ == "__main__":
    main()
