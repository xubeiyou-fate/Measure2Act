"""Audit regression-versus-winner-score gradients on the trained C98 A0 model."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from experiments.edfa_ascent.data import build_scene_dates
from c98_stif_ascent.model import build_model as build_c98_model
from model.utils import TrajectoryDataset, seq_collate

from .live_report import update_status
from .protocol import load_protocol, sha256


GROUP_PREFIXES = {
    "shared_context": (
        "agent_blks.",
        "agent_xy_proj.",
        "agent_z_proj.",
        "agent_ts_proj.",
        "pos_embed.",
        "norm.",
        "type_embed",
    ),
    "mode_embedding": ("mode1_embed",),
    "geometry_decoder": ("fp1.", "fp2.", "fp3."),
    "score_head": ("pi.",),
}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move(data: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in data.items()
    }


def parameter_groups(model: torch.nn.Module) -> dict[str, list[torch.nn.Parameter]]:
    groups: dict[str, list[torch.nn.Parameter]] = {name: [] for name in GROUP_PREFIXES}
    for name, parameter in model.named_parameters():
        for group, prefixes in GROUP_PREFIXES.items():
            if any(name == prefix or name.startswith(prefix) for prefix in prefixes):
                groups[group].append(parameter)
                break
    if not groups["shared_context"] or not groups["mode_embedding"]:
        raise RuntimeError("C99 gradient audit could not locate shared ASCENT parameters")
    return groups


def _gradient_pair(
    first_loss: torch.Tensor,
    second_loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
) -> dict[str, float | bool]:
    first = torch.autograd.grad(
        first_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    second = torch.autograd.grad(
        second_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    dot = first_loss.new_zeros(())
    first_sq = first_loss.new_zeros(())
    second_sq = first_loss.new_zeros(())
    for left, right in zip(first, second):
        if left is not None:
            first_sq = first_sq + left.detach().float().square().sum()
        if right is not None:
            second_sq = second_sq + right.detach().float().square().sum()
        if left is not None and right is not None:
            dot = dot + (left.detach().float() * right.detach().float()).sum()
    first_norm = first_sq.sqrt()
    second_norm = second_sq.sqrt()
    denominator = first_norm * second_norm
    cosine = dot / denominator.clamp_min(1e-20)
    ratio = second_norm / first_norm.clamp_min(1e-20)
    both_nonzero = bool(denominator.detach().cpu() > 0)
    return {
        "regression_norm": float(first_norm.detach().cpu()),
        "classification_norm": float(second_norm.detach().cpu()),
        "classification_to_regression_ratio": float(ratio.detach().cpu()),
        "cosine": float(cosine.detach().cpu()) if both_nonzero else 0.0,
        "both_nonzero": both_nonzero,
    }


def _summarize(rows: list[dict[str, float | bool]]) -> dict[str, float | int]:
    valid = [row for row in rows if bool(row["both_nonzero"])]
    ratios = [float(row["classification_to_regression_ratio"]) for row in valid]
    cosines = [float(row["cosine"]) for row in valid]
    regression = [float(row["regression_norm"]) for row in rows]
    classification = [float(row["classification_norm"]) for row in rows]
    return {
        "batches": len(rows),
        "both_nonzero_batches": len(valid),
        "median_regression_norm": median(regression) if regression else 0.0,
        "median_classification_norm": median(classification) if classification else 0.0,
        "median_classification_to_regression_ratio": median(ratios) if ratios else 0.0,
        "median_cosine": median(cosines) if cosines else 0.0,
        "negative_cosine_fraction": (
            sum(value < 0.0 for value in cosines) / len(cosines) if cosines else 0.0
        ),
    }


def _date_blocks(dates: list[str], count: int) -> list[tuple[list[str], list[int]]]:
    unique = sorted(set(dates))
    if len(unique) < count:
        raise RuntimeError("not enough train dates for C99 gradient-audit blocks")
    blocks = np.array_split(np.asarray(unique, dtype=object), count)
    result = []
    for values in blocks:
        selected = {str(value) for value in values.tolist()}
        indices = [index for index, date in enumerate(dates) if date in selected]
        result.append((sorted(selected), indices))
    return result


def run(
    output: Path | None = None,
    *,
    device_name: str | None = None,
    batches_override: int | None = None,
    batch_size_override: int | None = None,
) -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_development_sealed()
    settings = protocol.payload["gradient_audit"]
    seed = int(settings["seed"])
    _set_seed(seed)
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint_path = protocol.repository_root / str(settings["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_c98_model("a0_base", batch_size=int(settings["batch_size"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    groups = parameter_groups(model)

    dataset = TrajectoryDataset(
        protocol.split_path("train").as_posix(),
        obs_len=16,
        obs_steps=1,
        pred_len=120,
        pred_step=5,
        delim=" ",
    )
    train_dates_path = (
        protocol.repository_root / "artifacts/experiments/dive_ascent/train_scene_dates.json"
    )
    if train_dates_path.is_file():
        scene_dates = json.loads(train_dates_path.read_text(encoding="utf-8"))["dates"]
    else:
        scene_dates = build_scene_dates(
            protocol.split_path("train"), protocol.manifest_path, "train"
        )
        train_dates_path.parent.mkdir(parents=True, exist_ok=True)
        train_dates_path.write_text(
            json.dumps({"dates": scene_dates}) + "\n", encoding="utf-8"
        )
    if len(scene_dates) != len(dataset):
        raise RuntimeError("C99 train scene-date index does not match TrajectoryDataset")

    requested_batches = int(batches_override or settings["batches"])
    batch_size = int(batch_size_override or settings["batch_size"])
    block_count = int(settings["date_blocks"])
    batches_by_block = [requested_batches // block_count] * block_count
    for index in range(requested_batches % block_count):
        batches_by_block[index] += 1
    generator = torch.Generator().manual_seed(seed)
    raw: dict[str, dict[str, list[dict[str, float | bool]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    loss_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    processed = 0
    blocks_metadata = []
    for block_index, ((dates, indices), block_batches) in enumerate(
        zip(_date_blocks(scene_dates, block_count), batches_by_block)
    ):
        required_scenes = min(len(indices), block_batches * batch_size)
        permutation = torch.randperm(len(indices), generator=generator)[:required_scenes]
        selected = [indices[int(position)] for position in permutation]
        loader = DataLoader(
            Subset(dataset, selected),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=seq_collate,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        block_name = f"block_{block_index + 1}"
        blocks_metadata.append(
            {
                "name": block_name,
                "first_date": dates[0],
                "last_date": dates[-1],
                "dates": len(dates),
                "sampled_scenes": len(selected),
                "requested_batches": block_batches,
            }
        )
        for data in loader:
            data = _move(data, device)
            target = data["pred_traj"].transpose(1, 0)
            predictions, logits, _ = model(data)
            displacement = torch.linalg.vector_norm(
                predictions - target[:, None], dim=-1
            )
            winner = displacement.sum(dim=-1).detach().argmin(dim=-1)
            actor = torch.arange(target.shape[0], device=device)
            regression = F.smooth_l1_loss(predictions[actor, winner], target)
            classification = F.cross_entropy(logits, winner)
            loss_rows[block_name].append(
                {
                    "regression": float(regression.detach().cpu()),
                    "classification": float(classification.detach().cpu()),
                }
            )
            for group_name, parameters in groups.items():
                raw[block_name][group_name].append(
                    _gradient_pair(regression, classification, parameters)
                )
            processed += 1
            update_status(
                protocol.repository_root,
                "p0_gradient_audit",
                {
                    "phase": "gradient_audit",
                    "variant": "c98_a0_checkpoint",
                    "epoch": block_index + 1,
                    "epochs": block_count,
                    "batch": processed,
                    "batches_per_epoch": requested_batches,
                    "running_loss": f"{float((regression + classification).detach()):.6f}",
                    "development_minfde": "sealed",
                },
            )
            del predictions, logits, regression, classification

    summaries = {
        block: {group: _summarize(rows) for group, rows in groups_rows.items()}
        for block, groups_rows in raw.items()
    }
    all_groups: dict[str, list[dict[str, float | bool]]] = defaultdict(list)
    for groups_rows in raw.values():
        for group, rows in groups_rows.items():
            all_groups[group].extend(rows)
    overall = {group: _summarize(rows) for group, rows in all_groups.items()}
    ratio_gate = float(settings["minimum_classification_to_regression_norm_ratio"])
    conflict_gate = float(settings["minimum_negative_cosine_fraction"])
    block_checks = {}
    for block, summary in summaries.items():
        shared = summary["shared_context"]
        block_checks[block] = {
            "ratio": shared["median_classification_to_regression_ratio"] >= ratio_gate,
            "conflict": shared["negative_cosine_fraction"] >= conflict_gate,
        }
        block_checks[block]["passed"] = all(block_checks[block].values())
    passing_blocks = sum(bool(check["passed"]) for check in block_checks.values())
    mechanism_supported = passing_blocks >= int(settings["minimum_date_blocks_passing"])
    result = {
        "format_version": 1,
        "cycle": protocol.payload["cycle"],
        "protocol_sha256": sha256(protocol.path),
        "manifest_sha256": sha256(protocol.manifest_path),
        "checkpoint": checkpoint_path.relative_to(protocol.repository_root).as_posix(),
        "checkpoint_sha256": sha256(checkpoint_path),
        "device": str(device),
        "requested_batches": requested_batches,
        "processed_batches": processed,
        "batch_size_scenes": batch_size,
        "date_blocks": blocks_metadata,
        "loss": {
            block: {
                "median_regression": median(row["regression"] for row in rows),
                "median_classification": median(row["classification"] for row in rows),
            }
            for block, rows in loss_rows.items()
        },
        "groups": summaries,
        "overall": overall,
        "frozen_thresholds": {
            "minimum_classification_to_regression_norm_ratio": ratio_gate,
            "minimum_negative_cosine_fraction": conflict_gate,
            "minimum_date_blocks_passing": int(settings["minimum_date_blocks_passing"]),
        },
        "block_checks": block_checks,
        "passing_date_blocks": passing_blocks,
        "mechanism_supported": mechanism_supported,
        "training_authorized": mechanism_supported,
        "locked_test_used": False,
    }
    if not math.isfinite(
        float(overall["shared_context"]["median_classification_to_regression_ratio"])
    ):
        raise RuntimeError("C99 gradient audit produced non-finite shared-context statistics")
    output = output or protocol.repository_root / "artifacts/experiments/dive_ascent/gradient_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    update_status(
        protocol.repository_root,
        "p0_gradient_audit",
        {
            "phase": "complete",
            "variant": "c98_a0_checkpoint",
            "epoch": block_count,
            "epochs": block_count,
            "batch": processed,
            "batches_per_epoch": requested_batches,
            "running_loss": "-",
            "development_minfde": "sealed",
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--batches", type=int)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    result = run(
        args.output,
        device_name=args.device,
        batches_override=args.batches,
        batch_size_override=args.batch_size,
    )
    print(
        json.dumps(
            {
                "processed_batches": result["processed_batches"],
                "passing_date_blocks": result["passing_date_blocks"],
                "mechanism_supported": result["mechanism_supported"],
                "shared_context": result["overall"]["shared_context"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
