"""Read-only multi-horizon diagnosis of frozen C127 B0 and C129 checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from experiments.metric_exact.folds import indices_for_fold
from experiments.metric_exact.model import build_model as build_c127_model
from experiments.joint_coupled.model import build_model as build_c129_model
from experiments.joint_coupled.train import move, set_seed
from model.utils import TrajectoryDataset, seed_worker, seq_collate

from .protocol import load_protocol, sha256


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/decision_regret"
HORIZON_STEPS = {"30": 5, "60": 11, "90": 17, "120": 23}


def _paths(fold: int, family: str) -> tuple[Path, Path]:
    if family == "C129_J1":
        root = ROOT / "runs/joint_coupled" / (
            f"J1_joint_coupled_dual_fold{fold}_seed42_formal"
        )
    elif family == "C127_B0":
        phase = "P1" if fold == 0 else "P2"
        root = ROOT / "runs/metric_exact" / (
            f"{phase}_B0_signed_coupled_fold{fold}_seed42_formal"
        )
    else:
        raise ValueError(f"unknown diagnostic family: {family}")
    return root / "last.pt", root / "training_summary.json"


def _loader(dataset, *, workers: int, prefetch: int, batch_size: int) -> DataLoader:
    options = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": seq_collate,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
    }
    if workers > 0:
        options.update({"persistent_workers": True, "prefetch_factor": prefetch})
    return DataLoader(**options)


def _build(family: str, batch_size: int):
    if family == "C129_J1":
        return build_c129_model(batch_size=batch_size)
    if family == "C127_B0":
        return build_c127_model("B0_signed_coupled", batch_size=batch_size)
    raise ValueError(f"unknown diagnostic family: {family}")


@torch.no_grad()
def _evaluate(model, loader: DataLoader, device: torch.device) -> dict[str, object]:
    model.eval()
    actors = 0
    horizons = {
        name: {"top1_fde_sum": 0.0, "minfde_sum": 0.0, "regret_sum": 0.0}
        for name in HORIZON_STEPS
    }
    transitions = {"30_to_60": 0, "60_to_90": 0, "90_to_120": 0, "30_to_120": 0}
    pair_concordant = 0
    pair_comparable = 0
    ade_fde_overlap = 0
    combined_top1_overlap = 0
    oracle_ade_rank_sum = 0.0
    oracle_fde_rank_sum = 0.0
    oracle_ade_rank1 = 0
    oracle_fde_rank1 = 0

    for data in loader:
        data = move(data, device)
        target = data["pred_traj"].transpose(1, 0)
        predictions, logits, _ = model(data)
        batch = int(predictions.shape[0])
        actors += batch
        displacement = torch.linalg.vector_norm(
            predictions - target[:, None], dim=-1
        )
        ade = displacement.mean(dim=-1)
        fde = displacement[..., -1]
        top1 = logits.argmax(dim=1)
        rows = torch.arange(batch, device=device)
        winners = {}
        for name, step in HORIZON_STEPS.items():
            endpoint_error = displacement[..., step]
            winner = endpoint_error.argmin(dim=1)
            winners[name] = winner
            minimum = endpoint_error[rows, winner]
            selected = endpoint_error[rows, top1]
            horizons[name]["top1_fde_sum"] += float(selected.sum())
            horizons[name]["minfde_sum"] += float(minimum.sum())
            horizons[name]["regret_sum"] += float((selected - minimum).sum())
        for left, right in (("30", "60"), ("60", "90"), ("90", "120"), ("30", "120")):
            transitions[f"{left}_to_{right}"] += int((winners[left] != winners[right]).sum())

        ade_winner = ade.argmin(dim=1)
        fde_winner = fde.argmin(dim=1)
        combined = ade / 0.2760537266731262 + fde / 0.4783363938331604
        combined_winner = combined.argmin(dim=1)
        ade_fde_overlap += int((ade_winner == fde_winner).sum())
        combined_top1_overlap += int((combined_winner == top1).sum())

        score_order = logits.argsort(dim=1, descending=True)
        inverse_rank = torch.empty_like(score_order)
        inverse_rank.scatter_(1, score_order, torch.arange(5, device=device).expand(batch, -1))
        ade_rank = inverse_rank[rows, ade_winner] + 1
        fde_rank = inverse_rank[rows, fde_winner] + 1
        oracle_ade_rank_sum += float(ade_rank.sum())
        oracle_fde_rank_sum += float(fde_rank.sum())
        oracle_ade_rank1 += int((ade_rank == 1).sum())
        oracle_fde_rank1 += int((fde_rank == 1).sum())

        for first in range(5):
            for second in range(first + 1, 5):
                score_difference = logits[:, first] - logits[:, second]
                cost_difference = combined[:, first] - combined[:, second]
                comparable = (score_difference != 0) & (cost_difference != 0)
                pair_comparable += int(comparable.sum())
                pair_concordant += int(
                    ((score_difference * cost_difference < 0) & comparable).sum()
                )

    if actors <= 0 or pair_comparable <= 0:
        raise RuntimeError("empty C133 diagnostic evaluation")
    result = {
        "actors": actors,
        "horizons": {
            name: {
                "top1_fde": values["top1_fde_sum"] / actors,
                "minfde": values["minfde_sum"] / actors,
                "top1_regret": values["regret_sum"] / actors,
            }
            for name, values in horizons.items()
        },
        "oracle_winner_transition_rate": {
            name: count / actors for name, count in transitions.items()
        },
        "ade_fde_winner_overlap": ade_fde_overlap / actors,
        "combined_winner_top1_overlap": combined_top1_overlap / actors,
        "pairwise_score_cost_concordance": pair_concordant / pair_comparable,
        "pairwise_comparisons": pair_comparable,
        "oracle_ade_rank": oracle_ade_rank_sum / actors,
        "oracle_fde_rank": oracle_fde_rank_sum / actors,
        "oracle_ade_rank1": oracle_ade_rank1 / actors,
        "oracle_fde_rank1": oracle_fde_rank1 / actors,
    }
    values = []
    for horizon in result["horizons"].values():
        values.extend(horizon.values())
    values.extend(result["oracle_winner_transition_rate"].values())
    values.extend(
        result[name]
        for name in (
            "ade_fde_winner_overlap",
            "combined_winner_top1_overlap",
            "pairwise_score_cost_concordance",
            "oracle_ade_rank",
            "oracle_fde_rank",
            "oracle_ade_rank1",
            "oracle_fde_rank1",
        )
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise RuntimeError("non-finite C133 diagnostic metric")
    return result


def run(fold: int, device_name: str, workers: int, prefetch: int, batch_size: int) -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_boundaries()
    if fold not in protocol.payload["diagnostic"]["folds"]:
        raise RuntimeError("diagnostic fold is outside the frozen protocol")
    set_seed(42)
    dataset = TrajectoryDataset(
        protocol.split_path("train").as_posix(),
        obs_len=16,
        obs_steps=1,
        pred_len=120,
        pred_step=5,
        delim=" ",
    )
    expected = protocol.payload["dataset"]
    if len(dataset) != int(expected["expected_train_scenes"]):
        raise RuntimeError("C133 diagnostic cohort mismatch")
    dates = json.loads(
        (ROOT / str(expected["train_scene_dates"])).read_text(encoding="utf-8")
    )["dates"]
    folds = json.loads(
        (ROOT / str(expected["date_folds"])).read_text(encoding="utf-8")
    )
    _, validation_indices, _ = indices_for_fold(dates, folds, fold)
    validation = Subset(dataset, validation_indices)
    loader = _loader(
        validation, workers=workers, prefetch=prefetch, batch_size=batch_size
    )
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    families = {}
    for family in ("C127_B0", "C129_J1"):
        checkpoint_path, summary_path = _paths(fold, family)
        expected_hash = protocol.payload["diagnostic"]["checkpoint_sha256"][family][fold]
        if sha256(checkpoint_path) != expected_hash:
            raise RuntimeError(f"{family} fold {fold} checkpoint hash mismatch")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("complete") is not True
            or summary.get("formal") is not True
            or summary.get("fold") != fold
            or summary.get("seed") != 42
            or summary.get("fixed_final_epoch") != 20
            or summary.get("locked_test_used") is not False
        ):
            raise RuntimeError(f"{family} fold {fold} summary identity mismatch")
        model = _build(family, batch_size).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        families[family] = {
            "checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
            "checkpoint_sha256": expected_hash,
            "summary": summary_path.relative_to(ROOT).as_posix(),
            "summary_sha256": sha256(summary_path),
            "metrics": _evaluate(model, loader, device),
        }
        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "format_version": 1,
        "cycle": protocol.payload["cycle"],
        "phase": "P0_zero_training_read_only",
        "fold": fold,
        "protocol_sha256": sha256(protocol.path),
        "validation_scenes": len(validation),
        "families": families,
        "training_performed": False,
        "adaptive_selection_performed": False,
        "locked_test_used": False,
        "development_used": False,
    }
    output = ARTIFACT_ROOT / f"diagnostic_fold{fold}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.fold,
                args.device,
                args.num_workers,
                args.prefetch_factor,
                args.batch_size,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
