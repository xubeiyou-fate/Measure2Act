"""C134 P0-B continuous physical action-space capacity and solver audit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from experiments.metric_exact.folds import indices_for_fold
from experiments.joint_coupled.train import move, set_seed
from model.utils import TrajectoryDataset, seed_worker, seq_collate

from .physical_oracle import FREE_KNOTS, continuous_physical_oracle, pose_from_history
from .protocol import load_protocol, sha256


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/energy_predict_optimize"
DESIGN_PATH = ARTIFACT_ROOT / "p0_b_design.json"


def physical_violation_mask(
    parameters: torch.Tensor, candidates: torch.Tensor
) -> torch.Tensor:
    """Return a [B,K,T] mask for invalid physical controls or trajectories."""
    if parameters.shape != candidates.shape or parameters.shape[-1] != 3:
        raise ValueError("parameters and candidates must share shape [B,K,T,3]")
    return (
        (parameters[..., 0] < -1e-10)
        | (parameters[..., 2].abs() > math.pi / 2 + 1e-7)
        | ~torch.isfinite(parameters).all(dim=-1)
        | ~torch.isfinite(candidates).all(dim=-1)
    )


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


def _control(design: dict[str, object], fold: int) -> tuple[dict[str, object], Path]:
    entry = design["controls"][f"fold{fold}"]
    path = ROOT / str(entry["summary"])
    if sha256(path) != str(entry["summary_sha256"]):
        raise RuntimeError(f"C134 P0-B fold {fold} control hash mismatch")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if (
        summary.get("complete") is not True
        or summary.get("formal") is not True
        or summary.get("fold") != fold
        or summary.get("seed") != 42
        or summary.get("fixed_final_epoch") != 20
        or summary.get("locked_test_used") is not False
    ):
        raise RuntimeError(f"C134 P0-B fold {fold} control identity mismatch")
    return summary, path


def run(
    device_name: str, workers: int, prefetch: int, batch_size: int
) -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_boundaries()
    protocol_hash = sha256(protocol.path)
    p0_a_path = ARTIFACT_ROOT / "p0_a_decision.json"
    p0_a = json.loads(p0_a_path.read_text(encoding="utf-8"))
    if (
        p0_a.get("decision") != "P0_B_AUTHORIZED"
        or p0_a.get("protocol_sha256") != protocol_hash
        or p0_a.get("training_performed") is not False
        or p0_a.get("locked_test_used") is not False
    ):
        raise RuntimeError("C134 P0-A did not authorize P0-B")
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    if design.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("C134 P0-B design protocol mismatch")
    if tuple(design["operator"]["free_piecewise_linear_control_knots"]) != FREE_KNOTS:
        raise RuntimeError("C134 P0-B implementation does not match frozen knot bases")
    set_seed(42)
    expected = protocol.payload["dataset"]
    dataset = TrajectoryDataset(
        protocol.split_path("train").as_posix(),
        obs_len=16,
        obs_steps=1,
        pred_len=120,
        pred_step=5,
        delim=" ",
    )
    if (
        len(dataset) != int(expected["expected_train_scenes"])
        or int(dataset.obs_traj.shape[0]) != int(expected["expected_train_actors"])
    ):
        raise RuntimeError("C134 P0-B train cohort identity mismatch")
    dates = json.loads(
        (ROOT / str(expected["train_scene_dates"])).read_text(encoding="utf-8")
    )["dates"]
    folds = json.loads(
        (ROOT / str(expected["date_folds"])).read_text(encoding="utf-8")
    )
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    fold_results = {}
    total_actors = 0
    total_duplicate_actors = 0
    physical_violations = 0
    maximum_kkt_residual = 0.0
    reverse_prefix_unchanged = True
    all_finite = True
    for fold in (0, 1, 2):
        control, control_path = _control(design, fold)
        _, validation_indices, _ = indices_for_fold(dates, folds, fold)
        loader = _loader(
            Subset(dataset, validation_indices),
            workers=workers,
            prefetch=prefetch,
            batch_size=batch_size,
        )
        minade_chunks = []
        minfde_chunks = []
        fold_actors = 0
        fold_duplicates = 0
        with torch.no_grad():
            for batch_index, data in enumerate(loader):
                data = move(data, device)
                history = data["obs_traj"].transpose(1, 0)
                target = data["pred_traj"].transpose(1, 0)
                center, yaw, pitch = pose_from_history(history)
                candidates, auxiliary = continuous_physical_oracle(
                    target, center, yaw, pitch
                )
                displacement = torch.linalg.vector_norm(
                    candidates - target[:, None], dim=-1
                )
                minade = displacement.mean(dim=-1).min(dim=1).values
                minfde = displacement[..., -1].min(dim=1).values
                minade_chunks.append(minade.cpu().numpy())
                minfde_chunks.append(minfde.cpu().numpy())
                pairwise = torch.linalg.vector_norm(
                    candidates[:, :, None] - candidates[:, None, :], dim=-1
                ).mean(dim=-1)
                eye = torch.eye(5, device=device, dtype=torch.bool)[None]
                duplicate = (pairwise.masked_fill(eye, math.inf) < float(
                    design["operational_checks"][
                        "duplicate_pair_mean_trajectory_distance_below"
                    ]
                )).any(dim=(1, 2))
                parameters = auxiliary["flight_parameters"]
                violation = physical_violation_mask(parameters, candidates)
                physical_violations += int(violation.sum())
                all_finite = all_finite and bool(torch.isfinite(displacement).all())
                maximum_kkt_residual = max(
                    maximum_kkt_residual,
                    float(auxiliary["kkt_residual"].max()),
                )
                fold_duplicates += int(duplicate.sum())
                fold_actors += int(target.shape[0])

                if batch_index == 0:
                    altered = torch.cat(
                        (history[:, :-2].flip(1), history[:, -2:]), dim=1
                    )
                    altered_pose = pose_from_history(altered)
                    altered_candidates, _ = continuous_physical_oracle(
                        target, *altered_pose
                    )
                    reverse_prefix_unchanged = reverse_prefix_unchanged and torch.equal(
                        candidates, altered_candidates
                    )

        minade_array = np.concatenate(minade_chunks)
        minfde_array = np.concatenate(minfde_chunks)
        metrics = {
            "actors": fold_actors,
            "minade": float(minade_array.mean()),
            "minfde": float(minfde_array.mean()),
            "minfde_p95": float(np.quantile(minfde_array, 0.95)),
            "duplicate_actor_rate": fold_duplicates / fold_actors,
        }
        baseline = control["validation_metrics"]["overall"]
        comparisons = {
            "minade_relative_gain_vs_C129": 1.0
            - metrics["minade"] / float(baseline["minade"]),
            "minfde_relative_gain_vs_C129": 1.0
            - metrics["minfde"] / float(baseline["minfde"]),
            "minfde_p95_ratio_vs_C129": metrics["minfde_p95"]
            / float(baseline["minfde_p95"]),
        }
        fold_results[str(fold)] = {
            "validation_scenes": len(validation_indices),
            "metrics": metrics,
            "C129_control": {
                "summary": control_path.relative_to(ROOT).as_posix(),
                "summary_sha256": sha256(control_path),
                "metrics": {
                    key: float(baseline[key])
                    for key in ("minade", "minfde", "minfde_p95")
                },
            },
            "comparisons": comparisons,
        }
        total_actors += fold_actors
        total_duplicate_actors += fold_duplicates

    required_gain = float(
        design["metric_gate"]["every_fold_minade_relative_gain_vs_C129_at_least"]
    )
    p95_factor = float(
        design["metric_gate"][
            "every_fold_minfde_p95_not_worse_than_C129_factor"
        ]
    )
    aggregate_duplicate_rate = total_duplicate_actors / total_actors
    checks = {
        "every_fold_minade_gain_at_least_1pct": all(
            row["comparisons"]["minade_relative_gain_vs_C129"] >= required_gain
            for row in fold_results.values()
        ),
        "every_fold_minfde_gain_at_least_1pct": all(
            row["comparisons"]["minfde_relative_gain_vs_C129"] >= required_gain
            for row in fold_results.values()
        ),
        "every_fold_minfde_p95_within_guard": all(
            row["comparisons"]["minfde_p95_ratio_vs_C129"] <= p95_factor
            for row in fold_results.values()
        ),
        "physical_violation_count_zero": physical_violations == 0,
        "duplicate_actor_rate_below_1pct": aggregate_duplicate_rate
        < float(design["operational_checks"]["duplicate_actor_rate_below"]),
        "reverse_history_prefix_check": reverse_prefix_unchanged,
        "kkt_residual_within_frozen_tolerance": maximum_kkt_residual
        <= float(
            design["operational_checks"]["kkt_max_absolute_residual_at_most"]
        ),
        "all_metrics_finite": all_finite
        and all(
            math.isfinite(float(value))
            for row in fold_results.values()
            for section in (row["metrics"], row["comparisons"])
            for value in section.values()
        ),
    }
    passed = all(checks.values())
    result = {
        "format_version": 1,
        "cycle": protocol.payload["cycle"],
        "phase": "P0_B_continuous_physical_action_capacity",
        "protocol_sha256": protocol_hash,
        "p0_a_decision_sha256": sha256(p0_a_path),
        "p0_b_design": DESIGN_PATH.relative_to(ROOT).as_posix(),
        "p0_b_design_sha256": sha256(DESIGN_PATH),
        "free_knots": FREE_KNOTS,
        "folds": fold_results,
        "aggregate": {
            "actors": total_actors,
            "duplicate_actor_rate": aggregate_duplicate_rate,
            "physical_violation_count": physical_violations,
            "maximum_kkt_residual": maximum_kkt_residual,
        },
        "checks": checks,
        "passed": passed,
        "decision": "P1_FOLD0_AUTHORIZED" if passed else "C134_CLOSED_P0_B_FAILED",
        "training_performed": False,
        "diagnostic_target_used": True,
        "target_used_for_deployable_inference": False,
        "adaptive_model_or_parameter_selection": False,
        "locked_test_used": False,
        "development_used": False,
        "claim_boundary": protocol.payload["claim_boundary"],
    }
    output = ARTIFACT_ROOT / "p0_b_decision.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.device, args.num_workers, args.prefetch_factor, args.batch_size),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
