"""C134 P0-A target-free fixed-geometry probability audit."""

from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from experiments.metric_exact.folds import indices_for_fold
from experiments.joint_coupled.train import move, set_seed
from experiments.dual_expected_risk.model import build_model as build_c130_model
from experiments.dual_expected_risk.objective import ADE_SCALE
from model.utils import TrajectoryDataset, seed_worker, seq_collate

from .evaluation import RankingMetricAccumulator, compute_batch_metrics
from .protocol import load_protocol, sha256
from .solver import energy_objective, energy_optimal_probabilities


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/energy_predict_optimize"


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


def _control_summary(protocol, family: str) -> tuple[dict[str, object], Path]:
    entry = protocol.payload["controls"][family]
    summary_path = ROOT / str(entry["fold0_summary"])
    if sha256(summary_path) != str(entry["fold0_summary_sha256"]):
        raise RuntimeError(f"C134 {family} summary hash mismatch")
    protocol_path = ROOT / str(entry["protocol"])
    if sha256(protocol_path) != str(entry["protocol_sha256"]):
        raise RuntimeError(f"C134 {family} protocol hash mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("complete") is not True
        or summary.get("formal") is not True
        or summary.get("fold") != 0
        or summary.get("seed") != 42
        or summary.get("fixed_final_epoch") != 20
        or summary.get("locked_test_used") is not False
    ):
        raise RuntimeError(f"C134 {family} summary identity mismatch")
    return summary, summary_path


def _overall(summary: dict[str, object]) -> dict[str, object]:
    return summary["validation_metrics"]["overall"]


def _finite(metrics: dict[str, object]) -> bool:
    return all(
        isinstance(value, (int, float)) and math.isfinite(float(value))
        for value in metrics.values()
    )


def run(
    device_name: str, workers: int, prefetch: int, batch_size: int
) -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_boundaries()
    protocol_hash = sha256(protocol.path)
    set_seed(42)

    control_payloads = {}
    for family in ("C127_B0", "C129_J1", "C130_R1", "C131_R2", "C133_D1"):
        summary, summary_path = _control_summary(protocol, family)
        control_payloads[family] = {
            "summary": summary_path.relative_to(ROOT).as_posix(),
            "summary_sha256": sha256(summary_path),
            "metrics": _overall(summary),
        }

    checkpoint_path = protocol.control_path("C130_R1", "fold0_checkpoint")
    checkpoint_hash = str(
        protocol.payload["controls"]["C130_R1"]["fold0_checkpoint_sha256"]
    )
    if sha256(checkpoint_path) != checkpoint_hash:
        raise RuntimeError("C134 C130 checkpoint hash mismatch")

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
        raise RuntimeError("C134 P0-A train cohort identity mismatch")
    dates = json.loads(
        (ROOT / str(expected["train_scene_dates"])).read_text(encoding="utf-8")
    )["dates"]
    folds = json.loads(
        (ROOT / str(expected["date_folds"])).read_text(encoding="utf-8")
    )
    _, validation_indices, _ = indices_for_fold(dates, folds, 0)
    validation = Subset(dataset, validation_indices)
    loader = _loader(
        validation, workers=workers, prefetch=prefetch, batch_size=batch_size
    )

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but PyTorch cannot use CUDA")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = build_c130_model(batch_size=batch_size).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    accumulators = {
        "native_C130_softmax": RankingMetricAccumulator(),
        "C134_energy_optimal": RankingMetricAccumulator(),
    }
    maximum_simplex_error = 0.0
    minimum_probability = 1.0
    maximum_predicted_objective_violation = -math.inf
    decision_probability_argmax_disagreements = 0
    actors = 0
    with torch.no_grad():
        for data in loader:
            data = move(data, device)
            target = data["pred_traj"].transpose(1, 0)
            predictions, logits, auxiliary = model(data)
            centered_risks = auxiliary["centered_risk_predictions"]
            predicted_target_distance = centered_risks[..., 0] * ADE_SCALE
            pairwise_distance = torch.linalg.vector_norm(
                predictions[:, :, None] - predictions[:, None, :], dim=-1
            ).mean(dim=-1)
            probabilities = energy_optimal_probabilities(
                predicted_target_distance, pairwise_distance
            )
            native_probabilities = torch.softmax(logits, dim=1)
            decision_mode = centered_risks.sum(dim=-1).argmin(dim=1)
            if not torch.equal(decision_mode, logits.argmax(dim=1)):
                raise RuntimeError("C134 independent decision does not replay C130 top1")

            uniform = torch.full_like(probabilities, 1.0 / probabilities.shape[1])
            optimized_objective = energy_objective(
                probabilities, predicted_target_distance, pairwise_distance
            )
            uniform_objective = energy_objective(
                uniform, predicted_target_distance, pairwise_distance
            )
            maximum_predicted_objective_violation = max(
                maximum_predicted_objective_violation,
                float((optimized_objective - uniform_objective).max()),
            )
            maximum_simplex_error = max(
                maximum_simplex_error,
                float((probabilities.sum(dim=1) - 1.0).abs().max()),
            )
            minimum_probability = min(minimum_probability, float(probabilities.min()))
            decision_probability_argmax_disagreements += int(
                (decision_mode != probabilities.argmax(dim=1)).sum()
            )
            actors += int(probabilities.shape[0])
            accumulators["native_C130_softmax"].update(
                compute_batch_metrics(
                    predictions, native_probabilities, decision_mode, target
                )
            )
            accumulators["C134_energy_optimal"].update(
                compute_batch_metrics(predictions, probabilities, decision_mode, target)
            )

    metrics = {name: accumulator.summarize() for name, accumulator in accumulators.items()}
    c130 = control_payloads["C130_R1"]["metrics"]
    b0 = control_payloads["C127_B0"]["metrics"]
    replay_keys = (
        "top1_ade",
        "top1_fde",
        "minade",
        "minfde",
        "energy_score",
        "nll",
        "brier",
        "ece",
    )
    replay_tolerance = float(
        protocol.payload["p0_a"]["gate"]["native_metric_replay_absolute_tolerance"]
    )
    replay_errors = {
        key: abs(float(metrics["native_C130_softmax"][key]) - float(c130[key]))
        for key in replay_keys
    }
    energy_factor = float(
        protocol.payload["p0_a"]["gate"]["energy_not_worse_than_C127_B0_factor"]
    )
    objective_tolerance = float(
        protocol.payload["p0_a"]["gate"][
            "per_batch_objective_not_worse_than_uniform_tolerance"
        ]
    )
    simplex_tolerance = float(
        protocol.payload["p0_a"]["gate"]["simplex_tolerance"]
    )
    solver_parameters = tuple(inspect.signature(energy_optimal_probabilities).parameters)
    checks = {
        "native_C130_metrics_replay": all(
            error <= replay_tolerance for error in replay_errors.values()
        ),
        "C134_top1_replays_C130": all(
            abs(float(metrics["C134_energy_optimal"][key]) - float(c130[key]))
            <= replay_tolerance
            for key in ("top1_ade", "top1_fde", "minade", "minfde")
        ),
        "C134_energy_within_1pct_of_C127_B0": (
            float(metrics["C134_energy_optimal"]["energy_score"])
            <= float(b0["energy_score"]) * energy_factor
        ),
        "all_metrics_finite": all(_finite(family) for family in metrics.values()),
        "solver_API_has_no_target": "target" not in solver_parameters
        and "ground_truth" not in solver_parameters,
        "simplex_valid": maximum_simplex_error <= simplex_tolerance
        and minimum_probability >= -simplex_tolerance,
        "predicted_objective_not_worse_than_uniform": (
            maximum_predicted_objective_violation <= objective_tolerance
        ),
    }
    passed = all(checks.values())
    result = {
        "format_version": 1,
        "cycle": protocol.payload["cycle"],
        "phase": "P0_A_target_free_fixed_geometry_probability_audit",
        "protocol_sha256": protocol_hash,
        "fold": 0,
        "seed": 42,
        "validation_scenes": len(validation),
        "validation_actors": actors,
        "checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": checkpoint_hash,
        "metrics": metrics,
        "controls": control_payloads,
        "diagnostics": {
            "solver_parameters": solver_parameters,
            "maximum_simplex_error": maximum_simplex_error,
            "minimum_probability": minimum_probability,
            "maximum_predicted_objective_violation_vs_uniform": maximum_predicted_objective_violation,
            "decision_probability_argmax_disagreement_rate": (
                decision_probability_argmax_disagreements / actors
            ),
            "native_replay_absolute_errors": replay_errors,
            "C127_B0_energy_threshold": float(b0["energy_score"]) * energy_factor,
        },
        "checks": checks,
        "passed": passed,
        "decision": "P0_B_AUTHORIZED" if passed else "C134_CLOSED_P0_A_FAILED",
        "training_performed": False,
        "adaptive_model_or_parameter_selection": False,
        "oracle_probabilities_used": False,
        "locked_test_used": False,
        "development_used": False,
        "claim_boundary": protocol.payload["claim_boundary"],
    }
    output = ARTIFACT_ROOT / "p0_a_decision.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.device, args.num_workers, args.prefetch_factor, args.batch_size),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
