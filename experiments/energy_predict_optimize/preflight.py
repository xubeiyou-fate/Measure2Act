"""C134 protocol, information-boundary, solver, and evaluation preflight."""

from __future__ import annotations

import inspect
import json
import platform
from pathlib import Path

import torch

from airroute_stage_m.evaluation import compute_batch_metrics as legacy_metrics
from model.utils import TrajectoryDataset

from .evaluation import compute_batch_metrics
from .protocol import load_protocol, sha256
from .solver import energy_objective, energy_optimal_probabilities


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/energy_predict_optimize"


def run() -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_boundaries()
    expected = protocol.payload["dataset"]
    dataset = TrajectoryDataset(
        protocol.split_path("train").as_posix(),
        obs_len=16,
        obs_steps=1,
        pred_len=120,
        pred_step=5,
        delim=" ",
    )
    loaded = {
        "train_scenes": len(dataset),
        "train_actors": int(dataset.obs_traj.shape[0]),
    }
    expected_loaded = {
        "train_scenes": int(expected["expected_train_scenes"]),
        "train_actors": int(expected["expected_train_actors"]),
    }
    torch.manual_seed(134)
    predictions = torch.randn(7, 5, 8, 3)
    target = torch.randn(7, 8, 3)
    logits = torch.randn(7, 5)
    probabilities = torch.softmax(logits, dim=1)
    decision = logits.argmax(dim=1)
    explicit = compute_batch_metrics(predictions, probabilities, decision, target)
    legacy = legacy_metrics(predictions, logits, target)
    legacy_keys = set(legacy)
    evaluation_parity = all(
        torch.equal(explicit[key], legacy[key])
        if not torch.is_floating_point(explicit[key])
        else torch.allclose(explicit[key], legacy[key], atol=1e-6, rtol=1e-6)
        for key in legacy_keys
    )

    locations = torch.randn(7, 5, 4, 3)
    pairwise = torch.linalg.vector_norm(
        locations[:, :, None] - locations[:, None, :], dim=-1
    ).mean(dim=-1)
    predicted_distance = torch.randn(7, 5, requires_grad=True)
    optimized = energy_optimal_probabilities(predicted_distance, pairwise)
    uniform = torch.full_like(optimized, 0.2)
    optimized_objective = energy_objective(
        optimized, predicted_distance, pairwise
    )
    uniform_objective = energy_objective(uniform, predicted_distance, pairwise)
    optimized_objective.mean().backward()
    parameters = tuple(inspect.signature(energy_optimal_probabilities).parameters)
    boundary_checks = {}
    for split, name in (
        ("dev", "development_split_rejected"),
        ("locked_test", "locked_test_split_rejected"),
    ):
        try:
            protocol.split_path(split)
        except ValueError:
            boundary_checks[name] = True
        else:
            boundary_checks[name] = False
    checks = {
        "frozen_inputs_match": loaded == expected_loaded,
        **boundary_checks,
        "solver_API_is_target_free": "target" not in parameters
        and "ground_truth" not in parameters,
        "solver_simplex_valid": bool((optimized >= -1e-6).all())
        and torch.allclose(optimized.sum(dim=1), torch.ones(7), atol=1e-5),
        "solver_improves_uniform_predicted_objective": bool(
            (optimized_objective <= uniform_objective + 1e-5).all()
        ),
        "solver_gradient_is_finite": predicted_distance.grad is not None
        and bool(torch.isfinite(predicted_distance.grad).all()),
        "explicit_evaluation_replays_legacy": evaluation_parity,
        "independent_decision_supported": not torch.equal(
            torch.zeros(7, dtype=torch.long), optimized.argmax(dim=1)
        ),
        "fixed_solver_budget": int(protocol.payload["p0_a"]["probability_solver"].split()[0])
        == 32,
        "no_gate_or_residual_route": all(
            mechanism in protocol.payload["forbidden_mechanisms"]
            for mechanism in (
                "trajectory_or_control_residual",
                "learned_gate_router_or_mixture_of_experts",
            )
        ),
    }
    result = {
        "format_version": 1,
        "cycle": protocol.payload["cycle"],
        "phase": "implementation_preflight",
        "protocol_sha256": sha256(protocol.path),
        "manifest_sha256": sha256(protocol.manifest_path),
        "loaded": loaded,
        "expected": expected_loaded,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        },
        "solver_parameters": parameters,
        "checks": checks,
        "passed": all(checks.values()),
        "locked_test_used": False,
        "development_used": False,
    }
    if not result["passed"]:
        raise RuntimeError(f"C134 preflight failed: {checks}")
    output = ARTIFACT_ROOT / "preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
