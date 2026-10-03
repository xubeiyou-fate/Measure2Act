from __future__ import annotations

import inspect

import pytest
import torch

from airroute_stage_m.evaluation import compute_batch_metrics as legacy_metrics
from experiments.energy_predict_optimize.evaluation import compute_batch_metrics
from experiments.energy_predict_optimize.diagnose_physical import physical_violation_mask
from experiments.energy_predict_optimize.physical_oracle import (
    FREE_KNOTS,
    continuous_physical_oracle,
    piecewise_linear_basis,
    pose_from_history,
)
from experiments.energy_predict_optimize.model import build_model
from experiments.energy_predict_optimize.objective import energy_predict_optimize_objective
from experiments.energy_predict_optimize.protocol import load_protocol
from experiments.energy_predict_optimize.solver import (
    energy_objective,
    energy_optimal_probabilities,
)


def pairwise(batch: int = 3) -> torch.Tensor:
    torch.manual_seed(134)
    trajectories = torch.randn(batch, 5, 6, 3)
    return torch.linalg.vector_norm(
        trajectories[:, :, None] - trajectories[:, None, :], dim=-1
    ).mean(dim=-1)


def test_solver_is_target_free_and_stays_on_simplex() -> None:
    parameters = inspect.signature(energy_optimal_probabilities).parameters
    assert "target" not in parameters
    predicted = torch.randn(3, 5)
    probabilities = energy_optimal_probabilities(predicted, pairwise())
    assert torch.all(probabilities >= -1e-6)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(3), atol=1e-5)


def test_solver_never_worsens_uniform_predicted_objective() -> None:
    predicted = torch.tensor(
        [[0.2, 0.3, 0.5, 0.4, 0.8], [0.9, 0.1, 0.6, 0.3, 0.7]]
    )
    distances = pairwise(2)
    probabilities = energy_optimal_probabilities(predicted, distances)
    uniform = torch.full_like(probabilities, 0.2)
    assert torch.all(
        energy_objective(probabilities, predicted, distances)
        <= energy_objective(uniform, predicted, distances) + 1e-5
    )


def test_solver_has_finite_gradient_for_zero_curvature_geometry() -> None:
    predicted = torch.randn(3, 5, requires_grad=True)
    distances = torch.zeros(3, 5, 5)
    probabilities = energy_optimal_probabilities(predicted, distances)
    loss = energy_objective(probabilities, predicted, distances).mean()
    loss.backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()


def test_solver_is_invariant_to_per_actor_additive_risk_shift() -> None:
    predicted = torch.randn(3, 5)
    distances = pairwise()
    shifted = predicted + torch.tensor([[100.0], [-50.0], [17.0]])
    first = energy_optimal_probabilities(predicted, distances)
    second = energy_optimal_probabilities(shifted, distances)
    assert torch.allclose(first, second, atol=2e-5, rtol=1e-5)


def test_explicit_evaluation_replays_legacy_when_inputs_are_coupled() -> None:
    torch.manual_seed(7)
    predictions = torch.randn(4, 5, 8, 3)
    logits = torch.randn(4, 5)
    target = torch.randn(4, 8, 3)
    expected = legacy_metrics(predictions, logits, target)
    actual = compute_batch_metrics(
        predictions, torch.softmax(logits, dim=1), logits.argmax(dim=1), target
    )
    assert set(actual) == set(expected)
    for key in expected:
        assert torch.allclose(actual[key], expected[key], atol=1e-6, rtol=1e-6)


def test_nll_floor_does_not_clip_valid_small_probabilities() -> None:
    torch.manual_seed(1340)
    predictions = torch.randn(2, 5, 4, 3)
    target = torch.randn(2, 4, 3)
    logits = torch.tensor([[0.0, -40.0, -20.0, -10.0, -5.0], [40.0, 0.0, -5.0, -10.0, -20.0]])
    expected = legacy_metrics(predictions, logits, target)
    actual = compute_batch_metrics(
        predictions, torch.softmax(logits, dim=1), logits.argmax(dim=1), target
    )
    assert torch.allclose(actual["nll"], expected["nll"], atol=1e-5, rtol=1e-6)


def test_decision_index_can_differ_from_probability_argmax() -> None:
    predictions = torch.zeros(1, 5, 2, 3)
    predictions[0, :, :, 0] = torch.arange(5)[:, None]
    target = torch.zeros(1, 2, 3)
    probabilities = torch.tensor([[0.05, 0.65, 0.1, 0.1, 0.1]])
    metrics = compute_batch_metrics(
        predictions, probabilities, torch.tensor([0]), target
    )
    assert metrics["top1_mode"].item() == 0
    assert probabilities.argmax(dim=1).item() == 1
    assert metrics["top1_ade"].item() == 0.0


def test_protocol_rejects_non_train_splits() -> None:
    protocol = load_protocol()
    protocol.assert_boundaries()
    with pytest.raises(ValueError):
        protocol.split_path("dev")
    with pytest.raises(ValueError):
        protocol.split_path("locked_test")


def test_physical_oracle_has_five_absolute_control_bases() -> None:
    torch.manual_seed(1341)
    history = torch.cumsum(torch.randn(3, 16, 3) * 0.02, dim=1)
    target = history[:, -1, None] + torch.cumsum(
        torch.randn(3, 24, 3) * 0.02, dim=1
    )
    center, yaw, pitch = pose_from_history(history)
    candidates, auxiliary = continuous_physical_oracle(
        target, center, yaw, pitch
    )
    assert candidates.shape == (3, 5, 24, 3)
    assert auxiliary["flight_parameters"].shape == (3, 5, 24, 3)
    assert tuple(auxiliary["free_knots"].tolist()) == FREE_KNOTS
    assert torch.all(auxiliary["flight_parameters"][..., 0] >= 0)
    assert torch.all(auxiliary["flight_parameters"][..., 2].abs() <= torch.pi / 2)
    assert torch.all(auxiliary["kkt_residual"] <= 1e-6)
    assert torch.isfinite(candidates).all()


def test_piecewise_basis_keeps_zero_origin_fixed() -> None:
    basis = piecewise_linear_basis(
        24, 4, device=torch.device("cpu"), dtype=torch.float64
    )
    assert basis.shape == (24, 4)
    assert basis[0].sum() < 1.0
    assert torch.allclose(basis[-1].sum(), torch.tensor(1.0, dtype=torch.float64))


def test_reversing_unused_history_prefix_leaves_oracle_unchanged() -> None:
    torch.manual_seed(1342)
    history = torch.cumsum(torch.randn(2, 16, 3) * 0.02, dim=1)
    altered = torch.cat((history[:, :-2].flip(1), history[:, -2:]), dim=1)
    target = history[:, -1, None] + torch.cumsum(
        torch.randn(2, 24, 3) * 0.02, dim=1
    )
    first, _ = continuous_physical_oracle(target, *pose_from_history(history))
    second, _ = continuous_physical_oracle(target, *pose_from_history(altered))
    assert torch.equal(first, second)


def test_physical_violation_mask_reduces_the_coordinate_axis() -> None:
    parameters = torch.zeros(2, 5, 24, 3)
    candidates = torch.zeros_like(parameters)
    assert physical_violation_mask(parameters, candidates).shape == (2, 5, 24)
    assert not physical_violation_mask(parameters, candidates).any()
    parameters[0, 1, 2, 0] = -1.0
    candidates[1, 2, 3, 1] = torch.nan
    mask = physical_violation_mask(parameters, candidates)
    assert mask[0, 1, 2]
    assert mask[1, 2, 3]


def test_e1_model_separates_probabilities_and_decision() -> None:
    torch.manual_seed(1343)
    model = build_model(batch_size=3)
    data = {"obs_traj": torch.cumsum(torch.randn(16, 3, 3) * 0.03, dim=0)}
    predictions, probabilities, decision, auxiliary = model(data)
    assert predictions.shape == (3, 5, 24, 3)
    assert probabilities.shape == (3, 5)
    assert decision.shape == (3,)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(3), atol=1e-5)
    assert torch.equal(decision, auxiliary["decision_logits"].argmax(dim=1))
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())


def test_e1_objective_updates_only_the_cost_operator() -> None:
    torch.manual_seed(1344)
    model = build_model(batch_size=2).train()
    data = {"obs_traj": torch.cumsum(torch.randn(16, 2, 3) * 0.03, dim=0)}
    target = torch.cumsum(torch.randn(2, 24, 3) * 0.03, dim=1)
    predictions, probabilities, _, auxiliary = model(data)
    loss, diagnostics = energy_predict_optimize_objective(
        predictions,
        probabilities,
        auxiliary["predicted_normalized_ade_risk"],
        target,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert not diagnostics["target_normalized_ade_risk"].requires_grad
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.energy_cost_operator.parameters()
    )
