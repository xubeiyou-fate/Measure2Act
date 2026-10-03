from __future__ import annotations

import pytest
import torch

from experiments.decision_regret.model import VARIANT, ascent_config, build_model
from experiments.decision_regret.objective import (
    decision_cost_targets,
    decision_regret_objective,
    per_mode_errors,
    spo_plus_loss,
)
from experiments.decision_regret.protocol import load_protocol
from experiments.decision_regret.summarize import _fold_gate, _metrics


def batch(batch_size: int = 3) -> dict[str, torch.Tensor]:
    torch.manual_seed(133)
    return {"obs_traj": torch.cumsum(torch.randn(16, batch_size, 3) * 0.03, dim=0)}


def test_model_shapes_fixed_cost_and_forbidden_mechanisms() -> None:
    model = build_model(batch_size=3)
    predictions, logits, auxiliary = model(batch())
    assert VARIANT == "D1_native_decision_regret"
    assert predictions.shape == (3, 5, 24, 3)
    assert logits.shape == (3, 5)
    assert auxiliary["decision_costs"].shape == (3, 5)
    assert torch.allclose(
        logits, -auxiliary["centered_decision_costs"], atol=1e-6, rtol=1e-6
    )
    decoder = auxiliary["kinematic_decoder"]
    assert decoder["trajectory_residual"] is False
    assert decoder["control_residual"] is False
    assert decoder["learned_gate"] is False
    assert ascent_config()["token_codebook"] is False
    assert ascent_config()["future_autoregression"] is False
    assert ascent_config()["post_generation_selector"] is False


def test_spo_plus_is_shift_invariant_and_not_mse() -> None:
    costs = torch.tensor([[0.2, 1.0, 2.0, 3.0, 4.0]])
    predicted = torch.tensor([[0.0, 0.3, 0.4, 0.5, 0.6]], requires_grad=True)
    loss, winner = spo_plus_loss(predicted, costs)
    shifted, shifted_winner = spo_plus_loss(predicted + 17.0, costs)
    assert winner.tolist() == [0]
    assert torch.equal(winner, shifted_winner)
    assert torch.allclose(loss, shifted)
    assert not torch.allclose(loss, torch.nn.functional.mse_loss(predicted, costs))
    loss.backward()
    assert predicted.grad is not None and torch.isfinite(predicted.grad).all()


def test_objective_routes_geometry_and_decision_gradients() -> None:
    predictions = torch.randn(2, 5, 4, 3, requires_grad=True)
    costs = torch.randn(2, 5, requires_grad=True)
    centered = costs - costs.mean(dim=1, keepdim=True)
    logits = -centered
    target = torch.randn(2, 4, 3)
    loss, diagnostics = decision_regret_objective(
        predictions, logits, costs, target
    )
    assert torch.isfinite(loss)
    assert diagnostics["decision_cost_target"].requires_grad is False
    loss.backward()
    assert predictions.grad is not None and torch.isfinite(predictions.grad).all()
    assert costs.grad is not None and torch.isfinite(costs.grad).all()


def test_true_cost_uses_fixed_dual_metric_scales() -> None:
    ade = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    fde = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
    target = decision_cost_targets(ade, fde, ade_scale=1.0, fde_scale=1.0)
    assert target.argmin(dim=1).item() == 0
    assert not target.requires_grad


def test_exact_geometry_uses_independent_ade_and_fde_winners() -> None:
    predictions = torch.zeros(1, 5, 2, 3)
    target = torch.zeros(1, 2, 3)
    predictions[0, 0, 0, 0] = 0.0
    predictions[0, 0, 1, 0] = 2.0
    predictions[0, 1, 0, 0] = 3.0
    predictions[0, 1, 1, 0] = 0.0
    predictions[0, 2:] = 9.0
    ade, fde = per_mode_errors(predictions, target)
    assert ade.argmin(dim=1).item() == 0
    assert fde.argmin(dim=1).item() == 1


def test_protocol_rejects_non_train_splits() -> None:
    protocol = load_protocol()
    protocol.assert_boundaries()
    with pytest.raises(ValueError):
        protocol.split_path("dev")
    with pytest.raises(ValueError):
        protocol.split_path("locked_test")


def test_summary_reads_effective_modes_from_winner_distribution() -> None:
    overall = {
        "agents": 10,
        "top1_ade": 0.5,
        "top1_fde": 1.0,
        "minade": 0.2,
        "minfde": 0.4,
        "energy_score": 0.3,
        "oracle_ade_rank1": 0.2,
        "oracle_fde_rank1": 0.2,
        "minfde_p95": 1.2,
        "tail_minfde": 0.6,
        "winner_distribution": {"effective_modes": 4.8},
    }
    values = _metrics({"validation_metrics": {"overall": overall}})
    assert values["effective_modes"] == 4.8


def test_fold_gate_rejects_energy_failure_even_when_top1_passes() -> None:
    b0 = {
        "top1_ade": 1.0,
        "top1_fde": 2.0,
        "minade": 0.5,
        "minfde": 0.8,
        "energy_score": 0.4,
        "minfde_p95": 1.5,
        "tail_minfde": 0.7,
        "effective_modes": 4.9,
    }
    c129 = {**b0, "minade": 0.45, "minfde": 0.7}
    candidate = {
        **b0,
        "top1_ade": 0.8,
        "top1_fde": 1.7,
        "minade": 0.45,
        "minfde": 0.7,
        "energy_score": 0.48,
    }
    gate = _fold_gate(candidate, b0, c129)
    assert gate["top1_fde_relative_gain_at_least"] is True
    assert gate["energy_within_b0_guard"] is False
