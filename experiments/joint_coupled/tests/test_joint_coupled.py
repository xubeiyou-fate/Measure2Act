from __future__ import annotations

import pytest
import torch

from experiments.joint_coupled.model import VARIANT, ascent_config, build_model
from experiments.joint_coupled.objective import joint_coupled_dual_objective
from experiments.joint_coupled.protocol import load_protocol


def batch(batch_size: int = 3) -> dict[str, torch.Tensor]:
    torch.manual_seed(129)
    return {"obs_traj": torch.cumsum(torch.randn(16, batch_size, 3) * 0.03, dim=0)}


def test_native_shape_and_forbidden_mechanisms() -> None:
    model = build_model(batch_size=3)
    prediction, logits, auxiliary = model(batch())
    assert VARIANT == "J1_joint_coupled_dual"
    assert prediction.shape == (3, 5, 24, 3)
    assert logits.shape == (3, 5)
    decoder = auxiliary["kinematic_decoder"]
    assert decoder["trajectory_residual"] is False
    assert decoder["learned_gate"] is False
    assert ascent_config()["token_codebook"] is False
    assert ascent_config()["future_autoregression"] is False
    assert ascent_config()["post_generation_selector"] is False


def test_joint_objective_routes_distinct_geometry_and_score_targets() -> None:
    predictions = torch.randn(2, 5, 4, 3, requires_grad=True)
    logits = torch.randn(2, 5, requires_grad=True)
    target = torch.randn(2, 4, 3)
    loss, diagnostics = joint_coupled_dual_objective(predictions, logits, target)
    assert torch.isfinite(loss)
    assert diagnostics["score_winner"].requires_grad is False
    assert diagnostics["ade_winner"].requires_grad is False
    assert diagnostics["fde_winner"].requires_grad is False
    loss.backward()
    assert predictions.grad is not None and torch.isfinite(predictions.grad).all()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_score_only_gradient_reaches_shared_parameters() -> None:
    model = build_model(batch_size=3)
    prediction, logits, _ = model(batch())
    target = torch.randn(3, 24, 3)
    _, diagnostics = joint_coupled_dual_objective(prediction, logits, target)
    score_loss = torch.nn.functional.cross_entropy(logits, diagnostics["score_winner"])
    score_loss.backward()
    pi_grad = sum(
        float(parameter.grad.abs().sum())
        for name, parameter in model.named_parameters()
        if name.startswith("pi.") and parameter.grad is not None
    )
    shared_grad = sum(
        float(parameter.grad.abs().sum())
        for name, parameter in model.named_parameters()
        if not name.startswith("pi.") and parameter.grad is not None
    )
    assert pi_grad > 0
    assert shared_grad > 0


def test_protocol_rejects_non_train_split() -> None:
    protocol = load_protocol()
    protocol.assert_boundaries()
    with pytest.raises(ValueError):
        protocol.split_path("dev")
    with pytest.raises(ValueError):
        protocol.split_path("locked_test")
