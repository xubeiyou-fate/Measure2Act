from __future__ import annotations

import pytest
import torch

from experiments.dual_expected_risk.model import VARIANT, ascent_config, build_model
from experiments.dual_expected_risk.objective import (
    centered_dual_risk_targets,
    dual_expected_risk_objective,
    per_mode_errors,
)
from experiments.dual_expected_risk.protocol import load_protocol


def batch(batch_size: int = 3) -> dict[str, torch.Tensor]:
    torch.manual_seed(130)
    return {"obs_traj": torch.cumsum(torch.randn(16, batch_size, 3) * 0.03, dim=0)}


def test_model_shapes_fixed_score_and_forbidden_mechanisms() -> None:
    model = build_model(batch_size=3)
    predictions, logits, auxiliary = model(batch())
    assert VARIANT == "R1_dual_expected_risk"
    assert predictions.shape == (3, 5, 24, 3)
    assert logits.shape == (3, 5)
    assert auxiliary["risk_predictions"].shape == (3, 5, 2)
    assert torch.allclose(
        logits, -auxiliary["centered_risk_predictions"].sum(dim=-1)
    )
    decoder = auxiliary["kinematic_decoder"]
    assert decoder["trajectory_residual"] is False
    assert decoder["control_residual"] is False
    assert decoder["learned_gate"] is False
    assert ascent_config()["token_codebook"] is False
    assert ascent_config()["future_autoregression"] is False
    assert ascent_config()["post_generation_selector"] is False


def test_centering_is_shift_invariant_and_preserves_mode_order() -> None:
    ade = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    fde = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
    target = centered_dual_risk_targets(ade, fde, ade_scale=1.0, fde_scale=1.0)
    shifted = centered_dual_risk_targets(
        ade + 100.0, fde - 50.0, ade_scale=1.0, fde_scale=1.0
    )
    assert torch.allclose(target, shifted)
    assert torch.equal(target[..., 0].argsort(dim=1), ade.argsort(dim=1))
    assert torch.equal(target[..., 1].argsort(dim=1), fde.argsort(dim=1))


def test_objective_routes_geometry_and_full_risk_gradients() -> None:
    predictions = torch.randn(2, 5, 4, 3, requires_grad=True)
    raw_risks = torch.randn(2, 5, 2, requires_grad=True)
    centered = raw_risks - raw_risks.mean(dim=1, keepdim=True)
    logits = -centered.sum(dim=-1)
    target = torch.randn(2, 4, 3)
    loss, diagnostics = dual_expected_risk_objective(
        predictions, logits, raw_risks, target
    )
    assert torch.isfinite(loss)
    assert diagnostics["risk_target"].requires_grad is False
    loss.backward()
    assert predictions.grad is not None and torch.isfinite(predictions.grad).all()
    assert raw_risks.grad is not None and torch.isfinite(raw_risks.grad).all()


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
