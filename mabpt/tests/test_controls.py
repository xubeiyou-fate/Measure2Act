from __future__ import annotations

import torch

from experiments.metric_exact.model import ascent_config
from model.ascent import Ascent
from mabpt.controls import build_control_model
from mabpt.train_controls import original_ascent_objective


def _data(batch: int = 3) -> dict[str, torch.Tensor]:
    return {
        "obs_traj": torch.randn(16, batch, 3),
        "adj": torch.arange(batch),
    }


def test_widened_parameter_budget() -> None:
    model = build_control_model("widened_ascent")
    assert sum(p.numel() for p in model.parameters()) == 2_508_801
    assert abs(2_508_801 / 2_551_427 - 1.0) < 0.02


def test_shared_encoder_dual_decoder_shapes_and_boundaries() -> None:
    model = build_control_model("shared_encoder_dual_decoder").eval()
    predictions, logits, auxiliary = model(_data())
    assert predictions.shape == (3, 10, 24, 3)
    assert logits.shape == (3, 10)
    assert auxiliary["encoder_forward_count"] == 1
    assert auxiliary["shared_encoder"] is True
    assert auxiliary["trajectory_residual"] is False
    assert auxiliary["learned_gate"] is False


def test_b0_control_is_native_ascent() -> None:
    model = build_control_model("b0_extended")
    reference = Ascent(ascent_config("B0_signed_coupled"))
    assert type(model) is type(reference)
    assert sum(p.numel() for p in model.parameters()) == 1_238_849


def test_generic_objective_accepts_ten_modes() -> None:
    prediction = torch.randn(4, 10, 24, 3, requires_grad=True)
    logits = torch.randn(4, 10, requires_grad=True)
    target = torch.randn(4, 24, 3)
    loss, diagnostics = original_ascent_objective(prediction, logits, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(diagnostics) == {
        "loss", "regression", "classification", "minade", "minfde"
    }
