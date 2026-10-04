from __future__ import annotations

import json

import torch

from Measure2Act_forecasting import Ascent, ConstantVelocityModel
from Measure2Act_probability_transfer import energy_kl_projection, support_cost

def test_public_forecasting_facade_is_data_free() -> None:
    assert Ascent is not None
    assert ConstantVelocityModel is not None


def test_ascent_boundary_is_explicit() -> None:
    notice = __import__("pathlib").Path("docs/ASCENT_NOTICE.md").read_text()
    assert "https://github.com/a-pru/ascent" in notice
    assert "ASCENT-inspired / architecture-informed independently authored implementation" in notice
    assert "not official ASCENT weights" in notice


def test_model_release_records_architecture_reference_boundary() -> None:
    record = json.loads(__import__("pathlib").Path("model_release.json").read_text())
    upstream = record["upstream_implementation"]
    assert upstream["repository"] == "https://github.com/a-pru/ascent"
    assert upstream["commit"] is None
    assert "Architectural reference only" in upstream["scope"]


def test_public_probability_transfer_facade() -> None:
    source = torch.zeros(1, 5, 4, 3, dtype=torch.float64)
    target = source.clone()
    probabilities = torch.full((1, 5), 0.2, dtype=torch.float64)
    cost = support_cost(source, target)
    projected, _ = energy_kl_projection(
        probabilities,
        cost.diagonal(dim1=1, dim2=2),
        torch.zeros(1, 5, 5, dtype=torch.float64),
    )
    assert projected.shape == (1, 5)
    assert torch.allclose(projected.sum(dim=1), torch.ones(1, dtype=torch.float64))
