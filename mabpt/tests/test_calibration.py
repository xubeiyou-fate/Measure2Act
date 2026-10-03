from __future__ import annotations

import math

import torch

from mabpt.calibration import (
    continuous_mixture_nll,
    event_probabilities,
    kernel_event_probabilities,
)


def test_event_probabilities_push_mode_mass() -> None:
    probabilities = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float64)
    events = torch.tensor([[0, 1, 1, 8]])
    result = event_probabilities(probabilities, events)
    assert result[0, 0] == 0.1
    assert result[0, 1] == 0.5
    assert result[0, 8] == 0.4
    assert torch.allclose(result.sum(dim=1), torch.ones(1, dtype=torch.float64))


def test_single_component_mixture_nll_matches_gaussian() -> None:
    prediction = torch.zeros(1, 1, 2, 3, dtype=torch.float64)
    target = torch.zeros(1, 2, 3, dtype=torch.float64)
    probability = torch.ones(1, 1, dtype=torch.float64)
    variance = torch.ones(2, 3, dtype=torch.float64)
    nll = continuous_mixture_nll(prediction, probability, target, variance)
    assert torch.allclose(
        nll, torch.tensor([3.0 * math.log(2.0 * math.pi)], dtype=torch.float64)
    )


def test_kernel_event_probability_marginalizes_modes() -> None:
    probabilities = torch.tensor([[0.25, 0.75]], dtype=torch.float64)
    membership = torch.full((1, 2, 9), 1.0 / 9.0, dtype=torch.float64)
    result = kernel_event_probabilities(probabilities, membership)
    assert torch.all(result > 0)
    assert torch.allclose(result.sum(dim=1), torch.ones(1, dtype=torch.float64))
