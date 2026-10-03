from __future__ import annotations

import torch

from experiments.energy_predict_optimize.solver import energy_optimal_probabilities
from experiments.tpmo_ascent.operator import transported_prior, tpmo_probabilities
from experiments.mabpt_ascent.operator import mass_aware_transported_prior


def test_probability_operators_preserve_simplex() -> None:
    torch.manual_seed(1)
    cost = torch.rand(2, 5, 5, dtype=torch.float64)
    source = torch.softmax(torch.randn(2, 5, dtype=torch.float64), dim=1)
    prior_c162 = transported_prior(source, cost)["transported"]
    prior_c165 = mass_aware_transported_prior(source, cost)["transported"]
    target = torch.rand(2, 5, dtype=torch.float64)
    pairwise = torch.rand(2, 5, 5, dtype=torch.float64)
    pairwise = 0.5 * (pairwise + pairwise.transpose(1, 2))
    pairwise[:, torch.arange(5), torch.arange(5)] = 0.0
    refined, _ = tpmo_probabilities(prior_c162, target, pairwise)
    energy = energy_optimal_probabilities(target, pairwise)
    for value in (prior_c162, prior_c165, refined, energy):
        assert torch.isfinite(value).all()
        assert torch.all(value >= 0)
        assert torch.allclose(value.sum(dim=1), torch.ones(2, dtype=value.dtype), atol=1e-6)
