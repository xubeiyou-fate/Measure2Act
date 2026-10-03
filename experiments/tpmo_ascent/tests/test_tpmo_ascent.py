from __future__ import annotations

import torch

from experiments.tpmo_ascent.operator import (
    all_permutations,
    candidate_pairwise_distance,
    cross_support_cost,
    tpmo_objective,
    tpmo_probabilities,
    transported_prior,
)
from experiments.tpmo_ascent.protocol import load_protocol


def test_all_permutations_is_exact_factorial() -> None:
    permutations = all_permutations()
    assert permutations.shape == (120, 5)
    assert torch.unique(permutations, dim=0).shape[0] == 120
    assert torch.equal(torch.sort(permutations, dim=1).values, torch.arange(5).expand(120, 5))


def test_transport_uniform_assignment_marginal_is_simplex() -> None:
    q = torch.tensor([[0.02, 0.08, 0.15, 0.25, 0.50]], dtype=torch.float64)
    cost = torch.ones((1, 5, 5), dtype=torch.float64)
    result = transported_prior(q, cost)
    assert torch.allclose(result["transported"], torch.full((1, 5), 0.2, dtype=torch.float64))
    assert torch.allclose(result["transported"].sum(dim=1), torch.ones(1, dtype=torch.float64))


def test_tpmo_newton_is_simplex_preserving_and_nonincreasing() -> None:
    generator = torch.Generator().manual_seed(162)
    support = torch.randn((4, 5, 10, 3), generator=generator, dtype=torch.float64)
    candidate = support + 0.1 * torch.randn(support.shape, generator=generator, dtype=torch.float64)
    prior = transported_prior(
        torch.softmax(torch.randn((4, 5), generator=generator, dtype=torch.float64), dim=1),
        cross_support_cost(support, candidate),
    )["transported"]
    distance = torch.rand((4, 5), generator=generator, dtype=torch.float64)
    pairwise = candidate_pairwise_distance(candidate)
    probabilities, diagnostics = tpmo_probabilities(prior, distance, pairwise)
    assert torch.isfinite(probabilities).all()
    assert torch.all(probabilities > 0)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(4, dtype=torch.float64), atol=1e-12)
    assert torch.all(diagnostics["final_objective"] <= diagnostics["initial_objective"] + 1e-10)
    assert torch.allclose(
        diagnostics["final_objective"],
        tpmo_objective(probabilities, prior, distance, pairwise),
        atol=1e-10,
        rtol=0.0,
    )


def test_protocol_forbids_rejected_shortcuts() -> None:
    payload = load_protocol().payload
    forbidden = set(payload["forbidden_mechanisms"])
    assert "trajectory_or_control_residual" in forbidden
    assert "learned_gate_router_or_mixture_of_experts" in forbidden
    assert "temperature_or_assignment_scale_search" in forbidden
    assert "future_target_in_forward" in forbidden
    assert payload["algorithm"]["trainable_parameters"] == 0
    assert payload["algorithm"]["target_in_forward"] is False
