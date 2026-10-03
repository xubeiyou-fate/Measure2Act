from __future__ import annotations

import torch

from experiments.tpmo_ascent.operator import all_permutations
from experiments.mabpt_ascent.operator import mass_aware_transported_prior
from experiments.mabpt_ascent.protocol import load_protocol


def test_mass_aware_transport_preserves_simplex_and_uses_mass() -> None:
    probabilities = torch.tensor([[0.90, 0.05, 0.03, 0.01, 0.01]], dtype=torch.float64)
    cost = torch.tensor(
        [[[0.0, 0.2, 10.0, 10.0, 10.0],
          [0.2, 0.0, 10.0, 10.0, 10.0],
          [10.0, 10.0, 0.0, 10.0, 10.0],
          [10.0, 10.0, 10.0, 0.0, 10.0],
          [10.0, 10.0, 10.0, 10.0, 0.0]]], dtype=torch.float64)
    result = mass_aware_transported_prior(probabilities, cost)
    assert torch.all(result["transported"] >= 0)
    assert torch.allclose(result["transported"].sum(dim=1), torch.ones(1, dtype=torch.float64), atol=1e-12)
    assert result["assignment_cost"].shape == (1, 120)
    # The dominant source atom participates in the matching objective.
    assert float(result["assignment_cost"].std()) > 0.0


def test_uniform_mass_reduces_to_mean_cost() -> None:
    probabilities = torch.full((2, 5), 0.2, dtype=torch.float64)
    generator = torch.Generator().manual_seed(165)
    cost = torch.rand((2, 5, 5), generator=generator, dtype=torch.float64)
    result = mass_aware_transported_prior(probabilities, cost)
    permutation = all_permutations()
    assigned = cost[:, None].expand(-1, permutation.shape[0], -1, -1).gather(
        3, permutation[None, :, :, None].expand(2, -1, -1, -1)
    ).squeeze(-1)
    expected = assigned.mean(dim=-1)
    # For uniform source mass, the weighted cost is exactly the row mean.
    assert torch.allclose(result["assignment_cost"], expected, atol=1e-12, rtol=0.0)
    assert torch.allclose(result["transported"].sum(dim=1), torch.ones(2, dtype=torch.float64), atol=1e-12)


def test_protocol_excludes_post_hoc_repairs() -> None:
    payload = load_protocol().payload
    assert payload["algorithm"]["target_in_validation_probability_forward"] is False
    forbidden = set(payload["forbidden_mechanisms"])
    assert {"trajectory_or_control_residual", "learned_gate_router_or_mixture_of_experts", "temperature_or_assignment_scale_search"}.issubset(forbidden)
