from __future__ import annotations

import inspect

import pytest
import torch

from experiments.mabpt_ascent.operator import mass_aware_transported_prior
from mabpt.operator import (
    all_permutations,
    energy_kl_projection,
    exact_gibbs_transport,
    hard_bijection_transport,
    row_softmax_transport,
    sinkhorn_transport,
    top_m_gibbs_transport,
)


def _problem(batch: int, modes: int, seed: int = 165):
    generator = torch.Generator().manual_seed(seed)
    probabilities = torch.rand(batch, modes, generator=generator, dtype=torch.float64)
    probabilities /= probabilities.sum(dim=1, keepdim=True)
    cost = torch.rand(batch, modes, modes, generator=generator, dtype=torch.float64)
    risk = torch.rand(batch, modes, generator=generator, dtype=torch.float64)
    points = torch.rand(batch, modes, 7, 3, generator=generator, dtype=torch.float64)
    pairwise = torch.linalg.vector_norm(
        points[:, :, None] - points[:, None, :], dim=-1
    ).mean(dim=-1)
    return probabilities, cost, risk, pairwise


def test_exact_gibbs_reproduces_frozen_c165() -> None:
    probabilities, cost, _, _ = _problem(8, 5)
    legacy = mass_aware_transported_prior(probabilities, cost)
    standalone = exact_gibbs_transport(probabilities, cost, mass_weighted=True)
    assert torch.equal(legacy["assignment_cost"], standalone["assignment_cost"])
    assert torch.equal(legacy["assignment_weights"], standalone["assignment_weights"])
    assert torch.equal(legacy["transported"], standalone["transported"])


@pytest.mark.parametrize("modes", [3, 5, 7])
def test_transport_arms_preserve_simplex(modes: int) -> None:
    probabilities, cost, _, _ = _problem(3, modes)
    outputs = [
        hard_bijection_transport(probabilities, cost, mass_weighted=False),
        hard_bijection_transport(probabilities, cost, mass_weighted=True),
        exact_gibbs_transport(probabilities, cost, mass_weighted=False),
        exact_gibbs_transport(probabilities, cost, mass_weighted=True),
        row_softmax_transport(probabilities, cost),
        sinkhorn_transport(probabilities, cost),
    ]
    for output in outputs:
        transported = output["transported"]
        assert torch.all(transported >= -1e-12)
        assert torch.allclose(
            transported.sum(dim=1), torch.ones(3, dtype=torch.float64), atol=1e-8
        )


def test_gibbs_marginal_is_bistochastic() -> None:
    probabilities, cost, _, _ = _problem(11, 5)
    marginal = exact_gibbs_transport(
        probabilities, cost, mass_weighted=True
    )["marginal"]
    ones = torch.ones(11, 5, dtype=torch.float64)
    assert torch.allclose(marginal.sum(dim=1), ones, atol=1e-12, rtol=0)
    assert torch.allclose(marginal.sum(dim=2), ones, atol=1e-12, rtol=0)


def test_sinkhorn_reports_approximation_residuals() -> None:
    probabilities, cost, _, _ = _problem(11, 5)
    result = sinkhorn_transport(probabilities, cost)
    assert torch.allclose(
        result["transported"].sum(dim=1),
        torch.ones(11, dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )
    assert torch.all(result["row_error"] >= 0)
    assert torch.all(result["column_error"] >= 0)


def test_source_and_target_permutation_equivariance() -> None:
    probabilities, cost, risk, pairwise = _problem(7, 5)
    source = torch.tensor([2, 4, 0, 1, 3])
    target = torch.tensor([4, 1, 3, 0, 2])
    original = exact_gibbs_transport(
        probabilities, cost, mass_weighted=True
    )["transported"]
    transformed = exact_gibbs_transport(
        probabilities[:, source],
        cost[:, source][:, :, target],
        mass_weighted=True,
    )["transported"]
    assert torch.allclose(transformed, original[:, target], atol=1e-12, rtol=0)

    projected, _ = energy_kl_projection(original, risk, pairwise)
    transformed_projected, _ = energy_kl_projection(
        transformed, risk[:, target], pairwise[:, target][:, :, target]
    )
    assert torch.allclose(
        transformed_projected, projected[:, target], atol=2e-10, rtol=0
    )


def test_projection_decreases_objective_and_has_small_kkt_residual() -> None:
    probabilities, _, risk, pairwise = _problem(32, 5)
    projected, diagnostics = energy_kl_projection(
        probabilities, risk, pairwise, max_iterations=24
    )
    assert torch.all(projected > 0)
    assert torch.allclose(
        projected.sum(dim=1), torch.ones(32, dtype=torch.float64), atol=1e-12
    )
    assert torch.all(diagnostics["objective_gain"] >= -1e-12)
    assert float(diagnostics["kkt_residual"].max()) < 2e-8


def test_projection_hessian_is_positive_on_simplex_tangent() -> None:
    probabilities, _, _, pairwise = _problem(1000, 5)
    generator = torch.Generator().manual_seed(166)
    tangent = torch.randn(1000, 5, generator=generator, dtype=torch.float64)
    tangent -= tangent.mean(dim=1, keepdim=True)
    quadratic = (tangent.square() / probabilities).sum(dim=1) - torch.einsum(
        "bi,bij,bj->b", tangent, pairwise, tangent
    )
    assert torch.all(quadratic > 0)


def test_probability_forward_api_has_no_target() -> None:
    functions = [
        exact_gibbs_transport,
        hard_bijection_transport,
        row_softmax_transport,
        sinkhorn_transport,
        energy_kl_projection,
    ]
    forbidden = {"target", "future", "ground_truth", "label"}
    for function in functions:
        assert forbidden.isdisjoint(inspect.signature(function).parameters)


def test_permutation_counts_are_exact() -> None:
    assert all_permutations(3).shape == (6, 3)
    assert all_permutations(5).shape == (120, 5)
    assert all_permutations(7).shape == (5040, 7)


def test_top_m_converges_exactly_at_factorial() -> None:
    probabilities, cost, _, _ = _problem(5, 5)
    exact = exact_gibbs_transport(probabilities, cost, mass_weighted=True)
    approximate = top_m_gibbs_transport(
        probabilities, cost, top_m=120, mass_weighted=True
    )
    assert torch.allclose(
        approximate["transported"], exact["transported"], atol=1e-12, rtol=0
    )
    assert torch.allclose(
        approximate["retained_posterior_mass"],
        torch.ones(5, dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )
