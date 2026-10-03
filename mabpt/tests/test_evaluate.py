from __future__ import annotations

import inspect

import torch

from mabpt.evaluate import probability_arms


def test_probability_arms_are_complete_and_target_free() -> None:
    generator = torch.Generator().manual_seed(165)
    batch, modes = 4, 5
    source = torch.rand(batch, modes, generator=generator, dtype=torch.float64)
    source /= source.sum(dim=1, keepdim=True)
    native = torch.rand(batch, modes, generator=generator, dtype=torch.float64)
    native /= native.sum(dim=1, keepdim=True)
    energy = torch.rand(batch, modes, generator=generator, dtype=torch.float64)
    energy /= energy.sum(dim=1, keepdim=True)
    cost = torch.rand(batch, modes, modes, generator=generator, dtype=torch.float64)
    risk = torch.rand(batch, modes, generator=generator, dtype=torch.float64)
    points = torch.rand(batch, modes, 6, 3, generator=generator, dtype=torch.float64)
    pairwise = torch.linalg.vector_norm(
        points[:, :, None] - points[:, None, :], dim=-1
    ).mean(dim=-1)
    arms, diagnostics = probability_arms(source, native, energy, cost, risk, pairwise)
    expected = {
        "target_native_logits",
        "target_energy_single_support",
        "identity_full_projection",
        "ordinary_hungarian_full_projection",
        "mass_hungarian_full_projection",
        "row_softmax_full_projection",
        "sinkhorn_full_projection",
        "unweighted_gibbs_full_projection",
        "mabpt_u_only",
        "mabpt_risk_kl",
        "mabpt_diversity_kl",
        "uniform_energy_kl",
        "mabpt_no_kl",
        "mabpt",
    }
    assert set(arms) == expected
    assert diagnostics
    for probabilities in arms.values():
        assert torch.allclose(
            probabilities.sum(dim=1), torch.ones(batch, dtype=torch.float64), atol=1e-8
        )
    forbidden = {"target", "future", "ground_truth", "label"}
    assert forbidden.isdisjoint(inspect.signature(probability_arms).parameters)
