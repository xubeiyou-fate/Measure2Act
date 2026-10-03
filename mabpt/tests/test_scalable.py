from __future__ import annotations

import torch

from experiments.energy_predict_optimize.solver import energy_optimal_probabilities
from mabpt.scalable import (
    generalized_energy_probabilities,
    scalable_ascent_config,
    scalable_decision_objective,
)
from mabpt.evaluate_e9 import TOP_M, e9_probability_arms
from mabpt.train_e9 import build_model


def test_scalable_configs_are_native_cardinality_without_forbidden_mechanisms() -> None:
    for modes in (3, 7):
        for role in ("source", "decision"):
            config = scalable_ascent_config(modes, role=role, batch_size=8)
            assert config["k"] == modes
            assert config["trajectory_residual"] is False
            assert config["control_residual"] is False
            assert config["learned_gate"] is False


def test_generalized_energy_solver_replays_k5_solver() -> None:
    generator = torch.Generator().manual_seed(9)
    risk = torch.randn((4, 5), generator=generator, dtype=torch.float64)
    support = torch.randn((4, 5, 6, 3), generator=generator, dtype=torch.float64)
    pairwise = torch.linalg.vector_norm(
        support[:, :, None] - support[:, None, :], dim=-1
    ).mean(dim=-1)
    observed = generalized_energy_probabilities(risk, pairwise)
    expected = energy_optimal_probabilities(risk, pairwise)
    assert torch.allclose(observed, expected, atol=1e-12, rtol=0)


def test_generalized_energy_solver_is_permutation_equivariant() -> None:
    generator = torch.Generator().manual_seed(10)
    for modes in (3, 7):
        risk = torch.randn((2, modes), generator=generator, dtype=torch.float64)
        support = torch.randn((2, modes, 4, 3), generator=generator, dtype=torch.float64)
        pairwise = torch.linalg.vector_norm(
            support[:, :, None] - support[:, None, :], dim=-1
        ).mean(dim=-1)
        permutation = torch.randperm(modes, generator=generator)
        direct = generalized_energy_probabilities(risk, pairwise)
        permuted = generalized_energy_probabilities(
            risk[:, permutation], pairwise[:, permutation][:, :, permutation]
        )
        assert torch.allclose(permuted, direct[:, permutation], atol=1e-12, rtol=0)


def test_scalable_decision_objective_accepts_k3_and_k7() -> None:
    generator = torch.Generator().manual_seed(11)
    for modes in (3, 7):
        prediction = torch.randn((2, modes, 5, 3), generator=generator)
        target = torch.randn((2, 5, 3), generator=generator)
        costs = torch.randn((2, modes), generator=generator, requires_grad=True)
        logits = -(costs - costs.mean(dim=1, keepdim=True))
        loss, diagnostics = scalable_decision_objective(
            prediction, logits, costs, target
        )
        assert torch.isfinite(loss)
        assert diagnostics["minade"].ndim == 0
        loss.backward()
        assert costs.grad is not None


def test_e9_training_entry_builds_each_registered_stage() -> None:
    for stage in ("source", "decision", "energy"):
        model = build_model(stage, modes=3, batch_size=2)
        assert sum(parameter.numel() for parameter in model.parameters()) > 0


def test_e9_probability_arms_preserve_simplex_and_exact_top_m() -> None:
    generator = torch.Generator().manual_seed(12)
    batch, modes = 3, 3
    probability = torch.softmax(torch.randn((batch, modes), generator=generator), dim=1)
    source = torch.randn((batch, modes, 5, 3), generator=generator)
    target = torch.randn((batch, modes, 5, 3), generator=generator)
    cross = torch.linalg.vector_norm(
        source[:, :, None] - target[:, None, :], dim=-1
    ).mean(dim=-1).to(torch.float64)
    pairwise = torch.linalg.vector_norm(
        target[:, :, None] - target[:, None, :], dim=-1
    ).mean(dim=-1).to(torch.float64)
    risk = torch.randn((batch, modes), generator=generator, dtype=torch.float64)
    arms, diagnostics = e9_probability_arms(
        probability, cross, risk, pairwise, modes=modes
    )
    assert TOP_M[modes] == (6,)
    assert torch.allclose(arms["mabpt_exact"], arms["mabpt_top6"], atol=1e-12, rtol=0)
    for value in arms.values():
        assert torch.allclose(value.sum(dim=1), torch.ones(batch, dtype=value.dtype))
        assert bool((value > 0).all())
    assert torch.allclose(diagnostics["top6_retained_mass"], torch.ones(batch, dtype=torch.float64))


def test_e9_k5_has_registered_approximation_arms() -> None:
    generator = torch.Generator().manual_seed(13)
    batch, modes = 2, 5
    probability = torch.softmax(torch.randn((batch, modes), generator=generator), dim=1)
    source = torch.randn((batch, modes, 4, 3), generator=generator)
    target = torch.randn((batch, modes, 4, 3), generator=generator)
    cross = torch.linalg.vector_norm(
        source[:, :, None] - target[:, None, :], dim=-1
    ).mean(dim=-1).to(torch.float64)
    pairwise = torch.linalg.vector_norm(
        target[:, :, None] - target[:, None, :], dim=-1
    ).mean(dim=-1).to(torch.float64)
    risk = torch.randn((batch, modes), generator=generator, dtype=torch.float64)
    arms, diagnostics = e9_probability_arms(
        probability, cross, risk, pairwise, modes=modes
    )
    assert TOP_M[modes] == (8, 32)
    assert tuple(arms) == (
        "mabpt_exact",
        "mabpt_top8",
        "mabpt_top32",
        "mabpt_sinkhorn",
    )
    for value in arms.values():
        assert torch.allclose(value.sum(dim=1), torch.ones(batch, dtype=value.dtype))
        assert bool((value > 0).all())
    assert bool((diagnostics["top8_retained_mass"] <= 1.0).all())
    assert bool((diagnostics["top32_retained_mass"] <= 1.0).all())
