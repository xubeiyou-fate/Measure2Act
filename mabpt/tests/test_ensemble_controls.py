from __future__ import annotations

import torch

from mabpt.ensemble_controls import (
    exact_weighted_kmedoids_compression,
    hungarian_aligned_average,
    union_measure,
)


def _measure(batch: int = 4, modes: int = 5):
    support = torch.randn(batch, modes, 6, 3)
    probability = torch.rand(batch, modes)
    probability /= probability.sum(dim=1, keepdim=True)
    return support, probability


def test_union_is_a_ten_atom_measure() -> None:
    one, p_one = _measure()
    two, p_two = _measure()
    support, probability = union_measure(one, p_one, two, p_two)
    assert support.shape == (4, 10, 6, 3)
    assert torch.allclose(probability.sum(1), torch.ones(4))


def test_exact_compression_preserves_mass_and_uses_input_atoms() -> None:
    one, p_one = _measure()
    two, p_two = _measure()
    support, probability = union_measure(one, p_one, two, p_two)
    compressed, mass, auxiliary = exact_weighted_kmedoids_compression(
        support, probability
    )
    assert compressed.shape == (4, 5, 6, 3)
    assert torch.allclose(mass.sum(1), torch.ones(4))
    for batch in range(4):
        assert torch.equal(
            compressed[batch], support[batch, auxiliary["selected_indices"][batch]]
        )


def test_compression_is_perfect_when_atoms_repeat_twice() -> None:
    support, probability = _measure()
    union_support = torch.cat((support, support), dim=1)
    union_probability = torch.cat((probability, probability), dim=1) * 0.5
    _, _, auxiliary = exact_weighted_kmedoids_compression(
        union_support, union_probability
    )
    assert torch.allclose(
        auxiliary["weighted_reconstruction_cost"], torch.zeros(4), atol=1e-7
    )


def test_hungarian_average_recovers_permutation() -> None:
    support, probability = _measure()
    order = torch.tensor([2, 4, 0, 1, 3])
    permuted_support = support[:, order]
    permuted_probability = probability[:, order]
    averaged, averaged_probability, _ = hungarian_aligned_average(
        support, probability, permuted_support, permuted_probability
    )
    assert torch.allclose(averaged, support)
    assert torch.allclose(averaged_probability, probability)
