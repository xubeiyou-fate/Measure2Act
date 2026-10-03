"""Target-blind finite-measure ensemble operators for experiments E6 and E8."""

from __future__ import annotations

import itertools

import torch

from .operator import hard_bijection_transport, support_cost


def union_measure(
    support_one: torch.Tensor,
    probability_one: torch.Tensor,
    support_two: torch.Tensor,
    probability_two: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if support_one.shape != support_two.shape:
        raise ValueError("ensemble supports must have identical shapes")
    if (
        probability_one.shape != support_one.shape[:2]
        or probability_two.shape != support_two.shape[:2]
    ):
        raise ValueError("ensemble probabilities have incompatible shapes")
    support = torch.cat((support_one, support_two), dim=1)
    probability = torch.cat((probability_one, probability_two), dim=1) * 0.5
    return support, probability


def exact_weighted_kmedoids_compression(
    support: torch.Tensor,
    probability: torch.Tensor,
    *,
    output_modes: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Compress a small finite measure using exact target-blind K-medoids."""
    if support.ndim != 4 or probability.shape != support.shape[:2]:
        raise ValueError("support/probability must be [B,K,T,3]/[B,K]")
    batch, input_modes = probability.shape
    if not 0 < output_modes <= input_modes:
        raise ValueError("output_modes must lie in [1, input_modes]")
    combinations = torch.tensor(
        list(itertools.combinations(range(input_modes), output_modes)),
        dtype=torch.long,
        device=support.device,
    )
    distance = torch.linalg.vector_norm(
        support[:, :, None] - support[:, None, :], dim=-1
    ).mean(dim=-1)
    # [B,C,input_modes,output_modes], then assign every atom to its closest
    # candidate medoid. torch.argmin and lexicographic combinations fix ties.
    candidate_distance = distance[:, None].expand(
        -1, combinations.shape[0], -1, -1
    ).gather(
        3,
        combinations[None, :, None].expand(batch, -1, input_modes, -1),
    )
    nearest_distance, nearest_local = candidate_distance.min(dim=-1)
    objective = (probability[:, None] * nearest_distance).sum(dim=-1)
    best = objective.argmin(dim=1)
    selected = combinations[best]
    rows = torch.arange(batch, device=support.device)
    compressed_support = support[rows[:, None], selected]
    selected_assignment = nearest_local[rows, best]
    compressed_probability = probability.new_zeros((batch, output_modes))
    compressed_probability.scatter_add_(1, selected_assignment, probability)
    return compressed_support, compressed_probability, {
        "selected_indices": selected,
        "source_assignment": selected_assignment,
        "weighted_reconstruction_cost": objective[rows, best],
    }


def hungarian_aligned_average(
    support_one: torch.Tensor,
    probability_one: torch.Tensor,
    support_two: torch.Tensor,
    probability_two: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Average two K-atom measures after ordinary target-blind assignment."""
    if support_one.shape != support_two.shape:
        raise ValueError("aligned supports must have identical shapes")
    matching = hard_bijection_transport(
        probability_one,
        support_cost(support_one, support_two),
        mass_weighted=False,
    )
    permutation = matching["permutation"]
    aligned_support = support_two.gather(
        1,
        permutation[:, :, None, None].expand_as(support_two),
    )
    aligned_probability = probability_two.gather(1, permutation)
    return (
        0.5 * (support_one + aligned_support),
        0.5 * (probability_one + aligned_probability),
        {
            "permutation": permutation,
            "selected_cost": matching["selected_cost"],
        },
    )


__all__ = [
    "exact_weighted_kmedoids_compression",
    "hungarian_aligned_average",
    "union_measure",
]
