"""Target-free finite-measure Energy optimization on the probability simplex."""

from __future__ import annotations

import torch
from torch.nn import functional as F


FRANK_WOLFE_STEPS = 32


def _validate(
    predicted_target_distance: torch.Tensor, pairwise_distance: torch.Tensor
) -> None:
    if predicted_target_distance.ndim != 2:
        raise ValueError("predicted_target_distance must have shape [B,K]")
    batch, modes = predicted_target_distance.shape
    if modes != 5:
        raise ValueError("C134 requires exactly five native modes")
    if pairwise_distance.shape != (batch, modes, modes):
        raise ValueError("pairwise_distance must have shape [B,K,K]")
    if not bool(torch.isfinite(predicted_target_distance).all()):
        raise ValueError("predicted target distances must be finite")
    if not bool(torch.isfinite(pairwise_distance).all()):
        raise ValueError("pairwise distances must be finite")
    if bool((pairwise_distance < -1e-6).any()):
        raise ValueError("pairwise distances must be non-negative")
    if not torch.allclose(
        pairwise_distance,
        pairwise_distance.transpose(1, 2),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("pairwise distances must be symmetric")


def energy_objective(
    probabilities: torch.Tensor,
    predicted_target_distance: torch.Tensor,
    pairwise_distance: torch.Tensor,
) -> torch.Tensor:
    """Per-actor predicted Energy objective for a finite trajectory measure."""
    _validate(predicted_target_distance, pairwise_distance)
    if probabilities.shape != predicted_target_distance.shape:
        raise ValueError("probabilities must have shape [B,K]")
    target_term = (probabilities * predicted_target_distance).sum(dim=1)
    diversity_term = 0.5 * torch.einsum(
        "bi,bij,bj->b", probabilities, pairwise_distance, probabilities
    )
    return target_term - diversity_term


def energy_optimal_probabilities(
    predicted_target_distance: torch.Tensor,
    pairwise_distance: torch.Tensor,
    *,
    steps: int = FRANK_WOLFE_STEPS,
) -> torch.Tensor:
    """Solve the target-free K=5 Energy problem with fixed Frank-Wolfe steps.

    The target is deliberately absent from this API. The fixed uniform start and
    iteration count make the deployment transformation deterministic.
    """
    _validate(predicted_target_distance, pairwise_distance)
    if steps < 1:
        raise ValueError("at least one Frank-Wolfe step is required")
    modes = predicted_target_distance.shape[1]
    probabilities = torch.full_like(predicted_target_distance, 1.0 / modes)
    for _ in range(steps):
        distance_times_probability = torch.einsum(
            "bij,bj->bi", pairwise_distance, probabilities
        )
        gradient = predicted_target_distance - distance_times_probability
        vertex = F.one_hot(gradient.argmin(dim=1), num_classes=modes).to(
            predicted_target_distance.dtype
        )
        direction = vertex - probabilities
        directional_derivative = (direction * gradient).sum(dim=1)
        curvature = -torch.einsum(
            "bi,bij,bj->b", direction, pairwise_distance, direction
        )
        safe_curvature = curvature.clamp_min(1e-12)
        step = torch.where(
            curvature > 1e-12,
            -directional_derivative / safe_curvature,
            (directional_derivative < 0).to(predicted_target_distance.dtype),
        ).clamp(min=0.0, max=1.0)
        probabilities = probabilities + step[:, None] * direction
    if bool((probabilities < -1e-6).any()) or not torch.allclose(
        probabilities.sum(dim=1),
        torch.ones_like(probabilities[:, 0]),
        atol=1e-5,
        rtol=0.0,
    ):
        raise RuntimeError("C134 solver left the probability simplex")
    return probabilities


__all__ = [
    "FRANK_WOLFE_STEPS",
    "energy_objective",
    "energy_optimal_probabilities",
]
