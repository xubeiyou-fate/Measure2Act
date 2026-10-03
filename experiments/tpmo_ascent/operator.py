"""GPU-batched cross-support transport and Energy-KL proximal inference."""

from __future__ import annotations

from itertools import permutations

import torch
from torch.nn import functional as F


MODES = 5
ADE_SCALE = 0.2760537266731262
FDE_SCALE = 0.4783363938331604
NEWTON_MAX_ITERATIONS = 16
BACKTRACKING_STEPS = 12
SOLVER_TOLERANCE = 1e-9


def all_permutations(*, device: torch.device | None = None) -> torch.Tensor:
    """Return all source-to-target mode bijections in stable lexical order."""
    return torch.tensor(
        list(permutations(range(MODES))), dtype=torch.long, device=device
    )


def _validate_supports(
    baseline_predictions: torch.Tensor, candidate_predictions: torch.Tensor
) -> None:
    if baseline_predictions.ndim != 4 or candidate_predictions.ndim != 4:
        raise ValueError("trajectory supports must have shape [B,5,T,3]")
    if baseline_predictions.shape != candidate_predictions.shape:
        raise ValueError("baseline and candidate supports must have identical shapes")
    if baseline_predictions.shape[1] != MODES or baseline_predictions.shape[-1] != 3:
        raise ValueError("C162 requires shape [B,5,T,3]")
    if not bool(torch.isfinite(baseline_predictions).all()) or not bool(
        torch.isfinite(candidate_predictions).all()
    ):
        raise ValueError("trajectory supports must be finite")


def cross_support_cost(
    baseline_predictions: torch.Tensor,
    candidate_predictions: torch.Tensor,
    *,
    ade_scale: float = ADE_SCALE,
    fde_scale: float = FDE_SCALE,
) -> torch.Tensor:
    """Dimensionless full-path cost from every B0 atom to every C161 atom."""
    _validate_supports(baseline_predictions, candidate_predictions)
    if ade_scale <= 0 or fde_scale <= 0:
        raise ValueError("metric scales must be positive")
    baseline = baseline_predictions.to(torch.float64)
    candidate = candidate_predictions.to(torch.float64)
    displacement = torch.linalg.vector_norm(
        baseline[:, :, None] - candidate[:, None, :], dim=-1
    )
    ade = displacement.mean(dim=-1)
    fde = displacement[..., -1]
    return ade / ade_scale + fde / fde_scale


def transported_prior(
    baseline_probabilities: torch.Tensor,
    support_cost: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Marginalize B0 probability mass over all 120 support bijections."""
    if support_cost.ndim != 3 or support_cost.shape[1:] != (MODES, MODES):
        raise ValueError("support_cost must have shape [B,5,5]")
    if baseline_probabilities.shape != support_cost.shape[:2]:
        raise ValueError("baseline_probabilities must have shape [B,5]")
    probabilities = baseline_probabilities.to(torch.float64)
    if bool((probabilities < 0).any()) or not bool(torch.isfinite(probabilities).all()):
        raise ValueError("baseline probabilities must be finite and nonnegative")
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
    permutation = all_permutations(device=support_cost.device)
    count = permutation.shape[0]
    batch = support_cost.shape[0]
    destinations = permutation[None, :, :, None].expand(batch, -1, -1, -1)
    assigned = support_cost[:, None].expand(-1, count, -1, -1).gather(
        3, destinations
    ).squeeze(-1)
    assignment_cost = assigned.mean(dim=-1)
    assignment_weights = torch.softmax(-assignment_cost, dim=1)
    assignment_matrix = F.one_hot(permutation, num_classes=MODES).to(torch.float64)
    mapped = torch.einsum("bj,sji->bsi", probabilities, assignment_matrix)
    transported = torch.einsum("bs,bsi->bi", assignment_weights, mapped)
    hard_index = assignment_cost.argmin(dim=1)
    hard = mapped[torch.arange(batch, device=support_cost.device), hard_index]
    identity = probabilities
    entropy = -(
        assignment_weights
        * assignment_weights.clamp_min(torch.finfo(torch.float64).tiny).log()
    ).sum(dim=1)
    if bool((transported < 0).any()) or not torch.allclose(
        transported.sum(dim=1),
        torch.ones_like(transported[:, 0]),
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError("transported C162 prior left the simplex")
    return {
        "identity": identity,
        "hard": hard,
        "transported": transported,
        "assignment_cost": assignment_cost,
        "assignment_weights": assignment_weights,
        "hard_permutation": permutation[hard_index],
        "identity_cost": support_cost.diagonal(dim1=1, dim2=2).mean(dim=1),
        "hard_cost": assignment_cost.gather(1, hard_index[:, None]).squeeze(1),
        "expected_cost": (assignment_weights * assignment_cost).sum(dim=1),
        "assignment_entropy": entropy,
        "normalized_assignment_entropy": entropy / torch.log(
            entropy.new_tensor(float(count))
        ),
    }


def candidate_pairwise_distance(candidate_predictions: torch.Tensor) -> torch.Tensor:
    if candidate_predictions.ndim != 4 or candidate_predictions.shape[1] != MODES:
        raise ValueError("candidate_predictions must have shape [B,5,T,3]")
    candidate = candidate_predictions.to(torch.float64)
    return torch.linalg.vector_norm(
        candidate[:, :, None] - candidate[:, None, :], dim=-1
    ).mean(dim=-1)


def tpmo_objective(
    probabilities: torch.Tensor,
    transported: torch.Tensor,
    predicted_target_distance: torch.Tensor,
    pairwise_distance: torch.Tensor,
) -> torch.Tensor:
    if probabilities.ndim < 2 or transported.ndim < 2:
        raise ValueError("TPMO probabilities must have at least two dimensions")
    if probabilities.shape[-1] != MODES or transported.shape[-1] != MODES:
        raise ValueError("TPMO probabilities must have final dimension 5")
    if predicted_target_distance.shape[-1] != MODES:
        raise ValueError("predicted target distance must have final dimension 5")
    if pairwise_distance.shape[-2:] != (MODES, MODES):
        raise ValueError("pairwise distance must have final shape [5,5]")
    try:
        torch.broadcast_shapes(
            probabilities.shape,
            transported.shape,
            predicted_target_distance.shape,
            pairwise_distance.shape[:-1],
        )
    except RuntimeError as exc:
        raise ValueError("TPMO objective inputs are not broadcast-compatible") from exc
    normalized_target = predicted_target_distance / ADE_SCALE
    normalized_pairwise = pairwise_distance / ADE_SCALE
    target_term = (probabilities * normalized_target).sum(dim=-1)
    diversity_term = 0.5 * torch.einsum(
        "...i,...ij,...j->...", probabilities, normalized_pairwise, probabilities
    )
    tiny = torch.finfo(probabilities.dtype).tiny
    kl = (
        probabilities
        * (
            probabilities.clamp_min(tiny).log()
            - transported.clamp_min(tiny).log()
        )
    ).sum(dim=-1)
    return target_term - diversity_term + kl


def tpmo_probabilities(
    transported: torch.Tensor,
    predicted_target_distance: torch.Tensor,
    pairwise_distance: torch.Tensor,
    *,
    max_iterations: int = NEWTON_MAX_ITERATIONS,
    backtracking_steps: int = BACKTRACKING_STEPS,
    tolerance: float = SOLVER_TOLERANCE,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Solve the strictly convex Energy-KL problem on the K=5 simplex."""
    if transported.ndim != 2 or transported.shape[1] != MODES:
        raise ValueError("transported must have shape [B,5]")
    if predicted_target_distance.shape != transported.shape:
        raise ValueError("predicted target distance must have shape [B,5]")
    if pairwise_distance.shape != (transported.shape[0], MODES, MODES):
        raise ValueError("pairwise distance must have shape [B,5,5]")
    if max_iterations < 1 or backtracking_steps < 1 or tolerance <= 0:
        raise ValueError("invalid fixed solver configuration")
    prior = transported.to(torch.float64)
    target = predicted_target_distance.to(torch.float64)
    pairwise = pairwise_distance.to(torch.float64)
    if not bool(torch.isfinite(prior).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("TPMO inputs must be finite")
    if not bool(torch.isfinite(pairwise).all()) or bool((pairwise < -1e-9).any()):
        raise ValueError("TPMO pairwise distances must be finite and nonnegative")
    tiny = torch.finfo(torch.float64).tiny
    prior = prior.clamp_min(tiny)
    prior = prior / prior.sum(dim=1, keepdim=True)
    probabilities = prior.clone()
    normalized_target = target / ADE_SCALE
    normalized_pairwise = pairwise / ADE_SCALE
    batch = probabilities.shape[0]
    ones = probabilities.new_ones((batch, MODES, 1))
    convergence_iteration = torch.full(
        (batch,), max_iterations, dtype=torch.int64, device=probabilities.device
    )
    initial_objective = tpmo_objective(probabilities, prior, target, pairwise)

    for iteration in range(max_iterations):
        gradient = (
            normalized_target
            - torch.einsum("bij,bj->bi", normalized_pairwise, probabilities)
            + (probabilities.log() - prior.log())
            + 1.0
        )
        projected = gradient - gradient.mean(dim=1, keepdim=True)
        residual = projected.abs().amax(dim=1)
        active = residual > tolerance
        newly_converged = (~active) & (convergence_iteration == max_iterations)
        convergence_iteration[newly_converged] = iteration
        if not bool(active.any()):
            break
        hessian = torch.diag_embed(probabilities.reciprocal()) - normalized_pairwise
        kkt = probabilities.new_zeros((batch, MODES + 1, MODES + 1))
        kkt[:, :MODES, :MODES] = hessian
        kkt[:, :MODES, MODES:] = ones
        kkt[:, MODES:, :MODES] = ones.transpose(1, 2)
        rhs = probabilities.new_zeros((batch, MODES + 1))
        rhs[:, :MODES] = -gradient
        solution = torch.linalg.solve(kkt, rhs.unsqueeze(-1)).squeeze(-1)
        direction = solution[:, :MODES]
        direction = torch.where(active[:, None], direction, torch.zeros_like(direction))
        negative = direction < 0
        positive_limit = torch.where(
            negative,
            -0.99 * probabilities / direction.clamp_max(-tiny),
            torch.full_like(direction, float("inf")),
        ).amin(dim=1)
        maximum_step = positive_limit.clamp(max=1.0)
        directional_derivative = (gradient * direction).sum(dim=1)
        current_objective = tpmo_objective(probabilities, prior, target, pairwise)
        factors = 0.5 ** torch.arange(
            backtracking_steps, dtype=torch.float64, device=probabilities.device
        )
        steps = maximum_step[:, None] * factors[None]
        candidates = probabilities[:, None] + steps[:, :, None] * direction[:, None]
        candidate_objective = tpmo_objective(
            candidates,
            prior[:, None],
            target[:, None],
            pairwise[:, None],
        )
        armijo = candidate_objective <= (
            current_objective[:, None]
            + 1e-4 * steps * directional_derivative[:, None]
        )
        acceptable = armijo & (candidates > 0).all(dim=2) & active[:, None]
        has_step = acceptable.any(dim=1)
        first = acceptable.to(torch.int64).argmax(dim=1)
        selected_step = steps.gather(1, first[:, None]).squeeze(1)
        selected_step = torch.where(has_step, selected_step, torch.zeros_like(selected_step))
        probabilities = probabilities + selected_step[:, None] * direction

    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
    final_gradient = (
        normalized_target
        - torch.einsum("bij,bj->bi", normalized_pairwise, probabilities)
        + (probabilities.log() - prior.log())
        + 1.0
    )
    final_residual = (
        final_gradient - final_gradient.mean(dim=1, keepdim=True)
    ).abs().amax(dim=1)
    final_objective = tpmo_objective(probabilities, prior, target, pairwise)
    if bool((probabilities <= 0).any()) or not torch.allclose(
        probabilities.sum(dim=1),
        torch.ones_like(probabilities[:, 0]),
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError("TPMO solver left the open probability simplex")
    if bool((final_objective > initial_objective + 1e-10).any()):
        raise RuntimeError("TPMO solver increased its frozen objective")
    return probabilities, {
        "initial_objective": initial_objective,
        "final_objective": final_objective,
        "objective_gain": initial_objective - final_objective,
        "kkt_residual": final_residual,
        "convergence_iteration": convergence_iteration,
        "minimum_probability": probabilities.amin(dim=1),
        "simplex_error": (probabilities.sum(dim=1) - 1.0).abs(),
    }


__all__ = [
    "ADE_SCALE",
    "BACKTRACKING_STEPS",
    "FDE_SCALE",
    "MODES",
    "NEWTON_MAX_ITERATIONS",
    "SOLVER_TOLERANCE",
    "all_permutations",
    "candidate_pairwise_distance",
    "cross_support_cost",
    "tpmo_objective",
    "tpmo_probabilities",
    "transported_prior",
]
