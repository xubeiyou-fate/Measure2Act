"""Standalone finite-measure operators for MABPT.

The probability forward is target-free. All assignment and projection
operations act only on predicted supports, predicted risks, and source mass.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import permutations
import math

import torch
from torch.nn import functional as F


DEFAULT_ADE_SCALE = 0.2760537266731262
DEFAULT_FDE_SCALE = 0.4783363938331604
DEFAULT_NEWTON_ITERATIONS = 16
DEFAULT_BACKTRACKING_STEPS = 12
DEFAULT_SINKHORN_ITERATIONS = 64
DEFAULT_TOLERANCE = 1e-9


@lru_cache(maxsize=8)
def _cpu_permutations(modes: int) -> torch.Tensor:
    if modes < 2:
        raise ValueError("modes must be at least two")
    if modes > 8:
        raise ValueError("exact permutation enumeration is limited to K <= 8")
    return torch.tensor(list(permutations(range(modes))), dtype=torch.long)


def all_permutations(
    modes: int, *, device: torch.device | str | None = None
) -> torch.Tensor:
    """Return the K! source-to-target bijections in lexical order."""
    return _cpu_permutations(modes).to(device=device)


def _normalized_mass(probabilities: torch.Tensor) -> torch.Tensor:
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must have shape [B,K] with K >= 2")
    probabilities = probabilities.to(torch.float64)
    if not bool(torch.isfinite(probabilities).all()) or bool(
        (probabilities < 0).any()
    ):
        raise ValueError("probabilities must be finite and nonnegative")
    total = probabilities.sum(dim=1, keepdim=True)
    if bool((total <= 0).any()):
        raise ValueError("each probability row must have positive mass")
    return probabilities / total


def _validate_cost(probabilities: torch.Tensor, cost: torch.Tensor) -> int:
    modes = probabilities.shape[1]
    if cost.shape != (probabilities.shape[0], modes, modes):
        raise ValueError("cost must have shape [B,K,K]")
    if not bool(torch.isfinite(cost).all()):
        raise ValueError("cost must be finite")
    return modes


def support_cost(
    source_support: torch.Tensor,
    target_support: torch.Tensor,
    *,
    ade_scale: float = DEFAULT_ADE_SCALE,
    fde_scale: float = DEFAULT_FDE_SCALE,
) -> torch.Tensor:
    """Compute dimensionless full-path source-to-target support cost."""
    if source_support.ndim != 4 or source_support.shape != target_support.shape:
        raise ValueError("supports must share shape [B,K,T,D]")
    if source_support.shape[1] < 2 or source_support.shape[-1] < 2:
        raise ValueError("supports require at least two modes and two coordinates")
    if ade_scale <= 0 or fde_scale <= 0:
        raise ValueError("metric scales must be positive")
    if not bool(torch.isfinite(source_support).all()) or not bool(
        torch.isfinite(target_support).all()
    ):
        raise ValueError("supports must be finite")
    displacement = torch.linalg.vector_norm(
        source_support.to(torch.float64)[:, :, None]
        - target_support.to(torch.float64)[:, None, :],
        dim=-1,
    )
    return displacement.mean(dim=-1) / ade_scale + displacement[..., -1] / fde_scale


def pairwise_trajectory_distance(support: torch.Tensor) -> torch.Tensor:
    """Return mean Euclidean trajectory distance between target atoms."""
    if support.ndim != 4 or support.shape[1] < 2:
        raise ValueError("support must have shape [B,K,T,D]")
    if not bool(torch.isfinite(support).all()):
        raise ValueError("support must be finite")
    support = support.to(torch.float64)
    return torch.linalg.vector_norm(
        support[:, :, None] - support[:, None, :], dim=-1
    ).mean(dim=-1)


def _permutation_components(
    probabilities: torch.Tensor,
    cost: torch.Tensor,
    *,
    mass_weighted: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probabilities = _normalized_mass(probabilities)
    modes = _validate_cost(probabilities, cost)
    permutation = all_permutations(modes, device=cost.device)
    count = permutation.shape[0]
    destinations = permutation[None, :, :, None].expand(
        cost.shape[0], -1, -1, -1
    )
    assigned = cost.to(torch.float64)[:, None].expand(
        -1, count, -1, -1
    ).gather(3, destinations).squeeze(-1)
    source_weight = probabilities if mass_weighted else torch.full_like(
        probabilities, 1.0 / modes
    )
    assignment_cost = (assigned * source_weight[:, None]).sum(dim=-1)
    matrices = F.one_hot(permutation, num_classes=modes).to(torch.float64)
    mapped = torch.einsum("bi,sij->bsj", probabilities, matrices)
    return permutation, assignment_cost, mapped


def hard_bijection_transport(
    probabilities: torch.Tensor,
    cost: torch.Tensor,
    *,
    mass_weighted: bool,
) -> dict[str, torch.Tensor]:
    """Solve the ordinary or mass-weighted linear assignment exactly.

    For K <= 8, vectorized enumeration returns the same optimum as Hungarian
    assignment while retaining deterministic tie-breaking on GPU.
    """
    permutation, assignment_cost, mapped = _permutation_components(
        probabilities, cost, mass_weighted=mass_weighted
    )
    hard_index = assignment_cost.argmin(dim=1)
    rows = torch.arange(cost.shape[0], device=cost.device)
    return {
        "transported": mapped[rows, hard_index],
        "permutation": permutation[hard_index],
        "assignment_cost": assignment_cost,
        "selected_cost": assignment_cost[rows, hard_index],
    }


def exact_gibbs_transport(
    probabilities: torch.Tensor,
    cost: torch.Tensor,
    *,
    mass_weighted: bool,
) -> dict[str, torch.Tensor]:
    """Marginalize source mass over the exact finite Gibbs posterior."""
    permutation, assignment_cost, mapped = _permutation_components(
        probabilities, cost, mass_weighted=mass_weighted
    )
    log_weights = -assignment_cost
    weights = torch.softmax(log_weights, dim=1)
    transported = torch.einsum("bs,bsj->bj", weights, mapped)
    entropy = -(weights * weights.clamp_min(torch.finfo(weights.dtype).tiny).log()).sum(
        dim=1
    )
    marginal = torch.einsum(
        "bs,sij->bij",
        weights,
        F.one_hot(permutation, num_classes=probabilities.shape[1]).to(torch.float64),
    )
    _assert_transport(transported, marginal)
    return {
        "transported": transported,
        "marginal": marginal,
        "assignment_cost": assignment_cost,
        "assignment_weights": weights,
        "assignment_entropy": entropy,
        "normalized_assignment_entropy": entropy / math.log(permutation.shape[0]),
        "expected_cost": (weights * assignment_cost).sum(dim=1),
    }


def top_m_gibbs_transport(
    probabilities: torch.Tensor,
    cost: torch.Tensor,
    *,
    top_m: int,
    mass_weighted: bool = True,
) -> dict[str, torch.Tensor]:
    """Approximate Gibbs marginalization with the lowest-cost M bijections."""
    permutation, assignment_cost, mapped = _permutation_components(
        probabilities, cost, mass_weighted=mass_weighted
    )
    if top_m < 1 or top_m > permutation.shape[0]:
        raise ValueError("top_m must be in [1, K!]")
    selected_cost, selected_index = torch.topk(
        assignment_cost, k=top_m, dim=1, largest=False, sorted=True
    )
    selected_mapped = mapped.gather(
        1, selected_index[:, :, None].expand(-1, -1, mapped.shape[2])
    )
    weights = torch.softmax(-selected_cost, dim=1)
    transported = torch.einsum("bm,bmj->bj", weights, selected_mapped)
    _assert_transport(transported)
    retained_log_mass = torch.logsumexp(-selected_cost, dim=1) - torch.logsumexp(
        -assignment_cost, dim=1
    )
    return {
        "transported": transported,
        "selected_permutation": permutation[selected_index],
        "selected_cost": selected_cost,
        "selected_weights": weights,
        "retained_posterior_mass": retained_log_mass.exp(),
    }


def row_softmax_transport(
    probabilities: torch.Tensor, cost: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Independent row-wise soft correspondence control."""
    probabilities = _normalized_mass(probabilities)
    _validate_cost(probabilities, cost)
    marginal = torch.softmax(-cost.to(torch.float64), dim=2)
    transported = torch.einsum("bi,bij->bj", probabilities, marginal)
    _assert_transport(transported)
    return {"transported": transported, "marginal": marginal}


def sinkhorn_transport(
    probabilities: torch.Tensor,
    cost: torch.Tensor,
    *,
    iterations: int = DEFAULT_SINKHORN_ITERATIONS,
) -> dict[str, torch.Tensor]:
    """Entropic bistochastic correspondence control with a fixed unit scale."""
    probabilities = _normalized_mass(probabilities)
    modes = _validate_cost(probabilities, cost)
    if iterations < 1:
        raise ValueError("Sinkhorn requires at least one iteration")
    log_kernel = -cost.to(torch.float64)
    log_row = log_kernel.new_full((cost.shape[0], modes), -math.log(modes))
    log_col = log_row.clone()
    dual_row = torch.zeros_like(log_row)
    dual_col = torch.zeros_like(log_col)
    for _ in range(iterations):
        dual_row = log_row - torch.logsumexp(
            log_kernel + dual_col[:, None, :], dim=2
        )
        dual_col = log_col - torch.logsumexp(
            log_kernel + dual_row[:, :, None], dim=1
        )
    coupling = torch.exp(
        log_kernel + dual_row[:, :, None] + dual_col[:, None, :]
    )
    marginal = coupling * modes
    transported = torch.einsum("bi,bij->bj", probabilities, marginal)
    # Fixed-iteration Sinkhorn is an approximation, especially when the unit-
    # scale kernel is nearly a hard permutation. Preserve probability mass and
    # expose both stochasticity residuals instead of hiding them with a tuned
    # temperature or a data-dependent iteration search.
    transported = transported / transported.sum(dim=1, keepdim=True)
    _assert_transport(transported)
    return {
        "transported": transported,
        "marginal": marginal,
        "row_error": (marginal.sum(dim=2) - 1.0).abs().amax(dim=1),
        "column_error": (marginal.sum(dim=1) - 1.0).abs().amax(dim=1),
    }


def _assert_transport(
    transported: torch.Tensor,
    marginal: torch.Tensor | None = None,
    *,
    atol: float = 1e-12,
) -> None:
    if bool((transported < -atol).any()) or not bool(torch.isfinite(transported).all()):
        raise RuntimeError("transported mass is invalid")
    if not torch.allclose(
        transported.sum(dim=1),
        torch.ones_like(transported[:, 0]),
        atol=atol,
        rtol=0.0,
    ):
        raise RuntimeError("transported mass left the simplex")
    if marginal is not None:
        ones = torch.ones_like(marginal[:, :, 0])
        if not torch.allclose(marginal.sum(dim=2), ones, atol=atol, rtol=0.0):
            raise RuntimeError("assignment marginal is not row stochastic")
        if not torch.allclose(marginal.sum(dim=1), ones, atol=atol, rtol=0.0):
            raise RuntimeError("assignment marginal is not column stochastic")


def energy_kl_objective(
    probabilities: torch.Tensor,
    prior: torch.Tensor,
    predicted_risk: torch.Tensor,
    pairwise_distance: torch.Tensor,
    *,
    risk_weight: float = 1.0,
    diversity_weight: float = 1.0,
    kl_weight: float = 1.0,
) -> torch.Tensor:
    """Evaluate risk minus diversity plus KL on a finite measure."""
    if min(risk_weight, diversity_weight, kl_weight) < 0:
        raise ValueError("objective weights must be nonnegative")
    if probabilities.ndim < 2 or prior.ndim < 2:
        raise ValueError("probabilities and prior need a batch and mode dimension")
    modes = probabilities.shape[-1]
    if prior.shape[-1] != modes or predicted_risk.shape[-1] != modes:
        raise ValueError("probabilities, prior, and risk must share the final K dimension")
    if pairwise_distance.shape[-2:] != (modes, modes):
        raise ValueError("pairwise_distance must have final shape [K,K]")
    try:
        torch.broadcast_shapes(
            probabilities.shape,
            prior.shape,
            predicted_risk.shape,
            pairwise_distance.shape[:-1],
        )
    except RuntimeError as exc:
        raise ValueError("objective inputs are not broadcast-compatible") from exc
    target_term = risk_weight * (probabilities * predicted_risk).sum(dim=-1)
    diversity = 0.5 * diversity_weight * torch.einsum(
        "...i,...ij,...j->...", probabilities, pairwise_distance, probabilities
    )
    if kl_weight == 0:
        kl = torch.zeros_like(target_term)
    else:
        tiny = torch.finfo(probabilities.dtype).tiny
        kl = kl_weight * (
            probabilities
            * (
                probabilities.clamp_min(tiny).log()
                - prior.clamp_min(tiny).log()
            )
        ).sum(dim=-1)
    return target_term - diversity + kl


def _frank_wolfe_energy(
    predicted_risk: torch.Tensor,
    pairwise_distance: torch.Tensor,
    *,
    risk_weight: float,
    diversity_weight: float,
    steps: int = 64,
) -> torch.Tensor:
    modes = predicted_risk.shape[1]
    probabilities = torch.full_like(predicted_risk, 1.0 / modes)
    for _ in range(steps):
        gradient = risk_weight * predicted_risk - diversity_weight * torch.einsum(
            "bij,bj->bi", pairwise_distance, probabilities
        )
        vertex = F.one_hot(gradient.argmin(dim=1), num_classes=modes).to(
            probabilities.dtype
        )
        direction = vertex - probabilities
        derivative = (direction * gradient).sum(dim=1)
        curvature = -diversity_weight * torch.einsum(
            "bi,bij,bj->b", direction, pairwise_distance, direction
        )
        step = torch.where(
            curvature > 1e-12,
            -derivative / curvature.clamp_min(1e-12),
            (derivative < 0).to(probabilities.dtype),
        ).clamp(0.0, 1.0)
        probabilities = probabilities + step[:, None] * direction
    return probabilities


def energy_kl_projection(
    prior: torch.Tensor,
    predicted_risk: torch.Tensor,
    pairwise_distance: torch.Tensor,
    *,
    risk_weight: float = 1.0,
    diversity_weight: float = 1.0,
    kl_weight: float = 1.0,
    max_iterations: int = DEFAULT_NEWTON_ITERATIONS,
    backtracking_steps: int = DEFAULT_BACKTRACKING_STEPS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Solve the MABPT finite-measure projection on the probability simplex."""
    prior = _normalized_mass(prior)
    if predicted_risk.shape != prior.shape:
        raise ValueError("predicted_risk must have shape [B,K]")
    batch, modes = prior.shape
    if pairwise_distance.shape != (batch, modes, modes):
        raise ValueError("pairwise_distance must have shape [B,K,K]")
    predicted_risk = predicted_risk.to(torch.float64)
    pairwise_distance = pairwise_distance.to(torch.float64)
    if not bool(torch.isfinite(predicted_risk).all()) or not bool(
        torch.isfinite(pairwise_distance).all()
    ):
        raise ValueError("projection inputs must be finite")
    if bool((pairwise_distance < -1e-9).any()):
        raise ValueError("pairwise distances must be nonnegative")
    if min(risk_weight, diversity_weight, kl_weight) < 0:
        raise ValueError("objective weights must be nonnegative")
    if risk_weight == diversity_weight == 0:
        return prior, {
            "initial_objective": torch.zeros(batch, dtype=prior.dtype, device=prior.device),
            "final_objective": torch.zeros(batch, dtype=prior.dtype, device=prior.device),
            "objective_gain": torch.zeros(batch, dtype=prior.dtype, device=prior.device),
            "kkt_residual": torch.zeros(batch, dtype=prior.dtype, device=prior.device),
            "minimum_probability": prior.amin(dim=1),
        }
    if kl_weight == 0:
        probabilities = _frank_wolfe_energy(
            predicted_risk,
            pairwise_distance,
            risk_weight=risk_weight,
            diversity_weight=diversity_weight,
        )
        objective = energy_kl_objective(
            probabilities,
            prior,
            predicted_risk,
            pairwise_distance,
            risk_weight=risk_weight,
            diversity_weight=diversity_weight,
            kl_weight=0.0,
        )
        return probabilities, {
            "initial_objective": energy_kl_objective(
                torch.full_like(prior, 1.0 / modes),
                prior,
                predicted_risk,
                pairwise_distance,
                risk_weight=risk_weight,
                diversity_weight=diversity_weight,
                kl_weight=0.0,
            ),
            "final_objective": objective,
            "objective_gain": torch.full_like(objective, float("nan")),
            "kkt_residual": torch.full_like(objective, float("nan")),
            "minimum_probability": probabilities.amin(dim=1),
        }
    if max_iterations < 1 or backtracking_steps < 1 or tolerance <= 0:
        raise ValueError("invalid fixed Newton configuration")

    tiny = torch.finfo(torch.float64).tiny
    prior = prior.clamp_min(tiny)
    prior = prior / prior.sum(dim=1, keepdim=True)
    probabilities = prior.clone()
    ones = probabilities.new_ones((batch, modes, 1))
    initial = energy_kl_objective(
        probabilities,
        prior,
        predicted_risk,
        pairwise_distance,
        risk_weight=risk_weight,
        diversity_weight=diversity_weight,
        kl_weight=kl_weight,
    )
    for _ in range(max_iterations):
        gradient = (
            risk_weight * predicted_risk
            - diversity_weight
            * torch.einsum("bij,bj->bi", pairwise_distance, probabilities)
            + kl_weight * (probabilities.log() - prior.log() + 1.0)
        )
        residual = (gradient - gradient.mean(dim=1, keepdim=True)).abs().amax(dim=1)
        active = residual > tolerance
        if not bool(active.any()):
            break
        hessian = kl_weight * torch.diag_embed(probabilities.reciprocal()) - (
            diversity_weight * pairwise_distance
        )
        kkt = probabilities.new_zeros((batch, modes + 1, modes + 1))
        kkt[:, :modes, :modes] = hessian
        kkt[:, :modes, modes:] = ones
        kkt[:, modes:, :modes] = ones.transpose(1, 2)
        rhs = probabilities.new_zeros((batch, modes + 1))
        rhs[:, :modes] = -gradient
        direction = torch.linalg.solve(kkt, rhs.unsqueeze(-1)).squeeze(-1)[:, :modes]
        direction = torch.where(active[:, None], direction, torch.zeros_like(direction))
        negative = direction < 0
        positive_limit = torch.where(
            negative,
            -0.99 * probabilities / direction.clamp_max(-tiny),
            torch.full_like(direction, float("inf")),
        ).amin(dim=1)
        maximum_step = positive_limit.clamp(max=1.0)
        derivative = (gradient * direction).sum(dim=1)
        current = energy_kl_objective(
            probabilities,
            prior,
            predicted_risk,
            pairwise_distance,
            risk_weight=risk_weight,
            diversity_weight=diversity_weight,
            kl_weight=kl_weight,
        )
        factors = 0.5 ** torch.arange(
            backtracking_steps, dtype=torch.float64, device=prior.device
        )
        steps = maximum_step[:, None] * factors[None]
        candidates = probabilities[:, None] + steps[:, :, None] * direction[:, None]
        candidate_objective = energy_kl_objective(
            candidates,
            prior[:, None],
            predicted_risk[:, None],
            pairwise_distance[:, None],
            risk_weight=risk_weight,
            diversity_weight=diversity_weight,
            kl_weight=kl_weight,
        )
        acceptable = (
            candidate_objective <= current[:, None] + 1e-4 * steps * derivative[:, None]
        ) & (candidates > 0).all(dim=2) & active[:, None]
        has_step = acceptable.any(dim=1)
        # Near the stationary point, the Armijo right-hand side can be below
        # float64 objective noise (the Newton direction is already ~1e-9).
        # Permit a numerically neutral candidate only when the strict search
        # found no step; this keeps descent semantics for material updates.
        neutral = (
            (candidate_objective <= current[:, None] + 1e-12)
            & (candidates > 0).all(dim=2)
            & active[:, None]
        )
        acceptable = torch.where(has_step[:, None], acceptable, neutral)
        has_step = acceptable.any(dim=1)
        first = acceptable.to(torch.int64).argmax(dim=1)
        selected = steps.gather(1, first[:, None]).squeeze(1)
        selected = torch.where(has_step, selected, torch.zeros_like(selected))
        probabilities = probabilities + selected[:, None] * direction

    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
    final_gradient = (
        risk_weight * predicted_risk
        - diversity_weight
        * torch.einsum("bij,bj->bi", pairwise_distance, probabilities)
        + kl_weight * (probabilities.log() - prior.log() + 1.0)
    )
    final_residual = (
        final_gradient - final_gradient.mean(dim=1, keepdim=True)
    ).abs().amax(dim=1)
    final = energy_kl_objective(
        probabilities,
        prior,
        predicted_risk,
        pairwise_distance,
        risk_weight=risk_weight,
        diversity_weight=diversity_weight,
        kl_weight=kl_weight,
    )
    if bool((probabilities <= 0).any()) or not torch.allclose(
        probabilities.sum(dim=1),
        torch.ones_like(probabilities[:, 0]),
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError("projection left the open probability simplex")
    if bool((final > initial + 1e-10).any()):
        raise RuntimeError("projection increased its objective")
    return probabilities, {
        "initial_objective": initial,
        "final_objective": final,
        "objective_gain": initial - final,
        "kkt_residual": final_residual,
        "minimum_probability": probabilities.amin(dim=1),
    }


__all__ = [
    "DEFAULT_ADE_SCALE",
    "DEFAULT_FDE_SCALE",
    "all_permutations",
    "energy_kl_objective",
    "energy_kl_projection",
    "exact_gibbs_transport",
    "hard_bijection_transport",
    "pairwise_trajectory_distance",
    "row_softmax_transport",
    "sinkhorn_transport",
    "support_cost",
    "top_m_gibbs_transport",
]
