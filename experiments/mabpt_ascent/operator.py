"""Mass-aware Bayesian permutation transport followed by the frozen C162 TPMO."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from experiments.tpmo_ascent.operator import MODES, all_permutations


def mass_aware_transported_prior(
    baseline_probabilities: torch.Tensor,
    support_cost: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Map B0 mass through an exact finite permutation posterior.

    Unlike C162's uniform source-mode average, each source atom contributes in
    proportion to its actor-specific B0 probability mass.
    """
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
    assignment_cost = (assigned * probabilities[:, None, :]).sum(dim=-1)
    assignment_weights = torch.softmax(-assignment_cost, dim=1)
    assignment_matrix = F.one_hot(permutation, num_classes=MODES).to(torch.float64)
    mapped = torch.einsum("bj,sji->bsi", probabilities, assignment_matrix)
    transported = torch.einsum("bs,bsi->bi", assignment_weights, mapped)
    hard_index = assignment_cost.argmin(dim=1)
    hard = mapped[torch.arange(batch, device=support_cost.device), hard_index]
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
        raise RuntimeError("C165 transported prior left the simplex")
    return {
        "identity": probabilities,
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


__all__ = ["mass_aware_transported_prior"]
