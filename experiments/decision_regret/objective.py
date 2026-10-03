"""C133 exact geometry plus Native-K decision-regret training."""

from __future__ import annotations

import torch


ADE_SCALE = 0.2760537266731262
FDE_SCALE = 0.4783363938331604


def _validate(
    predictions: torch.Tensor,
    logits: torch.Tensor,
    decision_costs: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if predictions.ndim != 4 or predictions.shape[-1] != 3:
        raise ValueError("predictions must have shape [B,K,T,3]")
    if predictions.shape[1] != 5:
        raise ValueError("C133 requires exactly five native modes")
    if target.shape != (predictions.shape[0], predictions.shape[2], 3):
        raise ValueError("target shape is incompatible with predictions")
    if logits.shape != predictions.shape[:2]:
        raise ValueError("logits must have shape [B,K]")
    if decision_costs.shape != predictions.shape[:2]:
        raise ValueError("decision_costs must have shape [B,K]")


def per_mode_errors(
    predictions: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if predictions.ndim != 4 or target.ndim != 3:
        raise ValueError("predictions/target must be [B,K,T,3]/[B,T,3]")
    displacement = torch.linalg.vector_norm(
        predictions - target[:, None], dim=-1
    )
    return displacement.mean(dim=-1), displacement[..., -1]


def decision_cost_targets(
    ade: torch.Tensor,
    fde: torch.Tensor,
    *,
    ade_scale: float = ADE_SCALE,
    fde_scale: float = FDE_SCALE,
) -> torch.Tensor:
    """Build the fixed deployment cost vector, without gradient leakage."""
    if ade_scale <= 0 or fde_scale <= 0:
        raise ValueError("metric scales must be positive")
    return (ade / ade_scale + fde / fde_scale).detach()


def spo_plus_loss(
    predicted_costs: torch.Tensor, true_costs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """SPO+ convex surrogate for choosing one of K native candidate actions.

    For a one-hot action set, the optimization oracle is simply ``argmin`` over
    the candidate cost vector. The true costs are full-information labels; the
    model is trained on decision regret, not on cost MSE or a winner CE.
    """
    if predicted_costs.shape != true_costs.shape or predicted_costs.ndim != 2:
        raise ValueError("SPO+ costs must have shape [B,K]")
    true_winner = true_costs.argmin(dim=1)
    rows = torch.arange(true_costs.shape[0], device=true_costs.device)
    max_term = (true_costs - 2.0 * predicted_costs).max(dim=1).values
    anchor = 2.0 * predicted_costs[rows, true_winner] - true_costs[
        rows, true_winner
    ]
    return (max_term + anchor).mean(), true_winner


def decision_regret_objective(
    predictions: torch.Tensor,
    logits: torch.Tensor,
    decision_costs: torch.Tensor,
    target: torch.Tensor,
    *,
    ade_scale: float = ADE_SCALE,
    fde_scale: float = FDE_SCALE,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train C129 geometry with a decision-focused Native-K score head."""
    _validate(predictions, logits, decision_costs, target)
    ade, fde = per_mode_errors(predictions, target)
    true_costs = decision_cost_targets(
        ade, fde, ade_scale=ade_scale, fde_scale=fde_scale
    )
    centered_costs = decision_costs - decision_costs.mean(dim=1, keepdim=True)
    expected_logits = -centered_costs
    if not torch.allclose(logits, expected_logits, atol=1e-6, rtol=1e-6):
        raise ValueError("logits must be the negative centered decision costs")

    geometry = (
        ade.min(dim=1).values / ade_scale
        + fde.min(dim=1).values / fde_scale
    ).mean()
    regret_surrogate, true_winner = spo_plus_loss(centered_costs, true_costs)
    loss = geometry + regret_surrogate
    top1_mode = logits.detach().argmax(dim=1)
    rows = torch.arange(predictions.shape[0], device=predictions.device)
    selected_cost = true_costs[rows, top1_mode]
    oracle_cost = true_costs[rows, true_winner]
    return loss, {
        "loss": loss.detach(),
        "geometry": geometry.detach(),
        "decision_regret_surrogate": regret_surrogate.detach(),
        "decision_cost_target": true_costs,
        "centered_decision_costs": centered_costs,
        "true_cost_winner": true_winner.detach(),
        "ade_winner": ade.detach().argmin(dim=1),
        "fde_winner": fde.detach().argmin(dim=1),
        "top1_mode": top1_mode,
        "top1_decision_regret": (selected_cost - oracle_cost).detach().mean(),
        "oracle_overlap": (
            (ade.detach().argmin(dim=1) == fde.detach().argmin(dim=1))
            .float()
            .mean()
        ),
        "top1_ade_overlap": (
            (top1_mode == ade.detach().argmin(dim=1)).float().mean()
        ),
        "top1_fde_overlap": (
            (top1_mode == fde.detach().argmin(dim=1)).float().mean()
        ),
        "batch_minade": ade.min(dim=1).values.mean().detach(),
        "batch_minfde": fde.min(dim=1).values.mean().detach(),
    }


__all__ = [
    "ADE_SCALE",
    "FDE_SCALE",
    "decision_cost_targets",
    "decision_regret_objective",
    "per_mode_errors",
    "spo_plus_loss",
]
