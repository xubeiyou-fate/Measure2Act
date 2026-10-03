"""Joint exact-dual geometry and native coupled score objective for C129."""

from __future__ import annotations

import torch
from torch.nn import functional as F


ADE_SCALE = 0.2760537266731262
FDE_SCALE = 0.4783363938331604


def _validate(
    predictions: torch.Tensor, logits: torch.Tensor, target: torch.Tensor
) -> None:
    if predictions.ndim != 4 or predictions.shape[-1] != 3:
        raise ValueError("predictions must have shape [B,K,T,3]")
    if predictions.shape[1] != 5:
        raise ValueError("C129 requires exactly five native modes")
    if target.shape != (predictions.shape[0], predictions.shape[2], 3):
        raise ValueError("target shape is incompatible with predictions")
    if logits.shape != predictions.shape[:2]:
        raise ValueError("logits must have shape [B,K]")


def per_mode_errors(
    predictions: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if predictions.ndim != 4 or target.ndim != 3:
        raise ValueError("predictions/target must be [B,K,T,3]/[B,T,3]")
    displacement = torch.linalg.vector_norm(
        predictions - target[:, None], dim=-1
    )
    return displacement.mean(dim=-1), displacement[..., -1]


def joint_coupled_dual_objective(
    predictions: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    ade_scale: float = ADE_SCALE,
    fde_scale: float = FDE_SCALE,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Optimize independent ADE/FDE oracle modes and one deployable top1 score."""
    if ade_scale <= 0 or fde_scale <= 0:
        raise ValueError("metric scales must be positive")
    _validate(predictions, logits, target)
    ade, fde = per_mode_errors(predictions, target)
    ade_winner = ade.detach().argmin(dim=1)
    fde_winner = fde.detach().argmin(dim=1)
    combined = ade.detach() / ade_scale + fde.detach() / fde_scale
    score_winner = combined.argmin(dim=1)

    regression = (
        ade.min(dim=1).values / ade_scale
        + fde.min(dim=1).values / fde_scale
    ).mean()
    classification = F.cross_entropy(logits, score_winner)
    loss = regression + classification
    return loss, {
        "loss": loss.detach(),
        "regression": regression.detach(),
        "classification": classification.detach(),
        "ade_winner": ade_winner,
        "fde_winner": fde_winner,
        "score_winner": score_winner,
        "top1_mode": logits.detach().argmax(dim=1),
        "oracle_overlap": (ade_winner == fde_winner).float().mean().detach(),
        "score_ade_overlap": (score_winner == ade_winner).float().mean().detach(),
        "score_fde_overlap": (score_winner == fde_winner).float().mean().detach(),
        "batch_minade": ade.min(dim=1).values.mean().detach(),
        "batch_minfde": fde.min(dim=1).values.mean().detach(),
    }


__all__ = [
    "ADE_SCALE",
    "FDE_SCALE",
    "joint_coupled_dual_objective",
    "per_mode_errors",
]
