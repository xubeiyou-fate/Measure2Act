"""Exact dual geometry plus full-candidate dual expected-risk regression."""

from __future__ import annotations

import torch
from torch.nn import functional as F


ADE_SCALE = 0.2760537266731262
FDE_SCALE = 0.4783363938331604


def _validate(
    predictions: torch.Tensor,
    logits: torch.Tensor,
    risk_predictions: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if predictions.ndim != 4 or predictions.shape[-1] != 3:
        raise ValueError("predictions must have shape [B,K,T,3]")
    if predictions.shape[1] != 5:
        raise ValueError("C130 requires exactly five native modes")
    if target.shape != (predictions.shape[0], predictions.shape[2], 3):
        raise ValueError("target shape is incompatible with predictions")
    if logits.shape != predictions.shape[:2]:
        raise ValueError("logits must have shape [B,K]")
    if risk_predictions.shape != (*predictions.shape[:2], 2):
        raise ValueError("risk_predictions must have shape [B,K,2]")


def per_mode_errors(
    predictions: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    displacement = torch.linalg.vector_norm(predictions - target[:, None], dim=-1)
    return displacement.mean(dim=-1), displacement[..., -1]


def centered_dual_risk_targets(
    ade: torch.Tensor,
    fde: torch.Tensor,
    *,
    ade_scale: float = ADE_SCALE,
    fde_scale: float = FDE_SCALE,
) -> torch.Tensor:
    targets = torch.stack((ade / ade_scale, fde / fde_scale), dim=-1).detach()
    return targets - targets.mean(dim=1, keepdim=True)


def dual_expected_risk_objective(
    predictions: torch.Tensor,
    logits: torch.Tensor,
    risk_predictions: torch.Tensor,
    target: torch.Tensor,
    *,
    ade_scale: float = ADE_SCALE,
    fde_scale: float = FDE_SCALE,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Regress relative ADE/FDE risk for all modes and preserve exact geometry."""
    if ade_scale <= 0 or fde_scale <= 0:
        raise ValueError("metric scales must be positive")
    _validate(predictions, logits, risk_predictions, target)
    ade, fde = per_mode_errors(predictions, target)
    risk_targets = centered_dual_risk_targets(
        ade, fde, ade_scale=ade_scale, fde_scale=fde_scale
    )
    centered_predictions = risk_predictions - risk_predictions.mean(
        dim=1, keepdim=True
    )
    expected_logits = -centered_predictions.sum(dim=-1)
    if not torch.allclose(logits, expected_logits, atol=1e-6, rtol=1e-6):
        raise ValueError("logits must be the fixed negative sum of centered risks")

    geometry = (
        ade.min(dim=1).values / ade_scale
        + fde.min(dim=1).values / fde_scale
    ).mean()
    risk_regression = F.mse_loss(centered_predictions, risk_targets)
    loss = geometry + risk_regression

    ade_winner = ade.detach().argmin(dim=1)
    fde_winner = fde.detach().argmin(dim=1)
    combined_winner = risk_targets.sum(dim=-1).argmin(dim=1)
    top1_mode = logits.detach().argmax(dim=1)
    return loss, {
        "loss": loss.detach(),
        "geometry": geometry.detach(),
        "risk_regression": risk_regression.detach(),
        "ade_winner": ade_winner,
        "fde_winner": fde_winner,
        "combined_winner": combined_winner,
        "top1_mode": top1_mode,
        "oracle_overlap": (ade_winner == fde_winner).float().mean().detach(),
        "top1_ade_overlap": (top1_mode == ade_winner).float().mean().detach(),
        "top1_fde_overlap": (top1_mode == fde_winner).float().mean().detach(),
        "batch_minade": ade.min(dim=1).values.mean().detach(),
        "batch_minfde": fde.min(dim=1).values.mean().detach(),
        "risk_target": risk_targets,
        "centered_risk_prediction": centered_predictions,
    }


__all__ = [
    "ADE_SCALE",
    "FDE_SCALE",
    "centered_dual_risk_targets",
    "dual_expected_risk_objective",
    "per_mode_errors",
]
