"""Metric-exact and matched-control objectives for C127."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .model import COUPLED_VARIANTS, VARIANTS


OBJECTIVES = {
    "B0_signed_coupled": "original_smooth_l1",
    "B1_positive_coupled": "original_smooth_l1",
    "B2_decoupled_original": "original_smooth_l1",
    "B3_exact_minade": "exact_minade",
    "B4_exact_minfde": "exact_minfde",
    "B5_single_combined": "exact_single_combined_oracle",
    "B6_dual_oracle": "exact_independent_minade_plus_minfde",
    "B7_signed_dual": "exact_independent_minade_plus_minfde",
}


def _validate(
    predictions: torch.Tensor, logits: torch.Tensor, target: torch.Tensor
) -> None:
    if predictions.ndim != 4 or predictions.shape[-1] != 3:
        raise ValueError("predictions must have shape [B,K,T,3]")
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


def objective_for_variant(
    variant: str,
    predictions: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    ade_scale: float = 0.2760537266731262,
    fde_scale: float = 0.4783363938331604,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown C127 variant: {variant}")
    if ade_scale <= 0 or fde_scale <= 0:
        raise ValueError("metric scales must be positive")
    _validate(predictions, logits, target)
    ade, fde = per_mode_errors(predictions, target)
    ade_winner = ade.detach().argmin(dim=1)
    fde_winner = fde.detach().argmin(dim=1)
    batch = torch.arange(target.shape[0], device=target.device)
    classification = logits.sum() * 0.0

    objective = OBJECTIVES[variant]
    if objective == "original_smooth_l1":
        regression = F.smooth_l1_loss(predictions[batch, ade_winner], target)
        primary_winner = ade_winner
    elif objective == "exact_minade":
        regression = ade.min(dim=1).values.mean()
        primary_winner = ade_winner
    elif objective == "exact_minfde":
        regression = fde.min(dim=1).values.mean()
        primary_winner = fde_winner
    elif objective == "exact_single_combined_oracle":
        combined = ade / ade_scale + fde / fde_scale
        primary_winner = combined.detach().argmin(dim=1)
        regression = combined[batch, primary_winner].mean()
    elif objective == "exact_independent_minade_plus_minfde":
        regression = (
            ade.min(dim=1).values / ade_scale
            + fde.min(dim=1).values / fde_scale
        ).mean()
        primary_winner = ade_winner
    else:
        raise RuntimeError(f"unimplemented C127 objective: {objective}")

    if variant in COUPLED_VARIANTS:
        classification = F.cross_entropy(logits, ade_winner)
    loss = regression + classification
    return loss, {
        "loss": loss.detach(),
        "regression": regression.detach(),
        "classification": classification.detach(),
        "ade_winner": ade_winner,
        "fde_winner": fde_winner,
        "primary_winner": primary_winner.detach(),
        "oracle_overlap": (ade_winner == fde_winner).float().mean().detach(),
        "batch_minade": ade.min(dim=1).values.mean().detach(),
        "batch_minfde": fde.min(dim=1).values.mean().detach(),
    }


__all__ = ["OBJECTIVES", "objective_for_variant", "per_mode_errors"]
