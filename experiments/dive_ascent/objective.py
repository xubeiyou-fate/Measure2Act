"""Coupled baseline and score-shielded DIVE objectives."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _validate(
    predictions: torch.Tensor,
    target: torch.Tensor,
    active_modes: int | None = None,
) -> int:
    if predictions.ndim != 4 or target.ndim != 3:
        raise ValueError("predictions must be [B,K,T,3] and target [B,T,3]")
    if predictions.shape[0] != target.shape[0] or predictions.shape[2:] != target.shape[1:]:
        raise ValueError("prediction and target shapes are incompatible")
    modes = predictions.shape[1] if active_modes is None else int(active_modes)
    if not 1 <= modes <= predictions.shape[1]:
        raise ValueError("active_modes is outside the prediction mode range")
    return modes


def winner_assignment(
    predictions: torch.Tensor,
    target: torch.Tensor,
    *,
    active_modes: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    modes = _validate(predictions, target, active_modes)
    displacement = torch.linalg.vector_norm(
        predictions[:, :modes] - target[:, None], dim=-1
    )
    full_trajectory_error = displacement.sum(dim=-1)
    winner = full_trajectory_error.detach().argmin(dim=-1)
    return winner, full_trajectory_error


def geometry_wta_loss(
    predictions: torch.Tensor,
    target: torch.Tensor,
    *,
    active_modes: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    modes = _validate(predictions, target, active_modes)
    winner, full_error = winner_assignment(
        predictions, target, active_modes=modes
    )
    actor = torch.arange(target.shape[0], device=target.device)
    regression = F.smooth_l1_loss(predictions[actor, winner], target)
    counts = torch.bincount(winner, minlength=modes)
    distortion_sum = full_error.new_zeros(modes)
    distortion_sum.index_add_(0, winner, full_error[actor, winner].detach())
    return regression, {
        "regression": regression.detach(),
        "winner": winner,
        "winner_counts": counts.detach(),
        "winner_distortion_sum": distortion_sum.detach(),
        "active_modes": regression.new_tensor(modes, dtype=torch.long),
    }


def voronoi_score_loss(
    logits: torch.Tensor,
    winner: torch.Tensor,
    *,
    active_modes: int | None = None,
) -> torch.Tensor:
    modes = logits.shape[1] if active_modes is None else int(active_modes)
    if logits.ndim != 2 or not 1 <= modes <= logits.shape[1]:
        raise ValueError("invalid score logits or active_modes")
    return F.cross_entropy(logits[:, :modes], winner.detach())


def coupled_wta_loss(
    predictions: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    regression, diagnostics = geometry_wta_loss(predictions, target)
    classification = voronoi_score_loss(logits, diagnostics["winner"])
    return regression + classification, {
        **diagnostics,
        "classification": classification.detach(),
        "score_gradient_reaches_geometry": torch.tensor(True, device=target.device),
    }


__all__ = [
    "coupled_wta_loss",
    "geometry_wta_loss",
    "voronoi_score_loss",
    "winner_assignment",
]
