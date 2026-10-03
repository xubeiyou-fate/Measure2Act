"""Annealed winner-takes-all objective for matched ASCENT comparisons."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def geometric_temperature(
    epoch: int,
    total_epochs: int,
    *,
    initial: float = 8.0,
    final: float = 0.05,
) -> float:
    """Return a fixed geometric schedule from ``initial`` to ``final``."""

    if total_epochs < 2:
        raise ValueError("total_epochs must be at least two")
    if not 1 <= epoch <= total_epochs:
        raise ValueError("epoch must be in [1, total_epochs]")
    if initial <= 0.0 or final <= 0.0 or final >= initial:
        raise ValueError("temperatures must satisfy initial > final > 0")
    fraction = (epoch - 1) / (total_epochs - 1)
    return float(math.exp(math.log(initial) * (1.0 - fraction) + math.log(final) * fraction))


def annealed_wta_objective(
    predictions: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    epoch: int,
    total_epochs: int,
    initial_temperature: float = 8.0,
    final_temperature: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply detached soft assignments to per-mode SmoothL1 regression."""

    if predictions.ndim != 4 or predictions.shape[-1] != 3:
        raise ValueError("predictions must have shape [B,K,T,3]")
    if target.shape != (predictions.shape[0], predictions.shape[2], 3):
        raise ValueError("target shape is incompatible with predictions")
    if logits.shape != predictions.shape[:2]:
        raise ValueError("logits must have shape [B,K]")

    displacement = torch.linalg.vector_norm(predictions - target[:, None], dim=-1)
    ade = displacement.mean(dim=-1)
    fde = displacement[..., -1]
    winner = ade.detach().argmin(dim=1)
    temperature = geometric_temperature(
        epoch,
        total_epochs,
        initial=initial_temperature,
        final=final_temperature,
    )
    assignment = torch.softmax(-ade / temperature, dim=1).detach()
    per_mode_regression = F.smooth_l1_loss(
        predictions,
        target[:, None].expand_as(predictions),
        reduction="none",
    ).mean(dim=(2, 3))
    regression = (assignment * per_mode_regression).sum(dim=1).mean()
    classification = F.cross_entropy(logits, winner)
    loss = regression + classification
    entropy = -(assignment * assignment.clamp_min(1e-12).log()).sum(dim=1).mean()
    return loss, {
        "loss": loss.detach(),
        "regression": regression.detach(),
        "classification": classification.detach(),
        "temperature": loss.new_tensor(temperature),
        "assignment_entropy": entropy.detach(),
        "batch_minade": ade.min(dim=1).values.mean().detach(),
        "batch_minfde": fde.min(dim=1).values.mean().detach(),
    }
