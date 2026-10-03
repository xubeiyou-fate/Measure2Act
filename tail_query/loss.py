"""Preregistered C7 TailQuery losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def trajectory_wta_loss(
    predictions: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    error = torch.linalg.vector_norm(
        predictions - target[:, None], dim=-1
    ).sum(dim=-1)
    winner = error.argmin(dim=-1)
    batch = torch.arange(predictions.shape[0], device=predictions.device)
    regression = F.smooth_l1_loss(predictions[batch, winner], target)
    classification = F.cross_entropy(logits, winner.detach())
    return regression + classification


def supervised_contrastive_loss(
    projection: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    if projection.ndim != 2 or labels.shape != (projection.shape[0],):
        raise ValueError("projection/label shapes are incompatible")
    similarity = projection @ projection.transpose(0, 1) / temperature
    identity = torch.eye(
        projection.shape[0], device=projection.device, dtype=torch.bool
    )
    positive = labels[:, None].eq(labels[None]) & ~identity
    logits_masked = similarity.masked_fill(identity, float("-inf"))
    log_probability = similarity - torch.logsumexp(logits_masked, dim=1, keepdim=True)
    positive_count = positive.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return projection.sum() * 0.0
    mean_positive = (
        log_probability.masked_fill(~positive, 0.0).sum(dim=1)
        / positive_count.clamp_min(1)
    )
    return -mean_positive[valid].mean()


def tail_query_loss(
    predictions: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
    pattern_logits: torch.Tensor,
    history_projection: torch.Tensor,
    pattern_labels: torch.Tensor,
    pattern_weight: float = 0.5,
    contrastive_weight: float = 0.1,
    temperature: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    trajectory = trajectory_wta_loss(predictions, logits, target)
    pattern = F.cross_entropy(pattern_logits, pattern_labels)
    contrastive = supervised_contrastive_loss(
        history_projection, pattern_labels, temperature
    )
    total = trajectory + pattern_weight * pattern + contrastive_weight * contrastive
    return total, {
        "trajectory_wta": float(trajectory.detach()),
        "pattern_classification": float(pattern.detach()),
        "supervised_contrastive": float(contrastive.detach()),
        "total": float(total.detach()),
    }
