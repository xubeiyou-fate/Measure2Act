"""C134 expected-distance operator and differentiable Energy objective."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from experiments.dual_expected_risk.objective import ADE_SCALE


def energy_predict_optimize_objective(
    predictions: torch.Tensor,
    probabilities: torch.Tensor,
    predicted_normalized_ade_risk: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if predictions.ndim != 4 or predictions.shape[1] != 5:
        raise ValueError("predictions must have shape [B,5,T,3]")
    if probabilities.shape != predictions.shape[:2]:
        raise ValueError("probabilities must have shape [B,5]")
    if predicted_normalized_ade_risk.shape != predictions.shape[:2]:
        raise ValueError("predicted risk must have shape [B,5]")
    if target.shape != (predictions.shape[0], predictions.shape[2], 3):
        raise ValueError("target shape is incompatible with predictions")
    displacement = torch.linalg.vector_norm(
        predictions - target[:, None], dim=-1
    )
    ade = displacement.mean(dim=-1)
    normalized_target = (ade / ADE_SCALE).detach()
    centered_target = normalized_target - normalized_target.mean(dim=1, keepdim=True)
    centered_prediction = predicted_normalized_ade_risk - predicted_normalized_ade_risk.mean(
        dim=1, keepdim=True
    )
    risk_regression = F.mse_loss(centered_prediction, centered_target)
    pairwise = torch.linalg.vector_norm(
        predictions[:, :, None] - predictions[:, None, :], dim=-1
    ).mean(dim=-1)
    energy = (probabilities * ade).sum(dim=1) - 0.5 * torch.einsum(
        "bi,bij,bj->b", probabilities, pairwise, probabilities
    )
    normalized_energy = energy.mean() / ADE_SCALE
    loss = risk_regression + normalized_energy
    return loss, {
        "loss": loss.detach(),
        "risk_regression": risk_regression.detach(),
        "normalized_energy": normalized_energy.detach(),
        "energy_score": energy.mean().detach(),
        "target_normalized_ade_risk": normalized_target,
        "centered_target_normalized_ade_risk": centered_target,
    }


__all__ = ["energy_predict_optimize_objective"]
