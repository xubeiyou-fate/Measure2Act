"""Proper scoring objectives for an equal-weight finite trajectory ensemble."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def _validate(
    prediction: torch.Tensor,
    target: torch.Tensor,
    axis_scale: torch.Tensor,
) -> torch.Tensor:
    if prediction.ndim != 4 or prediction.shape[-1] != 3:
        raise ValueError("prediction must have shape [B,K,T,3]")
    if target.shape != (prediction.shape[0], prediction.shape[2], 3):
        raise ValueError("target must have shape [B,T,3]")
    scale = torch.as_tensor(
        axis_scale, dtype=prediction.dtype, device=prediction.device
    )
    if scale.shape != (3,) or not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
        raise ValueError("axis_scale must contain three positive finite values")
    return scale


def trajectory_energy_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    axis_scale: torch.Tensor,
) -> torch.Tensor:
    """Per-sample Energy Score of a uniform empirical trajectory distribution."""
    scale = _validate(prediction, target, axis_scale)
    batch, modes, steps, coordinates = prediction.shape
    normalization = math.sqrt(steps * coordinates)
    residual = (prediction - target[:, None]) / scale
    target_distance = torch.linalg.vector_norm(
        residual.reshape(batch, modes, -1), dim=-1
    ) / normalization
    pairwise = (prediction[:, :, None] - prediction[:, None, :]) / scale
    pairwise_distance = torch.linalg.vector_norm(
        pairwise.reshape(batch, modes, modes, -1), dim=-1
    ) / normalization
    return target_distance.mean(dim=1) - 0.5 * pairwise_distance.mean(dim=(1, 2))


def temporal_variogram_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    axis_scale: torch.Tensor,
    lags: Sequence[int] = (1, 3, 6, 12),
) -> torch.Tensor:
    """Variogram Score over same-axis temporal pairs at fixed prediction lags."""
    scale = _validate(prediction, target, axis_scale)
    scaled_prediction = prediction / scale
    scaled_target = target / scale
    lag_scores = []
    for lag in lags:
        if lag <= 0 or lag >= prediction.shape[2]:
            raise ValueError(f"lag {lag} is invalid for {prediction.shape[2]} steps")
        target_variation = (
            scaled_target[:, lag:] - scaled_target[:, :-lag]
        ).abs()
        ensemble_variation = (
            scaled_prediction[:, :, lag:] - scaled_prediction[:, :, :-lag]
        ).abs().mean(dim=1)
        lag_scores.append(
            (target_variation - ensemble_variation).square().mean(dim=(1, 2))
        )
    return torch.stack(lag_scores, dim=-1).mean(dim=-1)


def proper_set_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    axis_scale: torch.Tensor,
    *,
    endpoint_indices: Sequence[int] = (5, 11, 17, 23),
    variogram_lags: Sequence[int] = (1, 3, 6, 12),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Equal-weight mean of trajectory ES, endpoint ES, and temporal VS."""
    _validate(prediction, target, axis_scale)
    if not endpoint_indices:
        raise ValueError("at least one endpoint index is required")
    if min(endpoint_indices) < 0 or max(endpoint_indices) >= prediction.shape[2]:
        raise ValueError("endpoint index is outside the prediction horizon")

    trajectory = trajectory_energy_score(prediction, target, axis_scale)
    endpoint_terms = [
        trajectory_energy_score(
            prediction[:, :, index : index + 1],
            target[:, index : index + 1],
            axis_scale,
        )
        for index in endpoint_indices
    ]
    endpoint = torch.stack(endpoint_terms, dim=-1).mean(dim=-1)
    variogram = temporal_variogram_score(
        prediction, target, axis_scale, lags=variogram_lags
    )
    per_sample = (trajectory + endpoint + variogram) / 3.0
    components = {
        "trajectory_energy": trajectory.mean(),
        "endpoint_energy": endpoint.mean(),
        "temporal_variogram": variogram.mean(),
        "total": per_sample.mean(),
    }
    return components["total"], components
