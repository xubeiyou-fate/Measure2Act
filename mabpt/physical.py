"""Training-envelope physical plausibility metrics for MABPT-ASCENT."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch


FEATURES = (
    "horizontal_speed_km_per_second",
    "absolute_vertical_speed_km_per_second",
    "horizontal_acceleration_km_per_second2",
    "vertical_acceleration_km_per_second2",
    "absolute_turn_rate_radians_per_second",
)
_TORCH_QUANTILE_MAX_ELEMENTS = 2**24


def _with_mode_axis(positions: torch.Tensor) -> torch.Tensor:
    if positions.ndim == 3:
        positions = positions[:, None]
    if positions.ndim != 4 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape [B,T,3] or [B,K,T,3]")
    if positions.shape[2] < 3:
        raise ValueError("at least three future positions are required")
    if not bool(torch.isfinite(positions).all()):
        raise ValueError("positions must be finite")
    return positions.to(torch.float64)


def kinematic_features(
    positions: torch.Tensor,
    *,
    initial_position: torch.Tensor | None = None,
    stride_seconds: float = 5.0,
) -> dict[str, torch.Tensor]:
    """Derive model-independent kinematics in dataset physical units."""
    positions = _with_mode_axis(positions)
    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be positive")
    if initial_position is not None:
        if initial_position.shape != (positions.shape[0], 3):
            raise ValueError("initial_position must have shape [B,3]")
        initial = initial_position.to(dtype=positions.dtype, device=positions.device)
        initial = initial[:, None, None].expand(-1, positions.shape[1], -1, -1)
        positions = torch.cat((initial, positions), dim=2)

    velocity = positions.diff(dim=2) / stride_seconds
    horizontal_velocity = velocity[..., :2]
    horizontal_speed = torch.linalg.vector_norm(horizontal_velocity, dim=-1)
    vertical_speed = velocity[..., 2]
    horizontal_acceleration = torch.linalg.vector_norm(
        horizontal_velocity.diff(dim=2) / stride_seconds,
        dim=-1,
    )
    vertical_acceleration = (
        vertical_speed.diff(dim=2) / stride_seconds
    ).abs()
    heading = torch.atan2(horizontal_velocity[..., 1], horizontal_velocity[..., 0])
    heading_change = heading.diff(dim=2)
    wrapped_change = torch.atan2(torch.sin(heading_change), torch.cos(heading_change))
    turn_rate = wrapped_change.abs() / stride_seconds
    return {
        "horizontal_speed_km_per_second": horizontal_speed,
        "absolute_vertical_speed_km_per_second": vertical_speed.abs(),
        "horizontal_acceleration_km_per_second2": horizontal_acceleration,
        "vertical_acceleration_km_per_second2": vertical_acceleration,
        "absolute_turn_rate_radians_per_second": turn_rate,
    }


def fit_training_envelope(
    features: Mapping[str, torch.Tensor],
    *,
    lower_quantile: float = 0.005,
    upper_quantile: float = 0.995,
) -> dict[str, dict[str, float]]:
    """Fit fixed plausibility bounds from training trajectories only."""
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("invalid quantile bounds")
    envelope: dict[str, dict[str, float]] = {}
    for name in FEATURES:
        values = features[name].detach().to(torch.float64).reshape(-1).cpu()
        if values.numel() == 0 or not bool(torch.isfinite(values).all()):
            raise ValueError(f"invalid training feature: {name}")
        quantiles = [lower_quantile, upper_quantile]
        if values.numel() > _TORCH_QUANTILE_MAX_ELEMENTS:
            bounds = np.quantile(values.numpy(), quantiles, method="linear")
        else:
            bounds = torch.quantile(
                values,
                torch.tensor(quantiles, dtype=torch.float64),
            ).numpy()
        envelope[name] = {
            "lower": float(bounds[0]),
            "upper": float(bounds[1]),
            "lower_quantile": lower_quantile,
            "upper_quantile": upper_quantile,
            "training_samples": int(values.numel()),
        }
    return envelope


def physical_summary(
    features: Mapping[str, torch.Tensor],
    envelope: Mapping[str, Mapping[str, float]],
    *,
    mode_probabilities: torch.Tensor | None = None,
) -> dict[str, dict[str, float]]:
    """Summarize probability-weighted support violations and feature means."""
    first = features[FEATURES[0]]
    batch, modes = first.shape[:2]
    if mode_probabilities is None:
        weights = torch.full(
            (batch, modes),
            1.0 / modes,
            dtype=torch.float64,
            device=first.device,
        )
    else:
        if mode_probabilities.shape != (batch, modes):
            raise ValueError("mode_probabilities must have shape [B,K]")
        weights = mode_probabilities.to(dtype=torch.float64, device=first.device)
        if bool((weights < 0).any()) or not bool(torch.isfinite(weights).all()):
            raise ValueError("mode_probabilities must be finite and nonnegative")
        totals = weights.sum(dim=1, keepdim=True)
        if bool((totals <= 0).any()):
            raise ValueError("each probability row must have positive mass")
        weights = weights / totals

    result: dict[str, dict[str, float]] = {}
    for name in FEATURES:
        values = features[name].to(torch.float64)
        if values.shape[:2] != (batch, modes):
            raise ValueError(f"feature shape mismatch: {name}")
        lower = float(envelope[name]["lower"])
        upper = float(envelope[name]["upper"])
        lower_rate_by_mode = (values < lower).to(torch.float64).mean(dim=2)
        upper_rate_by_mode = (values > upper).to(torch.float64).mean(dim=2)
        mean_by_mode = values.mean(dim=2)
        lower_rate = (lower_rate_by_mode * weights).sum(dim=1).mean()
        upper_rate = (upper_rate_by_mode * weights).sum(dim=1).mean()
        weighted_mean = (mean_by_mode * weights).sum(dim=1).mean()
        result[name] = {
            "mean": float(weighted_mean),
            "below_training_envelope_rate": float(lower_rate),
            "above_training_envelope_rate": float(upper_rate),
            "outside_training_envelope_rate": float(lower_rate + upper_rate),
        }
    return result


__all__ = [
    "FEATURES",
    "fit_training_envelope",
    "kinematic_features",
    "physical_summary",
]
