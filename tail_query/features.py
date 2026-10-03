"""Fixed history descriptors and whole-future 3D behavior signatures."""

from __future__ import annotations

import torch

from model.utils import ptsToLocal


HISTORY_DESCRIPTOR_DIMENSION = 30
FUTURE_SIGNATURE_NAMES = (
    "endpoint_x",
    "endpoint_y",
    "endpoint_z",
    "horizontal_path_length",
    "spatial_path_length",
    "mean_speed",
    "final_speed",
    "speed_change",
    "sin_final_heading",
    "cos_final_heading",
    "total_absolute_turn",
    "total_absolute_pitch_change",
    "altitude_range",
    "maximum_absolute_lateral_displacement",
)


def wrapped_angle_difference(current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    difference = current - previous
    return torch.atan2(torch.sin(difference), torch.cos(difference))


def localize_batch(
    observed: torch.Tensor,
    future: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Transform sampled observations and futures to the final-observation frame."""
    if observed.ndim != 3 or observed.shape[-1] != 3 or observed.shape[1] < 2:
        raise ValueError("observed must have shape [N,O,3] with O >= 2")
    if future.ndim != 3 or future.shape[0] != observed.shape[0] or future.shape[-1] != 3:
        raise ValueError("future must have shape [N,T,3]")
    center = observed[:, -1]
    displacement = observed[:, -1] - observed[:, -2]
    yaw = torch.atan2(displacement[:, 1], displacement[:, 0])
    horizontal = torch.linalg.vector_norm(displacement[:, :2], dim=-1) + 1e-8
    pitch = torch.atan2(displacement[:, 2], horizontal)
    local_observed = ptsToLocal(center, yaw, pitch, observed)
    local_future = ptsToLocal(center, yaw, pitch, future)
    return local_observed, local_future, {
        "center": center,
        "yaw": yaw,
        "pitch": pitch,
    }


def history_descriptor(local_observed: torch.Tensor) -> torch.Tensor:
    """Create the fixed 30D history-only descriptor used by the P0 probe."""
    if local_observed.ndim != 3 or local_observed.shape[1:] != (4, 3):
        raise ValueError("local_observed must have shape [N,4,3]")
    displacement = local_observed[:, 1:] - local_observed[:, :-1]
    speed = torch.linalg.vector_norm(displacement, dim=-1)
    heading = torch.atan2(displacement[..., 1], displacement[..., 0])
    heading_sincos = torch.stack((torch.sin(heading), torch.cos(heading)), dim=-1)
    descriptor = torch.cat(
        (
            local_observed.reshape(local_observed.shape[0], -1),
            displacement.reshape(local_observed.shape[0], -1),
            speed,
            heading_sincos.reshape(local_observed.shape[0], -1),
        ),
        dim=-1,
    )
    if descriptor.shape[1] != HISTORY_DESCRIPTOR_DIMENSION:
        raise RuntimeError("history descriptor dimension changed")
    return descriptor


def future_behavior_signature(
    local_future: torch.Tensor,
    step_seconds: float = 5.0,
) -> torch.Tensor:
    """Summarize a complete local future without creating coordinate prototypes."""
    if local_future.ndim != 3 or local_future.shape[-1] != 3:
        raise ValueError("local_future must have shape [N,T,3]")
    if step_seconds <= 0:
        raise ValueError("step_seconds must be positive")
    origin = torch.zeros_like(local_future[:, :1])
    states = torch.cat((origin, local_future), dim=1)
    displacement = states[:, 1:] - states[:, :-1]
    horizontal_length = torch.linalg.vector_norm(displacement[..., :2], dim=-1)
    spatial_length = torch.linalg.vector_norm(displacement, dim=-1)
    speed = spatial_length / step_seconds

    heading = torch.atan2(displacement[..., 1], displacement[..., 0])
    initial_heading = torch.zeros_like(heading[:, :1])
    heading_change = wrapped_angle_difference(
        heading,
        torch.cat((initial_heading, heading[:, :-1]), dim=1),
    )
    pitch = torch.atan2(displacement[..., 2], horizontal_length + 1e-8)
    initial_pitch = torch.zeros_like(pitch[:, :1])
    pitch_change = wrapped_angle_difference(
        pitch,
        torch.cat((initial_pitch, pitch[:, :-1]), dim=1),
    )
    altitude_range = states[..., 2].amax(dim=1) - states[..., 2].amin(dim=1)
    signature = torch.stack(
        (
            local_future[:, -1, 0],
            local_future[:, -1, 1],
            local_future[:, -1, 2],
            horizontal_length.sum(dim=1),
            spatial_length.sum(dim=1),
            speed.mean(dim=1),
            speed[:, -1],
            speed[:, -1] - speed[:, 0],
            torch.sin(heading[:, -1]),
            torch.cos(heading[:, -1]),
            heading_change.abs().sum(dim=1),
            pitch_change.abs().sum(dim=1),
            altitude_range,
            local_future[..., 1].abs().amax(dim=1),
        ),
        dim=-1,
    )
    if signature.shape[1] != len(FUTURE_SIGNATURE_NAMES):
        raise RuntimeError("future behavior signature dimension changed")
    return signature


def constant_velocity_fde(
    local_observed: torch.Tensor,
    local_future: torch.Tensor,
    observation_step_seconds: float = 5.0,
    prediction_horizon_seconds: float = 120.0,
) -> torch.Tensor:
    """Final error of a last-observation constant-velocity extrapolation."""
    if local_observed.ndim != 3 or local_observed.shape[-1] != 3:
        raise ValueError("local_observed must have shape [N,O,3]")
    if local_future.ndim != 3 or local_future.shape[-1] != 3:
        raise ValueError("local_future must have shape [N,T,3]")
    if observation_step_seconds <= 0 or prediction_horizon_seconds <= 0:
        raise ValueError("time intervals must be positive")
    last_displacement = local_observed[:, -1] - local_observed[:, -2]
    endpoint = last_displacement * (
        prediction_horizon_seconds / observation_step_seconds
    )
    return torch.linalg.vector_norm(endpoint - local_future[:, -1], dim=-1)
