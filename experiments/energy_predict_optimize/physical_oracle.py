"""Deterministic continuous physical-control capacity oracle for C134 P0-B."""

from __future__ import annotations

import torch

from model.utils import ptsToGlobal, ptsToLocal


FREE_KNOTS = (2, 3, 4, 6, 8)


def piecewise_linear_basis(
    steps: int, free_knots: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Basis from a fixed zero control at t=0 to equally spaced free knots."""
    if steps < 2 or free_knots < 2:
        raise ValueError("steps and free_knots must each be at least two")
    position = torch.arange(1, steps + 1, device=device, dtype=dtype)
    position = position * (free_knots / steps)
    left = position.floor().to(torch.long).clamp(max=free_knots - 1)
    fraction = position - left.to(dtype)
    full = torch.zeros(steps, free_knots + 1, device=device, dtype=dtype)
    full.scatter_add_(1, left[:, None], (1.0 - fraction)[:, None])
    full.scatter_add_(1, (left + 1)[:, None], fraction[:, None])
    return full[:, 1:]


def pose_from_history(history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ASCENT's current center, yaw, and pitch from [B,T,3] history."""
    if history.ndim != 3 or history.shape[-1] != 3 or history.shape[1] < 2:
        raise ValueError("history must have shape [B,T,3] with T >= 2")
    center = history[:, -1]
    delta = history[:, -1] - history[:, -2]
    yaw = torch.atan2(delta[:, 1], delta[:, 0])
    horizontal = torch.linalg.vector_norm(delta[:, :2], dim=-1) + 1e-8
    pitch = torch.atan2(delta[:, 2], horizontal)
    return center, yaw, pitch


def _physicalize(local_path: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    zero = torch.zeros_like(local_path[:, :1])
    increments = torch.diff(torch.cat((zero, local_path), dim=1), dim=1)
    horizontal_speed = torch.linalg.vector_norm(increments[..., :2], dim=-1)
    safe_speed = horizontal_speed.clamp_min(torch.finfo(local_path.dtype).eps)
    heading = torch.atan2(increments[..., 1], increments[..., 0])
    pitch = torch.asin((increments[..., 2] / safe_speed).clamp(-1.0, 1.0))
    physical_increments = torch.stack(
        (
            horizontal_speed * torch.cos(heading),
            horizontal_speed * torch.sin(heading),
            horizontal_speed * torch.sin(pitch),
        ),
        dim=-1,
    )
    parameters = torch.stack((horizontal_speed, heading, pitch), dim=-1)
    return torch.cumsum(physical_increments, dim=1), parameters


def continuous_physical_oracle(
    target: torch.Tensor,
    center: torch.Tensor,
    yaw: torch.Tensor,
    pitch: torch.Tensor,
    *,
    free_knots: tuple[int, ...] = FREE_KNOTS,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Project a diagnostic target into five absolute physical control spaces."""
    if target.ndim != 3 or target.shape[-1] != 3:
        raise ValueError("target must have shape [B,T,3]")
    if len(free_knots) != 5 or len(set(free_knots)) != 5:
        raise ValueError("C134 P0-B requires five distinct control bases")
    local_target = ptsToLocal(center, yaw, pitch, target).to(torch.float64)
    candidates = []
    controls = []
    residuals = []
    for knots in free_knots:
        basis = piecewise_linear_basis(
            target.shape[1], knots, device=target.device, dtype=torch.float64
        )
        pseudo_inverse = torch.linalg.pinv(basis)
        control_positions = torch.einsum("mt,btd->bmd", pseudo_inverse, local_target)
        fitted = torch.einsum("tm,bmd->btd", basis, control_positions)
        normal_residual = torch.einsum(
            "mt,btd->bmd", basis.transpose(0, 1), fitted - local_target
        )
        physical_local, parameters = _physicalize(fitted)
        candidates.append(
            ptsToGlobal(
                center,
                yaw,
                pitch,
                physical_local.to(dtype=target.dtype),
            )
        )
        controls.append(parameters)
        residuals.append(normal_residual.abs().amax(dim=(1, 2)))
    return torch.stack(candidates, dim=1), {
        "flight_parameters": torch.stack(controls, dim=1),
        "kkt_residual": torch.stack(residuals, dim=1),
        "free_knots": torch.tensor(free_knots, device=target.device),
    }


__all__ = [
    "FREE_KNOTS",
    "continuous_physical_oracle",
    "piecewise_linear_basis",
    "pose_from_history",
]
