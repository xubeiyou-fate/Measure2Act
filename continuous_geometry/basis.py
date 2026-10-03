"""Analytic curve bases used by the representation audit and spline decoder."""

from __future__ import annotations

import math

import torch


def _sample_parameters(
    num_steps: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    return torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)[1:]


def bezier_basis(
    num_control_points: int,
    num_steps: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a Bernstein basis sampled at future times in ``(0, 1]``."""
    if num_control_points < 2:
        raise ValueError("Bezier curves require at least two control points")
    degree = num_control_points - 1
    u = _sample_parameters(num_steps, device=device, dtype=dtype).unsqueeze(-1)
    indices = torch.arange(num_control_points, device=device, dtype=dtype)
    coefficients = torch.tensor(
        [math.comb(degree, index) for index in range(num_control_points)],
        device=device,
        dtype=dtype,
    )
    return coefficients * (1.0 - u).pow(degree - indices) * u.pow(indices)


def bspline_basis(
    num_control_points: int,
    num_steps: int,
    *,
    degree: int = 3,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return an open-uniform clamped B-spline basis at future sample times."""
    if degree < 1:
        raise ValueError("degree must be positive")
    if num_control_points <= degree:
        raise ValueError("num_control_points must exceed degree")

    spans = num_control_points - degree
    internal = torch.arange(1, spans, device=device, dtype=dtype) / float(spans)
    knots = torch.cat(
        [
            torch.zeros(degree + 1, device=device, dtype=dtype),
            internal,
            torch.ones(degree + 1, device=device, dtype=dtype),
        ]
    )
    u = _sample_parameters(num_steps, device=device, dtype=dtype)
    endpoint_mask = u == 1
    epsilon = torch.finfo(dtype).eps * 8
    safe_u = torch.where(endpoint_mask, torch.full_like(u, 1.0 - epsilon), u)

    basis = torch.stack(
        [
            ((safe_u >= knots[index]) & (safe_u < knots[index + 1])).to(dtype)
            for index in range(knots.numel() - 1)
        ],
        dim=-1,
    )
    for current_degree in range(1, degree + 1):
        columns = []
        for index in range(basis.shape[-1] - 1):
            left_denominator = knots[index + current_degree] - knots[index]
            right_denominator = (
                knots[index + current_degree + 1] - knots[index + 1]
            )
            left = torch.zeros_like(safe_u)
            right = torch.zeros_like(safe_u)
            if float(left_denominator) > 0.0:
                left = (
                    (safe_u - knots[index]) / left_denominator * basis[:, index]
                )
            if float(right_denominator) > 0.0:
                right = (
                    (knots[index + current_degree + 1] - safe_u)
                    / right_denominator
                    * basis[:, index + 1]
                )
            columns.append(left + right)
        basis = torch.stack(columns, dim=-1)

    if endpoint_mask.any():
        basis = basis.clone()
        basis[endpoint_mask] = 0.0
        basis[endpoint_mask, -1] = 1.0
    return basis


def cosine_anchor_basis(
    num_coefficients: int,
    num_steps: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Cosine basis shifted to make every reconstructed curve start at zero."""
    if num_coefficients < 1:
        raise ValueError("num_coefficients must be positive")
    u = _sample_parameters(num_steps, device=device, dtype=dtype).unsqueeze(-1)
    frequency = torch.arange(1, num_coefficients + 1, device=device, dtype=dtype)
    return torch.cos(math.pi * u * frequency) - 1.0


def reconstruct_curve(
    control_points: torch.Tensor, basis: torch.Tensor
) -> torch.Tensor:
    """Evaluate batched 3D control points against a shared curve basis."""
    if control_points.ndim < 2 or control_points.shape[-1] != 3:
        raise ValueError("control_points must end with [control_points, 3]")
    if basis.ndim != 2 or basis.shape[1] != control_points.shape[-2]:
        raise ValueError("basis/control-point dimensions do not match")
    return torch.einsum("tc,...cd->...td", basis, control_points)


def fit_fixed_endpoint_curve(
    targets: torch.Tensor,
    basis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Least-squares fit with the current position and final future fixed exactly."""
    if targets.ndim != 3 or targets.shape[-1] != 3:
        raise ValueError("targets must have shape [batch, steps, 3]")
    if basis.shape[0] != targets.shape[1] or basis.shape[1] < 2:
        raise ValueError("basis does not match targets")

    batch = targets.shape[0]
    controls = targets.new_zeros((batch, basis.shape[1], 3))
    controls[:, -1] = targets[:, -1]
    if basis.shape[1] > 2:
        middle_basis = basis[:, 1:-1]
        adjusted = targets - basis[:, -1].view(1, -1, 1) * targets[:, -1:]
        pseudo_inverse = torch.linalg.pinv(middle_basis)
        controls[:, 1:-1] = torch.einsum("ct,btd->bcd", pseudo_inverse, adjusted)
    reconstruction = reconstruct_curve(controls, basis)
    return controls, reconstruction
