"""Deterministic seed geometry for permutation-symmetric finite ensembles."""

from __future__ import annotations

import torch


def regular_simplex(
    vertices: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return ``vertices`` centered equidistant points in ``R^(vertices-1)``."""
    if vertices < 2:
        raise ValueError("a regular simplex requires at least two vertices")
    centering = torch.eye(vertices, dtype=torch.float64)
    centering -= torch.full((vertices, vertices), 1.0 / vertices, dtype=torch.float64)
    eigenvalues, eigenvectors = torch.linalg.eigh(centering)
    basis = eigenvectors[:, eigenvalues > 0.5]
    if basis.shape != (vertices, vertices - 1):
        raise RuntimeError("failed to construct the regular-simplex basis")
    return basis.to(dtype=dtype, device=device)
