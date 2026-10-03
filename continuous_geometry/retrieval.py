"""Train-only analogue-future retrieval utilities for the P2 audit."""

from __future__ import annotations

import torch

from model.utils import ptsToLocal


def actor_local_tensors(
    observations: torch.Tensor,
    futures: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transform observed and future positions to the final observed actor frame."""
    if observations.ndim != 3 or observations.shape[-1] != 3:
        raise ValueError("observations must have shape [actors, steps, 3]")
    if futures.ndim != 3 or futures.shape[-1] != 3:
        raise ValueError("futures must have shape [actors, steps, 3]")
    centers = observations[:, -1]
    delta = observations[:, -1] - observations[:, -2]
    yaw = torch.atan2(delta[:, 1], delta[:, 0])
    horizontal = torch.linalg.vector_norm(delta[:, :2], dim=-1).clamp_min(1e-8)
    pitch = torch.atan2(delta[:, 2], horizontal)
    return (
        ptsToLocal(centers, yaw, pitch, observations),
        ptsToLocal(centers, yaw, pitch, futures),
        centers,
    )


def kinematic_descriptors(
    local_observations: torch.Tensor,
    centers: torch.Tensor,
    *,
    step_seconds: float = 5.0,
) -> torch.Tensor:
    """Build deterministic history-only descriptors for analogue retrieval."""
    velocity = torch.diff(local_observations, dim=1) / step_seconds
    acceleration = torch.diff(velocity, dim=1) / step_seconds
    return torch.cat(
        [
            local_observations.flatten(1),
            velocity.flatten(1),
            acceleration.flatten(1),
            centers,
        ],
        dim=-1,
    )


def select_k_medoids(
    trajectories: torch.Tensor,
    modes: int,
    *,
    iterations: int = 5,
) -> torch.Tensor:
    """Select deterministic batched K-medoids from retrieved full trajectories."""
    if trajectories.ndim != 4 or trajectories.shape[-1] != 3:
        raise ValueError("trajectories must have shape [batch, pool, steps, 3]")
    batch, pool, _, _ = trajectories.shape
    if not 0 < modes <= pool:
        raise ValueError("modes must be between one and pool size")
    flattened = trajectories.flatten(2)
    distance = torch.cdist(flattened, flattened)
    batch_index = torch.arange(batch, device=trajectories.device)

    medoids = torch.zeros((batch, modes), device=trajectories.device, dtype=torch.long)
    minimum_distance = distance[:, :, 0]
    for mode_index in range(1, modes):
        next_medoid = minimum_distance.argmax(dim=-1)
        medoids[:, mode_index] = next_medoid
        candidate_distance = distance[batch_index, :, next_medoid]
        minimum_distance = torch.minimum(minimum_distance, candidate_distance)

    for _ in range(iterations):
        to_medoids = distance.gather(
            2, medoids[:, None, :].expand(batch, pool, modes)
        )
        assignment = to_medoids.argmin(dim=-1)
        updated = medoids.clone()
        for mode_index in range(modes):
            members = assignment == mode_index
            costs = (distance * members[:, None, :]).sum(dim=-1)
            costs = costs.masked_fill(~members, torch.inf)
            candidate = costs.argmin(dim=-1)
            nonempty = members.any(dim=-1)
            updated[nonempty, mode_index] = candidate[nonempty]
        if torch.equal(updated, medoids):
            break
        medoids = updated
    return medoids
