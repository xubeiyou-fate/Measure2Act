"""Scene-conditioned actor encoding for multi-aircraft trajectory prediction."""

from __future__ import annotations

import math

import torch
from torch import nn


class RelationalSceneEncoder(nn.Module):
    """Fuse actor tokens with observed relative 3D motion in the same scene.

    Scene members attend only to other actors from the same reconstructed scene.
    Singleton actors bypass the module exactly, which provides a built-in placebo
    subset for the E1 experiment.
    """

    edge_dim = 11

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        position_scale: float = 5.0,
        velocity_scale: float = 0.1,
        altitude_scale: float = 1.0,
        horizon_seconds: float = 120.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.position_scale = position_scale
        self.velocity_scale = velocity_scale
        self.altitude_scale = altitude_scale
        self.horizon_seconds = horizon_seconds

        self.actor_norm = nn.LayerNorm(embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.edge_embedding = nn.Sequential(
            nn.Linear(self.edge_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.edge_bias = nn.Sequential(
            nn.Linear(self.edge_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_heads),
        )
        self.message_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.fusion = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    @staticmethod
    def _inverse_rotation(yaw: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(yaw)
        ones = torch.ones_like(yaw)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        yaw_inverse = torch.stack(
            (
                torch.stack((cy, sy, zeros), dim=-1),
                torch.stack((-sy, cy, zeros), dim=-1),
                torch.stack((zeros, zeros, ones), dim=-1),
            ),
            dim=-2,
        )
        pitch_inverse = torch.stack(
            (
                torch.stack((cp, zeros, -sp), dim=-1),
                torch.stack((zeros, ones, zeros), dim=-1),
                torch.stack((sp, zeros, cp), dim=-1),
            ),
            dim=-2,
        )
        return torch.matmul(pitch_inverse, yaw_inverse)

    def _pack_scenes(
        self,
        actor_features: torch.Tensor,
        actor_centers: torch.Tensor,
        actor_velocities: torch.Tensor,
        yaw: torch.Tensor,
        pitch: torch.Tensor,
        scene_index: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        if scene_index.ndim != 1 or scene_index.shape[0] != actor_features.shape[0]:
            raise ValueError("scene_index must contain one scene ID per actor")
        if scene_index.numel() == 0:
            raise ValueError("scene_index cannot be empty")
        if not bool(torch.all(scene_index[1:] >= scene_index[:-1])):
            raise ValueError("actors must be contiguous and ordered by scene")

        _, inverse, counts = torch.unique(
            scene_index, sorted=True, return_inverse=True, return_counts=True
        )
        starts = torch.cumsum(counts, dim=0) - counts
        rank = torch.arange(scene_index.numel(), device=scene_index.device)
        rank = rank - torch.repeat_interleave(starts, counts)
        max_actors = int(counts.max().item())
        scene_count = counts.shape[0]
        valid = (
            torch.arange(max_actors, device=scene_index.device)[None]
            < counts[:, None]
        )

        def pack(values: torch.Tensor) -> torch.Tensor:
            shape = (scene_count, max_actors, *values.shape[1:])
            packed = values.new_zeros(shape)
            packed[inverse, rank] = values
            return packed

        packed = {
            "features": pack(actor_features),
            "centers": pack(actor_centers),
            "velocities": pack(actor_velocities),
            "yaw": pack(yaw),
            "pitch": pack(pitch),
            "valid": valid,
            "counts": counts,
        }
        return packed, inverse, rank

    def _edge_features(self, packed: dict[str, torch.Tensor]) -> torch.Tensor:
        centers = packed["centers"]
        velocities = packed["velocities"]
        relative_position = centers[:, None] - centers[:, :, None]
        relative_velocity = velocities[:, None] - velocities[:, :, None]

        rotation = self._inverse_rotation(packed["yaw"], packed["pitch"])
        local_position = torch.einsum(
            "sijc,sidc->sijd", relative_position, rotation
        )
        local_velocity = torch.einsum(
            "sijc,sidc->sijd", relative_velocity, rotation
        )

        horizontal_distance = torch.linalg.vector_norm(
            relative_position[..., :2], dim=-1, keepdim=True
        )
        distance_3d = torch.linalg.vector_norm(
            relative_position, dim=-1, keepdim=True
        )
        altitude_separation = relative_position[..., 2:].abs()
        velocity_squared = torch.square(relative_velocity).sum(dim=-1, keepdim=True)
        time_to_closest = -(
            relative_position * relative_velocity
        ).sum(dim=-1, keepdim=True) / velocity_squared.clamp_min(1e-8)
        time_to_closest = time_to_closest.clamp(0.0, self.horizon_seconds)
        closest_offset = relative_position + relative_velocity * time_to_closest
        distance_at_closest = torch.linalg.vector_norm(
            closest_offset, dim=-1, keepdim=True
        )

        return torch.cat(
            (
                local_position / self.position_scale,
                local_velocity / self.velocity_scale,
                horizontal_distance / self.position_scale,
                distance_3d / self.position_scale,
                altitude_separation / self.altitude_scale,
                time_to_closest / self.horizon_seconds,
                distance_at_closest / self.position_scale,
            ),
            dim=-1,
        )

    def forward(
        self,
        actor_features: torch.Tensor,
        actor_centers: torch.Tensor,
        actor_velocities: torch.Tensor,
        yaw: torch.Tensor,
        pitch: torch.Tensor,
        scene_index: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        packed, inverse, rank = self._pack_scenes(
            actor_features,
            actor_centers,
            actor_velocities,
            yaw,
            pitch,
            scene_index.to(dtype=torch.long),
        )
        features = packed["features"]
        scene_count, max_actors, _ = features.shape
        normalized = self.actor_norm(features)
        query = self.query(normalized).view(
            scene_count, max_actors, self.num_heads, self.head_dim
        )
        key = self.key(normalized).view(
            scene_count, max_actors, self.num_heads, self.head_dim
        )
        value = self.value(normalized).view(
            scene_count, max_actors, self.num_heads, self.head_dim
        )

        edge_features = self._edge_features(packed)
        scores = torch.einsum("sihd,sjhd->shij", query, key) * self.scale
        scores = scores + self.edge_bias(edge_features).permute(0, 3, 1, 2)

        eye = torch.eye(max_actors, dtype=torch.bool, device=features.device)[None]
        valid_pairs = (
            packed["valid"][:, :, None]
            & packed["valid"][:, None, :]
            & ~eye
        )
        scores = scores.masked_fill(~valid_pairs[:, None], -1e4)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * valid_pairs[:, None].to(attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        edge_value = self.edge_embedding(edge_features).view(
            scene_count,
            max_actors,
            max_actors,
            self.num_heads,
            self.head_dim,
        )
        neighbor_value = value[:, None].expand(
            -1, max_actors, -1, -1, -1
        )
        messages = neighbor_value + edge_value
        message = torch.einsum(
            "shij,sijhd->sihd", attention, messages
        ).reshape(scene_count, max_actors, self.embed_dim)
        message = self.message_projection(message)
        fused = self.fusion(torch.cat((features, message), dim=-1))

        flat_fused = fused[inverse, rank]
        scene_sizes = packed["counts"][inverse]
        interaction_mask = scene_sizes > 1
        output = torch.where(interaction_mask[:, None], flat_fused, actor_features)

        entropy = -(
            attention.clamp_min(1e-8).log() * attention
        ).sum(dim=-1).mean(dim=1)
        flat_entropy = entropy[inverse, rank]
        flat_entropy = torch.where(
            interaction_mask, flat_entropy, torch.zeros_like(flat_entropy)
        )
        return output, {
            "scene_sizes": scene_sizes,
            "interaction_mask": interaction_mask,
            "attention_entropy": flat_entropy,
        }
