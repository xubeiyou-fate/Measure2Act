"""Time-resolved continuous scene encoding for the forecasting model."""

from __future__ import annotations

import math

import torch
from torch import nn


class _AgentTimeBlock(nn.Module):
    """Full attention over actor-time nodes with continuous 3D relation bias."""

    relation_dim = 8

    def __init__(self, dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("agent-time attention dimension must divide heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.relation_bias = nn.Sequential(
            nn.Linear(self.relation_dim, dim),
            nn.GELU(),
            nn.Linear(dim, heads),
        )
        self.projection = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )

    def forward(
        self,
        nodes: torch.Tensor,
        relations: torch.Tensor,
        valid_pairs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scenes, node_count, _ = nodes.shape
        qkv = self.qkv(self.norm1(nodes)).view(
            scenes, node_count, 3, self.heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        scores = torch.einsum("sihd,sjhd->shij", query, key) * self.scale
        scores = scores + self.relation_bias(relations).permute(0, 3, 1, 2)
        scores = scores.masked_fill(~valid_pairs[:, None], -1e4)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * valid_pairs[:, None].to(attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        message = torch.einsum("shij,sjhd->sihd", attention, value).reshape(
            scenes, node_count, self.dim
        )
        nodes = nodes + self.dropout(self.projection(message))
        nodes = nodes + self.dropout(self.mlp(self.norm2(nodes)))
        return nodes, attention


class SpatioTemporalHistoryEncoder(nn.Module):
    """Encode all observed actor-time states before producing actor contexts.

    ``cross_actor=False`` is the parameter-exact temporal-memory control. With
    ``cross_actor=True``, nodes may attend across actors in the same scene.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        depth: int = 2,
        cross_actor: bool = True,
        position_scale_km: float = 5.0,
        altitude_scale_km: float = 1.0,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("agent-time encoder depth must be positive")
        self.dim = dim
        self.heads = heads
        self.cross_actor = cross_actor
        self.position_scale_km = position_scale_km
        self.altitude_scale_km = altitude_scale_km
        self.blocks = nn.ModuleList(
            _AgentTimeBlock(dim, heads) for _ in range(depth)
        )
        self.pool_norm = nn.LayerNorm(dim)
        self.pool_query = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.pool_query, std=0.02)

    @staticmethod
    def _scene_layout(
        scene_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if scene_index.ndim != 1 or scene_index.numel() == 0:
            raise ValueError("scene_index must be a nonempty vector")
        if not bool(torch.all(scene_index[1:] >= scene_index[:-1])):
            raise ValueError("actors must be contiguous and ordered by scene")
        _, inverse, counts = torch.unique(
            scene_index.to(torch.long),
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        starts = torch.cumsum(counts, dim=0) - counts
        rank = torch.arange(scene_index.numel(), device=scene_index.device)
        rank = rank - torch.repeat_interleave(starts, counts)
        valid_actor = (
            torch.arange(int(counts.max()), device=scene_index.device)[None]
            < counts[:, None]
        )
        return inverse, rank, counts, valid_actor

    def _pack(
        self,
        values: torch.Tensor,
        inverse: torch.Tensor,
        rank: torch.Tensor,
        valid_actor: torch.Tensor,
    ) -> torch.Tensor:
        packed = values.new_zeros(
            valid_actor.shape[0], valid_actor.shape[1], *values.shape[1:]
        )
        packed[inverse, rank] = values
        return packed

    def _relations(
        self,
        positions: torch.Tensor,
        actor_index: torch.Tensor,
        time_index: torch.Tensor,
    ) -> torch.Tensor:
        relative = positions[:, None, :, :] - positions[:, :, None, :]
        scaled = relative.clone()
        scaled[..., :2] = scaled[..., :2] / self.position_scale_km
        scaled[..., 2] = scaled[..., 2] / self.altitude_scale_km
        horizontal_distance = torch.linalg.vector_norm(
            relative[..., :2], dim=-1, keepdim=True
        ) / self.position_scale_km
        altitude_separation = relative[..., 2:].abs() / self.altitude_scale_km
        time_delta = (
            time_index[None, :] - time_index[:, None]
        ).to(positions.dtype)
        time_delta = time_delta / max(int(time_index.max()), 1)
        same_actor = actor_index[:, None] == actor_index[None, :]
        same_time = time_index[:, None] == time_index[None, :]
        scenes = positions.shape[0]
        return torch.cat(
            (
                scaled,
                horizontal_distance,
                altitude_separation,
                time_delta[None, :, :, None].expand(scenes, -1, -1, -1),
                same_actor[None, :, :, None]
                .expand(scenes, -1, -1, -1)
                .to(positions.dtype),
                same_time[None, :, :, None]
                .expand(scenes, -1, -1, -1)
                .to(positions.dtype),
            ),
            dim=-1,
        )

    def forward(
        self,
        history_tokens: torch.Tensor,
        global_positions: torch.Tensor,
        scene_index: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | bool]]:
        if history_tokens.ndim != 3 or global_positions.shape[:2] != history_tokens.shape[:2]:
            raise ValueError("history tokens and positions must have shape [actors,time,*]")
        actors, steps, _ = history_tokens.shape
        inverse, rank, counts, valid_actor = self._scene_layout(scene_index)
        packed_tokens = self._pack(
            history_tokens, inverse, rank, valid_actor
        )
        packed_positions = self._pack(
            global_positions, inverse, rank, valid_actor
        )
        max_actors = valid_actor.shape[1]
        node_count = max_actors * steps
        nodes = packed_tokens.reshape(valid_actor.shape[0], node_count, self.dim)
        positions = packed_positions.reshape(valid_actor.shape[0], node_count, 3)
        node_valid = valid_actor[:, :, None].expand(-1, -1, steps).reshape(
            valid_actor.shape[0], node_count
        )
        actor_index = torch.arange(max_actors, device=nodes.device).repeat_interleave(steps)
        time_index = torch.arange(steps, device=nodes.device).repeat(max_actors)
        same_actor = actor_index[:, None] == actor_index[None, :]
        valid_pairs = node_valid[:, :, None] & node_valid[:, None, :]
        if not self.cross_actor:
            valid_pairs = valid_pairs & same_actor[None]
        relations = self._relations(positions, actor_index, time_index)

        attention = None
        for block in self.blocks:
            nodes, attention = block(nodes, relations, valid_pairs)
        if attention is None:
            raise RuntimeError("agent-time encoder produced no attention")
        packed_output = nodes.reshape(
            valid_actor.shape[0], max_actors, steps, self.dim
        )
        actor_tokens = packed_output[inverse, rank]
        pool_scores = torch.einsum(
            "btd,d->bt", self.pool_norm(actor_tokens), self.pool_query
        ) / math.sqrt(self.dim)
        pool_weights = torch.softmax(pool_scores, dim=-1)
        actor_context = torch.einsum("bt,btd->bd", pool_weights, actor_tokens)

        cross_pair = ~same_actor
        cross_mass = (
            attention * cross_pair[None, None].to(attention.dtype)
        ).sum(dim=-1).mean(dim=1)
        cross_mass = cross_mass.reshape(
            valid_actor.shape[0], max_actors, steps
        ).mean(dim=-1)
        flat_cross_mass = cross_mass[inverse, rank]
        if not self.cross_actor:
            flat_cross_mass = torch.zeros_like(flat_cross_mass)
        return actor_context, {
            "scene_sizes": counts[inverse],
            "cross_actor": self.cross_actor,
            "cross_actor_attention_mass": flat_cross_mass,
            "temporal_pool_weights": pool_weights,
            "trajectory_residual": False,
            "learned_gate": False,
        }
