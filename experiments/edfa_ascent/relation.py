"""Observed encounter features, future relation labels, and DAG construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class PackedScenes:
    inverse: torch.Tensor
    rank: torch.Tensor
    counts: torch.Tensor
    valid: torch.Tensor
    global_index: torch.Tensor

    @property
    def scene_count(self) -> int:
        return int(self.counts.shape[0])

    @property
    def max_actors(self) -> int:
        return int(self.valid.shape[1])

    def pack(self, values: torch.Tensor) -> torch.Tensor:
        shape = (self.scene_count, self.max_actors, *values.shape[1:])
        output = values.new_zeros(shape)
        output[self.inverse, self.rank] = values
        return output


def pack_scenes(scene_index: torch.Tensor) -> PackedScenes:
    if scene_index.ndim != 1 or scene_index.numel() == 0:
        raise ValueError("scene_index must be a non-empty vector")
    scene_index = scene_index.to(torch.long)
    if not bool(torch.all(scene_index[1:] >= scene_index[:-1])):
        raise ValueError("actors must be contiguous and ordered by scene")
    _, inverse, counts = torch.unique(
        scene_index, sorted=True, return_inverse=True, return_counts=True
    )
    starts = torch.cumsum(counts, dim=0) - counts
    rank = torch.arange(scene_index.numel(), device=scene_index.device)
    rank = rank - torch.repeat_interleave(starts, counts)
    max_actors = int(counts.max().item())
    valid = (
        torch.arange(max_actors, device=scene_index.device)[None]
        < counts[:, None]
    )
    global_index = torch.zeros(
        (counts.shape[0], max_actors), dtype=torch.long, device=scene_index.device
    )
    global_index[inverse, rank] = torch.arange(
        scene_index.numel(), device=scene_index.device
    )
    return PackedScenes(inverse, rank, counts, valid, global_index)


def scene_size_matched_permutation(scene_index: torch.Tensor) -> torch.Tensor:
    """Cyclically replace each scene with another scene of identical size."""
    packed = pack_scenes(scene_index)
    permutation = torch.arange(scene_index.numel(), device=scene_index.device)
    for count in torch.unique(packed.counts).tolist():
        scenes = torch.nonzero(packed.counts == count, as_tuple=False).flatten()
        if scenes.numel() < 2:
            continue
        donors = torch.roll(scenes, shifts=-1)
        width = int(count)
        for target_scene, donor_scene in zip(scenes.tolist(), donors.tolist()):
            target = packed.global_index[target_scene, :width]
            donor = packed.global_index[donor_scene, :width]
            permutation[target] = donor
    return permutation


def future_relation_labels(
    future: torch.Tensor,
    scene_index: torch.Tensor,
    horizontal_threshold: float = 1.0,
    vertical_threshold: float = 0.3,
    minimum_arrival_separation_steps: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, PackedScenes]:
    """Label cross-time path encounters as none, row->column, or column->row.

    The labels are training-only supervision. The primary model never reads them at
    inference time.
    """
    if future.ndim != 3 or future.shape[-1] != 3:
        raise ValueError("future must have shape [actors, time, 3]")
    packed = pack_scenes(scene_index)
    trajectories = packed.pack(future)
    scenes, actors, steps, _ = trajectories.shape
    valid_pairs = (
        packed.valid[:, :, None]
        & packed.valid[:, None, :]
        & torch.triu(torch.ones(actors, actors, dtype=torch.bool, device=future.device), diagonal=1)[None]
    )
    pair_index = torch.nonzero(valid_pairs, as_tuple=False)
    if pair_index.numel() == 0:
        labels = torch.zeros(
            (scenes, actors, actors), dtype=torch.long, device=future.device
        )
        return labels, valid_pairs, packed

    pair_scenes, pair_rows, pair_columns = pair_index.unbind(dim=-1)
    row_trajectories = trajectories[pair_scenes, pair_rows]
    column_trajectories = trajectories[pair_scenes, pair_columns]
    relative = row_trajectories[:, :, None, :] - column_trajectories[:, None, :, :]
    horizontal = torch.linalg.vector_norm(relative[..., :2], dim=-1)
    vertical = relative[..., 2].abs()
    admissible = vertical <= vertical_threshold
    score = (horizontal / horizontal_threshold).masked_fill(~admissible, float("inf"))
    flat_score = score.flatten(start_dim=-2)
    flat_index = flat_score.argmin(dim=-1)
    minimum = flat_score.gather(-1, flat_index[..., None]).squeeze(-1)
    row_time = torch.div(flat_index, steps, rounding_mode="floor")
    column_time = flat_index % steps
    difference = column_time - row_time
    encounter = minimum <= 1.0
    pair_labels = torch.zeros(pair_index.shape[0], dtype=torch.long, device=future.device)
    pair_labels = torch.where(
        encounter & (difference >= minimum_arrival_separation_steps),
        torch.ones_like(pair_labels),
        pair_labels,
    )
    pair_labels = torch.where(
        encounter & (difference <= -minimum_arrival_separation_steps),
        torch.full_like(pair_labels, 2),
        pair_labels,
    )
    labels = torch.zeros((scenes, actors, actors), dtype=torch.long, device=future.device)
    labels[pair_scenes, pair_rows, pair_columns] = pair_labels
    reverse_labels = torch.where(
        pair_labels == 1,
        torch.full_like(pair_labels, 2),
        torch.where(pair_labels == 2, torch.ones_like(pair_labels), pair_labels),
    )
    labels[pair_scenes, pair_columns, pair_rows] = reverse_labels
    return labels, valid_pairs, packed


def directed_acyclic_parents(
    relation_logits: torch.Tensor,
    valid: torch.Tensor,
    labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert pair relations to an acyclic parent matrix and actor order."""
    if relation_logits.ndim != 4 or relation_logits.shape[-1] != 3:
        raise ValueError("relation_logits must have shape [S,A,A,3]")
    scenes, actors = relation_logits.shape[:2]
    if labels is None:
        relation = relation_logits.argmax(dim=-1)
        confidence = torch.softmax(relation_logits, dim=-1).amax(dim=-1)
    else:
        relation = labels
        confidence = torch.ones_like(relation, dtype=relation_logits.dtype)
    upper = torch.triu(
        torch.ones(actors, actors, dtype=torch.bool, device=relation.device),
        diagonal=1,
    )[None]
    valid_pairs = valid[:, :, None] & valid[:, None, :] & upper
    forward = valid_pairs & (relation == 1)
    reverse = valid_pairs & (relation == 2)
    adjacency = confidence.new_zeros((scenes, actors, actors))
    adjacency = adjacency + forward.to(confidence.dtype) * confidence
    adjacency = adjacency + (
        reverse.to(confidence.dtype) * confidence
    ).transpose(1, 2)
    priority = adjacency.sum(dim=-1) - adjacency.sum(dim=-2)
    priority = priority.masked_fill(~valid, -torch.inf)
    orders = torch.argsort(priority, dim=-1, descending=True, stable=True)
    positions = torch.empty_like(orders)
    positions.scatter_(
        1,
        orders,
        torch.arange(actors, device=orders.device)[None].expand(scenes, -1),
    )
    source_precedes_target = positions[:, :, None] < positions[:, None, :]
    acyclic_adjacency = (adjacency > 0) & source_precedes_target
    parents = acyclic_adjacency.transpose(1, 2)
    return parents, orders


class EncounterRelationEncoder(nn.Module):
    """Permutation-equivariant observed-scene encoder with physical pair tokens."""

    edge_dim = 23

    def __init__(self, embed_dim: int = 128, heads: int = 4) -> None:
        super().__init__()
        if embed_dim % heads:
            raise ValueError("embed_dim must be divisible by heads")
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        self.scale = self.head_dim ** -0.5
        self.actor_norm = nn.LayerNorm(embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.edge_embedding = nn.Sequential(
            nn.Linear(self.edge_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)
        )
        self.edge_bias = nn.Sequential(
            nn.Linear(self.edge_dim, embed_dim // 2), nn.GELU(), nn.Linear(embed_dim // 2, heads)
        )
        self.node_projection = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.relation_head = nn.Sequential(
            nn.Linear(4 * embed_dim, 2 * embed_dim),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3),
        )

    @staticmethod
    def _rotation(yaw: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(yaw)
        ones = torch.ones_like(yaw)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        yaw_inverse = torch.stack(
            (
                torch.stack((cy, sy, zeros), dim=-1),
                torch.stack((-sy, cy, zeros), dim=-1),
                torch.stack((zeros, zeros, ones), dim=-1),
            ), dim=-2,
        )
        pitch_inverse = torch.stack(
            (
                torch.stack((cp, zeros, -sp), dim=-1),
                torch.stack((zeros, ones, zeros), dim=-1),
                torch.stack((sp, zeros, cp), dim=-1),
            ), dim=-2,
        )
        return torch.matmul(pitch_inverse, yaw_inverse)

    def _edge_features(
        self,
        query_history: torch.Tensor,
        context_history: torch.Tensor,
        query_yaw: torch.Tensor,
        query_pitch: torch.Tensor,
    ) -> torch.Tensor:
        query_position = query_history[:, :, -1]
        context_position = context_history[:, :, -1]
        query_velocity = query_history[:, :, -1] - query_history[:, :, -2]
        context_velocity = context_history[:, :, -1] - context_history[:, :, -2]
        query_acceleration = query_velocity - (
            query_history[:, :, -2] - query_history[:, :, -3]
        )
        context_acceleration = context_velocity - (
            context_history[:, :, -2] - context_history[:, :, -3]
        )
        relative_position = context_position[:, None] - query_position[:, :, None]
        relative_velocity = context_velocity[:, None] - query_velocity[:, :, None]
        relative_acceleration = context_acceleration[:, None] - query_acceleration[:, :, None]
        rotation = self._rotation(query_yaw, query_pitch)
        local_position = torch.einsum("sijc,sidc->sijd", relative_position, rotation)
        local_velocity = torch.einsum("sijc,sidc->sijd", relative_velocity, rotation)
        local_acceleration = torch.einsum("sijc,sidc->sijd", relative_acceleration, rotation)
        horizontal = torch.linalg.vector_norm(relative_position[..., :2], dim=-1, keepdim=True)
        vertical = relative_position[..., 2:].abs()
        relative_speed = torch.linalg.vector_norm(relative_velocity, dim=-1, keepdim=True)
        velocity_squared = relative_velocity[..., :2].square().sum(dim=-1, keepdim=True)
        tcpa = -(
            relative_position[..., :2] * relative_velocity[..., :2]
        ).sum(dim=-1, keepdim=True) / velocity_squared.clamp_min(1e-8)
        tcpa = tcpa.clamp(0.0, 120.0)
        at_closest = relative_position + relative_velocity * tcpa
        dca_horizontal = torch.linalg.vector_norm(at_closest[..., :2], dim=-1, keepdim=True)
        dca_vertical = at_closest[..., 2:].abs()
        projections = []
        for seconds in (30.0, 60.0, 90.0, 120.0):
            projected = relative_position + seconds * relative_velocity
            projections.extend((
                torch.linalg.vector_norm(projected[..., :2], dim=-1, keepdim=True) / 5.0,
                projected[..., 2:].abs(),
            ))
        return torch.cat(
            (
                local_position / 5.0,
                local_velocity / 0.1,
                local_acceleration / 0.02,
                horizontal / 5.0,
                vertical,
                relative_speed / 0.1,
                tcpa / 120.0,
                dca_horizontal / 5.0,
                dca_vertical,
                *projections,
            ), dim=-1,
        )

    def forward(
        self,
        actor_features: torch.Tensor,
        history: torch.Tensor,
        yaw: torch.Tensor,
        pitch: torch.Tensor,
        scene_index: torch.Tensor,
        context_permutation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | PackedScenes]]:
        packed = pack_scenes(scene_index)
        query_features = packed.pack(actor_features)
        query_history = packed.pack(history)
        query_yaw = packed.pack(yaw)
        query_pitch = packed.pack(pitch)
        if context_permutation is None:
            context_features = query_features
            context_history = query_history
        else:
            context_features = packed.pack(actor_features[context_permutation])
            context_history = packed.pack(history[context_permutation])

        normalized_query = self.actor_norm(query_features)
        normalized_context = self.actor_norm(context_features)
        scenes, actors, _ = query_features.shape
        query = self.query(normalized_query).view(scenes, actors, self.heads, self.head_dim)
        key = self.key(normalized_context).view(scenes, actors, self.heads, self.head_dim)
        value = self.value(normalized_context).view(scenes, actors, self.heads, self.head_dim)
        edges = self._edge_features(query_history, context_history, query_yaw, query_pitch)
        edge_embedding = self.edge_embedding(edges)
        scores = torch.einsum("sihd,sjhd->shij", query, key) * self.scale
        scores = scores + self.edge_bias(edges).permute(0, 3, 1, 2)
        eye = torch.eye(actors, dtype=torch.bool, device=actor_features.device)[None]
        valid_pairs = packed.valid[:, :, None] & packed.valid[:, None, :] & ~eye
        scores = scores.masked_fill(~valid_pairs[:, None], -1e4)
        attention = torch.softmax(scores, dim=-1) * valid_pairs[:, None]
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        messages = value[:, None] + edge_embedding.view(
            scenes, actors, actors, self.heads, self.head_dim
        )
        message = torch.einsum("shij,sijhd->sihd", attention, messages).reshape(
            scenes, actors, self.embed_dim
        )
        node = self.node_projection(torch.cat((query_features, message), dim=-1))
        singleton = packed.counts[packed.inverse] == 1
        flat_node = node[packed.inverse, packed.rank]
        flat_node = torch.where(singleton[:, None], actor_features, flat_node)

        row = query_features[:, :, None].expand(-1, -1, actors, -1)
        column = context_features[:, None].expand(-1, actors, -1, -1)
        pair_input = torch.cat((row, column, edge_embedding, row - column), dim=-1)
        relation_logits = self.relation_head(pair_input)
        entropy = -(attention.clamp_min(1e-8).log() * attention).sum(dim=-1).mean(dim=1)
        return flat_node, {
            "packed": packed,
            "pair_embedding": edge_embedding,
            "relation_logits": relation_logits,
            "attention_entropy": entropy[packed.inverse, packed.rank],
            "interaction_mask": ~singleton,
        }
