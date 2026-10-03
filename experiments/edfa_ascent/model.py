"""Encounter-DAG factorized ASCENT.

The primary model changes the scene probability factorization. It does not add a
trajectory residual, score residual, learned fusion gate, temporal autoregressive
decoder, or post-generation selector.
"""

from __future__ import annotations

import torch
from torch import nn

from model.ascent import Ascent
from model.utils import flight_params_to_pos, ptsToGlobal, ptsToLocal

from .relation import (
    EncounterRelationEncoder,
    PackedScenes,
    directed_acyclic_parents,
)


class EDFAAscent(Ascent):
    """ASCENT with end-to-end scene modes and cross-aircraft factorization."""

    def __init__(self, config: dict) -> None:
        base_config = {**config, "scene_interaction": False, "scene_shared_logits": False}
        super().__init__(base_config)
        self.edfa_factorized = bool(config.get("edfa_factorized", True))
        self.relation_encoder = EncounterRelationEncoder(
            embed_dim=self.embed_dim,
            heads=int(config.get("edfa_attention_heads", 4)),
        )
        self.parent_trajectory_encoder = nn.Sequential(
            nn.LayerNorm(self.future_steps * 3),
            nn.Linear(self.future_steps * 3, 2 * self.embed_dim),
            nn.GELU(),
            nn.Linear(2 * self.embed_dim, self.embed_dim),
        )
        self.conditional_decoder = nn.Sequential(
            nn.LayerNorm(3 * self.embed_dim),
            nn.Linear(3 * self.embed_dim, 2 * self.embed_dim),
            nn.GELU(),
            nn.Linear(2 * self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )
        self.scene_mode_score = nn.Sequential(
            nn.LayerNorm(2 * self.embed_dim),
            nn.Linear(2 * self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, 1),
        )

    def _encode_actor_history(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history = observations.transpose(1, 0)
        batch = history.shape[0]
        if history.shape[1] < 3:
            raise ValueError("EDFA requires at least three observed states")
        time = torch.arange(
            history.shape[1], device=history.device, dtype=history.dtype
        ).view(1, -1, 1).repeat(batch, 1, 1)
        centers = history[:, -1]
        velocities = history[:, -1] - history[:, -2]
        angle_features, yaw, pitch = self.compute_angles(history)
        position_embedding = self.pos_embed(
            torch.cat((centers, angle_features), dim=-1)
        )
        local = history
        if self.normalize_coords:
            local = ptsToLocal(centers, yaw, pitch, history).float()
        tokens = self.agent_xy_proj(local[..., :2]) + self.agent_z_proj(
            local[..., 2:]
        )
        tokens = tokens + self.agent_ts_proj(time)
        for block in self.agent_blks:
            tokens = block(tokens)
        actors = torch.max(tokens, dim=1).values + self.type_embed[0][None]
        if self.global_pos_embedding and self.normalize_coords:
            actors = actors + position_embedding
        actors = self.norm(actors)
        return actors, {
            "history": history,
            "centers": centers,
            "velocities": velocities,
            "yaw": yaw,
            "pitch": pitch,
        }

    def _decode_flat_features(
        self,
        features: torch.Tensor,
        centers: torch.Tensor,
        yaw: torch.Tensor,
        pitch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = features.shape[0]
        speed = self.fp1(features)
        heading_raw = self.fp2(features).view(count, self.future_steps, 2)
        heading_norm = torch.sqrt(heading_raw.square().sum(dim=-1) + 1e-8)
        heading = torch.atan2(
            heading_raw[..., 0] / heading_norm,
            heading_raw[..., 1] / heading_norm,
        )
        pitch_raw = self.fp3(features).view(count, self.future_steps, 2)
        pitch_norm = torch.sqrt(pitch_raw.square().sum(dim=-1) + 1e-8)
        vertical_angle = torch.atan2(
            pitch_raw[..., 0] / pitch_norm,
            pitch_raw[..., 1] / pitch_norm,
        )
        parameters = torch.stack((speed, heading, vertical_angle), dim=-1)
        local = flight_params_to_pos(
            parameters, torch.zeros_like(centers)
        )[:, 1:]
        trajectories = (
            ptsToGlobal(centers, yaw, pitch, local)
            if self.normalize_coords
            else self.loc(features).view(count, self.future_steps, 3)
        )
        return trajectories, parameters

    def _scene_logits(
        self,
        node_features: torch.Tensor,
        packed: PackedScenes,
    ) -> torch.Tensor:
        scene_sum = node_features.new_zeros((packed.scene_count, self.embed_dim))
        scene_sum.index_add_(0, packed.inverse, node_features)
        pooled = scene_sum / packed.counts[:, None]
        modes = self._mode_embeddings()[None].expand(packed.scene_count, -1, -1)
        scene = pooled[:, None].expand(-1, self.k, -1)
        return self.scene_mode_score(torch.cat((scene, modes), dim=-1)).squeeze(-1)

    def _decode_unfactorized(
        self,
        node_features: torch.Tensor,
        actor_features: torch.Tensor,
        state: dict[str, torch.Tensor],
        packed: PackedScenes,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        modes = self._mode_embeddings()[None].expand(node_features.shape[0], -1, -1)
        multi = packed.counts[packed.inverse] > 1
        scene_modes = node_features[:, None] + modes
        baseline_modes = actor_features[:, None] + modes
        features = torch.where(multi[:, None, None], scene_modes, baseline_modes)
        flat = features.reshape(-1, self.embed_dim)
        centers = state["centers"][:, None].expand(-1, self.k, -1).reshape(-1, 3)
        yaw = state["yaw"][:, None].expand(-1, self.k).reshape(-1)
        pitch = state["pitch"][:, None].expand(-1, self.k).reshape(-1)
        trajectories, parameters = self._decode_flat_features(flat, centers, yaw, pitch)
        return (
            trajectories.view(node_features.shape[0], self.k, self.future_steps, 3),
            parameters.view(node_features.shape[0], self.k, self.future_steps, 3),
            features,
        )

    def _parent_summary(
        self,
        decoded: torch.Tensor,
        selected_scenes: torch.Tensor,
        child_slots: torch.Tensor,
        parents: torch.Tensor,
        packed: PackedScenes,
        centers: torch.Tensor,
        pair_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parent_mask = parents[selected_scenes, child_slots]
        parent_index = packed.global_index[selected_scenes]
        parent_paths = decoded[parent_index]
        child_centers = centers[packed.global_index[selected_scenes, child_slots]]
        scale = decoded.new_tensor((5.0, 5.0, 1.0))
        relative = (
            parent_paths - child_centers[:, None, None, None, :]
        ) / scale
        encoded = self.parent_trajectory_encoder(
            relative.reshape(-1, self.future_steps * 3)
        ).view(relative.shape[0], relative.shape[1], self.k, self.embed_dim)
        weights = parent_mask.to(encoded.dtype)
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        trajectory_summary = (
            encoded * weights[:, :, None, None]
        ).sum(dim=1) / denominator[:, :, None]
        edges = pair_embedding[selected_scenes, child_slots]
        edge_summary = (edges * weights[..., None]).sum(dim=1) / denominator
        return trajectory_summary, edge_summary

    def _decode_factorized(
        self,
        node_features: torch.Tensor,
        actor_features: torch.Tensor,
        state: dict[str, torch.Tensor],
        relation_aux: dict,
        relation_labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        packed: PackedScenes = relation_aux["packed"]
        relation_logits = relation_aux["relation_logits"]
        parents, orders = directed_acyclic_parents(
            relation_logits, packed.valid, relation_labels
        )
        batch = node_features.shape[0]
        decoded = node_features.new_zeros((batch, self.k, self.future_steps, 3))
        parameters = node_features.new_zeros((batch, self.k, self.future_steps, 3))
        features = node_features.new_zeros((batch, self.k, self.embed_dim))
        mode_bank = self._mode_embeddings()
        for rank in range(packed.max_actors):
            selected_scenes = torch.nonzero(
                packed.counts > rank, as_tuple=False
            ).flatten()
            child_slots = orders[selected_scenes, rank]
            global_index = packed.global_index[selected_scenes, child_slots]
            multi = packed.counts[selected_scenes] > 1
            base = node_features[global_index, None] + mode_bank[None]
            parent_summary, edge_summary = self._parent_summary(
                decoded,
                selected_scenes,
                child_slots,
                parents,
                packed,
                state["centers"],
                relation_aux["pair_embedding"],
            )
            conditioned = self.conditional_decoder(
                torch.cat(
                    (
                        base,
                        parent_summary,
                        edge_summary[:, None].expand(-1, self.k, -1),
                    ),
                    dim=-1,
                )
            )
            baseline = actor_features[global_index, None] + mode_bank[None]
            feature = torch.where(multi[:, None, None], conditioned, baseline)
            flat_feature = feature.reshape(-1, self.embed_dim)
            centers = state["centers"][global_index, None].expand(-1, self.k, -1)
            yaw = state["yaw"][global_index, None].expand(-1, self.k)
            pitch = state["pitch"][global_index, None].expand(-1, self.k)
            trajectory, flight_parameters = self._decode_flat_features(
                flat_feature,
                centers.reshape(-1, 3),
                yaw.reshape(-1),
                pitch.reshape(-1),
            )
            decoded[global_index] = trajectory.view(
                global_index.shape[0], self.k, self.future_steps, 3
            )
            parameters[global_index] = flight_parameters.view(
                global_index.shape[0], self.k, self.future_steps, 3
            )
            features[global_index] = feature
        return decoded, parameters, features, parents, orders

    def forward(self, data: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
        if "adj" not in data:
            raise KeyError("EDFA requires data['adj'] scene indices")
        actor_features, state = self._encode_actor_history(data["obs_traj"])
        context_permutation = data.get("neighbor_permutation")
        node_features, relation_aux = self.relation_encoder(
            actor_features,
            state["history"],
            state["yaw"],
            state["pitch"],
            data["adj"],
            context_permutation=context_permutation,
        )
        packed: PackedScenes = relation_aux["packed"]
        graph_source = data.get("graph_source", "predicted")
        if graph_source not in {"predicted", "oracle"}:
            raise ValueError("graph_source must be predicted or oracle")
        relation_labels = data.get("relation_labels") if graph_source == "oracle" else None
        if graph_source == "oracle" and relation_labels is None:
            raise KeyError("oracle graph_source requires relation_labels")

        if self.edfa_factorized:
            trajectories, parameters, mode_features, parents, orders = self._decode_factorized(
                node_features, actor_features, state, relation_aux, relation_labels
            )
        else:
            trajectories, parameters, mode_features = self._decode_unfactorized(
                node_features, actor_features, state, packed
            )
            parents = torch.zeros(
                (packed.scene_count, packed.max_actors, packed.max_actors),
                dtype=torch.bool,
                device=actor_features.device,
            )
            orders = torch.arange(
                packed.max_actors, device=actor_features.device
            )[None].expand(packed.scene_count, -1)

        independent_logits = self.pi(mode_features).squeeze(-1)
        scene_logits = self._scene_logits(node_features, packed)
        multi = packed.counts[packed.inverse] > 1
        logits = torch.where(
            multi[:, None], scene_logits[packed.inverse], independent_logits
        )
        auxiliary = {
            "flight_params": parameters,
            "actor_centers": state["centers"],
            "actor_angles": state["yaw"],
            "actor_pitch": state["pitch"],
            "actor_context": actor_features,
            "mode_features": mode_features,
            "scene_mode_logits": scene_logits,
            "relation_logits": relation_aux["relation_logits"],
            "relation_pair_embedding": relation_aux["pair_embedding"],
            "scene_packing": packed,
            "parents": parents,
            "orders": orders,
            "scene_interaction": {
                "interaction_mask": relation_aux["interaction_mask"],
                "attention_entropy": relation_aux["attention_entropy"],
                "scene_sizes": packed.counts[packed.inverse],
            },
            "edfa": {
                "factorized": self.edfa_factorized,
                "graph_source": graph_source,
                "trajectory_residual": False,
                "score_residual": False,
                "learned_fusion_gate": False,
                "temporal_autoregression": False,
                "post_generation_selector": False,
            },
        }
        return trajectories, logits, auxiliary
