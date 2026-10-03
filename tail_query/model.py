"""History-conditioned future-pattern mode queries for ASCENT."""

from __future__ import annotations

import torch
from torch import nn

from model.ascent import Ascent
from model.utils import flight_params_to_pos, ptsToGlobal, ptsToLocal
from tail_query.features import (
    constant_velocity_fde,
    future_behavior_signature,
)


class FuturePatternAssigner(nn.Module):
    """Training/evaluation-only assignment to the frozen train-only codebook."""

    def __init__(self, codebook: dict) -> None:
        super().__init__()
        if int(codebook["clusters"]) != 5:
            raise ValueError("C7 P1 requires the fixed five-pattern codebook")
        self.clusters = int(codebook["clusters"])
        self.tail_threshold_cv_fde = float(codebook["tail_threshold_cv_fde"])
        self.register_buffer("signature_mean", codebook["signature_mean"].float())
        self.register_buffer("signature_scale", codebook["signature_scale"].float())
        self.register_buffer(
            "cluster_centers", codebook["cluster_centers_standardized"].float()
        )

    @classmethod
    def from_path(cls, path, map_location="cpu") -> "FuturePatternAssigner":
        codebook = torch.load(path, map_location=map_location, weights_only=False)
        if not codebook.get("p1_authorized", False):
            raise ValueError("the supplied C7 codebook did not authorize P1")
        return cls(codebook)

    def forward(
        self,
        data: dict,
        target: torch.Tensor,
        auxiliary: dict,
    ) -> dict[str, torch.Tensor]:
        local_future = ptsToLocal(
            auxiliary["actor_centers"],
            auxiliary["actor_angles"],
            auxiliary["actor_pitch"],
            target,
        )
        observed = data["obs_traj"].transpose(1, 0)
        local_observed = ptsToLocal(
            auxiliary["actor_centers"],
            auxiliary["actor_angles"],
            auxiliary["actor_pitch"],
            observed,
        )
        signature = future_behavior_signature(local_future)
        standardized = (signature - self.signature_mean) / self.signature_scale
        distance = torch.cdist(standardized, self.cluster_centers)
        labels = distance.argmin(dim=-1)
        if local_observed.shape[1] == 16:
            tail_observed = local_observed[:, (0, 5, 10, 15)]
        elif local_observed.shape[1] == 4:
            tail_observed = local_observed
        else:
            raise ValueError("future-pattern assignment requires 4 or 16 observations")
        cv_fde = constant_velocity_fde(tail_observed, local_future)
        return {
            "labels": labels,
            "tail_mask": cv_fde > self.tail_threshold_cv_fde,
            "cv_fde": cv_fde,
            "local_future": local_future,
        }


class TailQueryAscent(nn.Module):
    """ASCENT with direct history-conditioned, pattern-specific mode features."""

    def __init__(
        self,
        backbone: Ascent,
        patterns: int = 5,
        hidden_dim: int = 128,
        projection_dim: int = 64,
    ) -> None:
        super().__init__()
        if backbone.k != patterns:
            raise ValueError("pattern count must match ASCENT mode count")
        if (
            backbone.scene_interaction
            or backbone.causal_mode_generation
            or backbone.continuous_geometry
            or backbone.wind_relative_motion
            or backbone.mode_state_query
        ):
            raise ValueError("TailQuery P1 must remain an isolated ASCENT variant")
        self.backbone = backbone
        self.patterns = patterns
        self.embed_dim = backbone.embed_dim
        self.future_steps = backbone.future_steps
        self.pattern_classifier = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, patterns),
        )
        self.pattern_embedding = nn.Parameter(torch.empty(patterns, self.embed_dim))
        self.query_generator = nn.Sequential(
            nn.Linear(self.embed_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )
        self.history_projection = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )
        nn.init.normal_(self.pattern_embedding, std=0.02)
        self.backbone.mode1_embed.requires_grad_(False)

    def _decode(
        self,
        feature: torch.Tensor,
        auxiliary: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, modes, _ = feature.shape
        logits = self.backbone.pi(feature).squeeze(-1)
        speed = self.backbone.fp1(feature)
        yaw_pair = self.backbone.fp2(feature).view(batch, modes, -1, 2)
        yaw_norm = torch.linalg.vector_norm(yaw_pair, dim=-1).clamp_min(1e-8)
        yaw = torch.atan2(yaw_pair[..., 0] / yaw_norm, yaw_pair[..., 1] / yaw_norm)
        pitch_pair = self.backbone.fp3(feature).view(batch, modes, -1, 2)
        pitch_norm = torch.linalg.vector_norm(pitch_pair, dim=-1).clamp_min(1e-8)
        pitch = torch.atan2(
            pitch_pair[..., 0] / pitch_norm,
            pitch_pair[..., 1] / pitch_norm,
        )
        flight_parameters = torch.stack((speed, yaw, pitch), dim=-1)
        initial = torch.zeros(
            batch * modes,
            3,
            device=feature.device,
            dtype=feature.dtype,
        )
        local = flight_params_to_pos(
            flight_parameters.view(batch * modes, -1, 3), initial
        ).view(batch, modes, -1, 3)[:, :, 1:]
        if self.backbone.normalize_coords:
            center = auxiliary["actor_centers"]
            actor_yaw = auxiliary["actor_angles"]
            actor_pitch = auxiliary["actor_pitch"]
            global_modes = []
            for mode_index in range(modes):
                global_modes.append(ptsToGlobal(
                    center, actor_yaw, actor_pitch, local[:, mode_index]
                ))
            trajectory = torch.stack(global_modes, dim=1)
        else:
            trajectory = self.backbone.loc(feature).view(
                batch, modes, self.future_steps, 3
            )
        return trajectory, logits, flight_parameters

    def forward(self, data: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
        _, _, backbone_auxiliary = self.backbone(data)
        actor_context = (
            backbone_auxiliary["mode_features"][:, 0]
            - self.backbone.mode1_embed[0]
        )
        pattern_logits = self.pattern_classifier(actor_context)
        pattern_probability = torch.softmax(pattern_logits, dim=-1)
        batch = actor_context.shape[0]
        actor = actor_context[:, None].expand(-1, self.patterns, -1)
        pattern = self.pattern_embedding[None].expand(batch, -1, -1)
        query_input = torch.cat(
            (actor, pattern, pattern_probability.unsqueeze(-1)), dim=-1
        )
        mode_features = self.query_generator(query_input)
        trajectory, logits, flight_parameters = self._decode(
            mode_features, backbone_auxiliary
        )
        projection = torch.nn.functional.normalize(
            self.history_projection(actor_context), dim=-1
        )
        return trajectory, logits, {
            "flight_params": flight_parameters,
            "actor_centers": backbone_auxiliary["actor_centers"],
            "actor_angles": backbone_auxiliary["actor_angles"],
            "actor_pitch": backbone_auxiliary["actor_pitch"],
            "mode_features": mode_features,
            "pattern_logits": pattern_logits,
            "pattern_probability": pattern_probability,
            "history_projection": projection,
            "tail_query": {
                "patterns": self.patterns,
                "direct_dynamic_queries": True,
                "static_query_residual": False,
                "trajectory_residual": False,
                "learned_gate": False,
                "reranker": False,
            },
        }
