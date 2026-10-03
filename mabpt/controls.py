"""Registered ASCENT capacity controls for MABPT experiments E6-E8."""

from __future__ import annotations

import torch
from torch import nn

from experiments.metric_exact.model import ascent_config
from model.ascent import Ascent
from model.utils import flight_params_to_pos, ptsToGlobal


CONTROL_VARIANTS = (
    "b0_extended",
    "widened_ascent",
    "shared_encoder_dual_decoder",
)


def _head(dim: int, output: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(dim, 256),
        nn.ReLU(),
        nn.Linear(256, dim),
        nn.ReLU(),
        nn.Linear(dim, output),
    )


class PhysicalDecoderBranch(nn.Module):
    """A second native ASCENT physical decoder without another encoder."""

    def __init__(self, *, modes: int = 5, dim: int = 128, future_steps: int = 24):
        super().__init__()
        self.modes = modes
        self.dim = dim
        self.future_steps = future_steps
        self.mode_embed = nn.Parameter(torch.empty(modes, dim))
        self.fp1 = _head(dim, future_steps)
        self.fp2 = _head(dim, future_steps * 2)
        self.fp3 = _head(dim, future_steps * 2)
        self.pi = _head(dim, 1)
        nn.init.normal_(self.mode_embed, std=0.02)

    def forward(
        self,
        actor_context: torch.Tensor,
        actor_centers: torch.Tensor,
        yaw: torch.Tensor,
        pitch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch = actor_context.shape[0]
        features = actor_context[:, None] + self.mode_embed[None]
        speed = self.fp1(features)
        heading_raw = self.fp2(features).view(
            batch, self.modes, self.future_steps, 2
        )
        heading_norm = torch.sqrt(heading_raw.square().sum(dim=-1) + 1e-8)
        heading = torch.atan2(
            heading_raw[..., 0] / heading_norm,
            heading_raw[..., 1] / heading_norm,
        )
        pitch_raw = self.fp3(features).view(
            batch, self.modes, self.future_steps, 2
        )
        pitch_norm = torch.sqrt(pitch_raw.square().sum(dim=-1) + 1e-8)
        pitch_angle = torch.atan2(
            pitch_raw[..., 0] / pitch_norm,
            pitch_raw[..., 1] / pitch_norm,
        )
        flight_parameters = torch.stack((speed, heading, pitch_angle), dim=-1)
        initial = torch.zeros_like(
            actor_centers[:, None].expand(batch, self.modes, 3).reshape(-1, 3)
        )
        local = flight_params_to_pos(
            flight_parameters.reshape(batch * self.modes, self.future_steps, 3),
            initial,
        ).view(batch, self.modes, -1, 3)[:, :, 1:]
        trajectories = local.clone()
        for mode in range(self.modes):
            trajectories[:, mode] = ptsToGlobal(
                actor_centers, yaw, pitch, local[:, mode]
            )
        return trajectories, self.pi(features).squeeze(-1), {
            "flight_params": flight_parameters,
            "mode_features": features,
        }


class SharedEncoderDualDecoderAscent(nn.Module):
    """One ASCENT encoder and two independently parameterized K=5 decoders."""

    def __init__(self, *, batch_size: int = 256):
        super().__init__()
        self.primary = Ascent(
            ascent_config("B0_signed_coupled", batch_size=batch_size)
        )
        self.secondary = PhysicalDecoderBranch()

    def forward(
        self, data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        primary_support, primary_logits, primary_aux = self.primary(data)
        secondary_support, secondary_logits, secondary_aux = self.secondary(
            primary_aux["actor_context"],
            primary_aux["actor_centers"],
            primary_aux["actor_angles"],
            primary_aux["actor_pitch"],
        )
        return (
            torch.cat((primary_support, secondary_support), dim=1),
            torch.cat((primary_logits, secondary_logits), dim=1),
            {
                "primary": primary_aux,
                "secondary": secondary_aux,
                "shared_encoder": True,
                "encoder_forward_count": 1,
                "decoder_branches": 2,
                "trajectory_residual": False,
                "learned_gate": False,
            },
        )


def build_control_model(variant: str, *, batch_size: int = 256) -> nn.Module:
    if variant == "b0_extended":
        return Ascent(ascent_config("B0_signed_coupled", batch_size=batch_size))
    if variant == "widened_ascent":
        config = ascent_config("B0_signed_coupled", batch_size=batch_size)
        config["embed_dim"] = 192
        return Ascent(config)
    if variant == "shared_encoder_dual_decoder":
        return SharedEncoderDualDecoderAscent(batch_size=batch_size)
    raise ValueError(f"unknown MABPT control: {variant}")


__all__ = [
    "CONTROL_VARIANTS",
    "PhysicalDecoderBranch",
    "SharedEncoderDualDecoderAscent",
    "build_control_model",
]
