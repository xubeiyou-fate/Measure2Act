"""Time-aware mode/state decoder used by the C4 ASCENT experiment."""

from __future__ import annotations

import torch
from torch import nn


class ModeStateQueryDecoder(nn.Module):
    """Decode absolute kinematic states from mode and horizon queries.

    The mode feature represents a long-horizon intention.  A separate learned
    query is instantiated for every future timestamp, and self-attention lets
    those state queries model temporal evolution before physical integration.
    No trajectory residual, score gate, or post-generation correction is used.
    """

    def __init__(
        self,
        embed_dim: int,
        future_steps: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.future_steps = int(future_steps)
        self.time_embed = nn.Parameter(torch.empty(self.future_steps, embed_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.control_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 5),
        )
        nn.init.normal_(self.time_embed, std=0.02)

    def forward(
        self, mode_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return flight parameters ``[B,K,T,3]`` and state features.

        The five raw controls are speed, heading sine/cosine, and pitch
        sine/cosine.  Angles are normalized analytically before integration.
        """
        if mode_features.ndim != 3:
            raise ValueError("mode_features must have shape [B,K,D]")
        batch, modes, embed_dim = mode_features.shape
        if embed_dim != self.time_embed.shape[-1]:
            raise ValueError("mode feature dimension does not match decoder")

        queries = mode_features.unsqueeze(2) + self.time_embed.view(
            1, 1, self.future_steps, embed_dim
        )
        queries = queries.reshape(batch * modes, self.future_steps, embed_dim)
        attended, _ = self.attn(queries, queries, queries, need_weights=False)
        states = self.norm1(queries + attended)
        states = self.norm2(states + self.ffn(states))
        raw = self.control_head(states).view(batch, modes, self.future_steps, 5)

        speed = raw[..., 0]
        heading = torch.atan2(raw[..., 1], raw[..., 2])
        pitch = torch.atan2(raw[..., 3], raw[..., 4])
        parameters = torch.stack((speed, heading, pitch), dim=-1)
        return parameters, states.view(batch, modes, self.future_steps, embed_dim)
