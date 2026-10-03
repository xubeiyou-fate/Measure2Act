"""Single-layer causal mode memory for direct ASCENT trajectory generation."""

from __future__ import annotations

import torch
from torch import nn


class CausalModeMemory(nn.Module):
    """Condition each direct mode on modes already generated for the same actor."""

    VALID_VARIANTS = {"latent", "physical"}

    def __init__(
        self,
        embed_dim: int,
        future_steps: int,
        step_seconds: int,
        variant: str,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"Unknown causal mode variant: {variant}")
        self.variant = variant
        indices = sorted({
            min(max(seconds // step_seconds - 1, 0), future_steps - 1)
            for seconds in (15, 30, 60, 120)
        })
        self.register_buffer("anchor_indices", torch.tensor(indices, dtype=torch.long))

        self.query_norm = nn.LayerNorm(embed_dim)
        self.memory_norm = nn.LayerNorm(embed_dim)
        self.memory_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.output_norm = nn.LayerNorm(embed_dim)
        self.physical_projector = nn.Sequential(
            nn.Linear(7, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.physical_time_embedding = nn.Parameter(
            torch.empty(len(indices), embed_dim)
        )
        self.memory_type_embedding = nn.Parameter(torch.empty(2, embed_dim))
        nn.init.normal_(self.physical_time_embedding, std=0.02)
        nn.init.normal_(self.memory_type_embedding, std=0.02)

    def encode_physical(
        self,
        local_trajectory: torch.Tensor,
        flight_parameters: torch.Tensor,
    ) -> torch.Tensor:
        """Encode decoded positions and flight states at fixed future horizons."""
        if local_trajectory.shape[:2] != flight_parameters.shape[:2]:
            raise ValueError("Trajectory and flight-parameter horizons must match")
        speed, heading, pitch = flight_parameters.unbind(dim=-1)
        horizontal_speed = speed * torch.cos(pitch)
        vertical_speed = speed * torch.sin(pitch)
        state = torch.cat(
            [
                local_trajectory,
                horizontal_speed.unsqueeze(-1),
                torch.sin(heading).unsqueeze(-1),
                torch.cos(heading).unsqueeze(-1),
                vertical_speed.unsqueeze(-1),
            ],
            dim=-1,
        )
        selected = state.index_select(1, self.anchor_indices)
        tokens = self.physical_projector(selected) + self.physical_time_embedding
        return tokens.mean(dim=1)

    def forward(
        self,
        query: torch.Tensor,
        latent_memory: list[torch.Tensor],
        physical_memory: list[torch.Tensor],
    ) -> torch.Tensor:
        query_token = self.query_norm(query).unsqueeze(1)
        if latent_memory:
            latent_tokens = torch.stack(latent_memory, dim=1)
            latent_tokens = latent_tokens + self.memory_type_embedding[0]
            memory_tokens = [latent_tokens]
            if self.variant == "physical":
                if len(physical_memory) != len(latent_memory):
                    raise ValueError("Physical and latent memories must have equal length")
                physical_tokens = torch.stack(physical_memory, dim=1)
                physical_tokens = physical_tokens + self.memory_type_embedding[1]
                memory_tokens.append(physical_tokens)
            memory = self.memory_norm(torch.cat(memory_tokens, dim=1))
            conditioned, _ = self.memory_attention(
                query_token, memory, memory, need_weights=False
            )
        else:
            conditioned = query_token
        return self.output_norm(self.feed_forward(conditioned.squeeze(1)))
