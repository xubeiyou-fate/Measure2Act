"""Ordered history readouts for ASCENT ablations.

These modules operate only on observed history tokens. They do not alter the
trajectory candidates after generation and do not route samples or modes.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class StableDiagonalHistorySSM(nn.Module):
    """S4D-inspired bank of stable diagonal filters over observed tokens."""

    def __init__(self, dim: int = 128, state_size: int = 4) -> None:
        super().__init__()
        if state_size < 2:
            raise ValueError("state_size must be at least 2")
        self.dim = dim
        self.state_size = state_size
        self.input_norm = nn.LayerNorm(dim)
        self.input_projection = nn.Linear(dim, dim)
        self.output_norm = nn.LayerNorm(dim)
        self.output_projection = nn.Linear(dim, dim)

        time_constants = torch.logspace(0.0, math.log10(16.0), state_size)
        rates = time_constants.reciprocal().view(1, state_size).expand(dim, -1)
        inverse_softplus = torch.log(torch.expm1(rates))
        self.raw_rates = nn.Parameter(inverse_softplus.clone())
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_projection.weight, gain=0.5)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.eye_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def transition_decay(self) -> torch.Tensor:
        return torch.exp(-F.softplus(self.raw_rates))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.dim:
            raise ValueError("tokens must have shape [B, T, D]")
        projected = self.input_projection(self.input_norm(tokens))
        decay = self.transition_decay()
        state = projected.new_zeros(
            projected.shape[0], self.dim, self.state_size
        )
        for step in range(projected.shape[1]):
            observation = projected[:, step, :, None]
            state = decay[None] * state + (1.0 - decay[None]) * observation
        pooled = state.mean(dim=-1)
        return self.output_projection(self.output_norm(pooled))


class NeuralCDEHistory(nn.Module):
    """Euler-discretized observation-controlled history encoder.

    The vector field is driven by changes in the encoded observation path. This
    is a history encoder, not the future latent rollout used by C49-C51.
    """

    def __init__(self, dim: int = 128, control_dim: int = 8) -> None:
        super().__init__()
        if control_dim < 2:
            raise ValueError("control_dim must be at least 2")
        self.dim = dim
        self.control_dim = control_dim
        self.path_norm = nn.LayerNorm(dim)
        self.initial_projection = nn.Linear(dim, dim)
        self.control_projection = nn.Linear(dim, control_dim, bias=False)
        self.vector_field = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim * control_dim),
        )
        self.state_norm = nn.LayerNorm(dim)
        self.output_projection = nn.Linear(dim, dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.initial_projection.weight, gain=0.5)
        nn.init.zeros_(self.initial_projection.bias)
        nn.init.xavier_uniform_(self.control_projection.weight, gain=0.1)
        nn.init.xavier_uniform_(self.vector_field[1].weight, gain=0.5)
        nn.init.zeros_(self.vector_field[1].bias)
        nn.init.normal_(self.vector_field[-1].weight, std=0.01)
        nn.init.zeros_(self.vector_field[-1].bias)
        nn.init.eye_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.dim:
            raise ValueError("tokens must have shape [B, T, D]")
        if tokens.shape[1] < 2:
            raise ValueError("Neural CDE history requires at least two observations")
        path = self.path_norm(tokens)
        state = self.initial_projection(path[:, 0])
        scale = math.sqrt(float(self.control_dim))
        for step in range(1, path.shape[1]):
            control = self.control_projection(path[:, step] - path[:, step - 1])
            field = self.vector_field(state).view(
                state.shape[0], self.dim, self.control_dim
            )
            update = torch.einsum("bdc,bc->bd", field, control) / scale
            state = self.state_norm(state + update)
        return self.output_projection(state)


class ModeHistoryCrossAttention(nn.Module):
    """Let each fixed mode query read the complete observed token sequence."""

    def __init__(self, dim: int = 128, heads: int = 4) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.query_norm = nn.LayerNorm(dim)
        self.history_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, heads, dropout=0.0, batch_first=True
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(
        self, queries: torch.Tensor, history: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if queries.ndim != 3 or history.ndim != 3:
            raise ValueError("queries and history must have shape [B, S, D]")
        attended, weights = self.attention(
            self.query_norm(queries),
            self.history_norm(history),
            self.history_norm(history),
            need_weights=True,
            average_attn_weights=False,
        )
        return self.fusion(torch.cat((queries, attended), dim=-1)), weights
