"""Post-generation trajectory-token mode reranking for ASCENT candidates."""

from __future__ import annotations

import torch
from torch import nn


RANKING_SOURCES = ("mode_only", "endpoint_only", "pooled_mlp", "trajectory")
ROUTE_VARIANTS = ("real", "shuffled", "random")


def flight_parameters_to_tokens(flight_parameters: torch.Tensor) -> torch.Tensor:
    """Convert ``[B,K,T,3]`` speed/yaw/pitch sequences into local 3D tokens."""
    if flight_parameters.ndim != 4 or flight_parameters.shape[-1] != 3:
        raise ValueError("flight_parameters must have shape [B,K,T,3]")
    speed, yaw, pitch = flight_parameters.unbind(dim=-1)
    horizontal = speed * torch.cos(pitch)
    step = torch.stack(
        [horizontal * torch.cos(yaw), horizontal * torch.sin(yaw), speed * torch.sin(pitch)],
        dim=-1,
    )
    position = torch.cumsum(step, dim=2)
    yaw_delta = torch.diff(yaw, dim=2, prepend=yaw[:, :, :1])
    pitch_delta = torch.diff(pitch, dim=2, prepend=pitch[:, :, :1])
    time = torch.linspace(
        0.0, 1.0, flight_parameters.shape[2], device=flight_parameters.device,
        dtype=flight_parameters.dtype,
    ).view(1, 1, -1).expand_as(speed)
    return torch.cat(
        [
            position,
            speed.unsqueeze(-1),
            torch.sin(yaw).unsqueeze(-1),
            torch.cos(yaw).unsqueeze(-1),
            torch.sin(pitch).unsqueeze(-1),
            torch.cos(pitch).unsqueeze(-1),
            torch.sin(yaw_delta).unsqueeze(-1),
            torch.sin(pitch_delta).unsqueeze(-1),
            time.unsqueeze(-1),
        ],
        dim=-1,
    )


def route_descriptors(tokens: torch.Tensor) -> torch.Tensor:
    """Summarize a full token sequence without discarding vertical or turn dynamics."""
    if tokens.ndim < 3 or tokens.shape[-1] != 11:
        raise ValueError("tokens must end in [T,11]")
    return torch.cat(
        [tokens.mean(dim=-2), tokens.std(dim=-2, unbiased=False), tokens[..., -1, :]],
        dim=-1,
    )


class AirRouteStageMScorer(nn.Module):
    """Residual scorer that cannot modify ASCENT's generated candidates."""

    def __init__(
        self,
        source: str = "trajectory",
        mode_dim: int = 128,
        hidden_dim: int = 128,
        heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if source not in RANKING_SOURCES:
            raise ValueError(f"source must be one of {RANKING_SOURCES}, got {source!r}")
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.source = source
        self.hidden_dim = hidden_dim
        if source in {"mode_only", "pooled_mlp", "trajectory"}:
            self.mode_projection = nn.Sequential(
                nn.Linear(mode_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
            )
        if source in {"pooled_mlp", "trajectory"}:
            self.token_projection = nn.Sequential(
                nn.Linear(11, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
            )
        if source in {"endpoint_only", "pooled_mlp", "trajectory"}:
            self.endpoint_projection = nn.Sequential(
                nn.Linear(11, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
            )
        if source == "trajectory":
            self.trajectory_attention = nn.MultiheadAttention(
                hidden_dim, heads, dropout=dropout, batch_first=True
            )
            self.trajectory_norm = nn.LayerNorm(hidden_dim)
            self.trajectory_ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.trajectory_ffn_norm = nn.LayerNorm(hidden_dim)
        feature_multipliers = {
            "mode_only": 1,
            "endpoint_only": 1,
            "pooled_mlp": 3,
            "trajectory": 3,
        }
        input_dim = hidden_dim * feature_multipliers[source]
        self.score_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        mode_features: torch.Tensor,
        flight_parameters: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> torch.Tensor:
        if mode_features.shape[:2] != flight_parameters.shape[:2]:
            raise ValueError("mode features and flight parameters disagree on [B,K]")
        if base_logits.shape != mode_features.shape[:2]:
            raise ValueError("base logits must have shape [B,K]")

        batch, modes = base_logits.shape
        raw_tokens = flight_parameters_to_tokens(flight_parameters)

        if self.source == "mode_only":
            mode_context = self.mode_projection(mode_features)
            rank_features = mode_context
        elif self.source == "endpoint_only":
            endpoint_context = self.endpoint_projection(raw_tokens[:, :, -1])
            rank_features = endpoint_context
        elif self.source == "pooled_mlp":
            mode_context = self.mode_projection(mode_features)
            token_context = self.token_projection(raw_tokens)
            endpoint_context = self.endpoint_projection(raw_tokens[:, :, -1])
            pooled = token_context.mean(dim=2)
            rank_features = torch.cat([mode_context, pooled, endpoint_context], dim=-1)
        else:
            mode_context = self.mode_projection(mode_features)
            token_context = self.token_projection(raw_tokens)
            endpoint_context = self.endpoint_projection(raw_tokens[:, :, -1])
            query = mode_context.reshape(batch * modes, 1, self.hidden_dim)
            memory = token_context.reshape(
                batch * modes, token_context.shape[2], self.hidden_dim
            )
            attended, _ = self.trajectory_attention(query, memory, memory, need_weights=False)
            summary = self.trajectory_norm(query + attended)
            summary = self.trajectory_ffn_norm(summary + self.trajectory_ffn(summary))
            summary = summary.reshape(batch, modes, self.hidden_dim)
            rank_features = torch.cat([mode_context, summary, endpoint_context], dim=-1)

        delta = self.score_head(rank_features).squeeze(-1)
        return base_logits + self.residual_scale * delta

    def extra_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class AirRouteStageMRouteScorer(nn.Module):
    """Trajectory-token scorer conditioned on train-only route prototypes."""

    def __init__(
        self,
        route_centers: torch.Tensor,
        descriptor_mean: torch.Tensor,
        descriptor_scale: torch.Tensor,
        mode_dim: int = 128,
        hidden_dim: int = 128,
        heads: int = 4,
        dropout: float = 0.0,
        route_topk: int = 8,
        route_variant: str = "real",
        refinement_steps: int = 0,
        variant_seed: int = 20260715,
    ) -> None:
        super().__init__()
        if route_variant not in ROUTE_VARIANTS:
            raise ValueError(f"route_variant must be one of {ROUTE_VARIANTS}")
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if route_centers.ndim != 2 or route_centers.shape[1] != 33:
            raise ValueError("route_centers must have shape [M,33]")
        self.hidden_dim = hidden_dim
        self.route_topk = min(route_topk, route_centers.shape[0])
        self.route_variant = route_variant
        self.refinement_steps = refinement_steps
        self.register_buffer("route_keys", route_centers.float().clone())
        self.register_buffer("descriptor_mean", descriptor_mean.float().clone())
        self.register_buffer("descriptor_scale", descriptor_scale.float().clone())
        generator = torch.Generator().manual_seed(variant_seed)
        if route_variant == "real":
            route_values = route_centers.float().clone()
        elif route_variant == "shuffled":
            route_values = route_centers[torch.randperm(route_centers.shape[0], generator=generator)].float()
        else:
            route_values = torch.randn(
                route_centers.shape, generator=generator, dtype=torch.float32
            )
        self.register_buffer("route_values", route_values)

        self.mode_projection = nn.Sequential(
            nn.Linear(mode_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.token_projection = nn.Sequential(
            nn.Linear(11, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.endpoint_projection = nn.Sequential(
            nn.Linear(11, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.route_projection = nn.Sequential(
            nn.Linear(33, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.route_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.route_norm = nn.LayerNorm(hidden_dim)
        self.route_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.route_ffn_norm = nn.LayerNorm(hidden_dim)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        mode_features: torch.Tensor,
        flight_parameters: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> torch.Tensor:
        batch, modes = base_logits.shape
        raw_tokens = flight_parameters_to_tokens(flight_parameters)
        token_context = self.token_projection(raw_tokens)
        pooled_context = token_context.mean(dim=2)
        mode_context = self.mode_projection(mode_features)
        endpoint_context = self.endpoint_projection(raw_tokens[:, :, -1])

        descriptor = route_descriptors(raw_tokens)
        standardized = (descriptor - self.descriptor_mean) / self.descriptor_scale
        flat = standardized.reshape(batch * modes, -1)
        nearest = torch.cdist(flat, self.route_keys).topk(
            self.route_topk, largest=False, dim=-1
        ).indices
        route_memory = self.route_projection(self.route_values[nearest])
        query = self.query_norm(mode_context + pooled_context).reshape(
            batch * modes, 1, self.hidden_dim
        )
        for _ in range(1 + self.refinement_steps):
            attended, _ = self.route_attention(
                query, route_memory, route_memory, need_weights=False
            )
            query = self.route_norm(query + attended)
            query = self.route_ffn_norm(query + self.route_ffn(query))
        route_context = query.reshape(batch, modes, self.hidden_dim)
        rank_features = torch.cat(
            [mode_context, pooled_context, endpoint_context, route_context], dim=-1
        )
        delta = self.score_head(rank_features).squeeze(-1)
        return base_logits + self.residual_scale * delta

    def extra_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def initialize_for_reproducibility(module: nn.Module, seed: int) -> None:
    """Reset linear/attention weights under a local seed without touching global data order."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for child in module.modules():
            if isinstance(child, nn.Linear):
                nn.init.xavier_uniform_(child.weight)
                if child.bias is not None:
                    nn.init.zeros_(child.bias)
            elif isinstance(child, nn.MultiheadAttention):
                nn.init.xavier_uniform_(child.in_proj_weight)
                if child.in_proj_bias is not None:
                    nn.init.zeros_(child.in_proj_bias)
        module.residual_scale.data.fill_(1.0)
