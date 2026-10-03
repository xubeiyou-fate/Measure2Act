"""Independent probabilistic trajectory baseline for E15.

The model is intentionally independent from ASCENT/MABPT: it consumes a single
actor's observed track, encodes it with a GRU, and predicts a categorical
mixture of K trajectory modes.  The mode probabilities are trained directly by
cross-entropy against the closest trajectory mode, rather than inherited from
an ASCENT support set.
"""

from __future__ import annotations

import torch
from torch import nn


class IndependentMixtureGRU(nn.Module):
    """Single-actor GRU mixture-of-trajectories predictor.

    Inputs and outputs use local coordinates relative to the last observation.
    The caller adds the last observed position back after decoding.
    """

    def __init__(
        self,
        *,
        obs_len: int = 16,
        pred_len: int = 24,
        coord_dim: int = 3,
        modes: int = 5,
        hidden_dim: int = 128,
        layers: int = 2,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if modes < 2:
            raise ValueError("E15 requires at least two mixture modes")
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.coord_dim = coord_dim
        self.modes = modes
        self.hidden_dim = hidden_dim
        self.input = nn.Linear(coord_dim * 2, hidden_dim)
        self.encoder = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.mode_embedding = nn.Embedding(modes, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, pred_len * coord_dim),
        )
        self.logit_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, modes),
        )

    def forward(self, observed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return trajectories ``[B,K,T,3]`` and logits ``[B,K]``.

        ``observed`` is ``[B,T,3]`` in local coordinates.  The second input
        channel is the first difference, which supplies a velocity cue without
        introducing airport-specific context.
        """
        if observed.ndim != 3 or observed.shape[1] != self.obs_len:
            raise ValueError(f"expected [B,{self.obs_len},3], got {tuple(observed.shape)}")
        delta = torch.zeros_like(observed)
        delta[:, 1:] = observed[:, 1:] - observed[:, :-1]
        encoded_input = self.input(torch.cat([observed, delta], dim=-1))
        encoded, _ = self.encoder(encoded_input)
        context = encoded[:, -1]
        logits = self.logit_head(context)
        mode_ids = torch.arange(self.modes, device=observed.device)
        mode_features = self.mode_embedding(mode_ids).unsqueeze(0).expand(observed.shape[0], -1, -1)
        repeated_context = context.unsqueeze(1).expand(-1, self.modes, -1)
        decoded = self.decoder(torch.cat([repeated_context, mode_features], dim=-1))
        trajectories = decoded.view(observed.shape[0], self.modes, self.pred_len, self.coord_dim)
        return trajectories, logits


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
