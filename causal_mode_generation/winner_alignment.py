"""Read-only diagnostics for WTA assignment and evaluation-metric alignment."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@torch.no_grad()
def batch_winner_alignment(
    predictions: torch.Tensor,
    target: torch.Tensor,
    horizon_indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if predictions.ndim != 4 or predictions.shape[-1] != 3:
        raise ValueError("predictions must have shape [B,K,T,3]")
    if target.shape != (predictions.shape[0], predictions.shape[2], 3):
        raise ValueError("target must have shape [B,T,3]")
    if horizon_indices.ndim != 1:
        raise ValueError("horizon_indices must be one-dimensional")
    if int(horizon_indices.min()) < 0 or int(horizon_indices.max()) >= predictions.shape[2]:
        raise ValueError("horizon index is outside the prediction window")

    distance = torch.linalg.vector_norm(predictions - target[:, None], dim=-1)
    training_winner = distance.sum(dim=-1).argmin(dim=-1)
    horizon_distance = distance.index_select(2, horizon_indices)
    horizon_winners = horizon_distance.argmin(dim=1)
    batch = torch.arange(predictions.shape[0], device=predictions.device)
    final_winner = horizon_winners[:, -1]
    final_oracle = horizon_distance[:, :, -1].min(dim=-1).values
    training_final = horizon_distance[batch, training_winner, -1]
    distinct_horizon_winners = torch.tensor(
        [
            horizon_winners[row].unique().numel()
            for row in range(horizon_winners.shape[0])
        ],
        device=horizon_winners.device,
    )
    return {
        "training_winner": training_winner,
        "horizon_winners": horizon_winners,
        "training_final_error": training_final,
        "final_oracle_error": final_oracle,
        "training_final_regret": training_final - final_oracle,
        "training_final_agreement": (training_winner == final_winner).float(),
        "horizon_transition": (
            horizon_winners[:, 1:] != horizon_winners[:, :-1]
        ).float(),
        "distinct_horizon_winners": distinct_horizon_winners,
        "oracle_error_by_horizon": horizon_distance.min(dim=1).values,
    }


def _mode_distribution(values: np.ndarray, modes: int) -> dict:
    counts = np.bincount(values.astype(np.int64), minlength=modes)
    fractions = counts / counts.sum()
    positive = fractions[fractions > 0]
    entropy = float(-(positive * np.log(positive)).sum())
    return {
        "counts": counts.tolist(),
        "fractions": fractions.tolist(),
        "effective_modes": float(np.exp(entropy)),
    }


@dataclass
class WinnerAlignmentAccumulator:
    modes: int
    horizon_seconds: tuple[int, ...]
    values: dict[str, list[np.ndarray]] = field(default_factory=dict)

    def update(self, statistics: dict[str, torch.Tensor]) -> None:
        for name, value in statistics.items():
            self.values.setdefault(name, []).append(value.detach().cpu().numpy())

    def summarize(self) -> dict:
        arrays = {
            name: np.concatenate(chunks, axis=0)
            for name, chunks in self.values.items()
            if chunks
        }
        if not arrays:
            raise RuntimeError("No winner-alignment statistics were accumulated")
        horizon_winners = arrays["horizon_winners"]
        return {
            "agents": int(horizon_winners.shape[0]),
            "training_vs_final_winner_agreement": float(
                arrays["training_final_agreement"].mean()
            ),
            "training_winner_final_regret_mean": float(
                arrays["training_final_regret"].mean()
            ),
            "training_winner_final_regret_p95": float(
                np.quantile(arrays["training_final_regret"], 0.95)
            ),
            "mean_distinct_horizon_winners": float(
                arrays["distinct_horizon_winners"].mean()
            ),
            "fraction_with_multiple_horizon_winners": float(
                (arrays["distinct_horizon_winners"] > 1).mean()
            ),
            "transition_rate_by_interval": {
                f"{self.horizon_seconds[index]}-{self.horizon_seconds[index + 1]}s": float(
                    arrays["horizon_transition"][:, index].mean()
                )
                for index in range(len(self.horizon_seconds) - 1)
            },
            "oracle_fde_by_horizon": {
                f"{seconds}s": float(arrays["oracle_error_by_horizon"][:, index].mean())
                for index, seconds in enumerate(self.horizon_seconds)
            },
            "training_winner_distribution": _mode_distribution(
                arrays["training_winner"], self.modes
            ),
            "endpoint_winner_distribution_by_horizon": {
                f"{seconds}s": _mode_distribution(horizon_winners[:, index], self.modes)
                for index, seconds in enumerate(self.horizon_seconds)
            },
        }
