"""Candidate-set diagnostics that do not alter model predictions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@torch.no_grad()
def batch_coverage_statistics(
    predictions: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return per-agent coverage statistics for ``[B,K,T,3]`` predictions."""
    if predictions.ndim != 4 or predictions.shape[-1] != 3:
        raise ValueError("predictions must have shape [B,K,T,3]")
    batch, modes, steps, _ = predictions.shape
    if logits.shape != (batch, modes):
        raise ValueError("logits must have shape [B,K]")
    if target.shape != (batch, steps, 3):
        raise ValueError("target must have shape [B,T,3]")

    error = torch.linalg.vector_norm(predictions - target[:, None], dim=-1)
    ade = error.mean(dim=-1)
    fde = error[..., -1]
    training_winner = error.sum(dim=-1).argmin(dim=-1)
    score_order = logits.argsort(dim=-1, descending=True)

    score_topk_ade = []
    score_topk_fde = []
    for count in range(1, modes + 1):
        selected = score_order[:, :count]
        score_topk_ade.append(ade.gather(1, selected).min(dim=-1).values)
        score_topk_fde.append(fde.gather(1, selected).min(dim=-1).values)

    pair_ade = []
    pair_fde = []
    for first in range(modes):
        for second in range(first + 1, modes):
            distance = torch.linalg.vector_norm(
                predictions[:, first] - predictions[:, second], dim=-1
            )
            pair_ade.append(distance.mean(dim=-1))
            pair_fde.append(distance[:, -1])

    return {
        "training_winner": training_winner,
        "score_top1": score_order[:, 0],
        "ade_by_mode": ade,
        "fde_by_mode": fde,
        "score_topk_ade": torch.stack(score_topk_ade, dim=-1),
        "score_topk_fde": torch.stack(score_topk_fde, dim=-1),
        "prefix_ade": torch.cummin(ade, dim=-1).values,
        "prefix_fde": torch.cummin(fde, dim=-1).values,
        "pairwise_ade": torch.stack(pair_ade, dim=-1),
        "pairwise_fde": torch.stack(pair_fde, dim=-1),
    }


def _distribution_summary(indices: np.ndarray, modes: int) -> dict:
    counts = np.bincount(indices.astype(np.int64), minlength=modes)
    probabilities = counts / counts.sum()
    positive = probabilities[probabilities > 0]
    entropy = float(-(positive * np.log(positive)).sum())
    return {
        "counts": counts.tolist(),
        "fractions": probabilities.tolist(),
        "entropy_nats": entropy,
        "effective_modes": float(np.exp(entropy)),
    }


def _distance_summary(values: np.ndarray) -> dict:
    flat = values.reshape(-1).astype(np.float64)
    return {
        "mean": float(flat.mean()),
        "p05": float(np.quantile(flat, 0.05)),
        "p25": float(np.quantile(flat, 0.25)),
        "median": float(np.quantile(flat, 0.50)),
        "p75": float(np.quantile(flat, 0.75)),
        "p95": float(np.quantile(flat, 0.95)),
        "fraction_below_0_05": float((flat < 0.05).mean()),
        "fraction_below_0_10": float((flat < 0.10).mean()),
        "fraction_below_0_20": float((flat < 0.20).mean()),
    }


@dataclass
class CoverageAccumulator:
    """Accumulate exact D0 diagnostics while keeping only compact arrays."""

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
            raise RuntimeError("No coverage statistics were accumulated")
        modes = arrays["ade_by_mode"].shape[1]
        return {
            "agents": int(arrays["ade_by_mode"].shape[0]),
            "modes": int(modes),
            "wta_winner_distribution": _distribution_summary(
                arrays["training_winner"], modes
            ),
            "score_top1_distribution": _distribution_summary(
                arrays["score_top1"], modes
            ),
            "per_mode_mean_ade": arrays["ade_by_mode"].mean(axis=0).tolist(),
            "per_mode_mean_fde": arrays["fde_by_mode"].mean(axis=0).tolist(),
            "score_ordered_topk_minade": arrays["score_topk_ade"].mean(axis=0).tolist(),
            "score_ordered_topk_minfde": arrays["score_topk_fde"].mean(axis=0).tolist(),
            "fixed_prefix_minade": arrays["prefix_ade"].mean(axis=0).tolist(),
            "fixed_prefix_minfde": arrays["prefix_fde"].mean(axis=0).tolist(),
            "top1_to_oracle_ade_gap": float(
                arrays["score_topk_ade"][:, 0].mean()
                - arrays["score_topk_ade"][:, -1].mean()
            ),
            "top1_to_oracle_fde_gap": float(
                arrays["score_topk_fde"][:, 0].mean()
                - arrays["score_topk_fde"][:, -1].mean()
            ),
            "pairwise_ade": _distance_summary(arrays["pairwise_ade"]),
            "pairwise_fde": _distance_summary(arrays["pairwise_fde"]),
        }
