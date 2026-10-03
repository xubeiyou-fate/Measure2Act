"""Ranking, calibration, and 3D trajectory metrics for multimodal forecasts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F


def _gather_mode(values: torch.Tensor, mode: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(values.shape[0], device=values.device)
    return values[batch, mode]


def _expected_calibration_error(
    confidence: np.ndarray,
    correct: np.ndarray,
    bins: int,
) -> float:
    if confidence.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = confidence.size
    ece = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if not np.any(mask):
            continue
        ece += mask.sum() / total * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


@torch.no_grad()
def compute_batch_metrics(
    predictions: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
    miss_threshold: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """Compute per-agent metrics without reducing over the batch.

    Args:
        predictions: Candidate trajectories with shape ``[B, K, T, 3]``.
        logits: Candidate logits with shape ``[B, K]``.
        target: Ground-truth trajectory with shape ``[B, T, 3]``.
        miss_threshold: Optional final-displacement threshold in dataset units.
    """
    if predictions.ndim != 4 or predictions.shape[-1] != 3:
        raise ValueError(f"predictions must have shape [B,K,T,3], got {predictions.shape}")
    if logits.shape != predictions.shape[:2]:
        raise ValueError(f"logits shape {logits.shape} does not match {predictions.shape[:2]}")
    if target.shape != (predictions.shape[0], predictions.shape[2], 3):
        raise ValueError(f"target shape {target.shape} is incompatible with {predictions.shape}")

    displacement = torch.linalg.vector_norm(predictions - target.unsqueeze(1), dim=-1)
    ade_by_mode = displacement.mean(dim=-1)
    fde_by_mode = displacement[..., -1]

    horizontal = torch.linalg.vector_norm(
        predictions[..., :2] - target[:, None, :, :2], dim=-1
    )
    horizontal_ade_by_mode = horizontal.mean(dim=-1)
    horizontal_fde_by_mode = horizontal[..., -1]

    altitude = torch.abs(predictions[..., 2] - target[:, None, :, 2])
    altitude_ade_by_mode = altitude.mean(dim=-1)
    altitude_fde_by_mode = altitude[..., -1]

    probabilities = torch.softmax(logits, dim=-1)
    # Energy Score evaluates the probability-weighted candidate distribution
    # instead of selecting only its closest member. Distances are averaged over
    # the complete future trajectory.
    target_distance = displacement.mean(dim=-1)
    pairwise_distance = torch.linalg.vector_norm(
        predictions[:, :, None] - predictions[:, None, :], dim=-1
    ).mean(dim=-1)
    energy_score = (probabilities * target_distance).sum(dim=-1) - 0.5 * (
        probabilities[:, :, None]
        * probabilities[:, None, :]
        * pairwise_distance
    ).sum(dim=(-1, -2))
    top1_mode = logits.argmax(dim=-1)
    oracle_ade_mode = ade_by_mode.argmin(dim=-1)
    oracle_fde_mode = fde_by_mode.argmin(dim=-1)

    descending_modes = torch.argsort(logits, dim=-1, descending=True)
    oracle_ade_rank = (
        descending_modes == oracle_ade_mode.unsqueeze(-1)
    ).to(torch.int64).argmax(dim=-1) + 1
    oracle_fde_rank = (
        descending_modes == oracle_fde_mode.unsqueeze(-1)
    ).to(torch.int64).argmax(dim=-1) + 1

    one_hot = F.one_hot(oracle_ade_mode, num_classes=logits.shape[1]).to(probabilities.dtype)
    nll = F.cross_entropy(logits, oracle_ade_mode, reduction="none")
    brier = torch.square(probabilities - one_hot).sum(dim=-1)

    result = {
        "top1_mode": top1_mode,
        "oracle_ade_mode": oracle_ade_mode,
        "oracle_fde_mode": oracle_fde_mode,
        "top1_ade": _gather_mode(ade_by_mode, top1_mode),
        "top1_fde": _gather_mode(fde_by_mode, top1_mode),
        "minade": ade_by_mode.min(dim=-1).values,
        "minfde": fde_by_mode.min(dim=-1).values,
        "top1_horizontal_ade": _gather_mode(horizontal_ade_by_mode, top1_mode),
        "top1_horizontal_fde": _gather_mode(horizontal_fde_by_mode, top1_mode),
        "top1_altitude_ade": _gather_mode(altitude_ade_by_mode, top1_mode),
        "top1_altitude_fde": _gather_mode(altitude_fde_by_mode, top1_mode),
        "oracle_ade_rank": oracle_ade_rank,
        "oracle_fde_rank": oracle_fde_rank,
        "oracle_ade_rank1": (top1_mode == oracle_ade_mode).to(torch.float32),
        "oracle_fde_rank1": (top1_mode == oracle_fde_mode).to(torch.float32),
        "confidence": probabilities.max(dim=-1).values,
        "nll": nll,
        "brier": brier,
        "energy_score": energy_score,
    }
    if miss_threshold is not None:
        result["top1_miss"] = (result["top1_fde"] > miss_threshold).to(torch.float32)
        result["minfde_miss"] = (result["minfde"] > miss_threshold).to(torch.float32)
    return result


@dataclass
class RankingMetricAccumulator:
    """Accumulate exact per-agent metrics across arbitrarily sized batches."""

    calibration_bins: int = 15
    values: Dict[str, list[np.ndarray]] = field(default_factory=dict)

    def update(self, batch_metrics: Dict[str, torch.Tensor]) -> None:
        for name, tensor in batch_metrics.items():
            array = tensor.detach().cpu().numpy()
            self.values.setdefault(name, []).append(array)

    def arrays(self) -> Dict[str, np.ndarray]:
        return {
            name: np.concatenate(chunks, axis=0)
            for name, chunks in self.values.items()
            if chunks
        }

    def summarize(self) -> Dict[str, float]:
        arrays = self.arrays()
        if not arrays:
            raise RuntimeError("No metric batches were accumulated")

        summary: Dict[str, float] = {"agents": int(arrays["top1_ade"].shape[0])}
        excluded = {"top1_mode", "oracle_ade_mode", "oracle_fde_mode", "confidence"}
        for name, values in arrays.items():
            if name in excluded:
                continue
            summary[name] = float(values.mean())

        confidence = arrays["confidence"].astype(np.float64)
        correct = arrays["oracle_ade_rank1"].astype(np.float64)
        summary["ece"] = _expected_calibration_error(
            confidence, correct, self.calibration_bins
        )
        return summary


def concatenate_metadata(chunks: Iterable[Iterable[int]]) -> np.ndarray:
    """Flatten list-valued batch metadata such as ASCENT agent IDs."""
    return np.asarray([value for chunk in chunks for value in chunk], dtype=np.int64)
