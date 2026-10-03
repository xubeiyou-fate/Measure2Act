"""Metrics with independent probability measure and top1 decision inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import torch
from torch.nn import functional as F


def _gather(values: torch.Tensor, mode: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(values.shape[0], device=values.device)
    return values[batch, mode]


def _ece(confidence: np.ndarray, correct: np.ndarray, bins: int) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        upper = confidence <= edges[index + 1] if index == bins - 1 else confidence < edges[index + 1]
        mask = (confidence >= edges[index]) & upper
        if np.any(mask):
            result += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(result)


@torch.no_grad()
def compute_batch_metrics(
    predictions: torch.Tensor,
    probabilities: torch.Tensor,
    decision_mode: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Evaluate a finite measure and a separately supplied Bayes decision."""
    if predictions.ndim != 4 or predictions.shape[-1] != 3:
        raise ValueError("predictions must have shape [B,K,T,3]")
    batch, modes, steps, _ = predictions.shape
    if probabilities.shape != (batch, modes):
        raise ValueError("probabilities must have shape [B,K]")
    if decision_mode.shape != (batch,):
        raise ValueError("decision_mode must have shape [B]")
    if target.shape != (batch, steps, 3):
        raise ValueError("target shape is incompatible with predictions")
    if bool((probabilities < -1e-6).any()) or not torch.allclose(
        probabilities.sum(dim=1), torch.ones_like(probabilities[:, 0]), atol=1e-5, rtol=0.0
    ):
        raise ValueError("probabilities must lie on the simplex")
    if bool(((decision_mode < 0) | (decision_mode >= modes)).any()):
        raise ValueError("decision_mode is out of range")

    displacement = torch.linalg.vector_norm(predictions - target[:, None], dim=-1)
    ade = displacement.mean(dim=-1)
    fde = displacement[..., -1]
    horizontal = torch.linalg.vector_norm(
        predictions[..., :2] - target[:, None, :, :2], dim=-1
    )
    altitude = torch.abs(predictions[..., 2] - target[:, None, :, 2])
    pairwise = torch.linalg.vector_norm(
        predictions[:, :, None] - predictions[:, None, :], dim=-1
    ).mean(dim=-1)
    energy = (probabilities * ade).sum(dim=1) - 0.5 * torch.einsum(
        "bi,bij,bj->b", probabilities, pairwise, probabilities
    )
    oracle_ade = ade.argmin(dim=1)
    oracle_fde = fde.argmin(dim=1)
    probability_order = probabilities.argsort(dim=1, descending=True, stable=True)
    oracle_ade_rank = (probability_order == oracle_ade[:, None]).to(torch.int64).argmax(dim=1) + 1
    oracle_fde_rank = (probability_order == oracle_fde[:, None]).to(torch.int64).argmax(dim=1) + 1
    one_hot = F.one_hot(oracle_ade, num_classes=modes).to(probabilities.dtype)
    return {
        "top1_mode": decision_mode,
        "oracle_ade_mode": oracle_ade,
        "oracle_fde_mode": oracle_fde,
        "top1_ade": _gather(ade, decision_mode),
        "top1_fde": _gather(fde, decision_mode),
        "minade": ade.min(dim=1).values,
        "minfde": fde.min(dim=1).values,
        "top1_horizontal_ade": _gather(horizontal.mean(dim=-1), decision_mode),
        "top1_horizontal_fde": _gather(horizontal[..., -1], decision_mode),
        "top1_altitude_ade": _gather(altitude.mean(dim=-1), decision_mode),
        "top1_altitude_fde": _gather(altitude[..., -1], decision_mode),
        "oracle_ade_rank": oracle_ade_rank,
        "oracle_fde_rank": oracle_fde_rank,
        "oracle_ade_rank1": (decision_mode == oracle_ade).to(torch.float32),
        "oracle_fde_rank1": (decision_mode == oracle_fde).to(torch.float32),
        "confidence": _gather(probabilities, decision_mode),
        # Use the dtype floor only as a finite representation of -log(0). A
        # larger ad-hoc floor would make this evaluator fail exact legacy
        # cross-entropy replay for very small but valid softmax probabilities.
        "nll": -_gather(
            probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny),
            oracle_ade,
        ).log(),
        "brier": torch.square(probabilities - one_hot).sum(dim=1),
        "energy_score": energy,
    }


@dataclass
class RankingMetricAccumulator:
    calibration_bins: int = 15
    values: Dict[str, list[np.ndarray]] = field(default_factory=dict)

    def update(self, metrics: Dict[str, torch.Tensor]) -> None:
        for name, value in metrics.items():
            self.values.setdefault(name, []).append(value.detach().cpu().numpy())

    def arrays(self) -> Dict[str, np.ndarray]:
        return {name: np.concatenate(chunks) for name, chunks in self.values.items()}

    def summarize(self) -> Dict[str, float]:
        arrays = self.arrays()
        if not arrays:
            raise RuntimeError("no C134 metrics accumulated")
        excluded = {"top1_mode", "oracle_ade_mode", "oracle_fde_mode", "confidence"}
        result = {"agents": int(arrays["top1_ade"].shape[0])}
        for name, values in arrays.items():
            if name not in excluded:
                result[name] = float(values.mean())
        result["ece"] = _ece(
            arrays["confidence"].astype(np.float64),
            arrays["oracle_ade_rank1"].astype(np.float64),
            self.calibration_bins,
        )
        result["minfde_p95"] = float(np.quantile(arrays["minfde"], 0.95))
        tail = arrays["minfde"] >= result["minfde_p95"]
        result["tail_minfde"] = float(arrays["minfde"][tail].mean())
        return result


__all__ = ["RankingMetricAccumulator", "compute_batch_metrics"]
