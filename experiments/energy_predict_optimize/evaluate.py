"""Exact C134 evaluation with separate probability and top1 decision inputs."""

from __future__ import annotations

import math

import numpy as np
import torch

from .evaluation import RankingMetricAccumulator, compute_batch_metrics


@torch.no_grad()
def evaluate(model, loader, device: torch.device, *, tail_threshold: float) -> dict[str, object]:
    model.eval()
    accumulator = RankingMetricAccumulator()
    winner_counts = np.zeros(5, dtype=np.int64)
    tail_sum = 0.0
    tail_count = 0
    physical_violations = 0
    control_count = 0
    disagreement = 0
    actors = 0
    for data in loader:
        data = {
            name: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for name, value in data.items()
        }
        target = data["pred_traj"].transpose(1, 0)
        predictions, probabilities, decision_mode, auxiliary = model(data)
        metrics = compute_batch_metrics(
            predictions, probabilities, decision_mode, target
        )
        accumulator.update(metrics)
        winner_counts += np.bincount(
            metrics["oracle_fde_mode"].cpu().numpy(), minlength=5
        )
        endpoint_travel = torch.linalg.vector_norm(
            target[:, -1] - data["obs_traj"][-1], dim=-1
        )
        tail = endpoint_travel >= tail_threshold
        tail_sum += float(metrics["minfde"][tail].sum())
        tail_count += int(tail.sum())
        controls = auxiliary["horizontal_control"]
        physical_violations += int((controls < 0).sum())
        control_count += int(controls.numel())
        disagreement += int((decision_mode != probabilities.argmax(dim=1)).sum())
        actors += int(decision_mode.shape[0])
    overall = accumulator.summarize()
    fractions = winner_counts / winner_counts.sum()
    nonzero = fractions[fractions > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum())
    overall.update(
        {
            "tail_threshold": tail_threshold,
            "tail_samples": tail_count,
            "tail_minfde": tail_sum / max(tail_count, 1),
            "winner_distribution": {
                "counts": winner_counts.tolist(),
                "fractions": fractions.tolist(),
                "entropy_nats": entropy,
                "effective_modes": float(np.exp(entropy)),
            },
            "physical_violation_count": physical_violations,
            "negative_horizontal_control_rate": physical_violations
            / max(control_count, 1),
            "decision_probability_argmax_disagreement_rate": disagreement
            / max(actors, 1),
        }
    )
    if not all(
        math.isfinite(float(overall[name]))
        for name in (
            "top1_ade",
            "top1_fde",
            "minade",
            "minfde",
            "energy_score",
            "nll",
            "brier",
            "ece",
            "minfde_p95",
            "tail_minfde",
        )
    ):
        raise RuntimeError("non-finite C134 evaluation metric")
    return {"overall": overall}


__all__ = ["evaluate"]
