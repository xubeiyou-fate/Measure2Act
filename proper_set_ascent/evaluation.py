"""C15 development evaluation with equal-weight distribution metrics."""

from __future__ import annotations

import time

import torch

from proper_set_ascent.loss import proper_set_loss
from tail_query.evaluation import TrajectoryGroupAccumulator


def constant_velocity_fde(dataset) -> torch.Tensor:
    history = dataset.obs_traj.permute(0, 2, 1)
    future = dataset.pred_traj.permute(0, 2, 1)
    velocity = history[:, -1] - history[:, -2]
    endpoint = history[:, -1] + velocity * 120.0
    return torch.linalg.vector_norm(endpoint - future[:, -1], dim=-1)


@torch.no_grad()
def evaluate_equal_weight_set(
    model,
    loader,
    device: torch.device,
    axis_scale: torch.Tensor,
    dev_cv_fde: torch.Tensor,
    tail_threshold: float,
    limit_batches: int | None = None,
) -> dict:
    model.eval()
    overall = TrajectoryGroupAccumulator(modes=5)
    tail = TrajectoryGroupAccumulator(modes=5)
    component_sums = {
        "trajectory_energy": 0.0,
        "endpoint_energy": 0.0,
        "temporal_variogram": 0.0,
        "total": 0.0,
    }
    actors = 0
    cursor = 0
    finite = True
    started = time.perf_counter()
    for batch_index, data in enumerate(loader):
        if limit_batches is not None and batch_index >= limit_batches:
            break
        data = {
            name: value.to(device) if torch.is_tensor(value) else value
            for name, value in data.items()
        }
        prediction, _, auxiliary = model(data)
        target = data["pred_traj"].transpose(1, 0)
        logits = prediction.new_zeros(prediction.shape[:2])
        count = target.shape[0]
        mask = torch.ones(count, dtype=torch.bool, device=device)
        tail_mask = (dev_cv_fde[cursor : cursor + count] > tail_threshold).to(device)
        overall.update(prediction, logits, target, mask)
        tail.update(prediction, logits, target, tail_mask)
        _, components = proper_set_loss(prediction, target, axis_scale)
        for name, value in components.items():
            component_sums[name] += float(value) * count
        finite = finite and bool(torch.isfinite(prediction).all())
        if "flight_params" in auxiliary:
            finite = finite and bool(torch.isfinite(auxiliary["flight_params"]).all())
        actors += count
        cursor += count
    if actors == 0:
        raise RuntimeError("C15 evaluation received no actors")
    return {
        "overall": overall.summarize(),
        "tail": tail.summarize(),
        "proper_set_objective": {
            name: value / actors for name, value in component_sums.items()
        },
        "tail_definition": {
            "metric": "train_p80_constant_velocity_120s_fde",
            "threshold_km": tail_threshold,
            "evaluated_fraction": float(
                (dev_cv_fde[:cursor] > tail_threshold).to(torch.float32).mean()
            ),
        },
        "physical_outputs_finite": finite,
        "equal_weight_logits": True,
        "actors": actors,
        "evaluation_seconds": time.perf_counter() - started,
    }
