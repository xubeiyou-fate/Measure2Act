"""Self-contained C127 evaluation using the repository-audited metric kernel."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math

import numpy as np
import torch

from airroute_stage_m.evaluation import RankingMetricAccumulator, compute_batch_metrics
from experiments.edfa_ascent.objective import future_proximity_actor_mask
from experiments.edfa_ascent.relation import pack_scenes


@dataclass
class GroupAccumulator:
    modes: int = 5
    metrics: RankingMetricAccumulator = field(default_factory=RankingMetricAccumulator)
    winner_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(5, dtype=np.int64)
    )

    def update(self, batch: dict[str, torch.Tensor], mask: torch.Tensor) -> None:
        if not bool(mask.any()):
            return
        self.metrics.update({name: value[mask] for name, value in batch.items()})
        winner = batch["oracle_fde_mode"][mask].detach().cpu().numpy()
        self.winner_counts += np.bincount(winner, minlength=self.modes)

    def summarize(self) -> dict[str, object]:
        if not self.metrics.values:
            # Optional subgroup diagnostics can be empty in a valid split.
            return {
                "agents": 0,
                "empty_group": True,
                "minfde_p95": None,
                "endpoint_coverage": {},
                "winner_distribution": {
                    "counts": self.winner_counts.tolist(),
                    "fractions": [0.0] * self.modes,
                    "entropy_nats": 0.0,
                    "effective_modes": 0.0,
                },
            }
        summary = self.metrics.summarize()
        arrays = self.metrics.arrays()
        minfde = arrays["minfde"]
        fractions = self.winner_counts / self.winner_counts.sum()
        nonzero = fractions[fractions > 0]
        entropy = float(-(nonzero * np.log(nonzero)).sum())
        summary.update(
            {
                "minfde_p95": float(np.quantile(minfde, 0.95)),
                "endpoint_coverage": {
                    str(threshold): float((minfde <= threshold).mean())
                    for threshold in (0.25, 0.5, 1.0)
                },
                "winner_distribution": {
                    "counts": self.winner_counts.tolist(),
                    "fractions": fractions.tolist(),
                    "entropy_nats": entropy,
                    "effective_modes": float(np.exp(entropy)),
                },
            }
        )
        return summary


class DateAccumulator:
    def __init__(self) -> None:
        self.sums: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.counts: dict[str, int] = defaultdict(int)

    def update(
        self,
        dates: list[str],
        scene_inverse: np.ndarray,
        metrics: dict[str, np.ndarray],
    ) -> None:
        actor_dates = np.asarray(dates, dtype=object)[scene_inverse]
        for date in sorted(set(dates)):
            mask = actor_dates == date
            count = int(mask.sum())
            self.counts[date] += count
            for name in ("minade", "minfde", "energy_score"):
                self.sums[date][name] += float(metrics[name][mask].sum())

    def summarize(self) -> dict[str, dict[str, float]]:
        return {
            date: {
                "actors": self.counts[date],
                **{
                    name: value / self.counts[date]
                    for name, value in values.items()
                },
            }
            for date, values in sorted(self.sums.items())
        }


def target_tail_threshold(dataset, quantile: float = 0.75) -> float:
    displacement = dataset.pred_traj[:, :, -1] - dataset.obs_traj[:, :, -1]
    distance = torch.linalg.vector_norm(displacement, dim=1).float()
    return float(torch.quantile(distance, quantile))


@torch.no_grad()
def evaluate(
    model,
    loader,
    device: torch.device,
    *,
    scene_dates: list[str] | None,
    tail_threshold: float,
) -> dict[str, object]:
    model.eval()
    groups = {
        name: GroupAccumulator(modes=5)
        for name in ("overall", "multi_agent", "singleton", "interactive")
    }
    joint: dict[str, list[np.ndarray]] = defaultdict(list)
    date_metrics = DateAccumulator()
    scene_cursor = 0
    tail_sum = 0.0
    tail_count = 0
    negative_controls = 0
    control_count = 0

    for data in loader:
        data = {
            name: value.to(device) if torch.is_tensor(value) else value
            for name, value in data.items()
        }
        target = data["pred_traj"].transpose(1, 0)
        predictions, logits, auxiliary = model(data)
        batch_metrics = compute_batch_metrics(predictions, logits, target)
        packed = pack_scenes(data["adj"])
        multi = packed.counts[packed.inverse] > 1
        interactive = future_proximity_actor_mask(target, data["adj"])
        masks = {
            "overall": torch.ones_like(multi),
            "multi_agent": multi,
            "singleton": ~multi,
            "interactive": interactive,
        }
        for name, mask in masks.items():
            groups[name].update(batch_metrics, mask)

        scene_count = packed.scene_count
        dates = (
            ["unknown"] * scene_count
            if scene_dates is None
            else scene_dates[scene_cursor : scene_cursor + scene_count]
        )
        if len(dates) != scene_count:
            raise RuntimeError("C127 scene-date index does not match evaluation data")
        scene_cursor += scene_count
        batch_arrays = {
            name: value.detach().cpu().numpy()
            for name, value in batch_metrics.items()
        }
        date_metrics.update(
            dates,
            packed.inverse.detach().cpu().numpy(),
            batch_arrays,
        )

        displacement = torch.linalg.vector_norm(
            predictions - target[:, None], dim=-1
        )
        ade = displacement.mean(dim=-1)
        fde = displacement[..., -1]
        scene_ade = ade.new_zeros((scene_count, predictions.shape[1]))
        scene_fde = fde.new_zeros((scene_count, predictions.shape[1]))
        scene_ade.index_add_(0, packed.inverse, ade)
        scene_fde.index_add_(0, packed.inverse, fde)
        scene_ade = scene_ade / packed.counts[:, None]
        scene_fde = scene_fde / packed.counts[:, None]
        multi_scene = packed.counts > 1
        if bool(multi_scene.any()):
            joint["minade"].append(
                scene_ade[multi_scene].min(dim=-1).values.cpu().numpy()
            )
            joint["minfde"].append(
                scene_fde[multi_scene].min(dim=-1).values.cpu().numpy()
            )
            scene_logits = logits.new_zeros((scene_count, logits.shape[1]))
            scene_logits.index_add_(0, packed.inverse, logits)
            scene_logits = scene_logits / packed.counts[:, None]
            top1 = scene_logits.argmax(dim=-1)
            rows = torch.arange(scene_count, device=device)
            joint["top1_ade"].append(
                scene_ade[rows, top1][multi_scene].cpu().numpy()
            )
            joint["top1_fde"].append(
                scene_fde[rows, top1][multi_scene].cpu().numpy()
            )

        endpoint_travel = torch.linalg.vector_norm(
            target[:, -1] - data["obs_traj"][-1], dim=-1
        )
        tail = endpoint_travel >= tail_threshold
        tail_sum += float(batch_metrics["minfde"][tail].sum().cpu())
        tail_count += int(tail.sum().cpu())
        controls = auxiliary["horizontal_control"]
        negative_controls += int((controls < 0).sum().cpu())
        control_count += int(controls.numel())

    if scene_dates is not None and scene_cursor != len(scene_dates):
        raise RuntimeError("C127 evaluation did not consume every scene date")
    result = {name: accumulator.summarize() for name, accumulator in groups.items()}
    result["joint_multi_scene"] = {
        "scenes": int(sum(len(values) for values in joint["minfde"])),
        **{
            name: float(np.concatenate(values).mean())
            for name, values in joint.items()
        },
    }
    result["date_metrics"] = date_metrics.summarize()
    overall = result["overall"]
    fractions = overall["winner_distribution"]["fractions"]
    overall.update(
        {
            "tail_threshold": tail_threshold,
            "tail_samples": tail_count,
            "tail_minfde": tail_sum / max(tail_count, 1),
            "minimum_winner_fraction": min(fractions),
            "negative_horizontal_control_rate": negative_controls
            / max(control_count, 1),
        }
    )
    required = (
        "minade",
        "minfde",
        "minfde_p95",
        "energy_score",
        "tail_minfde",
        "negative_horizontal_control_rate",
    )
    if not all(math.isfinite(float(overall[name])) for name in required):
        raise RuntimeError("non-finite C127 evaluation metric")
    return result


__all__ = ["DateAccumulator", "GroupAccumulator", "evaluate", "target_tail_threshold"]
