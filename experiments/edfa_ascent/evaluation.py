"""Per-agent, interactive, scene-level, and date-clustered C96 metrics."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from airroute_stage_m.evaluation import compute_batch_metrics
from tail_query.evaluation import TrajectoryGroupAccumulator

from .objective import future_proximity_actor_mask
from .relation import pack_scenes, scene_size_matched_permutation


class DateMetricAccumulator:
    def __init__(self) -> None:
        self.sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.counts: dict[str, int] = defaultdict(int)

    def update(self, date: str, metrics: dict[str, torch.Tensor], mask: torch.Tensor) -> None:
        count = int(mask.sum().item())
        if not count:
            return
        self.counts[date] += count
        for name in ("minade", "minfde", "energy_score"):
            self.sums[date][name] += float(metrics[name][mask].sum().cpu())

    def update_batch(
        self,
        dates: list[str],
        scene_inverse: np.ndarray,
        metrics: dict[str, np.ndarray],
    ) -> None:
        scene_dates = np.asarray(dates, dtype=object)
        actor_dates = scene_dates[scene_inverse]
        for date in sorted(set(dates)):
            mask = actor_dates == date
            count = int(mask.sum())
            if not count:
                continue
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


def _scene_dates_for_batch(
    scene_dates: list[str] | None, cursor: int, scene_count: int
) -> tuple[list[str], int]:
    if scene_dates is None:
        return ["unknown"] * scene_count, cursor + scene_count
    selected = scene_dates[cursor:cursor + scene_count]
    if len(selected) != scene_count:
        raise RuntimeError("scene-date index does not match evaluation dataset")
    return selected, cursor + scene_count


def _metrics_to_numpy(metrics: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {
        name: values.detach().cpu().numpy()
        for name, values in metrics.items()
    }


def _update_groups(
    groups: dict[str, TrajectoryGroupAccumulator],
    metrics: dict[str, np.ndarray],
    masks: dict[str, torch.Tensor],
) -> None:
    cpu_masks = {
        name: mask.detach().cpu().numpy().astype(bool, copy=False)
        for name, mask in masks.items()
    }
    for group_name, mask in cpu_masks.items():
        if not np.any(mask):
            continue
        accumulator = groups[group_name]
        for metric_name, values in metrics.items():
            accumulator.metrics.values.setdefault(metric_name, []).append(values[mask])
        winner = metrics["oracle_fde_mode"][mask].astype(np.int64, copy=False)
        accumulator.winner_counts += np.bincount(
            winner, minlength=accumulator.modes
        )


@torch.no_grad()
def evaluate(
    model,
    loader,
    device: torch.device,
    scene_dates: list[str] | None = None,
    graph_source: str = "predicted",
    permute_neighbors: bool = False,
) -> dict:
    model.eval()
    groups = {
        name: TrajectoryGroupAccumulator(modes=5)
        for name in ("overall", "multi_agent", "singleton", "interactive")
    }
    joint: dict[str, list[np.ndarray]] = defaultdict(list)
    date_metrics = DateMetricAccumulator()
    relation_correct = relation_count = relation_positive = 0
    scene_cursor = 0
    for data in loader:
        for key, value in data.items():
            if torch.is_tensor(value):
                data[key] = value.to(device)
        target = data["pred_traj"].transpose(1, 0)
        packed = pack_scenes(data["adj"])
        if permute_neighbors:
            data["neighbor_permutation"] = scene_size_matched_permutation(data["adj"])
        if graph_source == "oracle":
            from .objective import relation_supervision
            labels, valid_pairs, _ = relation_supervision(target, data["adj"])
            data["relation_labels"] = labels
            data["graph_source"] = "oracle"
        predictions, logits, auxiliary = model(data)
        batch_metrics = compute_batch_metrics(predictions, logits, target)
        multi = packed.counts[packed.inverse] > 1
        interactive = future_proximity_actor_mask(target, data["adj"])
        masks = {
            "overall": torch.ones_like(multi),
            "multi_agent": multi,
            "singleton": ~multi,
            "interactive": interactive,
        }
        batch_arrays = _metrics_to_numpy(batch_metrics)
        _update_groups(groups, batch_arrays, masks)

        dates, scene_cursor = _scene_dates_for_batch(
            scene_dates, scene_cursor, packed.scene_count
        )
        date_metrics.update_batch(
            dates,
            packed.inverse.detach().cpu().numpy(),
            batch_arrays,
        )

        displacement = torch.linalg.vector_norm(predictions - target[:, None], dim=-1)
        ade = displacement.mean(dim=-1)
        fde = displacement[..., -1]
        scene_ade = ade.new_zeros((packed.scene_count, predictions.shape[1]))
        scene_fde = fde.new_zeros((packed.scene_count, predictions.shape[1]))
        scene_ade.index_add_(0, packed.inverse, ade)
        scene_fde.index_add_(0, packed.inverse, fde)
        scene_ade = scene_ade / packed.counts[:, None]
        scene_fde = scene_fde / packed.counts[:, None]
        multi_scene = packed.counts > 1
        joint["minade"].append(scene_ade[multi_scene].min(dim=-1).values.cpu().numpy())
        joint["minfde"].append(scene_fde[multi_scene].min(dim=-1).values.cpu().numpy())
        scene_logits = logits.new_zeros((packed.scene_count, logits.shape[1]))
        scene_logits.index_add_(0, packed.inverse, logits)
        scene_logits = scene_logits / packed.counts[:, None]
        top1 = scene_logits.argmax(dim=-1)
        rows = torch.arange(packed.scene_count, device=device)
        joint["top1_ade"].append(scene_ade[rows, top1][multi_scene].cpu().numpy())
        joint["top1_fde"].append(scene_fde[rows, top1][multi_scene].cpu().numpy())

        if "relation_logits" in auxiliary:
            if graph_source != "oracle":
                from .objective import relation_supervision
                labels, valid_pairs, _ = relation_supervision(target, data["adj"])
            predicted = auxiliary["relation_logits"].argmax(dim=-1)
            relation_correct += int(((predicted == labels) & valid_pairs).sum().cpu())
            relation_count += int(valid_pairs.sum().cpu())
            relation_positive += int(((labels != 0) & valid_pairs).sum().cpu())

    result = {name: accumulator.summarize() for name, accumulator in groups.items()}
    result["joint_multi_scene"] = {
        "scenes": int(sum(len(values) for values in joint["minfde"])),
        **{
            name: float(np.concatenate(values).mean())
            for name, values in joint.items()
        },
    }
    result["date_metrics"] = date_metrics.summarize()
    if relation_count:
        result["relation"] = {
            "pairs": relation_count,
            "accuracy": relation_correct / relation_count,
            "positive_fraction": relation_positive / relation_count,
        }
    result["control"] = {
        "graph_source": graph_source,
        "neighbor_permutation": permute_neighbors,
    }
    return result
