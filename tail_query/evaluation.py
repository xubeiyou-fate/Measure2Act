"""Overall, tail, and future-pattern evaluation for C7."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from airroute_stage_m.evaluation import RankingMetricAccumulator, compute_batch_metrics
from tail_query.model import FuturePatternAssigner


@dataclass
class TrajectoryGroupAccumulator:
    modes: int = 5
    metrics: RankingMetricAccumulator = field(default_factory=RankingMetricAccumulator)
    winner_counts: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.int64))

    def update(
        self,
        prediction: torch.Tensor,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if not mask.any():
            return
        batch = compute_batch_metrics(prediction[mask], logits[mask], target[mask])
        self.metrics.update(batch)
        winner = batch["oracle_fde_mode"].detach().cpu().numpy()
        self.winner_counts += np.bincount(winner, minlength=self.modes)

    def summarize(self) -> dict:
        summary = self.metrics.summarize()
        arrays = self.metrics.arrays()
        minfde = arrays["minfde"]
        fractions = self.winner_counts / self.winner_counts.sum()
        nonzero = fractions[fractions > 0]
        entropy = float(-(nonzero * np.log(nonzero)).sum())
        summary.update({
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
        })
        return summary


@torch.no_grad()
def evaluate_tail_groups(
    model,
    loader,
    assigner: FuturePatternAssigner,
    device: torch.device,
) -> dict:
    model.eval()
    groups = {
        "overall": TrajectoryGroupAccumulator(),
        "head": TrajectoryGroupAccumulator(),
        "tail": TrajectoryGroupAccumulator(),
        **{f"pattern_{index}": TrajectoryGroupAccumulator() for index in range(5)},
    }
    pattern_targets = []
    pattern_predictions = []
    cv_errors = []
    for data in loader:
        for name, value in data.items():
            if torch.is_tensor(value):
                data[name] = value.to(device)
        prediction, logits, auxiliary = model(data)
        target = data["pred_traj"].transpose(1, 0)
        assigned = assigner(data, target, auxiliary)
        labels = assigned["labels"]
        tail = assigned["tail_mask"]
        all_actors = torch.ones_like(tail, dtype=torch.bool)
        groups["overall"].update(prediction, logits, target, all_actors)
        groups["head"].update(prediction, logits, target, ~tail)
        groups["tail"].update(prediction, logits, target, tail)
        for index in range(5):
            groups[f"pattern_{index}"].update(
                prediction, logits, target, labels == index
            )
        if "pattern_logits" in auxiliary:
            pattern_targets.append(labels.cpu().numpy())
            pattern_predictions.append(
                auxiliary["pattern_logits"].argmax(dim=-1).cpu().numpy()
            )
        cv_errors.append(assigned["cv_fde"].cpu().numpy())
    result = {name: accumulator.summarize() for name, accumulator in groups.items()}
    result["tail_definition"] = {
        "metric": "constant_velocity_120s_FDE",
        "threshold": assigner.tail_threshold_cv_fde,
        "dev_fraction": result["tail"]["agents"] / result["overall"]["agents"],
        "cv_fde_p50": float(np.quantile(np.concatenate(cv_errors), 0.5)),
        "cv_fde_p95": float(np.quantile(np.concatenate(cv_errors), 0.95)),
    }
    if pattern_targets:
        target = np.concatenate(pattern_targets)
        prediction = np.concatenate(pattern_predictions)
        result["pattern_prediction"] = {
            "accuracy": float(accuracy_score(target, prediction)),
            "macro_f1": float(f1_score(target, prediction, average="macro")),
        }
    return result
