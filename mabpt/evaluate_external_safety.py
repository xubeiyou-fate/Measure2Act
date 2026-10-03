"""Evaluate frozen Part C checkpoints on an external airport safety proxy.

The event and alert thresholds are fitted on TrajAir training data.  This
module never fits, calibrates, or selects a threshold on the external data.
The reported near-conflict event is a research proxy, not a regulatory claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

# Required by torch deterministic mode for CUDA >= 10.2.  It must be set before
# importing torch and before any CUDA context can be initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch.utils.data import DataLoader, Subset

from experiments.edfa_ascent.relation import pack_scenes
from model.utils import TrajectoryDataset, seed_worker, seq_collate

from .partc_seed_evaluate import (
    _default_source,
    _default_target,
    _load_model_pair,
    _model_outputs,
)
from .traffic import _average_precision, _ece, pair_conflict_probabilities


ROOT = Path(__file__).resolve().parents[1]
EVENT_PROTOCOL = Path(__file__).with_name("e12_protocol.json")
MODEL_NAMES = ("original_ascent", "mabpt_ascent")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"external safety evaluation refuses to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _limited_dataset(dataset: TrajectoryDataset, maximum: int | None):
    if maximum is None or maximum >= len(dataset):
        return dataset
    if maximum < 1:
        raise ValueError("--max-scenes must be positive")
    indices = (
        torch.linspace(0, len(dataset) - 1, maximum)
        .round()
        .long()
        .unique()
        .tolist()
    )
    return Subset(dataset, indices)


def _dataset_fingerprint(path: Path) -> dict[str, object]:
    files = sorted(item for item in path.glob("*.txt") if item.is_file())
    if not files:
        raise FileNotFoundError(f"no .txt trajectories found directly under {path}")
    digest = hashlib.sha256()
    total_bytes = 0
    for item in files:
        size = item.stat().st_size
        total_bytes += size
        digest.update(item.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(_sha256(item).encode("ascii"))
        digest.update(b"\n")
    return {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "name_size_content_sha256": digest.hexdigest(),
    }


def _load_frozen_inputs(
    *,
    seed: int,
    threshold_artifact: Path,
    source_checkpoint: Path,
    target_checkpoint: Path,
) -> tuple[dict[str, float], dict[str, object]]:
    artifact = json.loads(threshold_artifact.read_text(encoding="utf-8"))
    if int(artifact.get("seed", -1)) != seed:
        raise RuntimeError("threshold artifact seed does not match --seed")
    integrity = artifact.get("integrity", {})
    if not integrity.get("conflict_thresholds_fit_on_training_only", False):
        raise RuntimeError("threshold artifact is not marked training-only")
    declared = artifact["inputs"]
    source_hash = _sha256(source_checkpoint)
    target_hash = _sha256(target_checkpoint)
    if source_hash != declared["source_checkpoint_sha256"]:
        raise RuntimeError("source checkpoint hash does not match threshold artifact")
    if target_hash != declared["target_checkpoint_sha256"]:
        raise RuntimeError("target checkpoint hash does not match threshold artifact")
    diagnostics = artifact["conflict_risk"]["training_thresholds"]
    thresholds = {
        model: float(diagnostics[model]["threshold"]) for model in MODEL_NAMES
    }
    return thresholds, {
        "path": threshold_artifact.as_posix(),
        "sha256": _sha256(threshold_artifact),
        "fit_split": "complete_TrajAir_training_only",
        "target_false_positive_rate": 0.05,
        "thresholds": thresholds,
        "training_diagnostics": diagnostics,
        "source_checkpoint_sha256": source_hash,
        "target_checkpoint_sha256": target_hash,
    }


def _valid_pair_indices(scene_index: torch.Tensor, device: torch.device):
    packed = pack_scenes(scene_index)
    actors = packed.max_actors
    valid = (
        packed.valid[:, :, None]
        & packed.valid[:, None, :]
        & torch.triu(
            torch.ones(actors, actors, dtype=torch.bool, device=device), diagonal=1
        )[None]
    )
    pair_index = torch.nonzero(valid, as_tuple=False)
    if not pair_index.numel():
        return None
    pair_scene, pair_row, pair_column = pair_index.unbind(dim=1)
    return (
        packed.global_index[pair_scene, pair_row],
        packed.global_index[pair_scene, pair_column],
    )


def pair_separation_statistics(
    predictions: torch.Tensor,
    probabilities: torch.Tensor,
    truth: torch.Tensor,
    scene_index: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return synchronous closest-approach proxies for every scene actor pair."""
    pair = _valid_pair_indices(scene_index, predictions.device)
    if pair is None:
        empty = probabilities.new_empty(0, dtype=torch.float64)
        return {
            "truth_min_horizontal_km": empty,
            "truth_min_3d_km": empty,
            "truth_vertical_at_min_horizontal_km": empty,
            "predicted_expected_min_horizontal_km": empty,
            "predicted_expected_min_3d_km": empty,
            "predicted_expected_vertical_at_min_horizontal_km": empty,
        }
    row, column = pair
    truth_relative = truth[row].to(torch.float64) - truth[column].to(torch.float64)
    truth_horizontal = torch.linalg.vector_norm(truth_relative[..., :2], dim=-1)
    truth_3d = torch.linalg.vector_norm(truth_relative, dim=-1)
    truth_horizontal_argmin = truth_horizontal.argmin(dim=-1, keepdim=True)

    relative = (
        predictions[row].to(torch.float64)[:, :, None]
        - predictions[column].to(torch.float64)[:, None, :]
    )
    horizontal = torch.linalg.vector_norm(relative[..., :2], dim=-1)
    distance_3d = torch.linalg.vector_norm(relative, dim=-1)
    horizontal_argmin = horizontal.argmin(dim=-1, keepdim=True)
    vertical_at_min_horizontal = relative[..., 2].abs().gather(
        dim=-1, index=horizontal_argmin
    ).squeeze(-1)
    joint_mass = probabilities[row].to(torch.float64)[:, :, None] * probabilities[
        column
    ].to(torch.float64)[:, None, :]
    return {
        "truth_min_horizontal_km": truth_horizontal.min(dim=-1).values,
        "truth_min_3d_km": truth_3d.min(dim=-1).values,
        "truth_vertical_at_min_horizontal_km": truth_relative[..., 2]
        .abs()
        .gather(dim=-1, index=truth_horizontal_argmin)
        .squeeze(-1),
        "predicted_expected_min_horizontal_km": (
            joint_mass * horizontal.min(dim=-1).values
        ).sum(dim=(1, 2)),
        "predicted_expected_min_3d_km": (
            joint_mass * distance_3d.min(dim=-1).values
        ).sum(dim=(1, 2)),
        "predicted_expected_vertical_at_min_horizontal_km": (
            joint_mass * vertical_at_min_horizontal
        ).sum(dim=(1, 2)),
    }


class SafetyAccumulator:
    def __init__(self) -> None:
        self.probability: list[np.ndarray] = []
        self.hard_probability: list[np.ndarray] = []
        self.label: list[np.ndarray] = []
        self.first_step: list[np.ndarray] = []
        self.separation: dict[str, list[np.ndarray]] = {}

    def update(
        self,
        conflict: dict[str, torch.Tensor],
        separation: dict[str, torch.Tensor],
    ) -> None:
        self.probability.append(conflict["probability"].detach().cpu().numpy())
        self.hard_probability.append(
            conflict["hard_probability"].detach().cpu().numpy()
        )
        self.label.append(
            conflict["label"].detach().cpu().numpy().astype(np.bool_)
        )
        self.first_step.append(
            conflict["first_true_step"].detach().cpu().numpy()
        )
        for key, value in separation.items():
            self.separation.setdefault(key, []).append(value.detach().cpu().numpy())

    def arrays(self):
        if not self.probability:
            raise RuntimeError("external evaluator received no batches")
        return (
            np.concatenate(self.probability),
            np.concatenate(self.hard_probability),
            np.concatenate(self.label),
            np.concatenate(self.first_step),
        )

    def summarize(self, *, alert_threshold: float) -> dict[str, object]:
        probability, hard_probability, label, first_step = self.arrays()
        if not len(label):
            return {
                "pairs": 0,
                "limitation": "selected scenes contain no multi-actor pairs",
                "alert_threshold": alert_threshold,
            }
        tiny = np.finfo(np.float64).tiny
        clipped = np.clip(probability, tiny, 1.0 - np.finfo(np.float64).eps)
        alert = probability >= alert_threshold
        tp = int((alert & label).sum())
        fp = int((alert & ~label).sum())
        fn = int((~alert & label).sum())
        tn = int((~alert & ~label).sum())
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if math.isfinite(precision + recall) and precision + recall
            else float("nan")
        )
        positive_first = first_step[label]
        time_to_event = (positive_first[positive_first >= 0] + 1) * 5.0
        result: dict[str, object] = {
            "pairs": int(len(label)),
            "positive_pairs": int(label.sum()),
            "truth_near_conflict_rate": float(label.mean()),
            "mean_soft_conflict_probability": float(probability.mean()),
            "mean_hard_support_conflict_probability": float(
                hard_probability.mean()
            ),
            "brier": float(np.square(probability - label).mean()),
            "nll": float(
                -(
                    label * np.log(clipped)
                    + (~label) * np.log1p(-clipped)
                ).mean()
            ),
            "auprc": _average_precision(probability, label),
            "ece_15_bin": _ece(probability, label, bins=15),
            "alert_threshold": alert_threshold,
            "alert_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "alert_precision": precision,
            "alert_recall": recall,
            "alert_f1": f1,
            "observed_fpr": fp / (fp + tn) if fp + tn else float("nan"),
            "hard_support_zero_rate_on_positive": (
                float((hard_probability[label] == 0).mean())
                if label.any()
                else float("nan")
            ),
            "mean_time_to_first_true_conflict_seconds": (
                float(time_to_event.mean()) if len(time_to_event) else float("nan")
            ),
            "median_time_to_first_true_conflict_seconds": (
                float(np.median(time_to_event))
                if len(time_to_event)
                else float("nan")
            ),
        }
        result["separation_proxy"] = {
            key: _distribution(np.concatenate(values))
            for key, values in sorted(self.separation.items())
        }
        return result


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _relative_gain(ascent: dict[str, object], mabpt: dict[str, object]):
    result = {}
    for metric in ("brier", "nll", "ece_15_bin"):
        baseline = float(ascent[metric])
        result[f"{metric}_reduction"] = (
            (baseline - float(mabpt[metric])) / baseline if baseline else float("nan")
        )
    baseline_auprc = float(ascent["auprc"])
    result["auprc_increase"] = (
        (float(mabpt["auprc"]) - baseline_auprc) / baseline_auprc
        if baseline_auprc
        else float("nan")
    )
    return result


@torch.inference_mode()
def run(
    *,
    dataset_path: Path,
    dataset_name: str,
    delimiter: str,
    seed: int,
    device: torch.device,
    workers: int,
    batch_size: int,
    max_scenes: int | None,
    source_checkpoint: Path | None = None,
    target_checkpoint: Path | None = None,
    threshold_artifact: Path | None = None,
) -> dict[str, object]:
    dataset_path = dataset_path.resolve()
    source_path = (source_checkpoint or _default_source(seed)).resolve()
    target_path = (target_checkpoint or _default_target(seed)).resolve()
    threshold_path = (
        threshold_artifact
        or ROOT
        / "artifacts/mabpt_partc_20260811"
        / f"seed{seed}_development_formal_v1.json"
    ).resolve()
    for path in (source_path, target_path, threshold_path, EVENT_PROTOCOL):
        if not path.is_file():
            raise FileNotFoundError(path)
    thresholds, threshold_record = _load_frozen_inputs(
        seed=seed,
        threshold_artifact=threshold_path,
        source_checkpoint=source_path,
        target_checkpoint=target_path,
    )
    event_protocol = json.loads(EVENT_PROTOCOL.read_text(encoding="utf-8"))
    event = event_protocol["event"]
    fingerprint = _dataset_fingerprint(dataset_path)
    dataset = TrajectoryDataset(
        dataset_path.as_posix(),
        obs_len=16,
        obs_steps=1,
        pred_len=120,
        pred_step=5,
        delim=delimiter,
    )
    evaluation_dataset = _limited_dataset(dataset, max_scenes)
    loader_options = {
        "dataset": evaluation_dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": seq_collate,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
    }
    if workers > 0:
        loader_options.update({"persistent_workers": True, "prefetch_factor": 4})
    data_loader = DataLoader(**loader_options)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    source, target_model = _load_model_pair(
        source_checkpoint=source_path,
        target_checkpoint=target_path,
        device=device,
        batch_size=batch_size,
    )
    states = {model: SafetyAccumulator() for model in MODEL_NAMES}
    actors = 0
    batches = 0
    for data in data_loader:
        data = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in data.items()
        }
        truth = data["pred_traj"].transpose(1, 0)
        outputs = _model_outputs(source, target_model, data)
        actors += int(truth.shape[0])
        batches += 1
        for model, values in outputs.items():
            conflict = pair_conflict_probabilities(
                values["support"],
                values["probability"],
                truth,
                data["adj"],
                horizontal_threshold=float(event["horizontal_threshold_km"]),
                vertical_threshold=float(event["vertical_threshold_km"]),
                horizontal_scale=float(event["horizontal_kernel_scale_km"]),
                vertical_scale=float(event["vertical_kernel_scale_km"]),
            )
            separation = pair_separation_statistics(
                values["support"],
                values["probability"],
                truth,
                data["adj"],
            )
            states[model].update(conflict, separation)
    results = {
        model: states[model].summarize(alert_threshold=thresholds[model])
        for model in MODEL_NAMES
    }
    if results["original_ascent"]["pairs"] != results["mabpt_ascent"]["pairs"]:
        raise RuntimeError("paired model pair counts differ")
    relative = (
        _relative_gain(results["original_ascent"], results["mabpt_ascent"])
        if results["original_ascent"]["pairs"]
        else {}
    )
    return _json_safe(
        {
            "format_version": 1,
            "experiment_id": "Tartan_external_safety_proxy",
            "dataset": dataset_name,
            "seed": seed,
            "scenes": len(evaluation_dataset),
            "actors": actors,
            "batches": batches,
            "event": event,
            "event_protocol_sha256": _sha256(EVENT_PROTOCOL),
            "results": {**results, "relative_gain_mabpt_vs_ascent": relative},
            "inputs": {
                "dataset_path": dataset_path.as_posix(),
                "dataset_fingerprint": fingerprint,
                "delimiter": delimiter,
                "source_checkpoint": source_path.as_posix(),
                "target_checkpoint": target_path.as_posix(),
                "threshold_artifact": threshold_record,
            },
            "integrity": {
                "zero_shot_external_evaluation": True,
                "external_finetuning_or_calibration": False,
                "event_definition_fit_on_TrajAir_training_only": True,
                "alert_thresholds_fit_on_TrajAir_training_only": True,
                "matched_seed_checkpoint_and_threshold_hashes": True,
                "external_result_based_selection": False,
                "deterministic_inference": True,
                "output_refuses_overwrite": True,
            },
            "claim_boundary": {
                "regulatory_claim": False,
                "description": (
                    "Synchronous horizontal/vertical near-conflict and closest-approach "
                    "research proxies; not a certified separation or collision-risk metric."
                ),
                "independence_assumption": (
                    "Actor mode probabilities are multiplied when marginalizing pair events."
                ),
                "external_dates_available": False,
            },
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--target-checkpoint", type=Path)
    parser.add_argument("--threshold-artifact", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(
            f"external safety evaluation refuses to overwrite {output}"
        )
    result = run(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        delimiter=args.delimiter,
        seed=args.seed,
        source_checkpoint=args.source_checkpoint,
        target_checkpoint=args.target_checkpoint,
        threshold_artifact=args.threshold_artifact,
        device=torch.device(args.device),
        workers=args.workers,
        batch_size=args.batch_size,
        max_scenes=args.max_scenes,
    )
    _atomic_json(output, result)
    print(
        json.dumps(
            {
                "output": output.as_posix(),
                "dataset": result["dataset"],
                "seed": result["seed"],
                "scenes": result["scenes"],
                "actors": result["actors"],
                "pairs": result["results"]["original_ascent"]["pairs"],
                "relative_gain": result["results"][
                    "relative_gain_mabpt_vs_ascent"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
