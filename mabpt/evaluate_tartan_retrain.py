"""Evaluate frozen Tartan-retrained ASCENT/MABPT checkpoints by calendar date."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from experiments.edfa_ascent.relation import pack_scenes
from experiments.energy_predict_optimize.evaluation import (
    RankingMetricAccumulator,
    compute_batch_metrics,
)
from model.utils import seed_worker, seq_collate

from .partc_seed_evaluate import _load_model_pair, _model_outputs
from .traffic import _average_precision, _ece, pair_conflict_probabilities
from .train_tartan_retrain import _dataset, _limited_indices


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("tartan_retrain_protocol_v1.json")
RUN_ROOT = ROOT / "runs/partc_tartan_retrain_20260812"
FREEZE_RECEIPT = (
    ROOT
    / "artifacts/partc_two_dataset_20260812/tartan_retrain_frozen_receipt_v1.json"
)
EVALUATION_RECEIPT = (
    ROOT
    / "artifacts/partc_two_dataset_20260812/tartan_retrain_evaluation_frozen_receipt_v1.json"
)
EVENT_PROTOCOL = Path(__file__).with_name("e12_protocol.json")
AIRPORTS = ("KAGC", "KBTP")
REGIMES = ("target_only", "full_finetune")
SPLITS = ("development", "test")
STAGES = ("ascent", "decision_support", "predicted_risk")
MODELS = ("original_ascent", "mabpt_ascent")
EVALUATED_ARMS = ("constant_velocity", *MODELS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return value


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"Tartan retrain evaluator refuses to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _checkpoint_path(
    root: Path,
    airport: str,
    regime: str,
    stage: str,
    seed: int,
    *,
    formal: bool = True,
) -> Path:
    return (
        root
        / "runs/partc_tartan_retrain_20260812"
        / airport
        / regime
        / "p100"
        / f"{stage}_seed{seed}_{'formal' if formal else 'smoke'}"
        / "last.pt"
    )


def _summary_path(checkpoint: Path) -> Path:
    return checkpoint.with_name("training_summary.json")


def _verify_freeze_receipt(
    *, root: Path = ROOT, receipt_path: Path = FREEZE_RECEIPT
) -> dict[str, object]:
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("locked_test_model_inference_completed") is not False:
        raise RuntimeError("freeze receipt no longer represents the unopened test state")
    verified = {}
    for relative, expected in receipt["files"].items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != expected["sha256"] or path.stat().st_size != int(expected["bytes"]):
            raise RuntimeError(f"freeze receipt mismatch: {relative}")
        verified[relative] = actual
    return {
        "path": receipt_path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": _sha256(receipt_path),
        "verified_files": verified,
        "locked_test_model_inference_completed_at_freeze": False,
    }


def _verify_evaluation_receipt(
    *, root: Path = ROOT, receipt_path: Path = EVALUATION_RECEIPT
) -> dict[str, object]:
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("locked_test_model_inference_completed_before_freeze") is not False:
        raise RuntimeError("evaluation receipt does not represent the unopened test state")
    verified = {}
    for relative, expected in receipt["files"].items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != expected["sha256"] or path.stat().st_size != int(expected["bytes"]):
            raise RuntimeError(f"evaluation receipt mismatch: {relative}")
        verified[relative] = actual
    return {
        "path": receipt_path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": _sha256(receipt_path),
        "verified_files": verified,
        "locked_test_model_inference_completed_before_freeze": False,
    }


def _validate_checkpoint(
    *,
    root: Path,
    protocol: dict[str, Any],
    airport: str,
    regime: str,
    stage: str,
    seed: int,
    load_checkpoint_metadata: bool,
    formal: bool = True,
) -> dict[str, object]:
    checkpoint = _checkpoint_path(
        root, airport, regime, stage, seed, formal=formal
    )
    summary_path = _summary_path(checkpoint)
    for path in (checkpoint, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_protocol = _sha256(root / "mabpt/tartan_retrain_protocol_v1.json")
    manifest = root / protocol["data"]["root"] / "manifest.json"
    expected_manifest = _sha256(manifest)
    expected = {
        "airport": airport,
        "regime": regime,
        "stage": stage,
        "seed": seed,
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"training summary identity mismatch: {summary_path}")
    if not summary.get("complete") or bool(summary.get("formal")) is not formal:
        expected_kind = "formal" if formal else "smoke"
        raise RuntimeError(f"checkpoint is not a completed {expected_kind} run: {checkpoint}")
    if float(summary.get("fraction", -1)) != 1.0:
        raise RuntimeError(f"checkpoint is not the registered 100-percent run: {checkpoint}")
    expected_epoch = (
        int(protocol["training"]["epochs"])
        if formal
        else int(summary.get("fixed_final_epoch", -1))
    )
    if expected_epoch < 1 or (formal and expected_epoch != int(protocol["training"]["epochs"])):
        raise RuntimeError(f"checkpoint is not the fixed final epoch: {checkpoint}")
    checkpoint_hash = _sha256(checkpoint)
    if summary.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError(f"checkpoint hash differs from training summary: {checkpoint}")
    if summary.get("protocol_sha256") != expected_protocol:
        raise RuntimeError(f"checkpoint protocol hash mismatch: {checkpoint}")
    if summary.get("data_manifest_sha256") != expected_manifest:
        raise RuntimeError(f"checkpoint data manifest hash mismatch: {checkpoint}")
    integrity = summary.get("integrity", {})
    if integrity.get("locked_test_used") is not False:
        raise RuntimeError(f"training summary reports locked-test use: {summary_path}")
    if load_checkpoint_metadata:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if any(state.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"checkpoint identity metadata mismatch: {checkpoint}")
        if int(state.get("epoch", -1)) != expected_epoch:
            raise RuntimeError(f"checkpoint epoch metadata mismatch: {checkpoint}")
        if float(state.get("fraction", -1)) != 1.0:
            raise RuntimeError(f"checkpoint fraction metadata mismatch: {checkpoint}")
        if state.get("protocol_sha256") != expected_protocol:
            raise RuntimeError(f"checkpoint protocol metadata mismatch: {checkpoint}")
        if state.get("data_manifest_sha256") != expected_manifest:
            raise RuntimeError(f"checkpoint manifest metadata mismatch: {checkpoint}")
        if state.get("locked_test_used") is not False:
            raise RuntimeError(f"checkpoint metadata reports locked-test use: {checkpoint}")
    return {
        "path": checkpoint.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": checkpoint_hash,
        "summary": summary_path.resolve().relative_to(root.resolve()).as_posix(),
        "summary_sha256": _sha256(summary_path),
        "bytes": checkpoint.stat().st_size,
    }


def _selected_checkpoint_triplet(
    *,
    root: Path,
    protocol: dict[str, Any],
    airport: str,
    regime: str,
    seed: int,
    formal: bool,
) -> dict[str, object]:
    return {
        stage: _validate_checkpoint(
            root=root,
            protocol=protocol,
            airport=airport,
            regime=regime,
            stage=stage,
            seed=seed,
            load_checkpoint_metadata=True,
            formal=formal,
        )
        for stage in STAGES
    }


def _formal_test_gate(
    *,
    root: Path,
    protocol: dict[str, Any],
    receipt_path: Path,
    evaluation_receipt_path: Path = EVALUATION_RECEIPT,
) -> dict[str, object]:
    receipt = _verify_freeze_receipt(root=root, receipt_path=receipt_path)
    evaluation_receipt = _verify_evaluation_receipt(
        root=root, receipt_path=evaluation_receipt_path
    )
    seeds = [int(value) for value in protocol["training"]["seeds"]]
    checkpoints = []
    for airport in AIRPORTS:
        for regime in REGIMES:
            for seed in seeds:
                for stage in STAGES:
                    checkpoints.append(
                        _validate_checkpoint(
                            root=root,
                            protocol=protocol,
                            airport=airport,
                            regime=regime,
                            stage=stage,
                            seed=seed,
                            load_checkpoint_metadata=True,
                        )
                    )
    expected = len(AIRPORTS) * len(REGIMES) * len(seeds) * len(STAGES)
    if len(checkpoints) != expected:
        raise RuntimeError("formal test gate did not verify the complete checkpoint grid")
    return {
        "passed": True,
        "required_checkpoint_count": expected,
        "verified_checkpoint_count": len(checkpoints),
        "freeze_receipt": receipt,
        "evaluation_freeze_receipt": evaluation_receipt,
        "checkpoint_receipt_sha256": hashlib.sha256(
            json.dumps(checkpoints, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


def _authorize_split(
    *,
    split: str,
    authorize_locked_test: bool,
    max_scenes: int | None,
    formal_gate,
) -> dict[str, object] | None:
    if split not in SPLITS:
        raise ValueError("split must be development or test")
    if split == "development":
        if authorize_locked_test:
            raise ValueError("--authorize-locked-test is invalid for development")
        return None
    if not authorize_locked_test:
        raise RuntimeError("locked test requires explicit --authorize-locked-test")
    if max_scenes is not None:
        raise RuntimeError("locked test forbids partial or smoke evaluation")
    # This call must precede any construction of the test dataset.
    return formal_gate()


class MetricStore:
    def __init__(self, *, retain_per_scene: bool = True) -> None:
        self.overall = RankingMetricAccumulator()
        self.by_date: dict[str, RankingMetricAccumulator] = {}
        self.by_scene: dict[str, RankingMetricAccumulator] = {}
        self.retain_per_scene = retain_per_scene

    @staticmethod
    def _slice(metrics: dict[str, torch.Tensor], mask: torch.Tensor):
        return {name: value[mask] for name, value in metrics.items()}

    def update(
        self,
        metrics: dict[str, torch.Tensor],
        inverse: torch.Tensor,
        scene_ids: list[str],
        dates: list[str],
    ) -> None:
        self.overall.update(metrics)
        date_scenes: dict[str, list[int]] = defaultdict(list)
        for local, (scene_id, date) in enumerate(zip(scene_ids, dates, strict=True)):
            date_scenes[date].append(local)
            if self.retain_per_scene:
                mask = inverse == local
                selected = self._slice(metrics, mask)
                self.by_scene.setdefault(scene_id, RankingMetricAccumulator()).update(selected)
        for date, local_scenes in date_scenes.items():
            scene_tensor = torch.as_tensor(local_scenes, device=inverse.device)
            mask = (inverse[:, None] == scene_tensor[None]).any(dim=1)
            self.by_date.setdefault(date, RankingMetricAccumulator()).update(
                self._slice(metrics, mask)
            )

    def summary(self) -> dict[str, object]:
        return {
            "overall": self.overall.summarize(),
            "per_date": {
                key: value.summarize() for key, value in sorted(self.by_date.items())
            },
            "per_scene": {
                key: value.summarize() for key, value in sorted(self.by_scene.items())
            },
        }


class SafetyStore:
    def __init__(self, *, retain_per_scene: bool = True) -> None:
        self.values: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
        self.retain_per_scene = retain_per_scene

    def _append(
        self,
        key: str,
        probability: np.ndarray,
        label: np.ndarray,
        first_step: np.ndarray,
    ) -> None:
        self.values[key].append((probability, label, first_step))

    def update(
        self,
        result: dict[str, torch.Tensor],
        scene_ids: list[str],
        dates: list[str],
    ) -> None:
        probability = result["probability"].detach().cpu().numpy().astype(np.float64)
        label = result["label"].detach().cpu().numpy().astype(np.bool_)
        first_step = result["first_true_step"].detach().cpu().numpy()
        pair_scene = result["pair_scene"].detach().cpu().numpy()
        self._append("overall", probability, label, first_step)
        date_scenes: dict[str, list[int]] = defaultdict(list)
        for local, (scene_id, date) in enumerate(zip(scene_ids, dates, strict=True)):
            date_scenes[date].append(local)
            mask = pair_scene == local
            if self.retain_per_scene:
                self._append(
                    f"scene:{scene_id}",
                    probability[mask],
                    label[mask],
                    first_step[mask],
                )
        for date, local_scenes in date_scenes.items():
            mask = np.isin(pair_scene, np.asarray(local_scenes))
            self._append(
                f"date:{date}", probability[mask], label[mask], first_step[mask]
            )

    @staticmethod
    def _summarize(
        chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        *,
        alert_threshold: float | None,
    ) -> dict[str, object]:
        probability = np.concatenate([item[0] for item in chunks])
        label = np.concatenate([item[1] for item in chunks])
        first_step = np.concatenate([item[2] for item in chunks])
        if not len(label):
            return {"pairs": 0, "positive_pairs": 0}
        tiny = np.finfo(np.float64).tiny
        clipped = np.clip(probability, tiny, 1.0 - np.finfo(np.float64).eps)
        result = {
            "pairs": int(len(label)),
            "positive_pairs": int(label.sum()),
            "prevalence": float(label.mean()),
            "mean_probability": float(probability.mean()),
            "brier": float(np.square(probability - label).mean()),
            "nll": float(
                -(label * np.log(clipped) + (~label) * np.log1p(-clipped)).mean()
            ),
            "auprc": _average_precision(probability, label),
            "ece_15_bin": _ece(probability, label, bins=15),
        }
        if alert_threshold is not None:
            alert = probability >= alert_threshold
            true_positive = alert & label
            false_positive = alert & ~label
            precision = float(true_positive.sum() / max(alert.sum(), 1))
            recall = float(true_positive.sum() / max(label.sum(), 1))
            lead = (first_step[true_positive] + 1) * 5.0
            result.update(
                {
                    "alert_threshold": alert_threshold,
                    "alerts": int(alert.sum()),
                    "true_positive_alerts": int(true_positive.sum()),
                    "precision_at_training_fixed_fpr": precision,
                    "recall_at_training_fixed_fpr": recall,
                    "f1_at_training_fixed_fpr": (
                        2.0 * precision * recall / (precision + recall)
                        if precision + recall
                        else 0.0
                    ),
                    "observed_fpr": float(false_positive.sum() / max((~label).sum(), 1)),
                    "mean_warning_lead_seconds": float(lead.mean()) if len(lead) else None,
                    "median_warning_lead_seconds": float(np.median(lead)) if len(lead) else None,
                }
            )
        return result

    def summary(self, *, alert_threshold: float | None = None) -> dict[str, object]:
        return {
            "overall": self._summarize(
                self.values["overall"], alert_threshold=alert_threshold
            ),
            "per_date": {
                key.removeprefix("date:"): self._summarize(
                    value, alert_threshold=alert_threshold
                )
                for key, value in sorted(self.values.items())
                if key.startswith("date:")
            },
            "per_scene": {
                key.removeprefix("scene:"): self._summarize(
                    value, alert_threshold=alert_threshold
                )
                for key, value in sorted(self.values.items())
                if key.startswith("scene:")
            },
        }


def _load_safety_thresholds(
    *,
    root: Path,
    airport: str,
    regime: str,
    seed: int,
    checkpoints: dict[str, object],
) -> tuple[dict[str, float], dict[str, object]]:
    path = (
        root
        / "artifacts/partc_two_dataset_20260812/tartan_safety_thresholds_v1"
        / f"{airport}_{regime}_seed{seed}_train_v1.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("airport") != airport
        or payload.get("regime") != regime
        or int(payload.get("seed", -1)) != seed
        or payload.get("fit_split") != "train"
    ):
        raise RuntimeError(f"safety threshold identity mismatch: {path}")
    if payload.get("integrity", {}).get("locked_test_accessed") is not False:
        raise RuntimeError(f"safety threshold was not fit without test access: {path}")
    declared = payload["inputs"]["checkpoints"]
    for stage in STAGES:
        if declared[stage]["sha256"] != checkpoints[stage]["sha256"]:
            raise RuntimeError(f"safety threshold checkpoint mismatch: {path}")
    thresholds = {
        model: float(payload["thresholds"][model]["threshold"])
        for model in EVALUATED_ARMS
    }
    return thresholds, {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "fit_split": "train",
        "thresholds": thresholds,
    }


def _relative_gain(ascent: dict[str, object], mabpt: dict[str, object]):
    result = {}
    for metric in (
        "top1_ade",
        "top1_fde",
        "minade",
        "minfde",
        "energy_score",
        "nll",
        "brier",
        "ece",
        "tail_minfde",
    ):
        baseline = float(ascent[metric])
        result[metric] = (
            (baseline - float(mabpt[metric])) / baseline if baseline else float("nan")
        )
    return result


def _constant_velocity(data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    observation = data["obs_traj"]
    velocity = observation[-1] - observation[-2]
    seconds = torch.arange(5, 121, 5, device=observation.device, dtype=observation.dtype)
    support = observation[-1, :, None] + velocity[:, None] * seconds[None, :, None]
    return {
        "support": support[:, None],
        "probability": torch.ones((support.shape[0], 1), device=support.device, dtype=support.dtype),
        "decision": torch.zeros(support.shape[0], device=support.device, dtype=torch.long),
    }


@torch.inference_mode()
def run(
    *,
    airport: str,
    checkpoint_airport: str | None = None,
    regime: str,
    seed: int,
    split: str,
    device: torch.device,
    workers: int,
    batch_size: int,
    max_scenes: int | None,
    authorize_locked_test: bool = False,
    checkpoint_kind: str = "formal",
    root: Path = ROOT,
) -> dict[str, object]:
    protocol_path = root / "mabpt/tartan_retrain_protocol_v1.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if airport not in AIRPORTS or airport not in protocol["data"]["airports"]:
        raise ValueError("airport lies outside the frozen Tartan registry")
    training_airport = checkpoint_airport or airport
    if training_airport not in AIRPORTS or training_airport not in protocol["data"]["airports"]:
        raise ValueError("checkpoint airport lies outside the frozen Tartan registry")
    if regime not in REGIMES:
        raise ValueError("regime lies outside the frozen Tartan registry")
    seeds = [int(value) for value in protocol["training"]["seeds"]]
    if seed not in seeds:
        raise ValueError("seed lies outside the frozen Tartan registry")
    if checkpoint_kind not in {"formal", "smoke"}:
        raise ValueError("checkpoint_kind must be formal or smoke")
    if training_airport != airport and (split != "test" or max_scenes is not None):
        raise RuntimeError("cross-airport evaluation is locked-test-only and forbids partial runs")
    if split == "test" and checkpoint_kind != "formal":
        raise RuntimeError("locked test requires formal checkpoints")
    receipt_path = (
        root
        / "artifacts/partc_two_dataset_20260812/tartan_retrain_frozen_receipt_v1.json"
    )
    test_gate = _authorize_split(
        split=split,
        authorize_locked_test=authorize_locked_test,
        max_scenes=max_scenes,
        formal_gate=lambda: _formal_test_gate(
            root=root, protocol=protocol, receipt_path=receipt_path
        ),
    )
    freeze_receipt = (
        test_gate["freeze_receipt"]
        if test_gate is not None
        else _verify_freeze_receipt(root=root, receipt_path=receipt_path)
    )
    checkpoints = _selected_checkpoint_triplet(
        root=root,
        protocol=protocol,
        airport=training_airport,
        regime=regime,
        seed=seed,
        formal=checkpoint_kind == "formal",
    )
    safety_thresholds = None
    safety_threshold_receipt = None
    if checkpoint_kind == "formal":
        safety_thresholds, safety_threshold_receipt = _load_safety_thresholds(
            root=root,
            airport=training_airport,
            regime=regime,
            seed=seed,
            checkpoints=checkpoints,
        )

    # Test data is not touched until every test authorization and integrity gate passes.
    dataset, all_dates, index_path = _dataset(protocol, airport, split)
    selected_indices = _limited_indices(len(dataset), max_scenes)
    selected_dates = [all_dates[index] for index in selected_indices]
    selected_scene_ids = [
        f"{airport}:{split}:{index:08d}" for index in selected_indices
    ]
    evaluation = Subset(dataset, selected_indices)
    options: dict[str, Any] = {
        "dataset": evaluation,
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": seq_collate,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }
    if workers > 0:
        options.update({"persistent_workers": True, "prefetch_factor": 4})
    loader = DataLoader(**options)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    source_path = root / checkpoints["ascent"]["path"]
    target_path = root / checkpoints["predicted_risk"]["path"]
    source, target = _load_model_pair(
        source_checkpoint=source_path,
        target_checkpoint=target_path,
        device=device,
        batch_size=batch_size,
    )
    event_protocol = json.loads((root / "mabpt/e12_protocol.json").read_text(encoding="utf-8"))
    event = event_protocol["event"]
    # Per-scene objects are useful for tiny engineering smoke runs but create
    # hundreds of thousands of redundant accumulators on the formal cohort.
    # Calendar date remains the frozen independent inference unit.
    retain_per_scene = max_scenes is not None
    metric_states = {
        name: MetricStore(retain_per_scene=retain_per_scene) for name in EVALUATED_ARMS
    }
    safety_states = {
        name: SafetyStore(retain_per_scene=retain_per_scene) for name in EVALUATED_ARMS
    }
    cursor = 0
    actors = 0
    batches = 0
    inference_seconds = 0.0
    started = time.perf_counter()
    for data in loader:
        data = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in data.items()
        }
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_started = time.perf_counter()
        truth = data["pred_traj"].transpose(1, 0).to(torch.float64)
        outputs = _model_outputs(source, target, data)
        outputs["constant_velocity"] = _constant_velocity(data)
        packed = pack_scenes(data["adj"])
        batch_dates = selected_dates[cursor : cursor + packed.scene_count]
        batch_scene_ids = selected_scene_ids[cursor : cursor + packed.scene_count]
        if len(batch_dates) != packed.scene_count:
            raise RuntimeError("scene/date index alignment failed")
        for model, values in outputs.items():
            metrics = compute_batch_metrics(
                values["support"].to(torch.float64),
                values["probability"].to(torch.float64),
                values["decision"],
                truth,
            )
            metric_states[model].update(
                metrics, packed.inverse, batch_scene_ids, batch_dates
            )
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
            safety_states[model].update(conflict, batch_scene_ids, batch_dates)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter() - inference_started
        cursor += packed.scene_count
        actors += int(truth.shape[0])
        batches += 1
    if cursor != len(evaluation):
        raise RuntimeError("evaluation did not consume the selected cohort exactly once")
    models = {name: state.summary() for name, state in metric_states.items()}
    safety = {
        name: state.summary(
            alert_threshold=(
                safety_thresholds[name] if safety_thresholds is not None else None
            )
        )
        for name, state in safety_states.items()
    }
    date_scene_counts = {
        date: selected_dates.count(date) for date in sorted(set(selected_dates))
    }
    paired_dates = {
        date: {
            "scenes": date_scene_counts[date],
            **{
                model: models[model]["per_date"][date]
                for model in EVALUATED_ARMS
            },
        }
        for date in sorted(set(selected_dates))
    }
    calibration_metrics = (
        "nll",
        "brier",
        "ece",
        "oracle_ade_rank1",
        "oracle_fde_rank1",
    )
    calibration = {
        model: {
            "overall": {
                metric: models[model]["overall"][metric]
                for metric in calibration_metrics
            },
            "per_date": {
                date: {
                    metric: values[metric] for metric in calibration_metrics
                }
                for date, values in models[model]["per_date"].items()
            },
        }
        for model in MODELS
    }
    return _json_safe(
        {
            "format_version": 1,
            "experiment_id": (
                "Tartan_retrain_locked_split_evaluation"
                if training_airport == airport
                else "Tartan_retrain_cross_airport_locked_test_evaluation"
            ),
            "evidence_class": (
                "locked_retrospective_test_single_pass"
                if split == "test"
                else "development_diagnostic"
            ),
            "airport": airport,
            "training_airport": training_airport,
            "evaluation_airport": airport,
            "cross_airport": training_airport != airport,
            "regime": regime,
            "seed": seed,
            "split": split,
            "scenes": len(evaluation),
            "dates": len(set(selected_dates)),
            "actors": actors,
            "models": models,
            "constant_velocity_publication_metrics": {
                metric: models["constant_velocity"]["overall"][metric]
                for metric in (
                    "top1_ade",
                    "top1_fde",
                    "minade",
                    "minfde",
                    "energy_score",
                    "minfde_p95",
                    "tail_minfde",
                )
            },
            "relative_gain_mabpt_vs_ascent": _relative_gain(
                models["original_ascent"]["overall"],
                models["mabpt_ascent"]["overall"],
            ),
            "paired_date_table": paired_dates,
            "calibration": calibration,
            "safety_proxy": {
                "event": event,
                "models": safety,
                "regulatory_claim": False,
                "threshold_fitted_on_evaluation_split": False,
                "claim_boundary": (
                    "Research proxy using the frozen E12 soft near-conflict event; "
                    "not regulatory safety assurance."
                ),
            },
            "inputs": {
                "protocol": protocol_path.relative_to(root).as_posix(),
                "protocol_sha256": _sha256(protocol_path),
                "freeze_receipt": freeze_receipt,
                "scene_date_index": index_path.relative_to(root).as_posix(),
                "scene_date_index_sha256": _sha256(index_path),
                "checkpoints": checkpoints,
                "safety_thresholds": safety_threshold_receipt,
                "formal_test_gate": test_gate,
            },
            "integrity": {
                "calendar_date_inference_unit": True,
                "per_scene_metrics_exported": retain_per_scene,
                "per_date_metrics_exported_for_paired_bootstrap": True,
                "overlapping_scenes_treated_as_independent_for_inference": False,
                "fixed_final_epoch": True,
                "fraction_100_percent": True,
                "matched_seed": True,
                "checkpoint_airport_differs_from_evaluation_airport": training_airport != airport,
                "target_in_probability_forward": False,
                "constant_velocity_probability_scores_publication_eligible": False,
                "test_dataset_constructed_only_after_gate": True,
                "locked_test_used": split == "test",
                "partial_locked_test": False,
                "development_smoke": split == "development" and max_scenes is not None,
                "formal_checkpoints": checkpoint_kind == "formal",
                "output_refuses_overwrite": True,
            },
            "efficiency": {
                "device": str(device),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "batches": batches,
                "inference_seconds": inference_seconds,
                "actors_per_second": actors / max(inference_seconds, 1e-12),
                "total_elapsed_seconds": time.perf_counter() - started,
                "peak_allocated_gpu_bytes": (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else 0
                ),
                "parameter_counts": {
                    "original_ascent": sum(value.numel() for value in source.parameters()),
                    "mabpt_ascent": sum(value.numel() for value in target.parameters()),
                },
                "checkpoint_bytes": {
                    stage: int(record["bytes"])
                    for stage, record in checkpoints.items()
                },
            },
            "claim_boundary": protocol["claim_boundaries"],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airport", choices=AIRPORTS, required=True)
    parser.add_argument("--checkpoint-airport", choices=AIRPORTS)
    parser.add_argument("--regime", choices=REGIMES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--authorize-locked-test", action="store_true")
    parser.add_argument("--checkpoint-kind", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.smoke:
        if args.split != "development":
            parser.error("--smoke is development-only")
        args.max_scenes = args.max_scenes or 8
    result = run(
        airport=args.airport,
        checkpoint_airport=args.checkpoint_airport,
        regime=args.regime,
        seed=args.seed,
        split=args.split,
        device=torch.device(args.device),
        workers=args.workers,
        batch_size=args.batch_size,
        max_scenes=args.max_scenes,
        authorize_locked_test=args.authorize_locked_test,
        checkpoint_kind=args.checkpoint_kind,
    )
    _atomic_json(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "output": args.output.resolve().as_posix(),
                "airport": result["airport"],
                "training_airport": result["training_airport"],
                "regime": result["regime"],
                "seed": result["seed"],
                "split": result["split"],
                "scenes": result["scenes"],
                "actors": result["actors"],
                "relative_gain": result["relative_gain_mabpt_vs_ascent"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
