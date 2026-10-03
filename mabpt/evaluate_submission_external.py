"""Evaluate five-seed frozen MABPT checkpoints on a sealed Tartan view."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import DataLoader, Subset

from experiments.energy_predict_optimize.evaluation import RankingMetricAccumulator, compute_batch_metrics
from model.utils import TrajectoryDataset, seed_worker, seq_collate

from .partc_seed_evaluate import _load_model_pair, _model_outputs


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("submission_external_protocol_v1.json")
ARMS = ("constant_velocity", "original_ascent", "mabpt_ascent")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"submission evaluator refuses to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _constant_velocity(data: dict[str, torch.Tensor]) -> torch.Tensor:
    observation = data["obs_traj"]
    velocity = observation[-1] - observation[-2]
    seconds = torch.arange(5, 121, 5, device=observation.device, dtype=observation.dtype)
    return observation[-1, :, None] + velocity[:, None] * seconds[None, :, None]


def _limited_dataset(dataset: TrajectoryDataset, maximum: int | None):
    if maximum is None or maximum >= len(dataset):
        return dataset
    if maximum < 1:
        raise ValueError("--max-scenes must be positive")
    indices = torch.linspace(0, len(dataset) - 1, maximum).round().long().unique().tolist()
    return Subset(dataset, indices)


def _verify_view(dataset_name: str, dataset_path: Path, protocol: dict[str, object]) -> dict[str, object]:
    manifest_record = protocol["view_manifest"]
    manifest_path = ROOT / manifest_record["path"]
    if _sha256(manifest_path) != manifest_record["sha256"]:
        raise RuntimeError("submission view manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["datasets"][dataset_name]["records"]
    digest = hashlib.sha256()
    total_bytes = 0
    for record in records:
        path = dataset_path / record["name"]
        if not path.is_file():
            raise FileNotFoundError(path)
        file_hash = _sha256(path)
        if file_hash != record["sha256"]:
            raise RuntimeError(f"submission view content mismatch: {path.name}")
        total_bytes += path.stat().st_size
        digest.update(path.name.encode("ascii"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    actual = sorted(path.name for path in dataset_path.glob("*.txt"))
    expected = sorted(record["name"] for record in records)
    if actual != expected:
        raise RuntimeError("submission view has missing or unregistered files")
    return {
        "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
        "manifest_sha256": manifest_record["sha256"],
        "files": len(records),
        "bytes": total_bytes,
        "ordered_name_content_sha256": digest.hexdigest(),
    }


@torch.inference_mode()
def run(
    *,
    dataset_name: str,
    seed: int,
    device: torch.device,
    workers: int,
    batch_size: int,
    max_scenes: int | None,
    protocol_path: Path = PROTOCOL,
) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if dataset_name not in protocol["datasets"]:
        raise ValueError("dataset lies outside the frozen submission registry")
    if seed not in [int(value) for value in protocol["seeds"]]:
        raise ValueError("seed lies outside the frozen submission registry")
    specification = protocol["datasets"][dataset_name]
    dataset_path = (ROOT / specification["path"]).resolve()
    view_record = _verify_view(dataset_name, dataset_path, protocol)
    checkpoint = protocol["checkpoints"][str(seed)]
    source_path = (ROOT / checkpoint["source"]).resolve()
    target_path = (ROOT / checkpoint["target"]).resolve()
    if _sha256(source_path) != checkpoint["source_sha256"]:
        raise RuntimeError("source checkpoint hash mismatch")
    if _sha256(target_path) != checkpoint["target_sha256"]:
        raise RuntimeError("target checkpoint hash mismatch")
    load_started = time.perf_counter()
    dataset = TrajectoryDataset(
        dataset_path.as_posix(),
        obs_len=16,
        obs_steps=1,
        pred_len=120,
        pred_step=5,
        delim=specification["delimiter"],
    )
    dataset_load_seconds = time.perf_counter() - load_started
    evaluation_dataset = _limited_dataset(dataset, max_scenes)
    options = {
        "dataset": evaluation_dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": seq_collate,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
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
    source, target = _load_model_pair(
        source_checkpoint=source_path,
        target_checkpoint=target_path,
        device=device,
        batch_size=batch_size,
    )
    states = {name: RankingMetricAccumulator() for name in ARMS}
    inference_started = time.perf_counter()
    batches = 0
    for data in loader:
        data = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in data.items()
        }
        truth = data["pred_traj"].transpose(1, 0).to(torch.float64)
        outputs = _model_outputs(source, target, data)
        cv = _constant_velocity(data)[:, None].to(torch.float64)
        measures = {
            "constant_velocity": (
                cv,
                torch.ones((truth.shape[0], 1), dtype=torch.float64, device=device),
                torch.zeros(truth.shape[0], dtype=torch.long, device=device),
            ),
            "original_ascent": (
                outputs["original_ascent"]["support"].to(torch.float64),
                outputs["original_ascent"]["probability"].to(torch.float64),
                outputs["original_ascent"]["decision"],
            ),
            "mabpt_ascent": (
                outputs["mabpt_ascent"]["support"].to(torch.float64),
                outputs["mabpt_ascent"]["probability"].to(torch.float64),
                outputs["mabpt_ascent"]["decision"],
            ),
        }
        for name, (support, probability, decision) in measures.items():
            states[name].update(compute_batch_metrics(support, probability, decision, truth))
        batches += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started
    arms = {name: state.summarize() for name, state in states.items()}
    actors = int(arms["original_ascent"]["agents"])
    if any(int(values["agents"]) != actors for values in arms.values()):
        raise RuntimeError("paired arm actor counts differ")
    return {
        "format_version": 1,
        "experiment_id": "Tartan_submission_zero_shot_K5",
        "evidence_class": protocol["evidence_class"],
        "dataset": dataset_name,
        "airport": specification["airport"],
        "seed": seed,
        "scenes": len(evaluation_dataset),
        "actors": actors,
        "arms": arms,
        "relative_gain_mabpt_vs_ascent": {
            metric: (arms["original_ascent"][metric] - arms["mabpt_ascent"][metric])
            / arms["original_ascent"][metric]
            for metric in ("top1_ade", "top1_fde", "minade", "minfde", "energy_score", "ece", "tail_minfde")
        },
        "inputs": {
            "protocol": protocol_path.relative_to(ROOT).as_posix(),
            "protocol_sha256": _sha256(protocol_path),
            "view": view_record,
            "source_checkpoint": checkpoint,
        },
        "integrity": {
            "zero_shot_external_evaluation": True,
            "external_finetuning_or_calibration": False,
            "matched_seed_checkpoint_hashes": True,
            "external_result_based_selection": False,
            "target_in_probability_forward": False,
            "deterministic_inference": True,
            "fresh_prospective_or_hidden_test": False,
        },
        "runtime": {
            "device": str(device),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "dataset_load_seconds": dataset_load_seconds,
            "inference_seconds": inference_seconds,
            "batches": batches,
            "actors_per_second": actors / max(inference_seconds, 1e-12),
            "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        },
        "claim_boundary": protocol["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.smoke and args.max_scenes is None:
        args.max_scenes = 16
    result = run(
        dataset_name=args.dataset,
        seed=args.seed,
        device=torch.device(args.device),
        workers=args.workers,
        batch_size=args.batch_size,
        max_scenes=args.max_scenes,
        protocol_path=args.protocol.resolve(),
    )
    _atomic_json(args.output.resolve(), result)
    print(json.dumps({
        "output": args.output.resolve().as_posix(),
        "dataset": result["dataset"],
        "seed": result["seed"],
        "scenes": result["scenes"],
        "actors": result["actors"],
        "relative_gain": result["relative_gain_mabpt_vs_ascent"],
    }, indent=2))


if __name__ == "__main__":
    main()
