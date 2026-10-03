"""Evaluate one official stochastic aviation baseline as a finite measure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .train_official_baseline import (
    DATASETS,
    FAMILIES,
    PROTOCOL,
    ROOT,
    SOURCE_RECEIPT,
    _load_official,
    _official_args,
    _seed_everything,
    _sha256,
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class MetricState:
    ACTOR_METRICS = (
        "sample1_ade",
        "sample1_fde",
        "expected_ade",
        "expected_fde",
        "minade",
        "minfde",
        "energy",
    )
    SCENE_METRICS = ("scene_oracle_ade", "scene_oracle_fde")

    def __init__(self) -> None:
        self.actor_count = 0
        self.scene_count = 0
        self.actor_sums = {key: 0.0 for key in self.ACTOR_METRICS}
        self.scene_sums = {key: 0.0 for key in self.SCENE_METRICS}
        self.actor_arrays: dict[str, list[np.ndarray]] = {
            key: [] for key in self.ACTOR_METRICS
        }

    def update(
        self,
        prediction: torch.Tensor,
        truth: torch.Tensor,
        *,
        independent_scenes: bool = False,
    ) -> None:
        # prediction [A,K,T,3], truth [A,T,3]
        displacement = torch.linalg.vector_norm(
            prediction.to(torch.float64) - truth.to(torch.float64)[:, None], dim=-1
        )
        ade = displacement.mean(dim=-1)
        fde = displacement[..., -1]
        pairwise = torch.linalg.vector_norm(
            prediction.to(torch.float64)[:, :, None]
            - prediction.to(torch.float64)[:, None, :],
            dim=-1,
        ).mean(dim=-1)
        values = {
            "sample1_ade": ade[:, 0],
            "sample1_fde": fde[:, 0],
            "expected_ade": ade.mean(dim=1),
            "expected_fde": fde.mean(dim=1),
            "minade": ade.min(dim=1).values,
            "minfde": fde.min(dim=1).values,
            "energy": ade.mean(dim=1) - 0.5 * pairwise.mean(dim=(1, 2)),
        }
        actors = truth.shape[0]
        self.actor_count += actors
        self.scene_count += actors if independent_scenes else 1
        for key, value in values.items():
            self.actor_sums[key] += float(value.sum())
            self.actor_arrays[key].append(value.detach().cpu().numpy())
        if independent_scenes:
            best = ade.argmin(dim=1)
            rows = torch.arange(actors, device=ade.device)
            self.scene_sums["scene_oracle_ade"] += float(ade[rows, best].sum())
            self.scene_sums["scene_oracle_fde"] += float(fde[rows, best].sum())
        else:
            scene_ade = ade.mean(dim=0)
            best = int(scene_ade.argmin())
            self.scene_sums["scene_oracle_ade"] += float(scene_ade[best])
            self.scene_sums["scene_oracle_fde"] += float(fde[:, best].mean())

    def summarize(self) -> dict[str, Any]:
        if not self.actor_count or not self.scene_count:
            raise RuntimeError("no official baseline metrics were accumulated")
        result: dict[str, Any] = {
            "actors": self.actor_count,
            "scenes": self.scene_count,
        }
        for key in self.ACTOR_METRICS:
            values = np.concatenate(self.actor_arrays[key])
            result[key] = self.actor_sums[key] / self.actor_count
            result[f"{key}_p50"] = float(np.quantile(values, 0.5))
            result[f"{key}_p95"] = float(np.quantile(values, 0.95))
        for key in self.SCENE_METRICS:
            result[key] = self.scene_sums[key] / self.scene_count
        return result


def _test_path(family: str, dataset: str) -> Path:
    if family == "trajairnet":
        return (
            ROOT / "dataset" / f"{dataset}_trajair_reconstructed"
            / "processed_data" / "test"
        )
    return (
        ROOT / "external" / "actrajnet_official" / "dataset"
        / f"{dataset}_no_social" / "test"
    )


@torch.inference_mode()
def evaluate(
    *,
    family: str,
    dataset_name: str,
    checkpoint_path: Path,
    checkpoint_label: str,
    device: torch.device,
    max_scenes: int | None,
    fuse_samples: bool = False,
    compile_model: bool = False,
) -> dict[str, Any]:
    if family not in FAMILIES or dataset_name not in DATASETS:
        raise ValueError("family or dataset lies outside the frozen E1 registry")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    evaluation_seed = int(protocol["evaluation"]["evaluation_seed"])
    _seed_everything(evaluation_seed)
    repository, model_class, utils = _load_official(family)
    data_path = _test_path(family, dataset_name)
    dataset = utils.TrajectoryDataset(
        str(data_path), obs_len=11, pred_len=120, step=10, delim=" "
    )
    if max_scenes is not None:
        dataset = Subset(dataset, range(min(max_scenes, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        collate_fn=utils.seq_collate,
    )
    model = model_class(_official_args()).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    inference_function = (
        torch.compile(model.inference, fullgraph=False, dynamic=False)
        if compile_model else model.inference
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    state = MetricState()
    started = time.perf_counter()
    for batch in loader:
        batch = [tensor.to(device) for tensor in batch]
        obs, pred, _obs_rel, _pred_rel, context, _seq_start = batch
        scene_sizes = _seq_start[:, 1] - _seq_start[:, 0]
        if family == "actrajnet" and not bool((scene_sizes == 1).all()):
            raise RuntimeError("ACTrajNet no-social data contains a multi-actor scene")
        for agents_tensor in scene_sizes.unique(sorted=True):
            agents = int(agents_tensor)
            scene_indices = (scene_sizes == agents_tensor).nonzero().flatten().tolist()
            scene_obs = torch.stack([
                obs[:, int(_seq_start[index, 0]):int(_seq_start[index, 1])].transpose(1, 2)
                for index in scene_indices
            ])
            scene_context = torch.stack([
                context[:, int(_seq_start[index, 0]):int(_seq_start[index, 1])].transpose(1, 2)
                for index in scene_indices
            ])
            scene_truth = torch.stack([
                pred[:, int(_seq_start[index, 0]):int(_seq_start[index, 1])].transpose(0, 1)
                for index in scene_indices
            ])

            def scene_inference(
                one_obs: torch.Tensor,
                one_context: torch.Tensor,
                latent: torch.Tensor,
            ) -> torch.Tensor:
                outputs = inference_function(
                    one_obs,
                    latent,
                    torch.ones((agents, agents), device=device),
                    one_context,
                )
                return torch.stack([output.transpose(0, 1) for output in outputs])

            group_scenes = len(scene_indices)
            if fuse_samples:
                # Preserve the registered five-call CUDA RNG stream exactly.
                latent = torch.stack([
                    torch.randn((group_scenes, 1, 1, 128), device=device)
                    for _ in range(5)
                ])
                flat_predictions = torch.vmap(scene_inference)(
                    scene_obs.repeat(5, 1, 1, 1),
                    scene_context.repeat(5, 1, 1, 1),
                    latent.flatten(0, 1),
                )
                predictions = flat_predictions.reshape(
                    5, group_scenes, *flat_predictions.shape[1:]
                ).permute(1, 2, 0, 3, 4)
            else:
                samples = []
                for _ in range(5):
                    latent = torch.randn((group_scenes, 1, 1, 128), device=device)
                    samples.append(torch.vmap(scene_inference)(
                        scene_obs, scene_context, latent
                    ))
                predictions = torch.stack(samples, dim=2)
            for index in range(group_scenes):
                state.update(predictions[index], scene_truth[index])
    summary = state.summarize()
    return {
        "format_version": 1,
        "experiment_id": "E1",
        "family": family,
        "dataset": dataset_name,
        "checkpoint_label": checkpoint_label,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "metrics": summary,
        "protocol_sha256": _sha256(PROTOCOL),
        "source_receipt_sha256": _sha256(SOURCE_RECEIPT),
        "evaluation": {
            "seed": evaluation_seed,
            "latent_samples": 5,
            "probabilities": [0.2] * 5,
            "history_steps": 11,
            "forecast_points": 12,
            "forecast_stride_seconds": 10,
            "test_path": str(data_path.relative_to(ROOT)),
            "maximum_scenes": max_scenes,
        },
        "runtime": {
            "device": str(device),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
        },
        "integrity": {
            "uniform_five_sample_measure": True,
            "oracle_sample_used_only_for_explicit_oracle_metrics": True,
            "test_selected_checkpoint": False,
            "temperature_or_probability_fit": False,
            "official_model_source_modified": False,
            "official_repository": str(repository.relative_to(ROOT)),
            "fused_sample_scene_vmap": fuse_samples,
            "torch_compile": compile_model,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--fuse-samples", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        family=args.family,
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint.resolve(),
        checkpoint_label=args.checkpoint_label,
        device=torch.device(args.device),
        max_scenes=args.max_scenes,
        fuse_samples=args.fuse_samples,
        compile_model=args.compile_model,
    )
    _atomic_json(args.output.resolve(), result)
    print(json.dumps({
        "output": str(args.output),
        "actors": result["metrics"]["actors"],
        "energy": result["metrics"]["energy"],
        "minade": result["metrics"]["minade"],
        "scene_oracle_ade": result["metrics"]["scene_oracle_ade"],
    }, indent=2))


if __name__ == "__main__":
    main()
