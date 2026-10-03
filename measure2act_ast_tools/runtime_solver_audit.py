"""Runtime and numerical solver audit for Measure2Act/AST closure."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("ASCENT_FULL_ROOT", DEFAULT_ROOT)).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mabpt.operator import (  # noqa: E402
    all_permutations,
    energy_kl_objective,
    energy_kl_projection,
    exact_gibbs_transport,
)


def load_runtime_files(root: Path) -> list[dict[str, Any]]:
    payloads = []
    if not root.exists():
        return payloads
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        runtime = payload.get("runtime")
        if not isinstance(runtime, dict):
            continue
        payloads.append(
            {
                "path": path.as_posix(),
                "experiment_id": payload.get("experiment_id"),
                "airport": payload.get("airport"),
                "regime": payload.get("regime"),
                "seed": payload.get("seed"),
                "split": payload.get("split"),
                "actors": payload.get("actors"),
                "runtime": runtime,
            }
        )
    return payloads


def summarize_runtime(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if not payloads:
        return {"files": 0}
    actors = np.asarray([float(p.get("actors") or 0.0) for p in payloads])
    elapsed = np.asarray([float(p["runtime"].get("total_elapsed_seconds") or 0.0) for p in payloads])
    inference = np.asarray([float(p["runtime"].get("inference_seconds") or 0.0) for p in payloads])
    aps = np.asarray([float(p["runtime"].get("actors_per_second") or 0.0) for p in payloads])
    peak = np.asarray([float(p["runtime"].get("peak_allocated_gpu_bytes") or 0.0) for p in payloads])
    return {
        "files": len(payloads),
        "total_actors": int(actors.sum()),
        "total_elapsed_seconds": float(elapsed.sum()),
        "total_inference_seconds": float(inference.sum()),
        "actors_per_second": {
            "min": float(aps.min()),
            "median": float(np.median(aps)),
            "max": float(aps.max()),
        },
        "peak_allocated_gpu_bytes": {
            "max": int(peak.max()),
            "median": int(np.median(peak)),
        },
        "devices": sorted({str(p["runtime"].get("device")) for p in payloads}),
        "torch_versions": sorted({str(p["runtime"].get("torch")) for p in payloads}),
        "cuda_versions": sorted({str(p["runtime"].get("cuda")) for p in payloads}),
    }


def solver_audit() -> dict[str, Any]:
    torch.manual_seed(20260916)
    batch = 64
    modes = 5
    raw = torch.rand(batch, modes, dtype=torch.float64)
    prior = raw / raw.sum(dim=1, keepdim=True)
    risk = torch.randn(batch, modes, dtype=torch.float64) * 0.25
    points = torch.randn(batch, modes, 3, dtype=torch.float64)
    pairwise = torch.linalg.vector_norm(points[:, :, None] - points[:, None, :], dim=-1)
    default_p, default_meta = energy_kl_projection(prior, risk, pairwise)
    strict_p, strict_meta = energy_kl_projection(
        prior,
        risk,
        pairwise,
        backtracking_steps=32,
        tolerance=1e-12,
    )
    cost = torch.rand(batch, modes, modes, dtype=torch.float64)
    gibbs = exact_gibbs_transport(prior, cost, mass_weighted=False)
    strict_obj = energy_kl_objective(strict_p, prior, risk, pairwise)
    return {
        "device": "cpu",
        "batch": batch,
        "modes": modes,
        "exact_permutation_count": int(all_permutations(modes).shape[0]),
        "expected_permutation_count": 120,
        "default_projection": {
            "sum_max_abs_error": float((default_p.sum(dim=1) - 1.0).abs().max()),
            "minimum_probability": float(default_p.min()),
            "max_kkt_residual": float(default_meta["kkt_residual"].max()),
            "min_objective_gain": float(default_meta["objective_gain"].min()),
        },
        "strict_projection": {
            "sum_max_abs_error": float((strict_p.sum(dim=1) - 1.0).abs().max()),
            "minimum_probability": float(strict_p.min()),
            "max_kkt_residual": float(strict_meta["kkt_residual"].max()),
            "min_objective_gain": float(strict_meta["objective_gain"].min()),
            "max_objective": float(strict_obj.max()),
        },
        "default_vs_strict": {
            "max_abs_probability_difference": float((default_p - strict_p).abs().max()),
            "mean_l1_probability_difference": float((default_p - strict_p).abs().sum(dim=1).mean()),
        },
        "exact_gibbs": {
            "transported_sum_max_abs_error": float((gibbs["transported"].sum(dim=1) - 1.0).abs().max()),
            "transported_minimum_probability": float(gibbs["transported"].min()),
            "assignment_entropy_mean": float(gibbs["assignment_entropy"].mean()),
        },
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    lines = [
        "# Runtime / Solver Audit",
        "",
        f"- Root: `{payload['root']}`",
        f"- Solver exact permutations K=5: `{payload['solver_audit']['exact_permutation_count']}`",
        f"- Strict max KKT residual: `{payload['solver_audit']['strict_projection']['max_kkt_residual']:.3e}`",
        f"- Strict simplex max error: `{payload['solver_audit']['strict_projection']['sum_max_abs_error']:.3e}`",
        "",
        "## Runtime Inputs",
    ]
    for name, summary in payload["runtime_summaries"].items():
        lines.extend(
            [
                f"### {name}",
                f"- files: `{summary.get('files', 0)}`",
                f"- total actors: `{summary.get('total_actors', 0)}`",
                f"- total elapsed seconds: `{summary.get('total_elapsed_seconds', 0.0):.3f}`",
                f"- median actors/sec: `{summary.get('actors_per_second', {}).get('median', 0.0):.3f}`",
                f"- max GPU bytes: `{summary.get('peak_allocated_gpu_bytes', {}).get('max', 0)}`",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("$LOCAL_WORKSPACE/measure2act_ast_runs_20260916"))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    runtime_roots = {
        "probability_ablation_v2": args.run_root / "probability_ablation_v2",
        "negative_controls": args.run_root / "negative_controls",
        "smoke": args.run_root / "smoke",
    }
    runtime_payloads = {name: load_runtime_files(path) for name, path in runtime_roots.items()}
    result = {
        "format_version": 1,
        "experiment_id": "measure2act_ast_runtime_solver_audit_v1",
        "root": ROOT.as_posix(),
        "run_root": args.run_root.resolve().as_posix(),
        "runtime_summaries": {
            name: summarize_runtime(payloads)
            for name, payloads in runtime_payloads.items()
        },
        "runtime_files": runtime_payloads,
        "solver_audit": solver_audit(),
        "claim_boundary": "CPU synthetic solver audit plus real local evaluator runtime receipts.",
    }
    atomic_json(args.output_json.resolve(), result)
    write_markdown(args.output_md.resolve(), result)
    print(json.dumps({"output": args.output_json.resolve().as_posix()}, indent=2))


if __name__ == "__main__":
    main()
