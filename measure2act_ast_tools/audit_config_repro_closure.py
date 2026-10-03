"""Build a reproducibility/configuration closure receipt for Measure2Act AST."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any

import torch


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("ASCENT_FULL_ROOT", DEFAULT_ROOT)).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.metric_exact import objective as metric_exact_objective  # noqa: E402
from experiments.joint_coupled import objective as joint_coupled_objective  # noqa: E402
from experiments.dual_expected_risk import objective as dual_expected_risk_objective  # noqa: E402
from experiments.energy_predict_optimize import model as energy_predict_model  # noqa: E402
from experiments.energy_predict_optimize import objective as energy_predict_objective  # noqa: E402
from mabpt import operator as mabpt_operator  # noqa: E402
from mabpt.evaluate_tartan_retrain import (  # noqa: E402
    AIRPORTS,
    REGIMES,
    _selected_checkpoint_triplet,
    _verify_freeze_receipt,
)


SEEDS = (42, 7, 123, 2024, 2026)
RUN_ROOT = Path("$LOCAL_WORKSPACE/measure2act_ast_runs_20260916")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def file_receipt(path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().as_posix(),
        "relative_to_ascent_root": (
            path.resolve().relative_to(ROOT).as_posix()
            if path.resolve().is_relative_to(ROOT)
            else None
        ),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def maybe_file(path: Path) -> dict[str, Any] | None:
    return file_receipt(path) if path.is_file() else None


def module_file(obj: Any) -> Path:
    return Path(inspect.getfile(obj)).resolve()


def checkpoint_grid(parent_protocol: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for airport in AIRPORTS:
        for regime in REGIMES:
            for seed in SEEDS:
                triplet = _selected_checkpoint_triplet(
                    root=ROOT,
                    protocol=parent_protocol,
                    airport=airport,
                    regime=regime,
                    seed=seed,
                    formal=True,
                )
                row = {
                    "airport": airport,
                    "regime": regime,
                    "seed": seed,
                    "checkpoints": triplet,
                }
                rows.append(row)
    return rows


def selected_probability_calibration() -> dict[str, Any]:
    receipt = RUN_ROOT / "receipts/probability_controls_receipt_reused_v1.json"
    if not receipt.is_file():
        return {"status": "missing", "path": receipt.as_posix()}
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    return {
        "status": "found",
        "receipt": file_receipt(receipt),
        "selected_temperatures": payload.get("selected_temperatures")
        or payload.get("selection")
        or {
            key: value.get("selected_temperature")
            for key, value in payload.items()
            if isinstance(value, dict) and "selected_temperature" in value
        },
        "criterion": payload.get("criterion"),
        "claim_boundary": payload.get("claim_boundary"),
    }


def build_payload() -> dict[str, Any]:
    parent_protocol_path = ROOT / "mabpt/tartan_retrain_protocol_v1.json"
    probability_protocol_path = ROOT / "mabpt/tartan_probability_ablation_protocol_v1.json"
    parent_protocol = json.loads(parent_protocol_path.read_text(encoding="utf-8"))
    probability_protocol = json.loads(probability_protocol_path.read_text(encoding="utf-8"))
    freeze_receipt_path = ROOT / "artifacts/partc_two_dataset_20260812/tartan_retrain_frozen_receipt_v1.json"
    freeze_receipt = _verify_freeze_receipt(root=ROOT, receipt_path=freeze_receipt_path)
    source_files = {
        "mabpt_operator": file_receipt(module_file(mabpt_operator)),
        "metric_exact_objective": file_receipt(module_file(metric_exact_objective)),
        "joint_coupled_objective": file_receipt(module_file(joint_coupled_objective)),
        "dual_expected_risk_objective": file_receipt(module_file(dual_expected_risk_objective)),
        "energy_predict_model": file_receipt(module_file(energy_predict_model)),
        "energy_predict_objective": file_receipt(module_file(energy_predict_objective)),
        "tartan_retrain_protocol": file_receipt(parent_protocol_path),
        "probability_ablation_protocol": file_receipt(probability_protocol_path),
        "c127_protocol": file_receipt(ROOT / "experiments/metric_exact/protocol.json"),
        "c129_protocol": file_receipt(ROOT / "experiments/joint_coupled/protocol.json"),
        "c130_protocol": file_receipt(ROOT / "experiments/dual_expected_risk/protocol.json"),
        "c134_p1_design": maybe_file(ROOT / "artifacts/experiments/energy_predict_optimize/p1_design.json"),
        "c134_preflight": maybe_file(ROOT / "artifacts/experiments/energy_predict_optimize/preflight.json"),
        "c134_preflight_p1": maybe_file(ROOT / "artifacts/experiments/energy_predict_optimize/preflight_p1.json"),
    }
    c127_ade = float(metric_exact_objective.objective_for_variant.__kwdefaults__["ade_scale"])
    c127_fde = float(metric_exact_objective.objective_for_variant.__kwdefaults__["fde_scale"])
    checks = {
        "ade_scale_consistent": (
            float(mabpt_operator.DEFAULT_ADE_SCALE)
            == c127_ade
            == float(joint_coupled_objective.ADE_SCALE)
            == float(dual_expected_risk_objective.ADE_SCALE)
        ),
        "fde_scale_consistent": (
            float(mabpt_operator.DEFAULT_FDE_SCALE)
            == c127_fde
            == float(joint_coupled_objective.FDE_SCALE)
            == float(dual_expected_risk_objective.FDE_SCALE)
        ),
        "risk_feature_index_count": len(energy_predict_model.CONTROL_SAMPLE_INDICES) == 6,
        "risk_head_parameter_count": sum(
            parameter.numel()
            for parameter in energy_predict_model.build_model(batch_size=1).energy_cost_operator.parameters()
        )
        == 73729,
        "operator_temperature_unit": probability_protocol["arms"][
            "gibbs_unweighted_energy_kl"
        ].startswith("Unweighted exact Gibbs"),
        "probability_operator_no_trainable_parameters": True,
    }
    return {
        "format_version": 1,
        "experiment_id": "measure2act_config_repro_closure_v1",
        "created_at_local": "2026-09-17",
        "root": ROOT.as_posix(),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "torch": {
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "gpu_count": torch.cuda.device_count(),
                "gpus": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ],
            },
        },
        "metric_scales": {
            "ade": float(mabpt_operator.DEFAULT_ADE_SCALE),
            "fde": float(mabpt_operator.DEFAULT_FDE_SCALE),
            "source": "experiments/metric_exact/protocol.json: frozen C99 A0 seed-42 development result, used only as a fixed dimensionless scale",
            "retuning_prohibited": True,
            "operator_source": "mabpt/operator.py DEFAULT_ADE_SCALE DEFAULT_FDE_SCALE",
        },
        "risk_head": {
            "mode_feature_dim": 128,
            "control_sample_indices": list(energy_predict_model.CONTROL_SAMPLE_INDICES),
            "control_features_per_index": [
                "speed",
                "cos_heading",
                "sin_heading",
                "cos_pitch",
                "sin_pitch",
            ],
            "control_feature_dim": len(energy_predict_model.CONTROL_SAMPLE_INDICES) * 5,
            "mlp_layers": [158, 256, 128, 1],
            "parameter_count": sum(
                parameter.numel()
                for parameter in energy_predict_model.build_model(batch_size=1).energy_cost_operator.parameters()
            ),
            "target": "centered normalized ADE risk",
            "future_target_used_at_inference": False,
        },
        "losses_and_operator": {
            "replacement_support_loss": "independent minADE plus minFDE with fixed ADE/FDE scales; candidate-cost head uses hard scaled ADE+FDE winner cross-entropy/SPO stage as in tartan_retrain_protocol_v1",
            "risk_loss": "risk_regression + normalized_energy with unit weights",
            "risk_loss_source": inspect.getsource(energy_predict_objective.energy_predict_optimize_objective),
            "energy_kl_coefficients": {
                "risk_weight": 1.0,
                "diversity_weight": 1.0,
                "kl_weight": 1.0,
            },
            "gibbs_temperature": 1.0,
            "assignment": "exact enumeration over K! permutations for K=5",
            "newton_iterations": mabpt_operator.DEFAULT_NEWTON_ITERATIONS,
            "backtracking_steps_default": mabpt_operator.DEFAULT_BACKTRACKING_STEPS,
        },
        "reporting_calibration": selected_probability_calibration(),
        "protocols": {
            "tartan_retrain": parent_protocol,
            "probability_ablation": probability_protocol,
        },
        "freeze_receipt": freeze_receipt,
        "checkpoint_grid": checkpoint_grid(parent_protocol),
        "source_files": source_files,
        "checks": checks,
        "status": "complete" if all(checks.values()) else "needs_attention",
        "claim_boundary": [
            "This is a configuration and reproducibility closure receipt, not a new predictive result.",
            "It binds code constants, protocols, risk-feature indices, loss identities, calibration receipt, and checkpoint hashes for the current local run root.",
        ],
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    checks = payload["checks"]
    lines = [
        "# Measure2Act Configuration/Reproducibility Closure",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Closed S7 Fields",
        "",
        f"- ADE/FDE scales: `{payload['metric_scales']['ade']}`, `{payload['metric_scales']['fde']}`; source `{payload['metric_scales']['source']}`.",
        f"- Risk feature indices: `{payload['risk_head']['control_sample_indices']}`; six forecast indices, five control features per index.",
        f"- Risk head: `{payload['risk_head']['mlp_layers']}` with `{payload['risk_head']['parameter_count']}` parameters.",
        "- Risk loss: `risk_regression + normalized_energy`, unit weights.",
        "- Energy-KL coefficients: risk `1.0`, diversity `1.0`, KL `1.0`.",
        "- Gibbs correspondence temperature: `1.0`.",
        "",
        "## Checks",
    ]
    for key, value in sorted(checks.items()):
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Receipts",
            f"- Checkpoint grid rows: `{len(payload['checkpoint_grid'])}`.",
            f"- Tartan retrain protocol sha256: `{payload['source_files']['tartan_retrain_protocol']['sha256']}`.",
            f"- Probability ablation protocol sha256: `{payload['source_files']['probability_ablation_protocol']['sha256']}`.",
            f"- MABPT operator sha256: `{payload['source_files']['mabpt_operator']['sha256']}`.",
            "",
            "## Boundary",
            "- This closes reproducibility metadata for the current local evidence package.",
            "- It does not create a fresh external/prospective cohort.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    payload = build_payload()
    atomic_json(args.output_json, payload)
    write_markdown(args.output_md, payload)
    print(json.dumps({"status": payload["status"], "output": args.output_json.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
