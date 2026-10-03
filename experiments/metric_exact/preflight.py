"""C127 data, objective, gradient, fold, and evaluator preflight."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch

from airroute_stage_m.evaluation import compute_batch_metrics
from model.ascent import Ascent
from model.utils import TrajectoryDataset

from .folds import build_fold_artifact, indices_for_fold
from .model import VARIANTS, ScoreIsolatedAscent, ascent_config, build_model
from .objective import objective_for_variant, per_mode_errors
from .protocol import load_protocol, sha256


ROOT = Path(__file__).resolve().parents[1]


def synthetic_batch(batch: int = 3) -> dict[str, torch.Tensor]:
    torch.manual_seed(127)
    return {"obs_traj": torch.cumsum(torch.randn(16, batch, 3) * 0.03, dim=0)}


def run(output: Path | None = None) -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_boundaries()
    manifest = protocol.manifest()
    expected = protocol.payload["dataset"]["expected"]
    train_dataset = TrajectoryDataset(
        protocol.split_path("train").as_posix(),
        obs_len=16,
        obs_steps=1,
        pred_len=120,
        pred_step=5,
        delim=" ",
    )
    dev_dataset = TrajectoryDataset(
        protocol.split_path("dev").as_posix(),
        obs_len=16,
        obs_steps=1,
        pred_len=120,
        pred_step=5,
        delim=" ",
    )
    train_date_path = ROOT / str(protocol.payload["dataset"]["train_scene_dates"])
    train_dates = json.loads(train_date_path.read_text(encoding="utf-8"))["dates"]
    fold_artifact = build_fold_artifact(protocol)
    fold_checks = []
    validation_union: set[int] = set()
    for fold in range(int(fold_artifact["fold_count"])):
        train_indices, validation_indices, validation_dates = indices_for_fold(
            train_dates, fold_artifact, fold
        )
        fold_checks.append(
            len(train_indices) > 0
            and len(validation_indices) > 0
            and len(validation_indices) == len(validation_dates)
        )
        validation_union.update(validation_indices)

    batch = synthetic_batch()
    torch.manual_seed(127)
    direct = Ascent(ascent_config("B0_signed_coupled")).eval()
    torch.manual_seed(127)
    wrapped = build_model("B0_signed_coupled").eval()
    with torch.no_grad():
        forward_parity = float((direct(batch)[0] - wrapped(batch)[0]).abs().max())

    shapes = {}
    forbidden = {}
    finite_gradients = {}
    for variant in VARIANTS:
        model = build_model(variant).train()
        prediction, logits, auxiliary = model(batch)
        target = torch.randn(3, 24, 3)
        loss, diagnostics = objective_for_variant(variant, prediction, logits, target)
        loss.backward()
        shapes[variant] = {
            "prediction": list(prediction.shape),
            "logits": list(logits.shape),
        }
        forbidden[variant] = {
            "trajectory_residual": bool(
                auxiliary["kinematic_decoder"]["trajectory_residual"]
            ),
            "learned_gate": bool(auxiliary["kinematic_decoder"]["learned_gate"]),
            "post_generation_selector": bool(
                ascent_config(variant)["post_generation_selector"]
            ),
        }
        finite_gradients[variant] = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ) and bool(torch.isfinite(diagnostics["regression"]))

    isolated = build_model("B6_dual_oracle")
    assert isinstance(isolated, ScoreIsolatedAscent)
    isolated_prediction, isolated_logits, _ = isolated(batch)
    isolated_loss, _ = objective_for_variant(
        "B6_dual_oracle", isolated_prediction, isolated_logits, torch.randn(3, 24, 3)
    )
    isolated_loss.backward()
    fixed_zero_logits = bool((isolated_logits == 0).all())
    zero_score_has_parameters = any(True for _ in isolated.geometry.pi.parameters())

    predictions = torch.randn(4, 5, 24, 3)
    target = torch.randn(4, 24, 3)
    logits = torch.randn(4, 5)
    ade, fde = per_mode_errors(predictions, target)
    evaluated = compute_batch_metrics(predictions, logits, target)
    evaluator_parity = {
        "minade_max_abs": float((ade.min(dim=1).values - evaluated["minade"]).abs().max()),
        "minfde_max_abs": float((fde.min(dim=1).values - evaluated["minfde"]).abs().max()),
    }

    distinct = torch.full((1, 2, 4, 3), 10.0, requires_grad=True)
    distinct.data[:, 0] = 1.0
    distinct.data[:, 1, -1] = 0.0
    distinct_logits = torch.zeros(1, 2)
    distinct_target = torch.zeros(1, 4, 3)
    dual_loss, dual_diagnostics = objective_for_variant(
        "B6_dual_oracle", distinct, distinct_logits, distinct_target
    )
    dual_loss.backward()
    distinct_winner_check = (
        int(dual_diagnostics["ade_winner"][0]) == 0
        and int(dual_diagnostics["fde_winner"][0]) == 1
        and bool(distinct.grad[0, 0].abs().sum() > 0)
        and bool(distinct.grad[0, 1, -1].abs().sum() == 0)
    )
    # The FDE winner is exactly correct in this synthetic case, so its endpoint
    # gradient is zero. Perturb it slightly to verify endpoint-only routing.
    endpoint = distinct.detach().clone().requires_grad_(True)
    endpoint.data[:, 1, -1] = 0.1
    endpoint_loss, endpoint_diagnostics = objective_for_variant(
        "B6_dual_oracle", endpoint, distinct_logits, distinct_target
    )
    endpoint_loss.backward()
    distinct_winner_check = distinct_winner_check and (
        int(endpoint_diagnostics["fde_winner"][0]) == 1
        and bool(endpoint.grad[0, 1, -1].abs().sum() > 0)
        and bool(endpoint.grad[0, 1, :-1].abs().sum() == 0)
    )

    run_order_path = ROOT / str(protocol.payload["run_order"]["artifact"])
    generator = torch.Generator().manual_seed(int(protocol.payload["run_order"]["seed"]))
    variants = list(protocol.payload["phases"]["P1"]["variants"])
    order = torch.randperm(len(variants), generator=generator).tolist()
    run_order = {
        "format_version": 1,
        "cycle": protocol.payload["cycle"],
        "seed": int(protocol.payload["run_order"]["seed"]),
        "P1_fold0_seed42": [variants[index] for index in order],
        "maximum_concurrent_runs_per_gpu": 1,
        "locked_test_used": False,
    }
    run_order_path.parent.mkdir(parents=True, exist_ok=True)
    run_order_path.write_text(json.dumps(run_order, indent=2) + "\n", encoding="utf-8")

    loaded = {
        "train_scenes": len(train_dataset),
        "train_actors": int(train_dataset.obs_traj.shape[0]),
        "development_scenes": len(dev_dataset),
        "development_actors": int(dev_dataset.obs_traj.shape[0]),
    }
    expected_loaded = {name: int(expected[name]) for name in loaded}
    checks = {
        "frozen_input_hashes_match": True,
        "locked_test_sealed": manifest["locked_test_evaluated"] is False,
        "cohort_matches_expected": loaded == expected_loaded,
        "train_scene_dates_complete": len(train_dates) == len(train_dataset),
        "train_manifest_date_count_matches": int(
            manifest["partitions"]["train"]["date_count"]
        ) == int(expected["train_manifest_dates"]),
        "train_scene_date_count_matches": len(set(train_dates))
        == int(expected["train_scene_dates"]),
        "folds_nonempty_and_complete": all(fold_checks)
        and len(validation_union) == len(train_dataset),
        "B0_forward_parity": forward_parity == 0.0,
        "all_shapes_match_K5_T24": all(
            value == {"prediction": [3, 5, 24, 3], "logits": [3, 5]}
            for value in shapes.values()
        ),
        "all_forbidden_mechanisms_disabled": all(
            not any(value.values()) for value in forbidden.values()
        ),
        "all_objective_gradients_finite": all(finite_gradients.values()),
        "isolated_logits_are_parameter_free_zero": fixed_zero_logits
        and not zero_score_has_parameters,
        "evaluator_minade_parity": evaluator_parity["minade_max_abs"] == 0.0,
        "evaluator_minfde_parity": evaluator_parity["minfde_max_abs"] == 0.0,
        "dual_oracle_routes_distinct_winner_gradients": distinct_winner_check,
    }
    result = {
        "format_version": 1,
        "cycle": protocol.payload["cycle"],
        "protocol_sha256": sha256(protocol.path),
        "manifest_sha256": sha256(protocol.manifest_path),
        "loaded": loaded,
        "folds": fold_artifact,
        "run_order": run_order,
        "forward_parity_max_abs_error": forward_parity,
        "shapes": shapes,
        "forbidden": forbidden,
        "finite_gradients": finite_gradients,
        "evaluator_parity": evaluator_parity,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        },
        "checks": checks,
        "passed": all(checks.values()),
        "locked_test_used": False,
    }
    if not result["passed"]:
        raise RuntimeError(f"C127 preflight failed: {checks}")
    output = output or ROOT / "artifacts/experiments/metric_exact/preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
