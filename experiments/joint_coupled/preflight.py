"""C129 train-only, gradient-path, and boundary preflight."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch

from model.utils import TrajectoryDataset

from .model import VARIANT, ascent_config, build_model
from .objective import joint_coupled_dual_objective
from .protocol import load_protocol, sha256


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/joint_coupled"


def synthetic_batch(batch: int = 4) -> dict[str, torch.Tensor]:
    torch.manual_seed(129)
    return {"obs_traj": torch.cumsum(torch.randn(16, batch, 3) * 0.03, dim=0)}


def run(output: Path | None = None) -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_boundaries()
    expected = protocol.payload["dataset"]
    dataset = TrajectoryDataset(
        protocol.split_path("train").as_posix(),
        obs_len=16,
        obs_steps=1,
        pred_len=120,
        pred_step=5,
        delim=" ",
    )
    loaded = {
        "train_scenes": len(dataset),
        "train_actors": int(dataset.obs_traj.shape[0]),
    }
    expected_loaded = {
        "train_scenes": int(expected["expected_train_scenes"]),
        "train_actors": int(expected["expected_train_actors"]),
    }

    boundary_checks: dict[str, bool] = {}
    try:
        protocol.split_path("dev")
    except ValueError:
        boundary_checks["development_split_rejected"] = True
    else:
        boundary_checks["development_split_rejected"] = False
    boundary_checks["locked_test_split_rejected"] = False
    try:
        protocol.split_path("locked_test")
    except ValueError:
        boundary_checks["locked_test_split_rejected"] = True

    batch = synthetic_batch()
    model = build_model(batch_size=4).train()
    predictions, logits, auxiliary = model(batch)
    target = torch.randn(4, 24, 3)
    loss, diagnostics = joint_coupled_dual_objective(predictions, logits, target)
    loss.backward()
    finite = bool(torch.isfinite(loss)) and all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )

    forbidden = auxiliary["kinematic_decoder"]
    forbidden_checks = {
        "trajectory_residual": not bool(forbidden["trajectory_residual"]),
        "learned_gate": not bool(forbidden["learned_gate"]),
        "token_codebook": not bool(forbidden.get("token_codebook", False)),
        "future_autoregression": not bool(forbidden.get("future_autoregression", False)),
        "post_generation_selector": not bool(
            auxiliary.get("post_generation_selector", False)
        ),
    }

    # Re-run only the score CE to prove that score learning is coupled to the
    # shared mode representation, while the hard target itself is detached.
    score_model = build_model(batch_size=4).train()
    score_predictions, score_logits, _ = score_model(batch)
    _, score_diagnostics = joint_coupled_dual_objective(
        score_predictions, score_logits, target
    )
    score_loss = torch.nn.functional.cross_entropy(
        score_logits, score_diagnostics["score_winner"]
    )
    score_loss.backward()
    pi_grad = sum(
        float(parameter.grad.detach().abs().sum())
        for name, parameter in score_model.named_parameters()
        if name.startswith("pi.") and parameter.grad is not None
    )
    shared_grad = sum(
        float(parameter.grad.detach().abs().sum())
        for name, parameter in score_model.named_parameters()
        if not name.startswith("pi.") and parameter.grad is not None
    )
    gradient_checks = {
        "score_loss_finite": bool(torch.isfinite(score_loss)),
        "score_head_receives_gradient": pi_grad > 0.0,
        "shared_mode_features_receive_gradient": shared_grad > 0.0,
        "score_target_is_detached": not score_diagnostics["score_winner"].requires_grad,
    }

    shape_checks = {
        "predictions": list(predictions.shape) == [4, 5, 24, 3],
        "logits": list(logits.shape) == [4, 5],
        "native_k5": int(ascent_config()["k"]) == 5,
        "finite_loss_and_gradients": finite,
    }
    checks = {
        "frozen_inputs_match": loaded == expected_loaded,
        **boundary_checks,
        **shape_checks,
        **forbidden_checks,
        **gradient_checks,
        "protocol_has_fixed_scales": bool(protocol.payload["metric_scales"]["retuning_prohibited"]),
    }
    result = {
        "format_version": 1,
        "cycle": protocol.payload["cycle"],
        "variant": VARIANT,
        "protocol_sha256": sha256(protocol.path),
        "manifest_sha256": sha256(protocol.manifest_path),
        "loaded": loaded,
        "expected": expected_loaded,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        },
        "gradient_norms": {"score_head": pi_grad, "shared": shared_grad},
        "checks": checks,
        "passed": all(checks.values()),
        "locked_test_used": False,
        "development_used": False,
    }
    if not result["passed"]:
        raise RuntimeError(f"C129 preflight failed: {checks}")
    output = output or ARTIFACT_ROOT / "preflight.json"
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
