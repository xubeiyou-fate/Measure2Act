"""C130 identity, boundary, architecture, and gradient-path preflight."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
from torch.nn import functional as F

from model.utils import TrajectoryDataset

from .model import VARIANT, ascent_config, build_model
from .objective import dual_expected_risk_objective, per_mode_errors
from .protocol import load_protocol, sha256


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/dual_expected_risk"


def synthetic_batch(batch: int = 4) -> dict[str, torch.Tensor]:
    torch.manual_seed(130)
    return {"obs_traj": torch.cumsum(torch.randn(16, batch, 3) * 0.03, dim=0)}


def _gradient_sum(model, prefix: str, *, invert: bool = False) -> float:
    return sum(
        float(parameter.grad.detach().abs().sum())
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and ((not name.startswith(prefix)) if invert else name.startswith(prefix))
    )


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
    boundary_checks = {}
    for split, name in (("dev", "development_split_rejected"), ("locked_test", "locked_test_split_rejected")):
        try:
            protocol.split_path(split)
        except ValueError:
            boundary_checks[name] = True
        else:
            boundary_checks[name] = False

    batch = synthetic_batch()
    target = torch.randn(4, 24, 3)
    model = build_model(batch_size=4).train()
    predictions, logits, auxiliary = model(batch)
    risks = auxiliary["risk_predictions"]
    loss, diagnostics = dual_expected_risk_objective(
        predictions, logits, risks, target
    )
    loss.backward()
    finite = bool(torch.isfinite(loss)) and all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )

    risk_model = build_model(batch_size=4).train()
    risk_predictions, risk_logits, risk_auxiliary = risk_model(batch)
    _, risk_diagnostics = dual_expected_risk_objective(
        risk_predictions,
        risk_logits,
        risk_auxiliary["risk_predictions"],
        target,
    )
    risk_only = F.mse_loss(
        risk_auxiliary["centered_risk_predictions"],
        risk_diagnostics["risk_target"],
    )
    risk_only.backward()
    risk_head_grad = _gradient_sum(risk_model, "pi.")
    shared_risk_grad = _gradient_sum(risk_model, "pi.", invert=True)

    geometry_model = build_model(batch_size=4).train()
    geometry_predictions, _, _ = geometry_model(batch)
    ade, fde = per_mode_errors(geometry_predictions, target)
    geometry_only = (
        ade.min(dim=1).values / float(protocol.payload["metric_scales"]["ade"])
        + fde.min(dim=1).values / float(protocol.payload["metric_scales"]["fde"])
    ).mean()
    geometry_only.backward()
    geometry_grad = sum(
        float(parameter.grad.detach().abs().sum())
        for name, parameter in geometry_model.named_parameters()
        if parameter.grad is not None and name.startswith(("fp1.", "fp2.", "fp3."))
    )

    decoder = auxiliary["kinematic_decoder"]
    source_checks = {
        "c127_protocol_hash": sha256(ROOT / "experiments/metric_exact/protocol.json")
        == protocol.payload["screening"]["c127_control_protocol_sha256"],
        "c129_protocol_hash": sha256(ROOT / "experiments/joint_coupled/protocol.json")
        == protocol.payload["screening"]["c129_protocol_sha256"],
        "c129_final_summary_hash": sha256(
            ROOT / "artifacts/experiments/joint_coupled/final_summary.json"
        )
        == protocol.payload["screening"]["c129_final_summary_sha256"],
    }
    checks = {
        "frozen_inputs_match": loaded == expected_loaded,
        **boundary_checks,
        **source_checks,
        "predictions_shape": list(predictions.shape) == [4, 5, 24, 3],
        "logits_shape": list(logits.shape) == [4, 5],
        "dual_risk_shape": list(risks.shape) == [4, 5, 2],
        "native_k5": int(ascent_config()["k"]) == 5,
        "fixed_logit_identity": torch.allclose(
            logits,
            -auxiliary["centered_risk_predictions"].sum(dim=-1),
            atol=1e-6,
            rtol=1e-6,
        ),
        "risk_targets_detached": not diagnostics["risk_target"].requires_grad,
        "risk_head_receives_gradient": risk_head_grad > 0,
        "shared_features_receive_risk_gradient": shared_risk_grad > 0,
        "geometry_heads_receive_exact_geometry_gradient": geometry_grad > 0,
        "finite_loss_and_gradients": finite,
        "trajectory_residual_absent": not bool(decoder["trajectory_residual"]),
        "control_residual_absent": not bool(decoder["control_residual"]),
        "learned_gate_absent": not bool(decoder["learned_gate"]),
        "token_codebook_absent": not bool(decoder.get("token_codebook", False)),
        "future_autoregression_absent": not bool(decoder.get("future_autoregression", False)),
        "post_generation_selector_absent": not bool(auxiliary.get("post_generation_selector", False)),
        "fixed_scales_and_weights": bool(protocol.payload["metric_scales"]["retuning_prohibited"])
        and bool(protocol.payload["objective_weights"]["search_prohibited"]),
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
        "gradient_norms": {
            "risk_head": risk_head_grad,
            "shared_from_risk": shared_risk_grad,
            "geometry_heads": geometry_grad,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "locked_test_used": False,
        "development_used": False,
    }
    if not result["passed"]:
        raise RuntimeError(f"C130 preflight failed: {checks}")
    output = output or ARTIFACT_ROOT / "preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
