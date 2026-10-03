"""C134 E1 identity, gradient, solver, and frozen-backbone preflight."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .model import CONTROL_SAMPLE_INDICES, VARIANT, build_model
from .objective import energy_predict_optimize_objective
from .protocol import load_protocol, sha256


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/energy_predict_optimize"
DESIGN_PATH = ARTIFACT_ROOT / "p1_design.json"


def run() -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_boundaries()
    protocol_hash = sha256(protocol.path)
    p0_b_path = ARTIFACT_ROOT / "p0_b_decision.json"
    p0_b = json.loads(p0_b_path.read_text(encoding="utf-8"))
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    if (
        p0_b.get("decision") != "P1_FOLD0_AUTHORIZED"
        or p0_b.get("protocol_sha256") != protocol_hash
        or p0_b.get("training_performed") is not False
    ):
        raise RuntimeError("C134 P0-B did not authorize E1")
    if design.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("C134 P1 design protocol mismatch")
    backbone_path = ROOT / str(design["backbone"]["checkpoint"])
    source_checks = {
        "backbone_checkpoint_hash": sha256(backbone_path)
        == design["backbone"]["checkpoint_sha256"],
        "backbone_summary_hash": sha256(ROOT / design["backbone"]["summary"])
        == design["backbone"]["summary_sha256"],
        "backbone_protocol_hash": sha256(ROOT / design["backbone"]["protocol"])
        == design["backbone"]["protocol_sha256"],
    }
    torch.manual_seed(134)
    model = build_model(batch_size=4)
    model.load_backbone(backbone_path, torch.device("cpu"))
    model.train()
    data = {"obs_traj": torch.cumsum(torch.randn(16, 4, 3) * 0.03, dim=0)}
    target = torch.cumsum(torch.randn(4, 24, 3) * 0.03, dim=1)
    predictions, probabilities, decision, auxiliary = model(data)
    loss, diagnostics = energy_predict_optimize_objective(
        predictions,
        probabilities,
        auxiliary["predicted_normalized_ade_risk"],
        target,
    )
    loss.backward()
    operator_gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.energy_cost_operator.parameters()
        if parameter.grad is not None
    )
    backbone_gradients = sum(
        int(parameter.grad is not None) for parameter in model.backbone.parameters()
    )
    decoder = auxiliary["kinematic_decoder"]
    checks = {
        **source_checks,
        "variant_identity": VARIANT == design["variant"],
        "control_samples_frozen": list(CONTROL_SAMPLE_INDICES)
        == design["algorithm"]["control_sample_indices"],
        "prediction_shape": list(predictions.shape) == [4, 5, 24, 3],
        "probability_shape": list(probabilities.shape) == [4, 5],
        "decision_shape": list(decision.shape) == [4],
        "probability_simplex": bool((probabilities >= -1e-6).all())
        and torch.allclose(probabilities.sum(dim=1), torch.ones(4), atol=1e-5),
        "decision_replays_backbone": torch.equal(
            decision, auxiliary["decision_logits"].argmax(dim=1)
        ),
        "finite_loss": bool(torch.isfinite(loss)),
        "operator_receives_gradient": operator_gradient > 0,
        "backbone_has_no_gradients": backbone_gradients == 0
        and not any(parameter.requires_grad for parameter in model.backbone.parameters()),
        "target_risk_detached": not diagnostics["target_normalized_ade_risk"].requires_grad,
        "positive_speed": bool((auxiliary["horizontal_control"] >= 0).all()),
        "forbidden_mechanisms_absent": not bool(decoder["trajectory_residual"])
        and not bool(decoder.get("control_residual", False))
        and not bool(decoder["learned_gate"])
        and not bool(auxiliary.get("post_generation_selector", False)),
    }
    result = {
        "format_version": 1,
        "cycle": protocol.payload["cycle"],
        "phase": "P1_E1_preflight",
        "variant": VARIANT,
        "protocol_sha256": protocol_hash,
        "p0_b_decision_sha256": sha256(p0_b_path),
        "p1_design_sha256": sha256(DESIGN_PATH),
        "gradient_l1": {
            "energy_cost_operator": operator_gradient,
            "backbone_parameters_with_grad": backbone_gradients,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "locked_test_used": False,
        "development_used": False,
    }
    if not result["passed"]:
        raise RuntimeError(f"C134 E1 preflight failed: {checks}")
    output = ARTIFACT_ROOT / "preflight_p1.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
