"""Native shared-feature ASCENT model with a decision-cost head."""

from __future__ import annotations

import torch

from experiments.metric_exact.model import ascent_config as c127_ascent_config
from model.ascent import Ascent


VARIANT = "D1_native_decision_regret"


def ascent_config(*, batch_size: int = 256) -> dict[str, object]:
    config = dict(c127_ascent_config("B6_dual_oracle", batch_size=batch_size))
    config.update(
        {
            "cycle": "C133_NATIVE_DECISION_REGRET",
            "variant": VARIANT,
            "mode_head_variant": "shared",
            "kinematic_decoder_variant": "positive_speed_pitch",
            "trajectory_residual": False,
            "control_residual": False,
            "learned_gate": False,
            "token_codebook": False,
            "future_autoregression": False,
            "post_generation_selector": False,
        }
    )
    if int(config["k"]) != 5:
        raise RuntimeError("C133 requires native K=5")
    return config


class DecisionRegretAscent(Ascent):
    """C129 positive-speed/pitch geometry with a scalar cost map per mode."""

    def __init__(self, *, batch_size: int = 256) -> None:
        super().__init__(ascent_config(batch_size=batch_size))

    def forward(
        self, data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        predictions, raw_costs, auxiliary = super().forward(data)
        if raw_costs.shape != predictions.shape[:2]:
            raise RuntimeError("C133 decision head must return [B,K] costs")
        centered_costs = raw_costs - raw_costs.mean(dim=1, keepdim=True)
        logits = -centered_costs
        decoder = {
            **auxiliary.get("kinematic_decoder", {}),
            "control_residual": False,
            "token_codebook": False,
        }
        return predictions, logits, {
            **auxiliary,
            "kinematic_decoder": decoder,
            "decision_costs": raw_costs,
            "centered_decision_costs": centered_costs,
            "decision_regret": {
                "candidate_scope": "all_five_native_modes",
                "deployment": "argmin_predicted_cost",
                "training_surrogate": "SPO_plus",
                "cost_shift_invariant": True,
                "risk_mse": False,
                "winner_cross_entropy": False,
                "routes_modes": False,
                "selects_candidates": False,
            },
        }


def build_model(*, batch_size: int = 256) -> DecisionRegretAscent:
    return DecisionRegretAscent(batch_size=batch_size)


__all__ = ["VARIANT", "DecisionRegretAscent", "ascent_config", "build_model"]
