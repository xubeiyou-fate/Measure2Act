"""ASCENT with a deployable full-candidate dual expected-risk head."""

from __future__ import annotations

import torch
from torch import nn

from experiments.metric_exact.model import ascent_config as c127_ascent_config
from model.ascent import Ascent


VARIANT = "R1_dual_expected_risk"


def ascent_config(*, batch_size: int = 256) -> dict[str, object]:
    config = dict(c127_ascent_config("B6_dual_oracle", batch_size=batch_size))
    config.update(
        {
            "cycle": "C130_DUAL_EXPECTED_RISK",
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
        raise RuntimeError("C130 requires native K=5")
    return config


class DualExpectedRiskAscent(Ascent):
    """C129 geometry with a two-output risk head on every native mode."""

    def __init__(self, *, batch_size: int = 256) -> None:
        super().__init__(ascent_config(batch_size=batch_size))
        # Replace the scalar categorical head; no unused scorer remains.
        self.pi = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 2),
        )

    def forward(
        self, data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        predictions, raw_risks, auxiliary = super().forward(data)
        if raw_risks.shape != (*predictions.shape[:2], 2):
            raise RuntimeError("C130 risk head must return [B,K,2]")
        centered_risks = raw_risks - raw_risks.mean(dim=1, keepdim=True)
        logits = -centered_risks.sum(dim=-1)
        decoder = {
            **auxiliary.get("kinematic_decoder", {}),
            "control_residual": False,
            "token_codebook": False,
        }
        return predictions, logits, {
            **auxiliary,
            "kinematic_decoder": decoder,
            "risk_predictions": raw_risks,
            "centered_risk_predictions": centered_risks,
            "dual_expected_risk": {
                "outputs_per_mode": ["normalized_ade", "normalized_fde"],
                "candidate_scope": "all_five_native_modes",
                "centering": "per_actor_across_modes",
                "deployment_logit": "negative_sum_of_centered_dual_risk",
                "risk_gradient_reaches_shared_mode_features": True,
                "routes_modes": False,
                "selects_candidates": False,
            },
        }


def build_model(*, batch_size: int = 256) -> DualExpectedRiskAscent:
    return DualExpectedRiskAscent(batch_size=batch_size)


__all__ = ["VARIANT", "DualExpectedRiskAscent", "ascent_config", "build_model"]
