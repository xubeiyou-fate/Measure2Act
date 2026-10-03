"""Native shared-feature ASCENT model for C129."""

from __future__ import annotations

import torch

from experiments.metric_exact.model import ascent_config as c127_ascent_config
from model.ascent import Ascent


VARIANT = "J1_joint_coupled_dual"


def ascent_config(*, batch_size: int = 256) -> dict[str, object]:
    config = dict(c127_ascent_config("B6_dual_oracle", batch_size=batch_size))
    config.update(
        {
            "cycle": "C129_JOINT_COUPLED_DUAL",
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
        raise RuntimeError("C129 requires native K=5")
    return config


class JointCoupledAscent(Ascent):
    """B6 geometry objective paired with ASCENT's native coupled score head."""

    def __init__(self, *, batch_size: int = 256) -> None:
        super().__init__(ascent_config(batch_size=batch_size))

    def forward(
        self, data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        predictions, logits, auxiliary = super().forward(data)
        decoder = {
            **auxiliary.get("kinematic_decoder", {}),
            "control_residual": False,
            "token_codebook": False,
        }
        return predictions, logits, {
            **auxiliary,
            "kinematic_decoder": decoder,
            "joint_coupling": {
                "native_shared_score_head": True,
                "score_gradient_reaches_shared_mode_features": True,
                "geometry_objective": "exact_independent_minade_plus_minfde",
                "score_target": "hard_scaled_ade_plus_fde_winner",
                "routes_modes": False,
                "selects_candidates": False,
            },
        }


def build_model(*, batch_size: int = 256) -> JointCoupledAscent:
    return JointCoupledAscent(batch_size=batch_size)


__all__ = ["VARIANT", "JointCoupledAscent", "ascent_config", "build_model"]
