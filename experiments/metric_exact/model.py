"""C127 model variants without residuals, gates, or candidate selection."""

from __future__ import annotations

import torch
from torch import nn

from experiments.dive_ascent.model import ZeroScore
from model.ascent import Ascent


VARIANTS = (
    "B0_signed_coupled",
    "B1_positive_coupled",
    "B2_decoupled_original",
    "B3_exact_minade",
    "B4_exact_minfde",
    "B5_single_combined",
    "B6_dual_oracle",
    "B7_signed_dual",
)

COUPLED_VARIANTS = {"B0_signed_coupled", "B1_positive_coupled"}
ISOLATED_VARIANTS = set(VARIANTS) - COUPLED_VARIANTS


def ascent_config(variant: str, *, batch_size: int = 256) -> dict[str, object]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown C127 variant: {variant}")
    signed = variant in {"B0_signed_coupled", "B7_signed_dual"}
    return {
        "lr": 1e-3,
        "epochs": 20,
        "k": 5,
        "obs_len": 16,
        "obs_steps": 1,
        "pred_len": 120,
        "pred_step": 5,
        "use_runway": False,
        "use_social": False,
        "use_weather": False,
        "ground_vs_airbourne": False,
        "split_xy_z": True,
        "normalize_coords": True,
        "global_pos_embedding": True,
        "mamba": False,
        "dataset_name": "trajair_111day_c12",
        "batch_size": batch_size,
        "loss_local": False,
        "flight_param_loss": False,
        "decoder": "new",
        "kinematic_decoder_variant": (
            "signed_speed_pitch" if signed else "positive_speed_pitch"
        ),
        "mode_head_variant": "shared",
        "history_encoder_variant": "max_pool",
        "history_capacity_variant": None,
        "trajectory_residual": False,
        "control_residual": False,
        "learned_gate": False,
        "token_codebook": False,
        "future_autoregression": False,
        "post_generation_selector": False,
        "cycle": "C127_metric_exact_score_isolated_ascent",
        "variant": variant,
    }


class ScoreIsolatedAscent(nn.Module):
    """ASCENT geometry with parameter-free logits during geometry screening."""

    def __init__(self, variant: str, *, batch_size: int = 256) -> None:
        super().__init__()
        if variant not in ISOLATED_VARIANTS:
            raise ValueError("ScoreIsolatedAscent requires an isolated C127 variant")
        self.variant = variant
        self.geometry = Ascent(ascent_config(variant, batch_size=batch_size))
        self.geometry.pi = ZeroScore()

    def forward(
        self, data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        predictions, logits, auxiliary = self.geometry(data)
        return predictions, logits, {
            **auxiliary,
            "score_isolation": {
                "geometry_stage_logits": "fixed_zero",
                "score_gradient_reaches_geometry": False,
                "routes_modes": False,
                "selects_candidates": False,
            },
        }


def build_model(variant: str, *, batch_size: int = 256) -> nn.Module:
    if variant not in VARIANTS:
        raise ValueError(f"unknown C127 variant: {variant}")
    if variant in COUPLED_VARIANTS:
        return Ascent(ascent_config(variant, batch_size=batch_size))
    return ScoreIsolatedAscent(variant, batch_size=batch_size)


def is_score_isolated(variant: str) -> bool:
    if variant not in VARIANTS:
        raise ValueError(f"unknown C127 variant: {variant}")
    return variant in ISOLATED_VARIANTS


__all__ = [
    "COUPLED_VARIANTS",
    "ISOLATED_VARIANTS",
    "ScoreIsolatedAscent",
    "VARIANTS",
    "ascent_config",
    "build_model",
    "is_score_isolated",
]
