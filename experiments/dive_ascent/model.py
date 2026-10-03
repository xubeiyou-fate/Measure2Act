"""Models for decoupled independent Voronoi-expert ASCENT (C99)."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn

from model.ascent import Ascent


C99_VARIANTS = (
    "a0_shared_signed",
    "a1_shared_positive",
    "a2_shared_decoupled",
    "a3_scaled_decoupled",
    "a4_independent_random",
    "a5_dive",
)

MODE_COUNT = 5


def ascent_config(
    variant: str,
    *,
    batch_size: int = 128,
    k: int | None = None,
) -> dict[str, object]:
    if variant not in C99_VARIANTS:
        raise ValueError(f"unknown C99 variant: {variant}")
    positive = variant != "a0_shared_signed"
    scaled = variant == "a3_scaled_decoupled"
    return {
        "lr": 1e-3,
        "epochs": 20,
        "k": MODE_COUNT if k is None else int(k),
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
        "variant": variant,
        "cycle": "C99_DIVE_ASCENT",
        "embed_dim": 208 if scaled else 128,
        "attn_depth": 5 if scaled else 2,
        "kinematic_decoder_variant": (
            "positive_speed_pitch" if positive else "signed_speed_pitch"
        ),
        "trajectory_residual": False,
        "control_residual": False,
        "learned_gate": False,
        "token_codebook": False,
        "future_autoregression": False,
        "post_generation_selector": False,
    }


class ZeroScore(nn.Module):
    """Parameter-free placeholder for a geometry-only ASCENT branch."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features.new_zeros((*features.shape[:-1], 1))


def _geometry_only_ascent(config: dict[str, object]) -> Ascent:
    model = Ascent(config)
    model.pi = ZeroScore()
    return model


class DetachedVoronoiScorer(nn.Module):
    """Estimate cell masses without sending score gradients to trajectories."""

    def __init__(self, *, batch_size: int = 128, modes: int = MODE_COUNT) -> None:
        super().__init__()
        config = ascent_config(
            "a2_shared_decoupled", batch_size=batch_size, k=1
        )
        self.history_encoder = _geometry_only_ascent(config)
        self.modes = modes
        self.context_dim = int(config["embed_dim"])
        trajectory_dim = int(config["pred_len"]) // int(config["pred_step"]) * 3
        self.trajectory_projection = nn.Sequential(
            nn.LayerNorm(trajectory_dim),
            nn.Linear(trajectory_dim, self.context_dim),
            nn.GELU(),
            nn.Linear(self.context_dim, self.context_dim),
        )
        self.mode_identity = nn.Parameter(torch.empty(modes, self.context_dim))
        self.score_head = nn.Sequential(
            nn.LayerNorm(self.context_dim * 3),
            nn.Linear(self.context_dim * 3, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        nn.init.normal_(self.mode_identity, std=0.02)

    def forward(
        self, data: dict[str, torch.Tensor], trajectories: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | bool | str]]:
        if trajectories.ndim != 4 or trajectories.shape[1] != self.modes:
            raise ValueError("trajectories must have shape [B,K,T,3]")
        _, _, auxiliary = self.history_encoder(data)
        context = auxiliary["actor_context"]
        centers = auxiliary["actor_centers"]
        detached = trajectories.detach()
        relative = detached - centers[:, None, None]
        descriptor = self.trajectory_projection(relative.flatten(start_dim=2))
        context = context[:, None].expand(-1, self.modes, -1)
        identity = self.mode_identity[None].expand(context.shape[0], -1, -1)
        logits = self.score_head(
            torch.cat((context, descriptor, identity), dim=-1)
        ).squeeze(-1)
        return logits, {
            "geometry_detached": True,
            "role": "voronoi_cell_mass_only",
            "routes_experts": False,
            "selects_candidates": False,
        }


class DecoupledAscent(nn.Module):
    """Shared or fully independent geometry with a separate score network."""

    def __init__(self, variant: str, *, batch_size: int = 128) -> None:
        super().__init__()
        if variant not in C99_VARIANTS[2:]:
            raise ValueError("DecoupledAscent requires an A2-A5 variant")
        self.variant = variant
        self.k = MODE_COUNT
        self.independent = variant in {
            "a4_independent_random",
            "a5_dive",
        }
        self.dac = variant == "a5_dive"
        if self.independent:
            self.experts = nn.ModuleList(
                [
                    _geometry_only_ascent(
                        ascent_config(variant, batch_size=batch_size, k=1)
                    )
                    for _ in range(self.k)
                ]
            )
            self.geometry = None
        else:
            self.geometry = _geometry_only_ascent(
                ascent_config(variant, batch_size=batch_size, k=self.k)
            )
            self.experts = nn.ModuleList()
        self.scorer = DetachedVoronoiScorer(batch_size=batch_size, modes=self.k)
        self.register_buffer(
            "active_experts",
            torch.tensor(1 if self.dac else self.k, dtype=torch.long),
            persistent=True,
        )
        if self.dac:
            self.initialize_dac()

    @torch.no_grad()
    def initialize_dac(self) -> None:
        if not self.dac:
            raise RuntimeError("DAC initialization is only defined for a5_dive")
        source = self.experts[0].state_dict()
        for expert in self.experts[1:]:
            expert.load_state_dict(source, strict=True)
        self.active_experts.fill_(1)

    def predict_geometry(
        self, data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, object]]:
        if not self.independent:
            trajectories, _, auxiliary = self.geometry(data)
            return trajectories, {
                "geometry": auxiliary,
                "all_experts_executed": True,
                "independent_experts": False,
            }
        trajectories = []
        auxiliaries = []
        for expert in self.experts:
            prediction, _, auxiliary = expert(data)
            trajectories.append(prediction[:, 0])
            auxiliaries.append(auxiliary)
        return torch.stack(trajectories, dim=1), {
            "geometry": auxiliaries,
            "all_experts_executed": True,
            "independent_experts": True,
        }

    def score(
        self, data: dict[str, torch.Tensor], trajectories: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, object]]:
        return self.scorer(data, trajectories)

    def forward(
        self, data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        trajectories, geometry_aux = self.predict_geometry(data)
        logits, score_aux = self.score(data, trajectories)
        auxiliary = {
            **geometry_aux,
            "score": score_aux,
            "active_training_experts": int(self.active_experts.item()),
            "trajectory_residual": False,
            "learned_gate": False,
            "token_codebook": False,
            "future_autoregression": False,
            "post_generation_selector": False,
        }
        geometry = geometry_aux["geometry"]
        first_geometry = geometry[0] if isinstance(geometry, list) else geometry
        for key in ("horizontal_control", "flight_params"):
            if key in first_geometry:
                if isinstance(geometry, list):
                    auxiliary[key] = torch.stack(
                        [item[key][:, 0] for item in geometry], dim=1
                    )
                else:
                    auxiliary[key] = first_geometry[key]
        return trajectories, logits, auxiliary

    def geometry_parameters(self) -> Iterable[nn.Parameter]:
        module: nn.Module = self.experts if self.independent else self.geometry
        return module.parameters()

    def scorer_parameters(self) -> Iterable[nn.Parameter]:
        return self.scorer.parameters()

    @torch.no_grad()
    def split_highest_distortion(
        self,
        distortions: torch.Tensor,
        *,
        perturbation_scale: float,
        seed: int,
    ) -> dict[str, int | float]:
        if not self.dac:
            raise RuntimeError("expert splitting is only defined for a5_dive")
        active = int(self.active_experts.item())
        if active >= self.k:
            raise RuntimeError("all DIVE experts are already active")
        if distortions.shape != (active,) or not torch.isfinite(distortions).all():
            raise ValueError("distortions must be finite and match active experts")
        parent_index = int(distortions.argmax().item())
        child_index = active
        parent = self.experts[parent_index]
        child = self.experts[child_index]
        child.load_state_dict(parent.state_dict(), strict=True)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        child_parameters = dict(child.named_parameters())
        perturbation_norm_sq = 0.0
        for name, parent_parameter in parent.named_parameters():
            if not parent_parameter.requires_grad:
                continue
            child_parameter = child_parameters[name]
            fan = max(parent_parameter.numel(), 1)
            base_scale = float(parent_parameter.detach().float().std(unbiased=False))
            if not math.isfinite(base_scale) or base_scale < 1e-6:
                base_scale = 1.0 / math.sqrt(fan)
            noise = torch.randn(
                parent_parameter.shape,
                generator=generator,
                device="cpu",
                dtype=torch.float32,
            ).to(device=parent_parameter.device, dtype=parent_parameter.dtype)
            noise.mul_(perturbation_scale * base_scale)
            parent_parameter.sub_(noise)
            child_parameter.add_(noise)
            perturbation_norm_sq += float(noise.float().square().sum().cpu())
        self.active_experts.add_(1)
        return {
            "parent": parent_index,
            "child": child_index,
            "active_experts": int(self.active_experts.item()),
            "perturbation_l2": math.sqrt(perturbation_norm_sq),
        }


def build_model(variant: str, *, batch_size: int = 128) -> nn.Module:
    if variant not in C99_VARIANTS:
        raise ValueError(f"unknown C99 variant: {variant}")
    if variant in C99_VARIANTS[:2]:
        return Ascent(ascent_config(variant, batch_size=batch_size))
    return DecoupledAscent(variant, batch_size=batch_size)


__all__ = [
    "C99_VARIANTS",
    "DecoupledAscent",
    "DetachedVoronoiScorer",
    "MODE_COUNT",
    "ascent_config",
    "build_model",
]
