"""Native-cardinality MABPT components for the trained E9 experiment."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from experiments.metric_exact.model import ascent_config as c127_ascent_config
from experiments.dual_expected_risk.objective import ADE_SCALE, FDE_SCALE
from model.ascent import Ascent


CONTROL_SAMPLE_INDICES = (3, 7, 11, 15, 19, 23)


def scalable_ascent_config(
    modes: int, *, role: str, batch_size: int
) -> dict[str, object]:
    if modes not in (3, 7):
        raise ValueError("trained scalable models are frozen to K=3 or K=7")
    if role not in ("source", "decision"):
        raise ValueError("role must be source or decision")
    variant = "B0_signed_coupled" if role == "source" else "B6_dual_oracle"
    config = dict(c127_ascent_config(variant, batch_size=batch_size))
    config.update(
        {
            "k": modes,
            "cycle": "MABPT_E9_TRAINED_CARDINALITY",
            "variant": f"K{modes}_{role}",
            "mode_head_variant": "shared",
            "kinematic_decoder_variant": (
                "signed_speed_pitch" if role == "source" else "positive_speed_pitch"
            ),
            "trajectory_residual": False,
            "control_residual": False,
            "learned_gate": False,
            "token_codebook": False,
            "future_autoregression": False,
            "post_generation_selector": False,
        }
    )
    return config


class ScalableDecisionAscent(Ascent):
    def __init__(self, modes: int, *, batch_size: int = 256) -> None:
        super().__init__(
            scalable_ascent_config(modes, role="decision", batch_size=batch_size)
        )
        self.modes = modes

    def forward(
        self, data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        predictions, raw_costs, auxiliary = super().forward(data)
        if raw_costs.shape != predictions.shape[:2]:
            raise RuntimeError("decision costs must have shape [B,K]")
        centered = raw_costs - raw_costs.mean(dim=1, keepdim=True)
        return predictions, -centered, {
            **auxiliary,
            "decision_costs": raw_costs,
            "centered_decision_costs": centered,
            "native_cardinality": self.modes,
            "trajectory_residual": False,
            "learned_gate": False,
        }


def generalized_energy_probabilities(
    predicted_target_distance: torch.Tensor,
    pairwise_distance: torch.Tensor,
    *,
    steps: int = 32,
) -> torch.Tensor:
    if predicted_target_distance.ndim != 2:
        raise ValueError("predicted distance must have shape [B,K]")
    batch, modes = predicted_target_distance.shape
    if modes < 2 or pairwise_distance.shape != (batch, modes, modes):
        raise ValueError("pairwise distance must have shape [B,K,K]")
    if steps < 1:
        raise ValueError("steps must be positive")
    probabilities = torch.full_like(predicted_target_distance, 1.0 / modes)
    for _ in range(steps):
        gradient = predicted_target_distance - torch.einsum(
            "bij,bj->bi", pairwise_distance, probabilities
        )
        vertex = F.one_hot(gradient.argmin(dim=1), num_classes=modes).to(
            predicted_target_distance.dtype
        )
        direction = vertex - probabilities
        directional_derivative = (direction * gradient).sum(dim=1)
        curvature = -torch.einsum(
            "bi,bij,bj->b", direction, pairwise_distance, direction
        )
        step = torch.where(
            curvature > 1e-12,
            -directional_derivative / curvature.clamp_min(1e-12),
            (directional_derivative < 0).to(predicted_target_distance.dtype),
        ).clamp(0.0, 1.0)
        probabilities = probabilities + step[:, None] * direction
    if bool((probabilities < -1e-6).any()) or not torch.allclose(
        probabilities.sum(dim=1), torch.ones_like(probabilities[:, 0]), atol=1e-5, rtol=0
    ):
        raise RuntimeError("generalized Energy solver left the simplex")
    return probabilities


class ScalableEnergyAscent(nn.Module):
    def __init__(self, modes: int, *, batch_size: int = 1024) -> None:
        super().__init__()
        self.modes = modes
        self.backbone = ScalableDecisionAscent(modes, batch_size=batch_size)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.energy_cost_operator = nn.Sequential(
            nn.Linear(128 + len(CONTROL_SAMPLE_INDICES) * 5, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def load_backbone(self, checkpoint_path: Path, device: torch.device) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.backbone.load_state_dict(checkpoint["model_state_dict"])
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    @staticmethod
    def control_features(flight_parameters: torch.Tensor) -> torch.Tensor:
        sampled = flight_parameters[:, :, CONTROL_SAMPLE_INDICES]
        speed, heading, pitch = sampled.unbind(dim=-1)
        return torch.stack(
            (
                speed,
                torch.cos(heading),
                torch.sin(heading),
                torch.cos(pitch),
                torch.sin(pitch),
            ),
            dim=-1,
        ).flatten(start_dim=2)

    def forward(
        self, data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
        with torch.no_grad():
            predictions, decision_logits, auxiliary = self.backbone(data)
        controls = self.control_features(auxiliary["flight_params"])
        operator_input = torch.cat((auxiliary["mode_features"], controls), dim=-1)
        raw_risk = self.energy_cost_operator(operator_input).squeeze(-1)
        centered_risk = raw_risk - raw_risk.mean(dim=1, keepdim=True)
        pairwise = torch.linalg.vector_norm(
            predictions[:, :, None] - predictions[:, None, :], dim=-1
        ).mean(dim=-1)
        probabilities = generalized_energy_probabilities(
            centered_risk * ADE_SCALE, pairwise
        )
        return predictions, probabilities, decision_logits.argmax(dim=1), {
            **auxiliary,
            "decision_logits": decision_logits,
            "predicted_normalized_ade_risk": raw_risk,
            "centered_predicted_normalized_ade_risk": centered_risk,
            "native_cardinality": self.modes,
            "trajectory_residual": False,
            "learned_gate": False,
        }


def scalable_decision_objective(
    predictions: torch.Tensor,
    logits: torch.Tensor,
    decision_costs: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    displacement = torch.linalg.vector_norm(predictions - target[:, None], dim=-1)
    ade = displacement.mean(dim=-1)
    fde = displacement[..., -1]
    true_costs = (ade / ADE_SCALE + fde / FDE_SCALE).detach()
    centered = decision_costs - decision_costs.mean(dim=1, keepdim=True)
    if not torch.allclose(logits, -centered, atol=1e-6, rtol=1e-6):
        raise ValueError("logits and centered decision costs disagree")
    true_winner = true_costs.argmin(dim=1)
    rows = torch.arange(predictions.shape[0], device=predictions.device)
    surrogate = (
        (true_costs - 2.0 * centered).max(dim=1).values
        + 2.0 * centered[rows, true_winner]
        - true_costs[rows, true_winner]
    ).mean()
    geometry = (
        ade.min(dim=1).values / ADE_SCALE
        + fde.min(dim=1).values / FDE_SCALE
    ).mean()
    loss = geometry + surrogate
    return loss, {
        "loss": loss.detach(),
        "geometry": geometry.detach(),
        "decision_regret_surrogate": surrogate.detach(),
        "minade": ade.min(dim=1).values.mean().detach(),
        "minfde": fde.min(dim=1).values.mean().detach(),
    }


def scalable_energy_objective(
    predictions: torch.Tensor,
    probabilities: torch.Tensor,
    predicted_risk: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    displacement = torch.linalg.vector_norm(predictions - target[:, None], dim=-1)
    ade = displacement.mean(dim=-1)
    normalized_target = (ade / ADE_SCALE).detach()
    centered_target = normalized_target - normalized_target.mean(dim=1, keepdim=True)
    centered_prediction = predicted_risk - predicted_risk.mean(dim=1, keepdim=True)
    risk_regression = F.mse_loss(centered_prediction, centered_target)
    pairwise = torch.linalg.vector_norm(
        predictions[:, :, None] - predictions[:, None, :], dim=-1
    ).mean(dim=-1)
    energy = (probabilities * ade).sum(dim=1) - 0.5 * torch.einsum(
        "bi,bij,bj->b", probabilities, pairwise, probabilities
    )
    normalized_energy = energy.mean() / ADE_SCALE
    loss = risk_regression + normalized_energy
    return loss, {
        "loss": loss.detach(),
        "risk_regression": risk_regression.detach(),
        "normalized_energy": normalized_energy.detach(),
        "energy_score": energy.mean().detach(),
    }


__all__ = [
    "ScalableDecisionAscent",
    "ScalableEnergyAscent",
    "generalized_energy_probabilities",
    "scalable_ascent_config",
    "scalable_decision_objective",
    "scalable_energy_objective",
]
