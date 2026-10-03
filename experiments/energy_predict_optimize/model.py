"""C134 Energy-consistent predict-and-optimize trajectory measure."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from experiments.dual_expected_risk.objective import ADE_SCALE
from experiments.decision_regret.model import build_model as build_c133_model

from .solver import energy_optimal_probabilities


VARIANT = "E1_energy_predict_optimize"
CONTROL_SAMPLE_INDICES = (3, 7, 11, 15, 19, 23)


class EnergyPredictOptimizeAscent(nn.Module):
    """Frozen physical decision backbone plus a learned Energy cost operator."""

    def __init__(self, *, batch_size: int = 1024) -> None:
        super().__init__()
        self.backbone = build_c133_model(batch_size=batch_size)
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
        if flight_parameters.ndim != 4 or flight_parameters.shape[-1] != 3:
            raise ValueError("flight_parameters must have shape [B,K,T,3]")
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
        pairwise_distance = torch.linalg.vector_norm(
            predictions[:, :, None] - predictions[:, None, :], dim=-1
        ).mean(dim=-1)
        probabilities = energy_optimal_probabilities(
            centered_risk * ADE_SCALE, pairwise_distance
        )
        decision_mode = decision_logits.argmax(dim=1)
        return predictions, probabilities, decision_mode, {
            **auxiliary,
            "decision_logits": decision_logits,
            "predicted_normalized_ade_risk": raw_risk,
            "centered_predicted_normalized_ade_risk": centered_risk,
            "energy_probabilities": probabilities,
            "energy_predict_optimize": {
                "probability_solver": "fixed_32_step_frank_wolfe",
                "top1_decision": "frozen_C133_SPO_plus",
                "decision_probability_decoupled": True,
                "trajectory_or_control_residual": False,
                "learned_gate": False,
                "temperature": False,
                "post_generation_selector": False,
            },
        }


def build_model(*, batch_size: int = 1024) -> EnergyPredictOptimizeAscent:
    return EnergyPredictOptimizeAscent(batch_size=batch_size)


__all__ = [
    "CONTROL_SAMPLE_INDICES",
    "EnergyPredictOptimizeAscent",
    "VARIANT",
    "build_model",
]
