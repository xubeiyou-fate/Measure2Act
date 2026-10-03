"""Full 2x2x2 fusion experiment for the unified MABPT-ASCENT model."""

from __future__ import annotations

from itertools import product

import torch

from .operator import (
    energy_kl_projection,
    exact_gibbs_transport,
    hard_bijection_transport,
)


CORRESPONDENCE_LEVELS = ("hard_bijection", "exact_gibbs")
MASS_LEVELS = ("uniform", "source_predicted_mass")
PROJECTION_LEVELS = ("transported_prior", "energy_kl")


def arm_name(correspondence: str, assignment_mass: str, projection: str) -> str:
    return (
        f"correspondence={correspondence}__"
        f"assignment_mass={assignment_mass}__projection={projection}"
    )


def design_matrix() -> list[dict[str, str]]:
    return [
        {
            "arm": arm_name(correspondence, assignment_mass, projection),
            "correspondence": correspondence,
            "assignment_cost_mass": assignment_mass,
            "projection": projection,
        }
        for correspondence, assignment_mass, projection in product(
            CORRESPONDENCE_LEVELS,
            MASS_LEVELS,
            PROJECTION_LEVELS,
        )
    ]


def factorial_probability_arms(
    source_probabilities: torch.Tensor,
    cross_cost: torch.Tensor,
    predicted_risk: torch.Tensor,
    pairwise_distance: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
    """Evaluate all fusion combinations without using a future target."""
    arms: dict[str, torch.Tensor] = {}
    diagnostics: dict[str, dict[str, torch.Tensor]] = {}
    for correspondence in CORRESPONDENCE_LEVELS:
        for assignment_mass in MASS_LEVELS:
            mass_weighted = assignment_mass == "source_predicted_mass"
            if correspondence == "hard_bijection":
                transport = hard_bijection_transport(
                    source_probabilities,
                    cross_cost,
                    mass_weighted=mass_weighted,
                )
            else:
                transport = exact_gibbs_transport(
                    source_probabilities,
                    cross_cost,
                    mass_weighted=mass_weighted,
                )
            prior = transport["transported"]
            for projection in PROJECTION_LEVELS:
                name = arm_name(correspondence, assignment_mass, projection)
                if projection == "energy_kl":
                    probabilities, projection_diagnostics = energy_kl_projection(
                        prior,
                        predicted_risk,
                        pairwise_distance,
                    )
                    diagnostics[name] = projection_diagnostics
                else:
                    probabilities = prior
                    diagnostics[name] = {}
                arms[name] = probabilities
    return arms, diagnostics


def full_model_arm() -> str:
    return arm_name("exact_gibbs", "source_predicted_mass", "energy_kl")


__all__ = [
    "CORRESPONDENCE_LEVELS",
    "MASS_LEVELS",
    "PROJECTION_LEVELS",
    "arm_name",
    "design_matrix",
    "factorial_probability_arms",
    "full_model_arm",
]
