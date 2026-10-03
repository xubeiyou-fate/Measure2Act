"""Public facade for the TPMO and MABPT finite-measure operators."""

from .operator import (
    energy_kl_objective,
    energy_kl_projection,
    exact_gibbs_transport,
    mass_aware_transported_prior,
    pairwise_trajectory_distance,
    support_cost,
    tpmo_probabilities,
    tpmo_transported_prior,
)

__all__ = [
    "energy_kl_objective",
    "energy_kl_projection",
    "exact_gibbs_transport",
    "mass_aware_transported_prior",
    "pairwise_trajectory_distance",
    "support_cost",
    "tpmo_probabilities",
    "tpmo_transported_prior",
]
