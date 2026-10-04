"""Stable public imports for the paper's probability-transfer operators.

The implementation remains in provenance-preserving compatibility modules.
This facade is the recommended import surface for new code.
"""

from experiments.tpmo_ascent.operator import (
    tpmo_probabilities,
    transported_prior as tpmo_transported_prior,
)
from experiments.mabpt_ascent.operator import mass_aware_transported_prior
from mabpt.operator import (
    energy_kl_objective,
    energy_kl_projection,
    exact_gibbs_transport,
    pairwise_trajectory_distance,
    support_cost,
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
