"""Submission-oriented experiments that leave frozen baselines unchanged."""

from .awta import annealed_wta_objective, geometric_temperature

__all__ = ["annealed_wta_objective", "geometric_temperature"]
