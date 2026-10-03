"""Tail-aware future-pattern utilities for C7 ASCENT experiments."""

from .features import (
    FUTURE_SIGNATURE_NAMES,
    HISTORY_DESCRIPTOR_DIMENSION,
    constant_velocity_fde,
    future_behavior_signature,
    history_descriptor,
    localize_batch,
)

__all__ = [
    "FUTURE_SIGNATURE_NAMES",
    "HISTORY_DESCRIPTOR_DIMENSION",
    "constant_velocity_fde",
    "future_behavior_signature",
    "history_descriptor",
    "localize_batch",
]
