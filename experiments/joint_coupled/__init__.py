"""C129 joint-coupled dual-metric ASCENT experiment."""

from .model import VARIANT, JointCoupledAscent, build_model
from .objective import joint_coupled_dual_objective, per_mode_errors

__all__ = [
    "VARIANT",
    "JointCoupledAscent",
    "build_model",
    "joint_coupled_dual_objective",
    "per_mode_errors",
]
