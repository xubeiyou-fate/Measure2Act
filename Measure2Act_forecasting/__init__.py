"""PatchTST-style public entrypoint for the aircraft forecasting backbone."""

from model.ascent import Ascent
from model.cv import ConstantVelocityModel

__all__ = ["Ascent", "ConstantVelocityModel"]
