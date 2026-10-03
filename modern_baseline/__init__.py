"""Isolated adapters for modern trajectory-prediction baselines."""

from .eqmotion_aviation import AviationSceneDataset, EqMotionAviation, aviation_collate

__all__ = ["AviationSceneDataset", "EqMotionAviation", "aviation_collate"]
