from __future__ import annotations

import torch

from mabpt.evaluate_external import _constant_velocity


def test_constant_velocity_uses_five_second_prediction_grid() -> None:
    observation = torch.zeros(16, 2, 3)
    observation[-1, :, 0] = 4.0
    observation[-2, :, 0] = 3.0
    support = _constant_velocity({"obs_traj": observation})
    assert support.shape == (2, 1, 24, 3)
    assert torch.equal(support[:, 0, 0, 0], torch.full((2,), 9.0))
    assert torch.equal(support[:, 0, -1, 0], torch.full((2,), 124.0))


def test_constant_velocity_supports_matched_official_ten_second_grid() -> None:
    observation = torch.zeros(11, 2, 3)
    observation[-1, :, 0] = 4.0
    observation[-2, :, 0] = 3.0
    support = _constant_velocity(
        {"obs_traj": observation}, prediction_stride_seconds=10
    )
    assert support.shape == (2, 1, 12, 3)
    assert torch.equal(support[:, 0, 0, 0], torch.full((2,), 14.0))
    assert torch.equal(support[:, 0, -1, 0], torch.full((2,), 124.0))
