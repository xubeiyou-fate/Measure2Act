from __future__ import annotations

import torch

from mabpt.events import event_labels, event_membership


def test_nine_event_labels_are_model_independent() -> None:
    observations = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=torch.float64
    )
    trajectories = torch.tensor(
        [[
            [[1.0, -2.0, -1.0]],
            [[3.0, 0.0, 0.0]],
            [[1.0, 2.0, 1.0]],
        ]],
        dtype=torch.float64,
    )
    labels = event_labels(
        observations,
        trajectories,
        turn_threshold=0.1,
        altitude_threshold=0.1,
    )
    assert labels.tolist() == [[0, 4, 8]]


def test_event_measurement_kernel_is_positive_and_normalized() -> None:
    observations = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=torch.float64
    )
    trajectories = torch.tensor(
        [[[[1.0, -2.0, -1.0]], [[3.0, 0.0, 0.0]], [[1.0, 2.0, 1.0]]]],
        dtype=torch.float64,
    )
    membership = event_membership(
        observations,
        trajectories,
        turn_threshold=0.5,
        altitude_threshold=0.25,
    )
    assert torch.all(membership > 0)
    assert torch.allclose(
        membership.sum(dim=-1), torch.ones(1, 3, dtype=torch.float64), atol=1e-12
    )
