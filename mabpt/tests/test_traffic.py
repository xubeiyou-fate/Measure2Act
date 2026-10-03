from __future__ import annotations

import torch

from mabpt.traffic import pair_conflict_probabilities


def test_pair_conflict_probability_and_truth() -> None:
    predictions = torch.tensor(
        [
            [[[0.0, 0.0, 0.0]], [[10.0, 0.0, 0.0]]],
            [[[0.2, 0.0, 0.0]], [[20.0, 0.0, 0.0]]],
        ],
        dtype=torch.float64,
    )
    probabilities = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    target = torch.tensor([[[0.0, 0.0, 0.0]], [[0.2, 0.0, 0.0]]], dtype=torch.float64)
    result = pair_conflict_probabilities(
        predictions,
        probabilities,
        target,
        torch.tensor([0, 0]),
        horizontal_threshold=1.0,
        vertical_threshold=0.3,
        horizontal_scale=0.1,
        vertical_scale=0.03,
    )
    assert result["label"].tolist() == [True]
    assert float(result["probability"][0]) > 0.5
    assert torch.allclose(result["hard_probability"], torch.ones(1, dtype=torch.float64))
