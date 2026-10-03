from __future__ import annotations

import torch

from mabpt.robustness import perturb


def _data():
    observations = torch.arange(16 * 3 * 3, dtype=torch.float32).reshape(16, 3, 3)
    return {
        "obs_traj": observations,
        "obs_traj_rel": torch.zeros_like(observations),
        "pred_traj": torch.zeros(24, 3, 3),
    }


def test_history_truncation_preserves_recent_observations() -> None:
    data = _data()
    result = perturb(
        data,
        "history_4",
        {"kind": "history_truncation", "retained_steps": 4},
        generator=torch.Generator().manual_seed(1),
    )
    assert torch.equal(result["obs_traj"][-4:], data["obs_traj"][-4:])
    assert torch.equal(
        result["obs_traj"][:-4], data["obs_traj"][-4].expand(12, -1, -1)
    )
    assert torch.equal(data["obs_traj"], _data()["obs_traj"])


def test_dropout_preserves_current_observation() -> None:
    data = _data()
    result = perturb(
        data,
        "dropout_50",
        {"kind": "ADS_B_dropout", "rate": 0.5},
        generator=torch.Generator().manual_seed(2),
    )
    assert torch.equal(result["obs_traj"][-1], data["obs_traj"][-1])


def test_position_noise_is_deterministic() -> None:
    specification = {
        "kind": "position_noise",
        "horizontal_sigma_km": 0.03,
        "vertical_sigma_km": 0.0075,
    }
    first = perturb(
        _data(),
        "noise_30m",
        specification,
        generator=torch.Generator().manual_seed(3),
    )
    second = perturb(
        _data(),
        "noise_30m",
        specification,
        generator=torch.Generator().manual_seed(3),
    )
    assert torch.equal(first["obs_traj"], second["obs_traj"])
