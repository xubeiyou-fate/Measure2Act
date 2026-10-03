from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from mabpt.evaluate_official_baseline import MetricState
from mabpt.train_official_baseline import _load_resume


def _resume_checkpoint(path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters())
    torch.save(
        {
            "epoch": 6,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "updates": 10,
            "history": [],
            "metadata": {
                "family": "trajairnet",
                "dataset": "7days1",
                "seed": 42,
                "protocol_sha256": "protocol",
            },
            "rng_state": {
                "python": __import__("random").getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": [torch.tensor([1], dtype=torch.uint8), torch.tensor([2], dtype=torch.uint8)],
            },
        },
        path,
    )


def test_resume_maps_one_checkpoint_cuda_rng_to_current_device(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint = tmp_path / "resume.pt"
    _resume_checkpoint(checkpoint)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters())
    selected = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda _states: pytest.fail("must not restore all devices"))
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state, device: selected.append((state, device)))
    start_epoch, updates, _history = _load_resume(
        checkpoint,
        model=model,
        optimizer=optimizer,
        expected={
            "family": "trajairnet",
            "dataset": "7days1",
            "seed": 42,
            "protocol_sha256": "protocol",
        },
        target_device=torch.device("cuda:0"),
        rng_source_device=1,
    )
    assert (start_epoch, updates) == (7, 10)
    assert len(selected) == 1
    assert selected[0][0].item() == 2
    assert selected[0][1] == torch.device("cuda:0")


def test_resume_requires_mapping_when_cuda_device_counts_differ(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint = tmp_path / "resume.pt"
    _resume_checkpoint(checkpoint)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="device counts differ"):
        _load_resume(
            checkpoint,
            model=model,
            optimizer=optimizer,
            expected={
                "family": "trajairnet",
                "dataset": "7days1",
                "seed": 42,
                "protocol_sha256": "protocol",
            },
            target_device=torch.device("cuda:0"),
            rng_source_device=None,
        )


def test_official_metric_state_preserves_measure_and_oracle_semantics() -> None:
    truth = torch.zeros((1, 2, 3), dtype=torch.float64)
    prediction = torch.zeros((1, 2, 2, 3), dtype=torch.float64)
    prediction[:, 1, :, 0] = 2.0
    state = MetricState()
    state.update(prediction, truth)
    result = state.summarize()
    assert result["sample1_ade"] == 0.0
    assert result["expected_ade"] == 1.0
    assert result["minade"] == 0.0
    # E[d(X,y)] - .5 E[d(X,X')] = 1 - .5 * 1.
    assert result["energy"] == 0.5
    assert result["scene_oracle_ade"] == 0.0
    assert result["scene_oracle_fde"] == 0.0


def test_scene_oracle_selects_one_shared_sample_for_all_actors() -> None:
    truth = torch.zeros((2, 1, 3), dtype=torch.float64)
    prediction = torch.zeros((2, 2, 1, 3), dtype=torch.float64)
    prediction[0, 0, 0, 0] = 0.0
    prediction[1, 0, 0, 0] = 4.0
    prediction[0, 1, 0, 0] = 3.0
    prediction[1, 1, 0, 0] = 0.0
    state = MetricState()
    state.update(prediction, truth)
    result = state.summarize()
    assert result["minade"] == 0.0
    assert result["scene_oracle_ade"] == 1.5
    assert result["scene_oracle_fde"] == 1.5


def test_independent_scene_oracle_does_not_mix_actrajnet_scenes() -> None:
    truth = torch.zeros((2, 1, 3), dtype=torch.float64)
    prediction = torch.zeros((2, 2, 1, 3), dtype=torch.float64)
    prediction[0, 0, 0, 0] = 0.0
    prediction[0, 1, 0, 0] = 3.0
    prediction[1, 0, 0, 0] = 4.0
    prediction[1, 1, 0, 0] = 0.0
    state = MetricState()
    state.update(prediction, truth, independent_scenes=True)
    result = state.summarize()
    assert result["scenes"] == 2
    assert result["scene_oracle_ade"] == 0.0
    assert result["scene_oracle_fde"] == 0.0
