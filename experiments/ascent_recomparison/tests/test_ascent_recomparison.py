from __future__ import annotations

import json

import numpy as np

from experiments.ascent_recomparison.evaluate_pair import paired_date_bootstrap
from experiments.ascent_recomparison.protocol import load_protocol


def test_protocol_is_train_only_and_forbids_residuals_and_gates() -> None:
    protocol = load_protocol()
    protocol.assert_boundaries()
    assert protocol.payload["dataset"]["allowed_split"] == "train_only"
    forbidden = set(protocol.payload["forbidden_mechanisms"])
    assert "trajectory_or_control_residual" in forbidden
    assert "learned_gate_router_or_mixture_of_experts" in forbidden
    assert protocol.payload["replication"]["folds"] == [1, 2]


def test_protocol_is_valid_json() -> None:
    protocol = load_protocol()
    assert json.loads(protocol.path.read_text(encoding="utf-8"))["format_version"] == 1


def test_paired_date_bootstrap_detects_known_gain() -> None:
    control = {
        "a": {"actors": 10, "top1_ade": 1.0},
        "b": {"actors": 20, "top1_ade": 2.0},
        "c": {"actors": 30, "top1_ade": 3.0},
    }
    candidate = {
        date: {"actors": values["actors"], "top1_ade": values["top1_ade"] - 0.2}
        for date, values in control.items()
    }
    result = paired_date_bootstrap(
        control, candidate, "top1_ade", replicates=2000, seed=161
    )
    assert np.isclose(result["point_absolute_gain"], 0.2)
    assert result["ci95"][0] > 0
