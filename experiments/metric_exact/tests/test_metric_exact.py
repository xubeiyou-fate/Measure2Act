"""Focused unit tests for the frozen C127 implementation."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from airroute_stage_m.evaluation import compute_batch_metrics
from experiments.metric_exact.folds import assign_date_folds, indices_for_fold
from experiments.metric_exact.locked_test import (
    control_confirmation_gates,
    validate_development_gate,
)
from experiments.metric_exact.model import VARIANTS, ScoreIsolatedAscent, build_model
from experiments.metric_exact.objective import objective_for_variant, per_mode_errors
from experiments.metric_exact.summarize import paired_date_bootstrap
from experiments.metric_exact.summarize_p2 import build_summary as build_p2_summary
from experiments.metric_exact.summarize_p3 import hierarchical_bootstrap


def test_model_interfaces_and_score_isolation() -> None:
    batch = {"obs_traj": torch.randn(16, 2, 3)}
    for variant in VARIANTS:
        model = build_model(variant, batch_size=2).eval()
        with torch.no_grad():
            predictions, logits, auxiliary = model(batch)
        assert predictions.shape == (2, 5, 24, 3)
        assert logits.shape == (2, 5)
        assert not auxiliary["kinematic_decoder"]["trajectory_residual"]
        assert not auxiliary["kinematic_decoder"]["learned_gate"]
        if variant not in {"B0_signed_coupled", "B1_positive_coupled"}:
            assert isinstance(model, ScoreIsolatedAscent)
            assert torch.count_nonzero(logits) == 0
            assert sum(parameter.numel() for parameter in model.geometry.pi.parameters()) == 0


def test_metric_errors_match_audited_evaluator() -> None:
    torch.manual_seed(127)
    predictions = torch.randn(7, 5, 24, 3)
    target = torch.randn(7, 24, 3)
    logits = torch.randn(7, 5)
    ade, fde = per_mode_errors(predictions, target)
    metrics = compute_batch_metrics(predictions, logits, target)
    torch.testing.assert_close(ade.min(dim=1).values, metrics["minade"])
    torch.testing.assert_close(fde.min(dim=1).values, metrics["minfde"])


def test_original_objectives_match_repository_loss() -> None:
    torch.manual_seed(127)
    predictions = torch.randn(4, 5, 24, 3)
    target = torch.randn(4, 24, 3)
    logits = torch.randn(4, 5)
    ade, _ = per_mode_errors(predictions, target)
    winner = ade.argmin(dim=1)
    batch = torch.arange(4)
    regression = F.smooth_l1_loss(predictions[batch, winner], target)

    coupled, _ = objective_for_variant(
        "B1_positive_coupled", predictions, logits, target
    )
    isolated, _ = objective_for_variant(
        "B2_decoupled_original", predictions, logits, target
    )
    torch.testing.assert_close(isolated, regression)
    torch.testing.assert_close(coupled, regression + F.cross_entropy(logits, winner))


def test_dual_oracle_routes_ade_and_fde_to_distinct_modes() -> None:
    target = torch.zeros(1, 4, 3)
    predictions = torch.full((1, 2, 4, 3), 10.0, requires_grad=True)
    with torch.no_grad():
        predictions[:, 0] = 1.0
        predictions[:, 1, -1] = 0.1
    logits = torch.zeros(1, 2)
    loss, diagnostics = objective_for_variant(
        "B6_dual_oracle", predictions, logits, target
    )
    loss.backward()
    assert int(diagnostics["ade_winner"][0]) == 0
    assert int(diagnostics["fde_winner"][0]) == 1
    assert predictions.grad[0, 0].abs().sum() > 0
    assert predictions.grad[0, 1, -1].abs().sum() > 0
    assert predictions.grad[0, 1, :-1].abs().sum() == 0


def test_date_folds_are_disjoint_complete_and_deterministic() -> None:
    dates = ["2022-01-01"] * 7 + ["2022-01-02"] * 4 + ["2022-01-03"] * 3
    dates += ["2022-01-04"] * 2 + ["2022-01-05"]
    first = assign_date_folds(dates, fold_count=3, seed=127)
    second = assign_date_folds(dates, fold_count=3, seed=127)
    assert first == second
    validation_union: set[int] = set()
    for fold in range(3):
        train, validation, validation_dates = indices_for_fold(dates, first, fold)
        assert not set(train) & set(validation)
        assert len(validation) == len(validation_dates)
        validation_union.update(validation)
    assert validation_union == set(range(len(dates)))


def test_cluster_bootstraps_preserve_known_paired_gain() -> None:
    control_dates = {
        f"date-{index}": {"actors": index + 1, "minade": 3.0, "minfde": 5.0}
        for index in range(11)
    }
    candidate_dates = {
        date: {"actors": values["actors"], "minade": 2.0, "minfde": 4.0}
        for date, values in control_dates.items()
    }
    paired = paired_date_bootstrap(
        control_dates, candidate_dates, "minade", replicates=100, seed=127
    )
    assert paired["point_absolute_gain"] == 1.0
    assert paired["ci95"] == [1.0, 1.0]

    controls = {seed: control_dates for seed in (7, 42)}
    candidates = {seed: candidate_dates for seed in (7, 42)}
    hierarchical = hierarchical_bootstrap(
        controls, candidates, "minfde", replicates=100, seed=127
    )
    assert hierarchical["point_absolute_gain"] == 1.0
    assert hierarchical["ci95"] == [1.0, 1.0]


def test_locked_confirmation_requires_both_b2_and_b0_claim_guards() -> None:
    seeds = (42, 7, 123, 2024, 2026)
    candidate = "B6_dual_oracle"

    def seed_metrics(value: float) -> dict[int, dict[str, object]]:
        return {
            seed: {
                "overall": {"minade": value, "minfde": value},
                "date_metrics": {
                    f"date-{index}": {
                        "actors": index + 1,
                        "minade": value,
                        "minfde": value,
                    }
                    for index in range(11)
                },
            }
            for seed in seeds
        }

    metrics = {
        candidate: seed_metrics(1.0),
        "B2_decoupled_original": seed_metrics(2.0),
        "B0_signed_coupled": seed_metrics(3.0),
    }
    aggregates = {
        variant: {
            metric: {"mean": values[42]["overall"][metric]}
            for metric in ("minade", "minfde")
        }
        for variant, values in metrics.items()
    }
    _, gates = control_confirmation_gates(metrics, aggregates, candidate)
    assert all(gates.values())

    metrics["B0_signed_coupled"] = seed_metrics(0.5)
    aggregates["B0_signed_coupled"] = {
        metric: {"mean": 0.5} for metric in ("minade", "minfde")
    }
    _, guarded = control_confirmation_gates(metrics, aggregates, candidate)
    assert all(value for name, value in guarded.items() if "B2" in name)
    assert not any(value for name, value in guarded.items() if "B0" in name)


def test_locked_development_gate_is_cross_checked_against_all_phases() -> None:
    protocol_hash = "frozen-protocol"
    candidate = "B6_dual_oracle"
    p3_gates = {"replicated": True}
    gate = {
        "protocol_sha256": protocol_hash,
        "candidate": candidate,
        "P1_passed": True,
        "P2_passed": True,
        "P3_gates": p3_gates,
        "passed": True,
        "decision": "LOCKED_TEST_AUTHORIZED",
        "checkpoint_policy": "all 15 frozen P3 checkpoints in one locked-test event",
        "locked_test_used": False,
    }
    p1 = {
        "decision": "P2_AUTHORIZED",
        "P2_selected_exact_candidate": candidate,
        "locked_test_used": False,
    }
    p2 = {
        "protocol_sha256": protocol_hash,
        "passed": True,
        "decision": "P3_AUTHORIZED",
        "P3_selected_exact_candidate": candidate,
        "locked_test_used": False,
    }
    p3 = {
        "protocol_sha256": protocol_hash,
        "candidate": candidate,
        "gates": p3_gates,
        "passed": True,
        "decision": "LOCKED_TEST_AUTHORIZED",
        "locked_test_used": False,
    }
    assert validate_development_gate(gate, p1, p2, p3, protocol_hash) == candidate

    corrupted = {**gate, "protocol_sha256": "different"}
    with pytest.raises(RuntimeError, match="gate_protocol"):
        validate_development_gate(corrupted, p1, p2, p3, protocol_hash)


def test_p2_untouched_replication_diagnostic_excludes_selection_fold() -> None:
    def result(value: float, date: str) -> dict[str, object]:
        return {
            "validation_metrics": {
                "overall": {"minade": value, "minfde": value},
                "date_metrics": {
                    date: {"actors": 1, "minade": value, "minfde": value}
                },
            }
        }

    candidate = "B6_dual_oracle"
    folds = {
        fold: {
            "B0_signed_coupled": result(2.0, f"date-{fold}"),
            "B2_decoupled_original": result(2.0, f"date-{fold}"),
            candidate: result(3.0 if fold == 0 else 1.0, f"date-{fold}"),
        }
        for fold in range(5)
    }
    summary = build_p2_summary(folds, candidate)
    diagnostic = summary["untouched_replication_diagnostic"]
    assert diagnostic["folds"] == [1, 2, 3, 4]
    assert diagnostic["date_count"] == 4
    assert diagnostic["same_direction_folds_vs_B2"] == 4
    assert diagnostic["paired_date_bootstrap_vs_B2"]["minade"]["ci95"] == [
        1.0,
        1.0,
    ]
