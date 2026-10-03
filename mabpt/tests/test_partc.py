from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

import mabpt.physical as physical
from mabpt.aggregate_partc_seeds import (
    hierarchical_paired_bootstrap,
    paired_seed_bootstrap,
)
from mabpt.aggregate_partc_hypotheses import holm_adjust
from mabpt.finalize_partc_package import _claim_boundary, _readme
from mabpt.operator import energy_kl_projection, exact_gibbs_transport
from mabpt.partc_design import blocked_schedule, factorial_runs, load_protocol
from mabpt.partc_factorial import (
    design_matrix,
    factorial_probability_arms,
    full_model_arm,
)
from mabpt.physical import (
    FEATURES,
    fit_training_envelope,
    kinematic_features,
    physical_summary,
)
from mabpt.train_partc_target import FORMAL_BATCHES
import mabpt.run_partc_queue as partc_queue


def _factorial_problem(batch: int = 4, modes: int = 5):
    generator = torch.Generator().manual_seed(20260811)
    probabilities = torch.rand(batch, modes, generator=generator, dtype=torch.float64)
    probabilities /= probabilities.sum(dim=1, keepdim=True)
    cost = torch.rand(batch, modes, modes, generator=generator, dtype=torch.float64)
    risk = torch.rand(batch, modes, generator=generator, dtype=torch.float64)
    support = torch.rand(batch, modes, 6, 3, generator=generator, dtype=torch.float64)
    pairwise = torch.linalg.vector_norm(
        support[:, :, None] - support[:, None, :], dim=-1
    ).mean(dim=-1)
    return probabilities, cost, risk, pairwise


def test_queue_reuses_aggregate_only_for_current_inputs(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.json"
    source.write_text('{"value": 1}\n', encoding="utf-8")
    aggregate = tmp_path / "aggregate.json"
    aggregate.write_text(
        json.dumps(
            {
                "inputs": [
                    {
                        "path": source.name,
                        "sha256": partc_queue._sha256(source),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(partc_queue, "ROOT", tmp_path)
    assert partc_queue._aggregate_inputs_current(aggregate)
    source.write_text('{"value": 2}\n', encoding="utf-8")
    assert not partc_queue._aggregate_inputs_current(aggregate)
    source.unlink()
    assert not partc_queue._aggregate_inputs_current(aggregate)


def test_partc_design_is_complete_and_deterministic() -> None:
    first = factorial_runs(seed=20260811)
    second = factorial_runs(seed=20260811)
    assert first == second
    assert len(first) == 8
    combinations = {
        (
            run["correspondence"],
            run["assignment_cost_mass"],
            run["projection"],
        )
        for run in first
    }
    assert len(combinations) == 8
    assert len(design_matrix()) == 8


def test_partc_schedule_is_blocked_by_all_five_seeds() -> None:
    protocol = load_protocol()
    schedule = blocked_schedule(protocol)
    assert len(schedule) == 5 * 10
    for seed in protocol["fixed_seeds"]:
        runs = [row for row in schedule if row["seed"] == seed]
        assert len(runs) == 10
        assert sorted(row["within_block_order"] for row in runs) == list(range(1, 11))


def test_formal_target_batches_match_frozen_routes() -> None:
    assert FORMAL_BATCHES == {
        "decision_support": (256, 512),
        "predicted_risk": (1024, 2048),
    }


def test_factorial_arms_preserve_mass_and_reproduce_full_model() -> None:
    probabilities, cost, risk, pairwise = _factorial_problem()
    arms, diagnostics = factorial_probability_arms(
        probabilities, cost, risk, pairwise
    )
    assert len(arms) == 8
    assert set(arms) == set(diagnostics)
    for value in arms.values():
        assert torch.all(value >= 0)
        assert torch.allclose(
            value.sum(dim=1),
            torch.ones(value.shape[0], dtype=value.dtype),
            atol=1e-12,
            rtol=0,
        )
    prior = exact_gibbs_transport(
        probabilities, cost, mass_weighted=True
    )["transported"]
    expected = energy_kl_projection(prior, risk, pairwise)[0]
    assert torch.allclose(arms[full_model_arm()], expected, atol=1e-12, rtol=0)


def test_factorial_probability_forward_has_no_target_argument() -> None:
    forbidden = {"target", "future", "ground_truth", "label"}
    assert forbidden.isdisjoint(
        inspect.signature(factorial_probability_arms).parameters
    )


def test_constant_velocity_physical_features() -> None:
    times = torch.arange(1, 7, dtype=torch.float64)
    positions = torch.zeros(2, 3, 6, 3, dtype=torch.float64)
    positions[..., 0] = times
    features = kinematic_features(positions, stride_seconds=1.0)
    assert set(features) == set(FEATURES)
    assert torch.allclose(
        features["horizontal_speed_km_per_second"],
        torch.ones(2, 3, 5, dtype=torch.float64),
    )
    for name in FEATURES[1:]:
        assert torch.count_nonzero(features[name]) == 0


def test_training_envelope_and_probability_weighted_summary() -> None:
    generator = torch.Generator().manual_seed(11)
    training = torch.cumsum(
        torch.randn(16, 8, 3, generator=generator, dtype=torch.float64), dim=1
    )
    training_features = kinematic_features(training, stride_seconds=5.0)
    envelope = fit_training_envelope(
        training_features, lower_quantile=0.0, upper_quantile=1.0
    )
    predictions = training[:4, None].repeat(1, 2, 1, 1)
    prediction_features = kinematic_features(predictions, stride_seconds=5.0)
    probabilities = torch.tensor(
        [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]],
        dtype=torch.float64,
    )
    summary = physical_summary(
        prediction_features, envelope, mode_probabilities=probabilities
    )
    for values in summary.values():
        assert values["outside_training_envelope_rate"] == 0.0


def test_large_training_envelope_fallback_matches_torch(monkeypatch) -> None:
    values = torch.linspace(-2.0, 3.0, 101, dtype=torch.float64)
    features = {name: values.clone() for name in FEATURES}
    expected = fit_training_envelope(features)
    monkeypatch.setattr(physical, "_TORCH_QUANTILE_MAX_ELEMENTS", 1)
    actual = fit_training_envelope(features)
    for name in FEATURES:
        assert abs(actual[name]["lower"] - expected[name]["lower"]) < 1e-12
        assert abs(actual[name]["upper"] - expected[name]["upper"]) < 1e-12
        assert actual[name]["training_samples"] == expected[name]["training_samples"]


def test_hierarchical_bootstrap_preserves_paired_seed_date_gain() -> None:
    control = {
        seed: {
            date: {"actors": 10, "energy_score": 2.0 + seed}
            for date in ("d1", "d2", "d3")
        }
        for seed in (1, 2)
    }
    candidate = {
        seed: {
            date: {"actors": 10, "energy_score": 1.5 + seed}
            for date in ("d1", "d2", "d3")
        }
        for seed in (1, 2)
    }
    result = hierarchical_paired_bootstrap(
        control, candidate, "energy_score", replicates=200, random_seed=3
    )
    assert result["absolute_gain"] == 0.5
    assert result["ci95"] == [0.5, 0.5]
    assert result["raw_one_sided_p"] == 1 / 201


def test_holm_adjustment_is_monotone_in_rank() -> None:
    adjusted = holm_adjust({"h1": 0.001, "h2": 0.01, "h3": 0.04, "h4": 0.2})
    assert adjusted == {"h1": 0.004, "h2": 0.03, "h3": 0.08, "h4": 0.2}


def test_paired_seed_bootstrap_preserves_constant_recall_gain() -> None:
    result = paired_seed_bootstrap(
        {seed: 0.5 for seed in range(5)},
        {seed: 0.6 for seed in range(5)},
        higher_is_better=True,
        replicates=100,
        random_seed=9,
    )
    assert abs(result["absolute_gain"] - 0.1) < 1e-12
    assert all(abs(value - 0.1) < 1e-12 for value in result["ci95"])
    assert result["raw_one_sided_p"] == 1 / 101


def test_paper_package_preserves_single_model_identity() -> None:
    documents = "\n".join((_claim_boundary(), _readme({"local_groups_total": 13})))
    lowered = documents.lower()
    assert "one paper model" in lowered
    assert "independent paper models" in lowered
    assert "system" not in lowered
    assert "architecture" not in lowered


def test_paper_package_preserves_confirmation_boundary() -> None:
    documents = "\n".join((_claim_boundary(), _readme({"local_groups_total": 13})))
    lowered = documents.lower()
    assert "fresh sealed confirmation" in lowered
    assert "new sealed later-period or airport cohort" in lowered
