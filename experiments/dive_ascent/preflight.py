"""Data, architecture, isolation, and DAC preflight for C99."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model.ascent import Ascent
from model.utils import TrajectoryDataset

from .model import C99_VARIANTS, DecoupledAscent, ascent_config, build_model
from .objective import geometry_wta_loss, voronoi_score_loss
from .protocol import load_protocol, sha256


def _columns(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return len(line.strip().split(" "))
    return 0


def _synthetic_batch(batch: int = 3) -> dict[str, torch.Tensor]:
    history = torch.cumsum(torch.randn(16, batch, 3) * 0.05, dim=0)
    return {"obs_traj": history}


def _has_finite_gradients(parameters) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters
    )


def run(output: Path | None = None, *, verify_all_hashes: bool = True) -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_development_sealed()
    manifest = protocol.manifest()
    names: dict[str, set[str]] = {}
    partitions = {}
    hash_failures: list[str] = []
    schema_failures: list[str] = []
    for split in ("train", "dev", "locked_test"):
        records = manifest["partitions"][split]["records"]
        directory = protocol.split_path(split)
        expected_names = {record["name"] for record in records}
        actual_names = {path.name for path in directory.iterdir() if path.is_file()}
        if expected_names != actual_names:
            raise RuntimeError(f"C99 {split} files differ from the frozen manifest")
        names[split] = expected_names
        for record in records:
            path = directory / record["name"]
            if verify_all_hashes and sha256(path) != record["sha256"]:
                hash_failures.append(f"{split}/{record['name']}")
            if _columns(path) != 7:
                schema_failures.append(f"{split}/{record['name']}")
        partitions[split] = {
            "dates": int(manifest["partitions"][split]["date_count"]),
            "files": len(records),
        }
    overlap = {
        "train_dev": len(names["train"] & names["dev"]),
        "train_locked": len(names["train"] & names["locked_test"]),
        "dev_locked": len(names["dev"] & names["locked_test"]),
    }
    datasets = {
        split: TrajectoryDataset(
            protocol.split_path(split).as_posix(),
            obs_len=16,
            obs_steps=1,
            pred_len=120,
            pred_step=5,
            delim=" ",
        )
        for split in ("train", "dev")
    }
    loaded = {
        "train_scenes": len(datasets["train"]),
        "train_actors": int(datasets["train"].obs_traj.shape[0]),
        "dev_scenes": len(datasets["dev"]),
        "dev_actors": int(datasets["dev"].obs_traj.shape[0]),
    }
    expected = protocol.payload["dataset"]["expected"]
    expected_loaded = {name: int(expected[name]) for name in loaded}

    torch.manual_seed(99)
    direct = Ascent(ascent_config("a0_shared_signed")).eval()
    torch.manual_seed(99)
    wrapped = build_model("a0_shared_signed").eval()
    batch = _synthetic_batch()
    with torch.no_grad():
        parity = float((direct(batch)[0] - wrapped(batch)[0]).abs().max())

    parameter_counts = {
        variant: sum(parameter.numel() for parameter in build_model(variant).parameters())
        for variant in C99_VARIANTS
    }
    capacity_ratio = parameter_counts["a3_scaled_decoupled"] / parameter_counts["a4_independent_random"]
    finite = {}
    forbidden_disabled = {}
    for variant in C99_VARIANTS:
        candidate = build_model(variant).eval()
        with torch.no_grad():
            prediction, logits, auxiliary = candidate(batch)
        finite[variant] = bool(torch.isfinite(prediction).all() and torch.isfinite(logits).all())
        forbidden_disabled[variant] = not any(
            bool(auxiliary.get(name, False))
            for name in (
                "trajectory_residual",
                "learned_gate",
                "token_codebook",
                "future_autoregression",
                "post_generation_selector",
            )
        )

    target = torch.randn(3, 24, 3)
    isolated = build_model("a2_shared_decoupled")
    assert isinstance(isolated, DecoupledAscent)
    prediction, _ = isolated.predict_geometry(batch)
    geometry_loss, diagnostics = geometry_wta_loss(prediction, target)
    geometry_loss.backward()
    geometry_has_grad = any(parameter.grad is not None for parameter in isolated.geometry_parameters())
    scorer_has_geometry_grad = any(parameter.grad is not None for parameter in isolated.scorer_parameters())
    isolated.zero_grad(set_to_none=True)
    with torch.no_grad():
        prediction, _ = isolated.predict_geometry(batch)
    logits, _ = isolated.score(batch, prediction)
    score_loss = voronoi_score_loss(logits, diagnostics["winner"])
    score_loss.backward()
    geometry_has_score_grad = any(parameter.grad is not None for parameter in isolated.geometry_parameters())
    scorer_has_score_grad = any(parameter.grad is not None for parameter in isolated.scorer_parameters())

    independent = build_model("a4_independent_random")
    assert isinstance(independent, DecoupledAscent)
    expert_parameter_ids = [
        {id(parameter) for parameter in expert.parameters()}
        for expert in independent.experts
    ]
    experts_parameter_disjoint = all(
        expert_parameter_ids[left].isdisjoint(expert_parameter_ids[right])
        for left in range(5)
        for right in range(left + 1, 5)
    )
    dive = build_model("a5_dive")
    assert isinstance(dive, DecoupledAscent)
    initial_equal = all(
        torch.equal(left, right)
        for left, right in zip(dive.experts[0].state_dict().values(), dive.experts[1].state_dict().values())
    )
    split = dive.split_highest_distortion(
        torch.tensor([1.0]), perturbation_scale=0.01, seed=99042
    )
    post_split_different = any(
        not torch.equal(left, right)
        for left, right in zip(dive.experts[0].state_dict().values(), dive.experts[1].state_dict().values())
        if torch.is_floating_point(left)
    )
    with torch.no_grad():
        _, _, dive_aux = dive(batch)

    checks = {
        "manifest_overlap_zero": all(value == 0 for value in overlap.values()),
        "locked_test_sealed": manifest["locked_test_evaluated"] is False,
        "all_sha256_match": not hash_failures,
        "seven_column_schema_complete": not schema_failures,
        "cohort_matches_frozen_counts": loaded == expected_loaded,
        "partition_date_counts_match": all(
            partitions[split]["dates"] == int(expected[f"{split}_dates"])
            for split in ("train", "dev", "locked_test")
        ),
        "a0_forward_parity": parity == 0.0,
        "all_outputs_finite": all(finite.values()),
        "capacity_control_within_three_percent": 0.97 <= capacity_ratio <= 1.03,
        "all_forbidden_mechanisms_disabled": all(forbidden_disabled.values()),
        "geometry_step_updates_only_geometry": geometry_has_grad and not scorer_has_geometry_grad,
        "score_step_updates_only_scorer": scorer_has_score_grad and not geometry_has_score_grad,
        "isolated_gradients_finite": _has_finite_gradients(isolated.parameters()),
        "independent_expert_parameters_disjoint": experts_parameter_disjoint,
        "dac_starts_as_one_cloned_expert": initial_equal,
        "dac_split_is_symmetric_and_nonzero": post_split_different and split["perturbation_l2"] > 0.0,
        "dac_forward_executes_all_five": dive_aux["all_experts_executed"] is True,
    }
    result = {
        "format_version": 1,
        "cycle": protocol.payload["cycle"],
        "protocol_sha256": sha256(protocol.path),
        "manifest_sha256": sha256(protocol.manifest_path),
        "partitions": partitions,
        "pairwise_filename_overlap": overlap,
        "loaded": loaded,
        "hash_failures": hash_failures,
        "schema_failures": schema_failures,
        "a0_forward_parity_max_abs_error": parity,
        "parameter_counts": parameter_counts,
        "a3_to_a4_parameter_ratio": capacity_ratio,
        "finite_outputs": finite,
        "forbidden_mechanisms_disabled": forbidden_disabled,
        "gradient_isolation": {
            "geometry_has_geometry_gradient": geometry_has_grad,
            "scorer_has_geometry_gradient": scorer_has_geometry_grad,
            "geometry_has_score_gradient": geometry_has_score_grad,
            "scorer_has_score_gradient": scorer_has_score_grad,
        },
        "dac_split": split,
        "checks": checks,
        "passed": all(checks.values()),
        "locked_test_used": False,
    }
    if not result["passed"]:
        raise RuntimeError(f"C99 preflight failed: {checks}")
    output = output or protocol.repository_root / "artifacts/experiments/dive_ascent/preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-hashes", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.output, verify_all_hashes=not args.skip_hashes), indent=2))


if __name__ == "__main__":
    main()
