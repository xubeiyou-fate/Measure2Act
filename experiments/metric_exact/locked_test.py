"""Execute C127's single authorized locked-test evaluation event."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiments.edfa_ascent.data import build_scene_dates
from model.utils import TrajectoryDataset, seq_collate

from .evaluation import evaluate
from .locking import exclusive_process_lock
from .model import build_model
from .protocol import load_protocol, sha256
from .summarize_p3 import SEEDS, aggregate, hierarchical_bootstrap


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/metric_exact"
RUN_ROOT = ROOT / "runs/metric_exact"
PLAN_PATH = Path(__file__).with_name("locked_analysis_plan.json")
CLAIM_GUARD_PATH = Path(__file__).with_name("locked_claim_guard_addendum.json")
EXPECTED_PLAN_SHA256 = "a57e7bd6d5c761ff4c0d4e65d70cec1f4dc975d202a281e7d0c069b80cda76a1"
EXPECTED_CLAIM_GUARD_SHA256 = (
    "624163fbac3866230e7611479d42ff66b9050637ab80025bf7c3dfd61da218b1"
)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_development_gate(
    gate: dict[str, object],
    p1: dict[str, object],
    p2: dict[str, object],
    p3: dict[str, object],
    protocol_hash: str,
) -> str:
    """Cross-check the locked-test authorization against every frozen phase."""

    candidate = gate.get("candidate")
    checks = {
        "gate_protocol": gate.get("protocol_sha256") == protocol_hash,
        "gate_flags": gate.get("P1_passed") is True
        and gate.get("P2_passed") is True
        and gate.get("passed") is True
        and gate.get("decision") == "LOCKED_TEST_AUTHORIZED"
        and gate.get("locked_test_used") is False,
        "checkpoint_policy": gate.get("checkpoint_policy")
        == "all 15 frozen P3 checkpoints in one locked-test event",
        "P1": p1.get("decision") == "P2_AUTHORIZED"
        and p1.get("P2_selected_exact_candidate") == candidate
        and p1.get("locked_test_used") is False,
        "P2": p2.get("protocol_sha256") == protocol_hash
        and p2.get("passed") is True
        and p2.get("decision") == "P3_AUTHORIZED"
        and p2.get("P3_selected_exact_candidate") == candidate
        and p2.get("locked_test_used") is False,
        "P3": p3.get("protocol_sha256") == protocol_hash
        and p3.get("passed") is True
        and p3.get("decision") == "LOCKED_TEST_AUTHORIZED"
        and p3.get("candidate") == candidate
        and p3.get("gates") == gate.get("P3_gates")
        and p3.get("locked_test_used") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed or not isinstance(candidate, str):
        details = ", ".join(failed or ["candidate"])
        raise RuntimeError(f"C127 development gate integrity check failed: {details}")
    return candidate


def _checkpoint(
    variant: str, seed: int, protocol_hash: str
) -> tuple[dict[str, object], Path, dict[str, object]]:
    directory = RUN_ROOT / f"P3_{variant}_all_train_seed{seed}_formal"
    summary_path = directory / "training_summary.json"
    checkpoint_path = directory / "last.pt"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("complete") is not True
        or summary.get("formal") is not True
        or summary.get("fixed_final_epoch") != 20
        or summary.get("protocol_sha256") != protocol_hash
        or summary.get("locked_test_used") is not False
    ):
        raise RuntimeError(f"invalid C127 P3 summary: {summary_path}")
    if sha256(checkpoint_path) != summary["checkpoint_sha256"]:
        raise RuntimeError(f"C127 checkpoint hash mismatch: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol_sha256") != protocol_hash:
        raise RuntimeError(f"C127 checkpoint protocol mismatch: {checkpoint_path}")
    return checkpoint, checkpoint_path, summary


def control_confirmation_gates(
    metrics: dict[str, dict[int, dict[str, object]]],
    aggregates: dict[str, dict[str, object]],
    candidate: str,
) -> tuple[dict[str, object], dict[str, bool]]:
    """Apply the frozen B2 gates and the conjunctive B0 claim guard."""

    candidate_dates = {
        seed: metrics[candidate][seed]["date_metrics"] for seed in SEEDS
    }
    comparison = {}
    for control in ("B2_decoupled_original", "B0_signed_coupled"):
        control_dates = {
            seed: metrics[control][seed]["date_metrics"] for seed in SEEDS
        }
        bootstrap = {
            metric: hierarchical_bootstrap(control_dates, candidate_dates, metric)
            for metric in ("minade", "minfde")
        }
        per_seed_both = sum(
            metrics[candidate][seed]["overall"]["minade"]
            < metrics[control][seed]["overall"]["minade"]
            and metrics[candidate][seed]["overall"]["minfde"]
            < metrics[control][seed]["overall"]["minfde"]
            for seed in SEEDS
        )
        comparison[control] = {
            "hierarchical_bootstrap": bootstrap,
            "same_direction_seeds": per_seed_both,
            "candidate_mean_better_for_both_metrics": all(
                aggregates[candidate][metric]["mean"]
                < aggregates[control][metric]["mean"]
                for metric in ("minade", "minfde")
            ),
        }
    b2_comparison = comparison["B2_decoupled_original"]
    b0_comparison = comparison["B0_signed_coupled"]
    success_gates = {
        "both_metrics_better_than_B2_in_at_least_four_of_five_seeds": (
            b2_comparison["same_direction_seeds"] >= 4
        ),
        "candidate_mean_better_than_B2_for_both_metrics": b2_comparison[
            "candidate_mean_better_for_both_metrics"
        ],
        "hierarchical_date_by_seed_bootstrap_vs_B2_ci_lower_bound_positive_for_both": all(
            b2_comparison["hierarchical_bootstrap"][metric]["ci95"][0] > 0
            for metric in ("minade", "minfde")
        ),
        "both_metrics_better_than_B0_in_at_least_four_of_five_seeds": (
            b0_comparison["same_direction_seeds"] >= 4
        ),
        "candidate_mean_better_than_B0_for_both_metrics": b0_comparison[
            "candidate_mean_better_for_both_metrics"
        ],
        "hierarchical_date_by_seed_bootstrap_vs_B0_ci_lower_bound_positive_for_both": all(
            b0_comparison["hierarchical_bootstrap"][metric]["ci95"][0] > 0
            for metric in ("minade", "minfde")
        ),
    }
    return comparison, success_gates


def run(output: Path, device_name: str) -> dict[str, object]:
    protocol = load_protocol()
    protocol.assert_boundaries()
    protocol_hash = sha256(protocol.path)
    plan_hash = sha256(PLAN_PATH)
    if plan_hash != EXPECTED_PLAN_SHA256:
        raise RuntimeError("C127 locked analysis plan hash mismatch")
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if plan.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("C127 locked analysis plan protocol mismatch")
    claim_guard_hash = sha256(CLAIM_GUARD_PATH)
    if claim_guard_hash != EXPECTED_CLAIM_GUARD_SHA256:
        raise RuntimeError("C127 locked claim-guard addendum hash mismatch")
    claim_guard = json.loads(CLAIM_GUARD_PATH.read_text(encoding="utf-8"))
    if (
        claim_guard.get("protocol_sha256") != protocol_hash
        or claim_guard.get("original_analysis_plan_sha256") != plan_hash
        or claim_guard.get("locked_test_used") is not False
    ):
        raise RuntimeError("C127 locked claim-guard addendum mismatch")
    gate_path = ROOT / str(
        protocol.payload["locked_test_policy"]["required_gate_artifact"]
    )
    if not gate_path.is_file():
        raise RuntimeError("C127 development gate is missing")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    phase_summaries = {
        phase: json.loads((ARTIFACT_ROOT / f"{phase}_summary.json").read_text(encoding="utf-8"))
        for phase in ("p1", "p2", "p3")
    }
    candidate = validate_development_gate(
        gate,
        phase_summaries["p1"],
        phase_summaries["p2"],
        phase_summaries["p3"],
        protocol_hash,
    )
    receipt_path = ROOT / str(protocol.payload["locked_test_policy"]["receipt"])
    if receipt_path.exists():
        raise RuntimeError("C127 locked test has already been consumed")

    variants = ("B0_signed_coupled", "B2_decoupled_original", candidate)
    checkpoints: dict[tuple[str, int], tuple[dict[str, object], Path, dict[str, object]]] = {}
    for variant in variants:
        for seed in SEEDS:
            checkpoints[(variant, seed)] = _checkpoint(variant, seed, protocol_hash)
    tail_thresholds = {
        float(values[2]["tail_threshold"]) for values in checkpoints.values()
    }
    if len(tail_thresholds) != 1:
        raise RuntimeError("C127 P3 train-derived tail thresholds differ")

    started = {
        "format_version": 1,
        "cycle": protocol.payload["cycle"],
        "status": "locked_test_event_started",
        "event_count": 1,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol_hash,
        "analysis_plan_sha256": plan_hash,
        "claim_guard_addendum_sha256": claim_guard_hash,
        "manifest_sha256_before": sha256(protocol.manifest_path),
    }
    atomic_json(receipt_path, started)

    dataset = TrajectoryDataset(
        protocol.split_path("locked_test").as_posix(),
        obs_len=16,
        obs_steps=1,
        pred_len=120,
        pred_step=5,
        delim=" ",
    )
    dates = build_scene_dates(
        protocol.split_path("locked_test"), protocol.manifest_path, "locked_test"
    )
    if len(dates) != len(dataset) or len(set(dates)) != 11:
        raise RuntimeError("C127 locked-test scene-date index mismatch")
    loader = DataLoader(
        dataset,
        batch_size=int(protocol.payload["training"]["evaluation_batch_size"]),
        shuffle=False,
        collate_fn=seq_collate,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )
    device = torch.device(device_name)
    metrics: dict[str, dict[int, dict[str, object]]] = {
        variant: {} for variant in variants
    }
    checkpoint_receipts = {}
    for variant in variants:
        for seed in SEEDS:
            checkpoint, checkpoint_path, _ = checkpoints[(variant, seed)]
            model = build_model(variant, batch_size=256).to(device)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            metrics[variant][seed] = evaluate(
                model,
                loader,
                device,
                scene_dates=dates,
                tail_threshold=next(iter(tail_thresholds)),
            )
            checkpoint_receipts[f"{variant}_seed{seed}"] = {
                "path": checkpoint_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(checkpoint_path),
            }
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    aggregates = {
        variant: aggregate(
            {
                seed: {"validation_metrics": metrics[variant][seed]}
                for seed in SEEDS
            }
        )
        for variant in variants
    }
    comparison, success_gates = control_confirmation_gates(
        metrics, aggregates, candidate
    )
    b2_comparison = comparison["B2_decoupled_original"]
    b0_comparison = comparison["B0_signed_coupled"]
    completed_utc = datetime.now(timezone.utc).isoformat()
    result = {
        "format_version": 1,
        "cycle": protocol.payload["cycle"],
        "event_count": 1,
        "candidate": candidate,
        "seeds": list(SEEDS),
        "selection_on_locked_test": False,
        "metrics": metrics,
        "aggregates": aggregates,
        "candidate_vs_controls": comparison,
        "candidate_vs_B2_hierarchical_bootstrap": b2_comparison[
            "hierarchical_bootstrap"
        ],
        "same_direction_seeds_vs_B2": b2_comparison["same_direction_seeds"],
        "candidate_vs_B0_hierarchical_bootstrap": b0_comparison[
            "hierarchical_bootstrap"
        ],
        "same_direction_seeds_vs_B0": b0_comparison["same_direction_seeds"],
        "success_gates": success_gates,
        "locked_confirmation_passed": all(success_gates.values()),
        "checkpoint_receipts": checkpoint_receipts,
        "protocol_sha256": protocol_hash,
        "analysis_plan_sha256": plan_hash,
        "claim_guard_addendum_sha256": claim_guard_hash,
        "completed_utc": completed_utc,
    }
    atomic_json(output, result)

    manifest = protocol.manifest()
    manifest["locked_test_evaluated"] = True
    manifest["locked_test_evaluation"] = {
        "cycle": protocol.payload["cycle"],
        "event_count": 1,
        "result": output.relative_to(ROOT).as_posix(),
        "completed_utc": completed_utc,
    }
    atomic_json(protocol.manifest_path, manifest)
    receipt = {
        **started,
        "status": "locked_test_event_complete",
        "completed_utc": completed_utc,
        "result": output.relative_to(ROOT).as_posix(),
        "result_sha256": sha256(output),
        "manifest_sha256_after": sha256(protocol.manifest_path),
        "locked_confirmation_passed": result["locked_confirmation_passed"],
    }
    atomic_json(receipt_path, receipt)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_ROOT / "locked_test_result.json",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    with exclusive_process_lock(ARTIFACT_ROOT / "locked_test.lock", "locked test"):
        result = run(args.output, args.device)
    print(
        json.dumps(
            {
                "event_count": result["event_count"],
                "candidate": result["candidate"],
                "locked_confirmation_passed": result[
                    "locked_confirmation_passed"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
