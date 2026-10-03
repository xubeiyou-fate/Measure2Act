"""Run the remaining registered journal-extension experiments sequentially."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable).resolve()
SEEDS = (42, 7, 123, 2024, 2026)
NEW_EQMOTION_SEEDS = (7, 123, 2024, 2026)


def run_command(arguments: Iterable[str], *, output: Path | None = None) -> None:
    if output is not None and output.is_file():
        print(json.dumps({"status": "skip_existing", "output": output.relative_to(ROOT).as_posix()}), flush=True)
        return
    command = [str(PYTHON), *map(str, arguments)]
    print(json.dumps({"status": "start", "command": command}), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def eqmotion_training(device: str) -> None:
    for seed in (123, 2024, 2026):
        run_dir = ROOT / f"runs/partc_two_dataset_20260812/eqmotion_tartan_multiseed_v2/KBTP/seed{seed}_formal"
        output = ROOT / f"artifacts/partc_two_dataset_20260812/modern_baseline/eqmotion_tartan_multiseed_v2/KBTP_target_only_seed{seed}_formal.json"
        arguments = [
            "-m", "modern_baseline.run_eqmotion_tartan_multiseed_v2",
            "--airport", "KBTP", "--seed", str(seed), "--device", device,
            "--run-dir", run_dir.relative_to(ROOT), "--output", output.relative_to(ROOT),
        ]
        if run_dir.joinpath("last.pt").is_file():
            arguments.append("--resume")
        run_command(arguments, output=output)


def eqmotion_locked(device: str) -> None:
    receipt = ROOT / "artifacts/partc_two_dataset_20260812/modern_baseline/eqmotion_tartan_multiseed_v2_evaluation_receipt.json"
    run_command(["-m", "scripts.freeze_eqmotion_tartan_multiseed_v2"], output=receipt)
    for airport in ("KAGC", "KBTP"):
        for seed in NEW_EQMOTION_SEEDS:
            output = ROOT / f"artifacts/partc_two_dataset_20260812/modern_baseline/eqmotion_tartan_multiseed_v2/locked/{airport}_target_only_seed{seed}_locked_test_v2.json"
            run_command([
                "-m", "modern_baseline.evaluate_eqmotion_tartan_multiseed_v2_locked",
                "--airport", airport, "--seed", str(seed), "--device", device,
                "--batch-size", "512", "--authorize-locked-test", "--output", output.relative_to(ROOT),
            ], output=output)
    summary = ROOT / "artifacts/journal_extension_20260814/eqmotion_five_seed_summary_v1.json"
    run_command(["-m", "experiments.journal_extension.aggregate_eqmotion_multiseed"], output=summary)


def probability_controls(device: str, workers: int) -> None:
    development_root = ROOT / "artifacts/journal_extension_20260814/probability_controls/development"
    test_root = ROOT / "artifacts/journal_extension_20260814/probability_controls/test"
    for split, root in (("development", development_root), ("test", test_root)):
        if split == "test":
            receipt = ROOT / "artifacts/journal_extension_20260814/probability_controls_receipt_v1.json"
            run_command(["-m", "experiments.journal_extension.freeze_probability_controls"], output=receipt)
        for airport in ("KAGC", "KBTP"):
            for seed in SEEDS:
                output = root / f"{airport}_seed{seed}_{split}_v1.json"
                arguments = [
                    "-m", "experiments.journal_extension.evaluate_probability_controls",
                    "--airport", airport, "--seed", str(seed), "--split", split,
                    "--device", device, "--workers", str(workers), "--batch-size", "512",
                    "--output", output.relative_to(ROOT),
                ]
                if split == "test":
                    arguments.append("--authorize-retrospective-test")
                run_command(arguments, output=output)
    summary = ROOT / "artifacts/journal_extension_20260814/probability_controls_summary_v1.json"
    run_command(["-m", "experiments.journal_extension.aggregate_probability_controls"], output=summary)


def awta_tartan(device: str, workers: int) -> None:
    for airport in ("KAGC", "KBTP"):
        for seed in SEEDS:
            run_dir = ROOT / f"runs/journal_extension_20260814/awta_tartan/{airport}/seed{seed}_formal"
            output = ROOT / f"artifacts/journal_extension_20260814/awta_tartan/development/{airport}_seed{seed}_formal.json"
            arguments = [
                "-m", "experiments.journal_extension.train_awta_tartan",
                "--airport", airport, "--seed", str(seed), "--device", device,
                "--workers", str(workers), "--run-dir", run_dir.relative_to(ROOT),
                "--output", output.relative_to(ROOT),
            ]
            if run_dir.joinpath("last.pt").is_file():
                arguments.append("--resume")
            run_command(arguments, output=output)


def awta_tartan_locked(device: str, workers: int) -> None:
    receipt = ROOT / "artifacts/journal_extension_20260814/awta_tartan_evaluation_receipt_v1.json"
    run_command(["-m", "experiments.journal_extension.freeze_awta_tartan_evaluation"], output=receipt)
    for airport in ("KAGC", "KBTP"):
        for seed in SEEDS:
            output = ROOT / f"artifacts/journal_extension_20260814/awta_tartan/test/{airport}_seed{seed}_test_v1.json"
            run_command([
                "-m", "experiments.journal_extension.evaluate_awta_tartan_locked",
                "--airport", airport, "--seed", str(seed), "--device", device,
                "--workers", str(workers), "--batch-size", "512",
                "--authorize-retrospective-test", "--output", output.relative_to(ROOT),
            ], output=output)


def awta_trajair(device: str, workers: int) -> None:
    for seed in SEEDS:
        run_dir = ROOT / f"runs/journal_extension_20260814/awta_trajair/seed{seed}_formal"
        output = ROOT / f"artifacts/journal_extension_20260814/awta_trajair/development/seed{seed}_formal.json"
        arguments = [
            "-m", "experiments.journal_extension.train_awta_trajair",
            "--seed", str(seed), "--device", device, "--workers", str(workers),
            "--run-dir", run_dir.relative_to(ROOT), "--output", output.relative_to(ROOT),
        ]
        if run_dir.joinpath("last.pt").is_file():
            arguments.append("--resume")
        run_command(arguments, output=output)
    summary = ROOT / "artifacts/journal_extension_20260814/awta_summary_v1.json"
    run_command(["-m", "experiments.journal_extension.aggregate_awta"], output=summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument(
        "--start-stage",
        choices=("eqmotion_training", "eqmotion_locked", "probability_controls", "awta_tartan", "awta_tartan_locked", "awta_trajair"),
        default="eqmotion_training",
    )
    args = parser.parse_args()
    if args.wait_pid is not None:
        print(json.dumps({"status": "waiting_for_pid", "pid": args.wait_pid}), flush=True)
        while True:
            try:
                os.kill(args.wait_pid, 0)
            except OSError:
                break
            time.sleep(10)
        print(json.dumps({"status": "wait_complete", "pid": args.wait_pid}), flush=True)
    stages = (
        ("eqmotion_training", lambda: eqmotion_training(args.device)),
        ("eqmotion_locked", lambda: eqmotion_locked(args.device)),
        ("probability_controls", lambda: probability_controls(args.device, args.workers)),
        ("awta_tartan", lambda: awta_tartan(args.device, args.workers)),
        ("awta_tartan_locked", lambda: awta_tartan_locked(args.device, args.workers)),
        ("awta_trajair", lambda: awta_trajair(args.device, args.workers)),
    )
    started = False
    for name, action in stages:
        started = started or name == args.start_stage
        if started:
            print(json.dumps({"status": "stage", "name": name}), flush=True)
            action()
    audit = ROOT / "artifacts/journal_extension_20260814/completion_audit_v1.json"
    run_command(["-m", "experiments.journal_extension.audit_completion"], output=audit)


if __name__ == "__main__":
    main()
