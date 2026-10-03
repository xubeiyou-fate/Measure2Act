"""Finalize the verified Part C paper kit for the single MABPT-ASCENT model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
PARTC = ROOT / "artifacts/mabpt_partc_20260811"
LEGACY = ROOT / "artifacts/mabpt"
DEFAULT_KIT = WORKSPACE / "MABPT_paper_kit_20260811"
DEFAULT_ARCHIVE = WORKSPACE / "MABPT_ASCENT_PartC_complete_20260811.zip"
DEFAULT_OLD_ARCHIVE = WORKSPACE / "MABPT_ASCENT_unified_paper_kit_20260811.zip"

PARTC_ARTIFACTS = (
    "design_v2.json",
    "event_definition_train_v1.json",
    "experiment_audit.json",
    "factorial_physical_fold1_formal_v1.json",
    "factorial_physical_fold2_formal_v1.json",
    "factorial_physical_summary_v1.json",
    "five_seed_development_summary_v1.json",
    "five_seed_robustness_summary_v1.json",
    "formal_queue_status.json",
    "hypothesis_summary_v1.json",
    *(f"seed{seed}_development_formal_v1.json" for seed in (42, 7, 123, 2024, 2026)),
    *(f"seed{seed}_robustness_formal_v1.json" for seed in (42, 7, 123, 2024, 2026)),
)
FIGURES = (
    "e13_target_blind_cases_fold1.json",
    "e13_target_blind_cases_fold1.pdf",
    "e13_target_blind_cases_fold1.png",
    "e13_target_blind_cases_fold2.json",
    "e13_target_blind_cases_fold2.pdf",
    "e13_target_blind_cases_fold2.png",
    "partc_primary_results.json",
    "partc_primary_results.pdf",
    "partc_primary_results.png",
    "partc_secondary_results.json",
    "partc_secondary_results.pdf",
    "partc_secondary_results.png",
)
LEGACY_ARTIFACTS = (
    "e1_matched_summary_v1.json",
    *(f"e1_trajairnet_7days{fold}_fixed_epoch10_formal_v1.json" for fold in range(1, 5)),
)
PROTOCOLS = (
    "partc_protocol.json",
    "partc_protocol_amendment_20260811.json",
    "partc_protocol_amendment_calibration_20260811.json",
)
NEW_SOURCE_NAMES = (
    "aggregate_partc_factorial.py",
    "aggregate_partc_hypotheses.py",
    "aggregate_partc_robustness.py",
    "aggregate_partc_seeds.py",
    "audit_partc_experiments.py",
    "finalize_partc_package.py",
    "fit_partc_event_definition.py",
    "partc_design.py",
    "partc_evaluate.py",
    "partc_factorial.py",
    "partc_seed_evaluate.py",
    "partc_seed_robustness.py",
    "physical.py",
    "render_partc_cases.py",
    "render_partc_primary.py",
    "render_partc_summary.py",
    "run_partc_queue.py",
    "train_partc_target.py",
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _mean(summary: dict[str, object], section: str, model: str, metric: str) -> float:
    return float(summary[section][model][metric]["mean"])


def _interval(effect: dict[str, object]) -> str:
    low, high = effect["ci95"]
    return f"[{float(low):.6f}, {float(high):.6f}]"


def _gain(effect: dict[str, object]) -> str:
    return f"{100 * float(effect['relative_gain']):.2f}%"


def _main_results(summary: dict[str, object]) -> str:
    rows = []
    labels = {
        "top1_ade": "Top-1 ADE",
        "top1_fde": "Top-1 FDE",
        "minade": "minADE@5",
        "minfde": "minFDE@5",
        "energy_score": "Energy Score",
    }
    for metric in labels:
        effect = summary["paired_hierarchical_bootstrap"][metric]
        rows.append(
            "| "
            + " | ".join(
                (
                    labels[metric],
                    f"{_mean(summary, 'aggregates', 'original_ascent', metric):.6f}",
                    f"{_mean(summary, 'aggregates', 'mabpt_ascent', metric):.6f}",
                    _gain(effect),
                    _interval(effect),
                )
            )
            + " |"
        )
    calibration = summary["fixed_event_calibration"]
    for metric, label in (
        ("event_nll", "Fixed-event NLL"),
        ("event_brier", "Fixed-event Brier"),
        ("mixture_nll", "Continuous mixture NLL"),
    ):
        effect = calibration["paired_hierarchical_bootstrap"][metric]
        rows.append(
            "| "
            + " | ".join(
                (
                    label,
                    f"{_mean(calibration, 'aggregates', 'original_ascent', metric):.6f}",
                    f"{_mean(calibration, 'aggregates', 'mabpt_ascent', metric):.6f}",
                    _gain(effect),
                    _interval(effect),
                )
            )
            + " |"
        )
    return "\n".join(rows)


def _conflict_results(summary: dict[str, object]) -> str:
    conflict = summary["conflict_risk"]
    rows = []
    for metric, label in (
        ("nll", "Conflict NLL"),
        ("brier", "Conflict Brier"),
        ("auprc", "AUPRC"),
        ("recall_at_fixed_fpr", "Recall at fixed training FPR"),
        ("mean_warning_lead_seconds", "Mean warning lead (s)"),
    ):
        original = _mean(conflict, "aggregates", "original_ascent", metric)
        candidate = _mean(conflict, "aggregates", "mabpt_ascent", metric)
        rows.append(f"| {label} | {original:.6f} | {candidate:.6f} |")
    return "\n".join(rows)


def _hypothesis_results(hypotheses: dict[str, object]) -> str:
    rows = []
    for name, result in hypotheses["tests"].items():
        rows.append(
            f"| {name} | {float(result['absolute_gain']):.6f} | "
            f"{_interval(result)} | {float(result['raw_one_sided_p']):.6g} | "
            f"{float(result['holm_adjusted_p']):.6g} | "
            f"{'yes' if result['holm_reject_at_0_05'] else 'no'} |"
        )
    return "\n".join(rows)


def _robustness_results(robustness: dict[str, object]) -> str:
    rows = []
    for name, result in robustness["conditions"].items():
        effect = result["paired_hierarchical_bootstrap"]["energy_score"]
        rows.append(
            f"| {name} | {float(effect['absolute_gain']):.6f} | "
            f"{_interval(effect)} | {_gain(effect)} |"
        )
    return "\n".join(rows)


def _factorial_results(factorial: dict[str, object]) -> str:
    rows = []
    effects = factorial["factorial_effects"]["energy_score"]
    for name, result in effects.items():
        rows.append(
            f"| {name} | {float(result['estimate']):.6f} | "
            f"{_interval(result)} | {float(result['two_sided_bootstrap_p']):.6g} |"
        )
    return "\n".join(rows)


def _results_document(
    summary: dict[str, object],
    hypotheses: dict[str, object],
    robustness: dict[str, object],
    factorial: dict[str, object],
) -> str:
    return f"""# MABPT-ASCENT Part C Results

Package date: 2026-08-11. Values below are generated directly from the packaged
machine-readable artifacts. MABPT-ASCENT is one new ASCENT-derived finite-measure
multimodal aircraft trajectory prediction model; route identifiers are internal
implementation provenance only.

## Five-seed matched development results

Five fixed seeds ({', '.join(str(seed) for seed in summary['seeds'])}) were run
with paired seed and date resampling. Lower is better for all rows.

| Endpoint | Original ASCENT | MABPT-ASCENT | Relative improvement | 95% CI for absolute improvement |
|---|---:|---:|---:|---:|
{_main_results(summary)}

These are development results on previously opened dates, not a fresh sealed
confirmation. The H2 minFDE noninferiority component remains unevaluated because
no operationally justified margin was prespecified before observing results.

## Multiplicity-controlled development hypotheses

Positive absolute gain favors MABPT-ASCENT. Holm correction covers the four
superiority components H1-H4.

| Test | Absolute gain | 95% CI | Raw one-sided p | Holm p | Reject at 0.05 |
|---|---:|---:|---:|---:|:---:|
{_hypothesis_results(hypotheses)}

These p values are retrospective development diagnostics and must not be called
confirmatory p values.

## Conflict-risk surrogate

This is a non-regulatory aircraft-pair surrogate. AUPRC and recall are higher-is-
better; NLL and Brier are lower-is-better. Warning lead is reported without an
improvement claim.

| Endpoint | Original ASCENT | MABPT-ASCENT |
|---|---:|---:|
{_conflict_results(summary)}

## Robustness

Energy improvements use the same perturbation for both models and nested
seed/date bootstrap. Positive values favor MABPT-ASCENT.

| Condition | Absolute Energy gain | 95% CI | Relative gain |
|---|---:|---:|---:|
{_robustness_results(robustness)}

## Unified-module factorial analysis

The eight arms form a paired 2x2x2 analysis within the same model. Effects are
high level minus low level, so a negative Energy effect favors the high level.

| Term | Energy effect | 95% CI | Two-sided bootstrap p |
|---|---:|---:|---:|
{_factorial_results(factorial)}

## Evidence boundary

All locally executable registered experiment groups are complete. The available
historical locked test was opened on 2026-08-05, and no new sealed later-period
or airport cohort exists locally. A fresh cohort is therefore still required
before making confirmatory generalization claims for a Part C submission.
"""


def _evidence_map() -> str:
    return """# Evidence Map

All paper-facing claims concern one historical model package, MABPT-ASCENT.
The selected final probability operator is exact Gibbs permutation transport
with Energy-KL projection; MABPT must not be expanded as "mass-aware" in the
manuscript. Historical route labels identify implementation provenance only.

| Evidence ID | Packaged source | Supports | Evidence class |
|---|---|---|---|
| P001 | `evidence/partc/five_seed_development_summary_v1.json` | Five-seed ADE/FDE, minADE/minFDE, Energy, calibration, conflict and runtime | paired development only |
| P002 | `evidence/partc/hypothesis_summary_v1.json` | Holm-adjusted H1-H4 superiority diagnostics | retrospective development only |
| P003 | `evidence/partc/factorial_physical_summary_v1.json` | Complete 2x2x2 module effects and physical-envelope diagnostics | retrospective development only |
| P004 | `evidence/partc/five_seed_robustness_summary_v1.json` | Perturbation and operating-stratum robustness | paired development only |
| P005 | `evidence/partc/experiment_audit.json` | 13-group local completion and fresh-confirmation boundary | computational audit |
| P006 | `evidence/baselines/e1_matched_summary_v1.json` | Matched-horizon official baseline comparison | reused cross-dataset evidence |
| P007 | `evidence/figures/partc_primary_results.*` | Five-seed, factorial and physical primary panels | generated from P001/P003 |
| P008 | `evidence/figures/partc_secondary_results.*` | Calibration, conflict and robustness panels | generated from P001/P004 |
| P009 | `evidence/figures/e13_target_blind_cases_fold*.{png,pdf,json}` | Target-blind qualitative cases | retrospective development only |
| P010 | `src/mabpt/partc_protocol.json` and amendments | Fixed seeds, endpoints, analysis and documented amendments | registered local protocol |

An accountable author must verify values, units, populations, directions and
citations before submission. No packaged artifact is a new sealed confirmation.
"""


def _claim_boundary() -> str:
    return """# MABPT-ASCENT Claim Boundary

MABPT-ASCENT is the historical identifier for one ASCENT-derived finite-measure
multimodal aircraft trajectory prediction model. The selected probability
operator uses uniform assignment weights, exact Gibbs permutation
marginalization, transported source probabilities and Energy-KL projection.

Supported wording: the complete model may be compared directly with original
matched ASCENT on five-seed development ADE/FDE, minADE/minFDE, Energy Score,
fixed-event NLL/Brier and continuous mixture NLL. Conflict prediction is a
non-regulatory surrogate. Robustness, physical-envelope and factorial results
are development analyses.

Required qualifications:

- the locally available dates and external views have already been opened;
- the H2 minFDE noninferiority component has no prespecified operational margin;
- exact permutation enumeration is justified only for small mode counts;
- null or adverse calibration, warning-lead, ensemble and module-ablation
  results must be reported wherever they remain in the final artifacts; and
- a new sealed later-period or airport cohort is required for confirmatory
  generalization claims.

The mass-weighted assignment control remains historical ablation evidence. Its
independent development contribution was not established, so it is not part of
the selected algorithm identity and must not be claimed as an innovation.

Do not claim regulatory validity, universal novelty, unrestricted transfer or
fresh confirmation from this package.
"""


def _methods_map() -> str:
    return """# MABPT-ASCENT Methods Map

The manuscript describes one complete model: **MABPT-ASCENT**. Internal route
identifiers record module provenance and are not model names.

| Unified model module | Primary packaged implementation | Paper role |
|---|---|---|
| ASCENT encoder and native decoder | `src/model/` | Motion encoding and native K=5 prediction |
| Physical dual-objective support | `src/experiments/metric_exact/`, `src/experiments/joint_coupled/` | Positive-speed, pitch-aware ADE/FDE support generation |
| Decision-aware top-1 output | `src/experiments/decision_regret/` | Deployment decision separated from probability argmax |
| Predicted finite-measure risk | `src/experiments/energy_predict_optimize/` | Target-free risk and support-diversity prediction |
| Retained enhanced branch | `src/experiments/ascent_recomparison/` | Joint support, decision and risk inference |
| Exact Gibbs support fusion | `src/mabpt/operator.py` | Uniform-cost Gibbs permutation marginalization and source-probability transport |
| Energy-KL projection | `src/mabpt/operator.py` | Unique convex probability solve on the enhanced support |
| Part C evaluation | `src/mabpt/partc_*.py`, `src/mabpt/aggregate_partc_*.py` | Five-seed geometry, calibration, robustness, physical and conflict evaluation |

The main comparison is complete MABPT-ASCENT versus original matched ASCENT.
Module variants belong in ablations named by the removed mechanism.

| Result group | Evidence of record |
|---|---|
| Five-seed main comparison | `evidence/partc/five_seed_development_summary_v1.json` |
| H1-H4 multiplicity analysis | `evidence/partc/hypothesis_summary_v1.json` |
| 2x2x2 module and physical analysis | `evidence/partc/factorial_physical_summary_v1.json` |
| Robustness and operating strata | `evidence/partc/five_seed_robustness_summary_v1.json` |
| Matched official baseline | `evidence/baselines/e1_matched_summary_v1.json` |
| Completion and evidence boundary | `evidence/partc/experiment_audit.json` |
"""


def _experiment_report(audit: dict[str, object]) -> str:
    rows = []
    for name, group in audit["groups"].items():
        rows.append(
            f"| {name} | {'complete' if group['local_computation_complete'] else 'pending'} | "
            f"{group['evidence_class']} |"
        )
    rows_text = "\n".join(rows)
    return f"""# Part C Experiment Completion Report

Paper model: MABPT-ASCENT.

| Experiment group | Local status | Evidence class |
|---|---|---|
{rows_text}

Local completion: {audit['local_groups_complete']}/{audit['local_groups_total']}.

Fresh confirmatory evidence: **not complete**. {audit['fresh_confirmatory_evidence']['reason']}
The required next input is a {audit['fresh_confirmatory_evidence']['required_next_input']}.
"""


def _readme(audit: dict[str, object]) -> str:
    return f"""# MABPT-ASCENT Part C Paper Kit

Package date: 2026-08-11.

This package is organized around one paper model: **MABPT-ASCENT**, a new
ASCENT-derived finite-measure multimodal aircraft trajectory prediction model.
Historical `C...` route labels are retained only for reproducibility and must
not be presented as independent paper models.

Start with `paper/UNIFIED_MODEL.md`, `paper/UNIFIED_RESULTS.md`,
`paper/PARTC_EXPERIMENT_REPORT.md`, `paper/EVIDENCE_MAP.md`, and
`CLAIM_BOUNDARY.md`. Machine-readable formal results are under
`evidence/partc/`; figures are under `evidence/figures/`; the observed software
environment and reference-environment deviation are in
`environment/FORMAL_EXECUTION_ENVIRONMENT.md`.

All {audit['local_groups_total']} locally executable registered experiment
groups are complete. This does not supply a fresh sealed confirmation: the
available locked cohort was opened previously and no new temporal or airport
cohort is present locally.

From the extracted package root, verify integrity with:

```sh
sha256sum -c MANIFEST.sha256
PYTHONPATH=src python -m pytest -q src/mabpt/tests
python -m compileall -q src
```

Datasets and large checkpoints are omitted. Redistribution rights are not
granted; see `LICENSING_NOTICE.md`. Accountable authors must verify all numeric
claims and complete authorship, ethics, funding, conflict, data/code availability
and AI-use disclosures before submission.
"""


def _environment_document(
    seed_payload: dict[str, object], training_summary: dict[str, object]
) -> str:
    evaluation = seed_payload["runtime"]
    training = training_summary["runtime"]
    return f"""# Formal Execution Environment

The formal five-seed extension was executed with the following observed
software stack, as recorded in the machine-readable run receipts:

- Python: `{evaluation['python']}`
- PyTorch: `{evaluation['torch']}`
- PyTorch CUDA build: `{evaluation['cuda']}`
- Example evaluation device locator: `{evaluation['device']}`
- Example training device locator: `{training['device']}`

The repository reference environment documents Python 3.11, PyTorch 2.8 and
CUDA 12.8. The formal extension therefore has an environment deviation and
must not be described as a bitwise reproduction of that reference stack.
Deterministic algorithms, fixed seeds, disabled AMP and disabled TF32 were
retained by the formal protocol. Full per-run timing and peak-memory receipts
are packaged under `evidence/partc/` and `evidence/training_receipts/`.
"""


def _copy_evidence(kit: Path) -> None:
    for name in PARTC_ARTIFACTS:
        _copy(PARTC / name, kit / "evidence/partc" / name)
    for name in FIGURES:
        _copy(PARTC / "figures" / name, kit / "evidence/figures" / name)
    for name in LEGACY_ARTIFACTS:
        _copy(LEGACY / name, kit / "evidence/baselines" / name)
    for name in PROTOCOLS:
        _copy(ROOT / "mabpt" / name, kit / "src/mabpt" / name)
    for name in NEW_SOURCE_NAMES:
        _copy(ROOT / "mabpt" / name, kit / "src/mabpt" / name)
    _copy(ROOT / "mabpt/tests/test_partc.py", kit / "src/mabpt/tests/test_partc.py")
    for seed in (42, 7, 123, 2024, 2026):
        for stage in ("decision_support", "predicted_risk"):
            source = (
                ROOT
                / "runs/mabpt_partc_20260811"
                / f"mabpt_ascent_{stage}_seed{seed}_formal"
            )
            for name in ("config.json", "training_summary.json"):
                _copy(
                    source / name,
                    kit / "evidence/training_receipts" / source.name / name,
                )
    baseline = ROOT / "runs/mabpt_official_baselines/trajairnet_7days1_seed42_formal"
    for name in ("config.json", "training_summary.json"):
        _copy(baseline / name, kit / "evidence/training_receipts" / baseline.name / name)


def _write_manifests(kit: Path) -> None:
    source_lines = []
    for path in sorted((kit / "src").rglob("*")):
        if path.is_file():
            source_lines.append(f"{_sha256(path)}  {path.relative_to(kit).as_posix()}")
    (kit / "SOURCE_MANIFEST.sha256").write_text(
        "\n".join(source_lines) + "\n", encoding="utf-8"
    )
    content_paths = sorted(
        path
        for path in kit.rglob("*")
        if path.is_file() and path.name not in {"CONTENTS.txt", "MANIFEST.sha256"}
    )
    (kit / "CONTENTS.txt").write_text(
        "\n".join(
            f"{path.stat().st_size}\t{path.relative_to(kit).as_posix()}"
            for path in content_paths
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_paths = sorted(
        path for path in kit.rglob("*") if path.is_file() and path.name != "MANIFEST.sha256"
    )
    (kit / "MANIFEST.sha256").write_text(
        "\n".join(
            f"{_sha256(path)}  ./{path.relative_to(kit).as_posix()}"
            for path in manifest_paths
        )
        + "\n",
        encoding="utf-8",
    )


def _make_archive(kit: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=archive.parent, prefix=archive.stem + ".", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as bundle:
            for path in sorted(kit.rglob("*")):
                if path.is_file():
                    bundle.write(path, (Path(kit.name) / path.relative_to(kit)).as_posix())
        with zipfile.ZipFile(temporary) as bundle:
            corrupt = bundle.testzip()
            if corrupt is not None:
                raise RuntimeError(f"archive CRC failure: {corrupt}")
        temporary.replace(archive)
    finally:
        if temporary.exists():
            temporary.unlink()


def finalize(kit: Path, archive: Path, old_archive: Path | None) -> dict[str, object]:
    audit = _load(PARTC / "experiment_audit.json")
    if not audit.get("all_local_computation_complete"):
        raise RuntimeError("the 13-group local experiment audit is not complete")
    queue = _load(PARTC / "formal_queue_status.json")
    if queue.get("phase") != "complete" or queue.get("pending"):
        raise RuntimeError("formal GPU queue is not complete")
    summary = _load(PARTC / "five_seed_development_summary_v1.json")
    hypotheses = _load(PARTC / "hypothesis_summary_v1.json")
    robustness = _load(PARTC / "five_seed_robustness_summary_v1.json")
    factorial = _load(PARTC / "factorial_physical_summary_v1.json")
    if any(
        payload.get("model") != "MABPT-ASCENT"
        for payload in (summary, hypotheses, robustness, factorial)
    ):
        raise RuntimeError("paper-model identity mismatch in formal summaries")

    _copy_evidence(kit)
    (kit / "paper/UNIFIED_RESULTS.md").write_text(
        _results_document(summary, hypotheses, robustness, factorial), encoding="utf-8"
    )
    (kit / "paper/EVIDENCE_MAP.md").write_text(_evidence_map(), encoding="utf-8")
    (kit / "paper/PARTC_EXPERIMENT_REPORT.md").write_text(
        _experiment_report(audit), encoding="utf-8"
    )
    (kit / "CLAIM_BOUNDARY.md").write_text(_claim_boundary(), encoding="utf-8")
    (kit / "METHODS_MAP.md").write_text(_methods_map(), encoding="utf-8")
    (kit / "README_PAPER_KIT.md").write_text(_readme(audit), encoding="utf-8")
    seed42 = _load(PARTC / "seed42_development_formal_v1.json")
    training42 = _load(
        ROOT
        / "runs/mabpt_partc_20260811"
        / "mabpt_ascent_predicted_risk_seed42_formal"
        / "training_summary.json"
    )
    (kit / "environment/FORMAL_EXECUTION_ENVIRONMENT.md").write_text(
        _environment_document(seed42, training42), encoding="utf-8"
    )
    (kit / "src/mabpt/EXPERIMENT_STATUS.md").write_text(
        _experiment_report(audit), encoding="utf-8"
    )
    _write_manifests(kit)
    _make_archive(kit, archive)
    if old_archive is not None and old_archive != archive and old_archive.is_file():
        old_archive.unlink()
    return {
        "model": "MABPT-ASCENT",
        "kit": str(kit),
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": _sha256(archive),
        "manifest_entries": sum(
            1 for _ in (kit / "MANIFEST.sha256").open(encoding="utf-8")
        ),
        "old_archive_removed": old_archive is not None and not old_archive.exists(),
        "local_experiment_groups": f"{audit['local_groups_complete']}/{audit['local_groups_total']}",
        "fresh_confirmatory_complete": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, default=DEFAULT_KIT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--old-archive", type=Path, default=DEFAULT_OLD_ARCHIVE)
    args = parser.parse_args()
    result = finalize(
        args.kit.resolve(), args.archive.resolve(), args.old_archive.resolve()
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
