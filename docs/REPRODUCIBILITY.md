# Reproducibility contract

## Level 1: source-only verification

No data or weights are needed:

```bash
python -m pip install -c constraints/requirements-cpu.txt ".[test]"
python -m pytest -q
python -m Measure2Act_probability_transfer.run --smoke
python measure2act_ast_tools/runtime_solver_audit.py --help
```

This verifies imports, simplex preservation, the finite-measure operators, and
the packaged public entrypoints. It does not reproduce manuscript estimates.

## Level 2: aggregate-table reconstruction

The release includes small, non-trajectory CSVs for the reported values in
Tables 3--7. Verify all expected files and exact SHA256 values with:

```bash
python scripts/verify_paper_summaries.py
```

The corresponding study aggregation implementations are:

| Manuscript evidence | Source entrypoint |
|---|---|
| Main/extended suite | `measure2act_ast_tools/aggregate_ast_extended_suite.py` |
| Fixed-support controls | `measure2act_ast_tools/aggregate_fixed_t_negative_controls.py` |
| Two-predictor capacity control | `measure2act_ast_tools/aggregate_two_ascent_budget.py` |
| EqMotion support transfer | `measure2act_ast_tools/aggregate_eqmotion_support_transfer.py` |
| AWTA and probability controls | `experiments/journal_extension/aggregate_*.py` |

The public CSVs are final numerical summaries, not raw observations and not a
replacement for full reanalysis. Per-case trajectories, intermediate
predictions, formal receipts, and the internal closure archive are not hosted
on GitHub. Their absence does not prevent exact checksum verification of the
reported table values, but it does prevent recomputing those values from the
lowest-level observations using this repository alone.

## Level 3: full evaluation or retraining

Requires the official upstream datasets plus the external model-weight record.
Follow [DATASETS.md](DATASETS.md) and [DATA_SOURCES.md](DATA_SOURCES.md), then
materialize the acquired data and published weights under a single read-only
asset root and set:

```bash
export MEASURE2ACT_ASSET_ROOT=/path/to/materialized/assets
```

The asset root must provide the relative paths declared by the frozen JSON
protocols, including `dataset/` and `checkpoints/` as applicable. Generated
`artifacts/` and `runs/` belong in a separate writable output directory. Never
point formal evaluation at mutable exploratory directories.

The main study uses TartanAviation KAGC/KBTP, target-only/full-finetune regimes,
five paired seeds (42, 7, 123, 2024, 2026), and K=5. The model record must map
each published role to an exact configuration, seed, protocol, and SHA256.

EqMotion evaluation additionally requires the upstream implementation at the
exact commit recorded by the published protocol. That upstream source is not
vendored here; its license and commit must be verified from the custodian.

## Determinism boundary

The probability-operator smoke is deterministic on CPU. Full neural training
may depend on framework, hardware, and kernel versions. Every formal rerun
must capture Python/PyTorch versions, hardware, seed, configuration, input
manifest, output manifest, runtime, and peak memory.
