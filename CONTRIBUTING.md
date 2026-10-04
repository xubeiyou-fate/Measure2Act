# Contributing

Measure2Act is a research-software release tied to frozen manuscript evidence.
Contributions are welcome when they preserve that distinction.

Participation is subject to [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Development setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -c constraints/requirements-cpu.txt ".[test]"
```

## Before opening a pull request

Run the same data-free checks used in continuous integration:

```bash
python scripts/audit_code_release.py
python scripts/audit_release_readiness.py
python -m pytest -q
python -m Measure2Act_probability_transfer.run --smoke
python -m build
```

Document behavior changes and update tests. Keep new user-facing functionality
behind the stable `Measure2Act_*` packages where practical.

## Research asset boundary

Do not commit datasets, derived cases, per-flight outputs, model checkpoints,
run directories, credentials, private links, or developer-local absolute
paths. Raw data remain at official upstream records; weights are versioned as
a GitHub Release asset and may receive a DOI later. Do not copy
third-party source into this repository without recording its exact origin,
commit, licence, and notice requirements.

The numbered `c*` modules and frozen JSON protocols are evidence-linked. Avoid
renaming or rewriting them unless the change is explicitly versioned as a new
protocol and the corresponding evidence is regenerated.

## Reporting results

New experimental claims must identify the dataset version, split, model role,
configuration, seed, aggregation rule, and exact code revision. Do not replace
the archived paper outputs in place.
