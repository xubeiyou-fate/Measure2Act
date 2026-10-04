# Public asset boundary

This GitHub repository contains source and small metadata only.

| Asset family | GitHub code repository | Separate release record |
|---|---:|---:|
| Author-maintained source and protocols | yes | optional archive copy |
| Prior aircraft-forecasting architecture | no source is vendored; implementation is independent | see the single upstream-reference notice for attribution |
| Public synthetic tests | yes | no |
| Aggregate paper-table CSVs | yes | archive with software release |
| Raw or processed third-party trajectories | no | retrieve from official custodian |
| Derived trajectory pools/case bundles | no | internal preservation only |
| Per-flight/per-window outputs | no | internal preservation only |
| Model checkpoints | no | model record |
| Formal logs, receipts, and large intermediate tables | no | internal preservation only |
| Official EqMotion source | no | retrieve from upstream at frozen commit |

The planned external weight set, model index, and external-record gate are
described in [`model_release.json`](../model_release.json) and
[`MODEL_RELEASE.md`](MODEL_RELEASE.md). Those descriptors contain no weights.

The `.gitignore` enforces the top-level data/model boundary, and
`scripts/audit_code_release.py` fails if a forbidden payload, archive, nested
dataset directory, or oversized file is introduced. Official access metadata
are in [`DATASETS.md`](DATASETS.md) and
[`dataset_sources.csv`](dataset_sources.csv).
