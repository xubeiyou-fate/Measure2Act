# Release audit snapshot

Audit date: 2026-10-04 (Asia/Shanghai). This snapshot records the public
v1.0.1 GitHub release. It does not fabricate a DOI or external archive record.

## Passed checks

| Check | Result |
|---|---|
| Source-only boundary (`scripts/audit_code_release.py`) | PASS |
| GitHub candidate layout (`scripts/audit_release_readiness.py`) | PASS |
| Public CPU tests | PASS, 6 tests |
| Paper aggregate summaries | PASS, 5 files and 38 rows; SHA256 matched |
| Clean-clone audit and tests | PASS |
| Wheel build | PASS; root Apache-2.0 license included in `.dist-info` |
| Forbidden payload scan | PASS; no raw datasets, checkpoints, archives, or local absolute paths in the candidate |
| Manifest | PASS; `MANIFEST.sha256` covers 405 source-only files and excludes caches/build output |

## Optional persistent identifiers

The strict metadata audit requires a release date and repository identity. A
software or model DOI is optional until an external archive returns a real
identifier. If created later, add it consistently to the manuscript and
repository metadata.

The model-index provenance and source/model boundary are recorded in the model
card and the single upstream-reference notice. The source tree and model asset
are uploaded and CI-verified; the release must be described as a GitHub
release, not as a DOI-tagged archive.
