# Release audit snapshot

Audit date: 2026-10-04 (Asia/Shanghai). This snapshot records the state of
the source-only GitHub candidate. It does not create a repository owner, DOI,
or licence approval that has not been supplied by the authors.

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
| Manifest | PASS; `MANIFEST.sha256` covers 404 source-only files and excludes caches/build output |

## Deliberate release gates

`python scripts/audit_release_readiness.py --strict` remains FAIL until the
authors provide the following real values or approvals:

1. The software archive DOI.
2. A public model record URL and DOI for the 70 checkpoint rows. The external
   model deposit is technically complete; its separate publication-readiness
   report still contains author-managed release fields.
3. The release date and matching software/model identifiers in the manuscript
   and Data/Code Availability statements.

These are metadata gates, not failing code tests. The ASCENT implementation
provenance is now recorded in the model index and model card. The separate
license matrix remains available for the authors' publication review but is
outside this technical-only audit. Until the metadata gates are closed, the
repository may be uploaded as an initial source-only GitHub candidate, but it
must not be presented as a finalized DOI-tagged `v1.0.0` release.
