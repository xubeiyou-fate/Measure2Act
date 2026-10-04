# External model release contract

Model binaries are not stored in Git history. The machine-readable
[`model_release.json`](../model_release.json) records the external model object
and deliberately uses `null` for identifiers that have not yet been minted.
Those nulls are a release gate, not a DOI and must not be copied into a paper.

The intended public route is a versioned GitHub Release attachment for direct
download, with a separate DOI-backed model archive for persistent citation.
As of 2026-10-04 neither the GitHub asset nor the model DOI has been published;
the verified archive remains local until its public metadata and rights
decisions are consistent.

The model deposit contains 60 core formal role checkpoints
(two airports, two regimes, five seeds, and three roles) and 10 EqMotion
support-control checkpoints (two airports and five seeds). Its authoritative
`paper_model_index.csv` must bind every file to its paper role, seed,
configuration, protocol, byte size, and SHA256. `MANIFEST.sha256` covers the
entire deposit, and `MODEL_CARD.md` records intended use, training-data
provenance, limitations, and model-specific terms.

The 60 core rows are bound to the independently authored Measure2Act source
repository and release commit recorded in `paper_model_index.csv`. ASCENT is
an architecture reference only (`https://github.com/a-pru/ascent`); no ASCENT
source or official ASCENT checkpoint is redistributed. The canonical
description is **ASCENT-inspired / architecture-informed independently
authored implementation**. Measure2Act-specific probability-transfer
operators, experiment adapters, and evaluation protocols remain in this source
repository.

Before release, create the public model record, replace `doi` and `record_url`
with the returned values, verify the published manifest against the staged
deposit, and update the README, `CITATION.cff`, and manuscript availability
statement consistently. The code repository must continue to contain no
checkpoint payloads. The archive-side weight terms are CC BY 4.0 as stated in
`deposits/models/MODEL_WEIGHTS_LICENSE.md`; this does not grant rights to
upstream data, source code, or official baseline weights.
