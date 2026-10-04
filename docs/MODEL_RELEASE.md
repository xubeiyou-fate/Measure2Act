# External model release contract

Model binaries are not stored in Git history. The machine-readable
[`model_release.json`](../model_release.json) records the public GitHub Release
asset and deliberately uses `null` for an optional DOI that has not been
minted. `null` is not a DOI and must not be copied into a paper.

The public route is the versioned GitHub Release attachment for direct
download. A separate DOI-backed model archive is optional for persistent
citation and can be added later. The v1.0.0 asset is released under CC BY 4.0.

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

Verify the GitHub asset's manifest and SHA256 against the staged deposit. The
code repository must continue to contain no checkpoint payloads. The archive-
side weight terms are CC BY 4.0 as stated in
`deposits/models/MODEL_WEIGHTS_LICENSE.md`; this does not grant rights to
upstream data, source code, or official baseline weights.
