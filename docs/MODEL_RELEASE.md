# External model release contract

Model binaries are not stored in Git history. The machine-readable
[`model_release.json`](../model_release.json) records the external model object.
In the release candidate, its unresolved fields remain explicit and do not
assert that a DOI or public URL already exists.

The model deposit contains 60 core formal role checkpoints
(two airports, two regimes, five seeds, and three roles) and 10 EqMotion
support-control checkpoints (two airports and five seeds). Its authoritative
`paper_model_index.csv` must bind every file to its paper role, seed,
configuration, protocol, byte size, and SHA256. `MANIFEST.sha256` covers the
entire deposit, and `MODEL_CARD.md` records intended use, training-data
provenance, limitations, and model-specific terms.

The 60 core rows are bound to the official ASCENT reproduction source at
`https://github.com/a-pru/ascent`, commit
`814e0a18a8a7500dfb0498ab2ee873d022874e8`. Measure2Act-specific
probability-transfer operators, experiment adapters, and evaluation protocols
remain in this source repository; the binding is provenance metadata, not a
claim that every local file is byte-identical to upstream.

Before release, replace both model-record placeholders, verify the published
manifest against the staged deposit, approve model and upstream-weight terms,
and update the README, `CITATION.cff`, and manuscript availability statement
consistently. The code repository must continue to contain no checkpoint
payloads.
