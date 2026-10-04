# ASCENT-inspired implementation notice

Measure2Act is **not** a redistribution of the ASCENT source tree and the
released Measure2Act weights are **not official ASCENT weights**. The
aircraft-forecasting implementation in this repository was independently
authored for Measure2Act, informed by the architectural ideas described in:

- Prutsch et al., *ASCENT: Transformer-Based Aircraft Trajectory Prediction in
  Non-Towered Terminal Airspace*;
- the public ASCENT project page: <https://github.com/a-pru/ascent>.

The reference is cited for scientific context and architectural comparison
(positional/angular motion representation, context encoding, multimodal mode
queries, and flight-parameter decoding). It is not a source-code dependency
for the Measure2Act implementation. No ASCENT source files, checkpoints,
copyright notices, or upstream license are redistributed by this repository or
by the separate Measure2Act model archive.

## Boundary of authorship

- `model/`, `Measure2Act_forecasting/`, and the probability-transfer packages
  are independently authored project code and are covered by the root
  Apache-2.0 license.
- The model archive contains author-trained Measure2Act weights and auxiliary
  EqMotion adaptations. Their exact files, hashes, configurations, and roles
  are listed in `paper_model_index.csv` in the model archive.
- The auxiliary EqMotion experiment is a separately documented baseline. Its
  upstream source remains available from
  <https://github.com/MediaBrain-SJTU/EqMotion> under its own terms; the
  upstream source is not vendored in this repository.

The canonical wording for the code and model provenance is **ASCENT-inspired / architecture-informed independently authored implementation**.
It must be used consistently in the manuscript, README, model card, Data/Code
Availability statements, and repository metadata.
