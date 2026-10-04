# Third-party notices and dependency boundary

This repository contains independently authored Measure2Act code informed by
published aircraft-trajectory architectures. It does not redistribute any
upstream source tree or official baseline checkpoint. The root Apache-2.0
license applies to original Measure2Act source, tests, documentation, and
project-authored figures only.

## Architecture reference

The single attribution and authorship boundary is documented in
[`docs/ARCHITECTURE_REFERENCE_NOTICE.md`](docs/ARCHITECTURE_REFERENCE_NOTICE.md).

## EqMotion baseline

The `modern_baseline/` directory contains Measure2Act adapters and frozen
protocols. It does not contain the official EqMotion source tree. The protocols
pin <https://github.com/MediaBrain-SJTU/EqMotion.git> at commit
`5aec2e0b61c511fa93a24138dd90da59a089084b`. Retrieve that source separately
under its MIT terms when running the optional baseline. MIT source terms do not
automatically license locally trained weights.

## Datasets

TrajAir is not bundled. The study cites KiltHub version 1,
`10.1184/R1/14866251.v1`, and records official download URLs and checksums in
`docs/dataset_sources.csv`.

TartanAviation raw ADS-B data are not bundled. The study pins the official
repository at commit `4065f5bb11c3d8e557dcaf20a56469e6b0738714` and delegates
acquisition to its official downloader and preprocessing scripts. The
repository source license does not establish a separate license for the
downloaded payload; the raw payload must not be redistributed without the
custodian's permission.

## Model weights

The separate Measure2Act model archive applies CC BY 4.0 to the author-created
weight files only, subject to the release declaration in its
`MODEL_WEIGHTS_LICENSE.md`. The license does not grant rights to third-party
datasets, upstream source code, or official baseline weights.
