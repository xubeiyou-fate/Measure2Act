# Third-party notices and dependency boundary

This repository depends on third-party Python packages declared in
`pyproject.toml`. Their copyright and license terms remain with their
respective maintainers. The generated SBOM records the resolved versions used
for a release build.

The `modern_baseline/` directory contains the project's aviation adapters and
protocols. It does not contain the official EqMotion source tree. The frozen
protocol records `https://github.com/MediaBrain-SJTU/EqMotion.git` at commit
`5aec2e0b61c511fa93a24138dd90da59a089084b` under its MIT source-code
license. Preserve that notice when retrieving the upstream implementation;
the MIT source license does not by itself license locally trained weights.

The forecasting source uses the publicly released ASCENT implementation and
the ASCENT architecture described by Prutsch et al. The source is pinned to
commit `814e0a18a8a7500dfb0498ab2ee873d022874e8` of
`https://github.com/a-pru/ascent`. The release owner has confirmed permission
to redistribute this upstream component. ASCENT copyright, attribution, and
upstream terms remain applicable; the root Measure2Act Apache-2.0 licence does
not relicense ASCENT files. See `docs/ASCENT_NOTICE.md` for the exact scope.
ASCENT checkpoints are not stored in this code repository and are governed by
the separate model record.

The ASCENT utility source retains its upstream provenance comment identifying
the DAG-Net and TrajAirNet projects. TrajAirNet's source repository publishes
the BSD-4-Clause licence; this repository does not vendor the TrajAirNet source
tree, and any redistributed derivative must retain its advertising clause. The
DAG-Net repository did not expose an explicit licence file on the verified
upstream page. Consequently, the ASCENT redistribution authorization must be
kept with the release records and must cover the inherited utility provenance;
otherwise `model/utils.py` must be replaced by an independently implemented
loader before a final version tag. The complete decision and evidence fields
are in `docs/THIRD_PARTY_LICENSE_MATRIX.md`.

TrajAir is not bundled. The study identifies KiltHub version 1 by the
versioned DOI `10.1184/R1/14866251.v1`; its official dataset record states
CC BY 4.0. The five official archive URLs and checksums are recorded in
`docs/dataset_sources.csv` solely to support acquisition from the custodian.

TartanAviation data are not bundled. The study pins the official repository
`https://github.com/castacks/TartanAviation.git` at commit
`4065f5bb11c3d8e557dcaf20a56469e6b0738714` and delegates acquisition and
preprocessing to `adsb/download.py` and `adsb/process.py`. BSD-3-Clause applies
to the verified repository source. No separate data-payload licence was found
on the verified official pages, so the source-code licence must not be treated
as permission to redistribute downloaded ADS-B payloads. Zenodo DOI
`10.5281/zenodo.14699102` describes a Software record, not a dataset DOI.

The complete dataset access and rights boundary is documented in
`docs/DATASETS.md` and `docs/DATA_SOURCES.md`.

The root `LICENSE` covers original Measure2Act material only. Third-party
source, datasets, and model weights retain their respective terms. No dataset
or model-weight redistribution permission is implied by this notice.
