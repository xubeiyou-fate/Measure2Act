# Data sources and GitHub boundary

This public repository does not contain third-party raw data, processed data,
or a derived trajectory dataset. It contains source code, official access
links, frozen source versions, preprocessing entry points, small aggregate
paper evidence, and data-free tests.

The machine-readable source registry is
[`data_sources.json`](data_sources.json). Checksums identify the exact upstream
assets used in the study; they do not grant redistribution rights.

## TrajAir

- Official versioned record: <https://doi.org/10.1184/R1/14866251.v1>
- KiltHub landing page:
  <https://kilthub.cmu.edu/articles/dataset/TrajAir_A_General_Aviation_Trajectory_Dataset/14866251>
- Record version used by the upstream download route: `1`
- Dataset license shown by KiltHub: `CC BY 4.0`
- Official supporting code: <https://github.com/castacks/trajairnet>

The versioned DOI is the dataset identifier. The ICRA article DOI is a publication
identifier and must not replace it in the Data Availability statement.

The KiltHub v1 API exposes direct official downloads for `111_days.zip` and
`7days1.zip` through `7days4.zip`. Their file IDs, URLs, official MD5 values,
and the study's SHA256 values are recorded in `data_sources.json` and
`dataset_sources.csv`.

## TartanAviation

- Official project page: <https://theairlab.org/tartanaviation/>
- Official source and downloader repository:
  <https://github.com/castacks/TartanAviation>
- Paper-frozen repository commit:
  `4065f5bb11c3d8e557dcaf20a56469e6b0738714`
- Official ADS-B downloader: `adsb/download.py`
- Official ADS-B preprocessor: `adsb/process.py`
- Zenodo software snapshot: <https://doi.org/10.5281/zenodo.14699102>,
  version `1.0`, resource type `Software`

The Zenodo record archives approximately 4.2 MB of repository software; it is
not a DOI for the ADS-B data payload. The repository's BSD-3-Clause license is
an explicit source-code license. The reviewed official project/repository
pages do not state a separate machine-readable license for the downloaded
trajectory payload, so this repository does not redistribute it or apply a
new license to it.

## Download outside the clone

The wrapper is dry-run by default and refuses to place data inside the GitHub
working tree:

```bash
python scripts/fetch_official_data.py --list
python scripts/fetch_official_data.py \
  --dataset trajair \
  --asset 111_days.zip \
  --destination /path/to/external-data \
  --accept-upstream-terms \
  --execute
```

For TartanAviation, install the dependencies declared by its official
downloader, then use:

```bash
python scripts/fetch_official_data.py \
  --dataset tartanaviation \
  --destination /path/to/external-data \
  --tartan-option Raw \
  --location Both \
  --accept-upstream-terms \
  --execute
```

The wrapper clones the official repository, checks out the paper-frozen
commit, and invokes its official ADS-B downloader. Availability of the
custodian's server remains outside this project's control.

## Project preprocessing

For the chronological KAGC/KBTP study representation, pass explicit archive
paths and an output path outside the Git checkout:

```bash
python scripts/build_tartan_target_domain.py \
  --archive KAGC=/path/to/tartan_kagc_raw_2022.zip \
  --archive KBTP=/path/to/tartan_kbtp_raw_2022.zip \
  --output /path/to/external-data/measure2act_tartan_target_domain
```

TrajAir processed data can be consumed directly by the forecasting entry
points. Raw-to-processed TrajAir conversion belongs to the official upstream
`adsb_preprocess` utilities so its upstream version and terms remain visible.

## DOI decision

A separate Measure2Act **data DOI is not required** for this GitHub strategy:
the reused TrajAir dataset already has its official versioned DOI,
TartanAviation is
accessed from its official custodian, and the small author-generated aggregate
table evidence is archived with the software release.

Create a separate data DOI only if the authors later publish the internal
per-cell/closure evidence as a distinct reusable dataset. Do not upload the
internal `deposits/data` mother archive to GitHub, and do not cite a data DOI
that has not actually been registered.
