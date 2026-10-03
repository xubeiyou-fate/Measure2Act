# Dataset access matrix

The public GitHub release is source-only with small aggregate manuscript
evidence. It does not redistribute a third-party raw or derived dataset.

| Dataset | Role | Official access | Frozen version | License status | Hosted here |
|---|---|---|---|---|---:|
| TrajAir | Reused KBTP trajectory data and public views | [KiltHub v1](https://doi.org/10.1184/R1/14866251.v1) | KiltHub version 1 | CC BY 4.0 on dataset record | no |
| TartanAviation ADS-B | KAGC/KBTP chronological trajectory data | [AirLab project](https://theairlab.org/tartanaviation/) and [official repository](https://github.com/castacks/TartanAviation) | commit `4065f5bb11c3d8e557dcaf20a56469e6b0738714` | BSD-3-Clause for source code; data-payload license not separately stated on verified pages | no |

TrajAir's `111_days.zip` and `7days1.zip` through `7days4.zip` are linked
individually in [`dataset_sources.csv`](dataset_sources.csv), with KiltHub's
official MD5 and the study's SHA256 for each file. TartanAviation acquisition
is delegated to the custodian's `adsb/download.py`; project preprocessing is
documented in [`DATA_SOURCES.md`](DATA_SOURCES.md).

The aggregate outputs under `results/paper_tables/` are numerical paper
evidence, not a trajectory dataset. The internal `deposits/data` archive is a
private preservation mother copy and is not part of the GitHub upload.

No separate Measure2Act data DOI is needed under this policy. A software DOI
must archive the tagged GitHub release. Create a data DOI only if a separately
curated author-generated evidence dataset is actually released later.
