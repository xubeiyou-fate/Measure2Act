# Dataset access matrix

The Git-tracked repository contains source and small aggregate manuscript
evidence. Model weights are a separate GitHub Release asset. The project does
not redistribute a third-party raw or derived dataset.

| Dataset | Role in this study | Official access | Frozen version | License status | Hosted here |
|---|---|---|---|---|---:|
| TrajAir | Reused KBTP trajectory data and public-view comparisons; see the frozen protocols for each experiment's exact use | [KiltHub v1](https://doi.org/10.1184/R1/14866251.v1) | KiltHub version 1 | CC BY 4.0 on dataset record | no |
| TartanAviation ADS-B | Main chronological KAGC/KBTP trajectory study | [AirLab project](https://theairlab.org/tartanaviation/) and [official repository at the paper-frozen commit](https://github.com/castacks/TartanAviation/tree/4065f5bb11c3d8e557dcaf20a56469e6b0738714) | commit `4065f5bb11c3d8e557dcaf20a56469e6b0738714` | BSD-3-Clause for source code; data-payload license not separately stated on verified official pages | no |

## Exact assets and checksums

The following TrajAir v1 files are the assets indexed by the study's source
registry. Use the KiltHub record for the complete dataset record and terms;
direct links below point to the official file objects.

| Official file | Official download | Official MD5 | Study SHA256 |
|---|---|---|---|
| `111_days.zip` | [download](https://ndownloader.figshare.com/files/28625106) | `50dc9f4d271da435b6f1be2d13e70202` | `2887bb00c1af7a1ea7c4f9434709d7ae66c95eafaf5cbf41712deb03ec929877` |
| `7days1.zip` | [download](https://ndownloader.figshare.com/files/28625121) | `ea45667239cc503767dc4baa9d125066` | `d058adbbc19393bdb08ace91e0505933c858b626e5098ed87dd7645ab045485c` |
| `7days2.zip` | [download](https://ndownloader.figshare.com/files/28625118) | `076085a72cba1b45ba90357fda666ed4` | `913b5b9bd69eaa344fff309058b85671ce19e54ac53d8252d838988a67ca8fdf` |
| `7days3.zip` | [download](https://ndownloader.figshare.com/files/28625115) | `5fee0d141054e7783e74d9f506b6198e` | `e83679a171cdb9e9753d6209555ba51404e8467210f4ef894b87146fc5afe080` |
| `7days4.zip` | [download](https://ndownloader.figshare.com/files/28625112) | `14028319829f45ae006f815076a00424` | `93362e3b12898be9869de33601b4da5f4c86742360d53a0c2ff498949ae9eb66` |

The KiltHub record also lists a weather archive and README. The paper's
protocols do not use measured weather as an input; the study-specific asset
registry therefore lists only the five trajectory archives it tracks. The
exact experiment protocols remain authoritative for whether a given run
consumes a raw dataset, a public-view file, or a pretrained model.

The exact TartanAviation 2022 archive objects used by the main study are
available from the official dataset server through the frozen upstream
downloader. These URLs follow the `download_file_from_bucket` endpoint and
object-key construction in [`adsb/download.py` at the pinned commit](https://raw.githubusercontent.com/castacks/TartanAviation/4065f5bb11c3d8e557dcaf20a56469e6b0738714/adsb/download.py).

| Study asset | Official 2022 object | Study SHA256 |
|---|---|---|
| KAGC raw 2022 | [download](https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330/tartanaviation-adsb/kagc/raw/2022.zip) | `3e94a0e1f1ee7db7e0180fa01690f9e88340312a3e6cb2744844f975221a109b` |
| KBTP raw 2022 | [download](https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330/tartanaviation-adsb/kbtp/raw/2022.zip) | `668d753a7f105f3026f07faef8ca5426a357256701b387fa970cd4ef6ca63354` |
| KAGC processed | [download](https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330/tartanaviation-adsb/kagc/processed.zip) | `bc0ce9796545c69239a3c1d1dc97a77bb65ad3e2ea6cb055d8c8ba57465d2700` |
| KBTP processed | [download](https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330/tartanaviation-adsb/kbtp/processed.zip) | `f131e41ddeb7dc2fcc8f28f654dcc33c9129a2ac44cb82475e1473fc1670891e` |

These are direct custodian-server links, not copies hosted by Measure2Act.
Availability depends on the upstream server. The downloader's `Raw` option
fetches additional years; the exact 2022 URLs above prevent accidentally
including those extra years when reconstructing this study.

TartanAviation is described in Patrikar, J., Dantas, J., Moon, B. et al.
“Image, speech, and ADS-B trajectory datasets for terminal airspace
operations.” *Scientific Data* 12, 468 (2025),
[https://doi.org/10.1038/s41597-025-04775-6](https://doi.org/10.1038/s41597-025-04775-6).

The machine-readable registry in [`data_sources.json`](data_sources.json) and
[`dataset_sources.csv`](dataset_sources.csv) remains the canonical source for
file identifiers, versions, official checksums, and redistribution flags.
Project preprocessing is documented in [`DATA_SOURCES.md`](DATA_SOURCES.md).

The aggregate outputs under `results/paper_tables/` are numerical paper
evidence, not a trajectory dataset. The internal `deposits/data` archive is a
private preservation mother copy and is not part of the GitHub upload.

No separate Measure2Act data DOI is claimed under this policy. A software DOI
may archive the tagged GitHub release later. Create a data DOI only if a
separately curated author-generated evidence dataset is actually released.
