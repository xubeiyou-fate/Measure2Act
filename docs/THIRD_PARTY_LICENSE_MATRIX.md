# Third-party license and provenance matrix

This matrix separates what is present in the source-only GitHub tree from
what is merely referenced. It is part of the release review and must be
updated if a source file, data asset, or checkpoint is added.

| Component | Repository content | Upstream record | Terms/status | Release action |
|---|---|---|---|---|
| Measure2Act original code and tests | Included | This repository | Apache-2.0 in the root `LICENSE` | Keep the root notice and file-level notices. |
| Workflow figure | Included | `docs/assets/measure2act_workflow.png` | Author-provided project figure; Apache-2.0 with the original project files | Confirm all authors own the figure before the final tag. |
| ASCENT backbone | Included, with local changes | [`a-pru/ascent`](https://github.com/a-pru/ascent), commit `814e0a18a8a7500dfb0498ab2ee873d022874e8` | Public upstream project; no standard `LICENSE` file was visible at the verified path. The release owner has recorded permission to redistribute the component in `ASCENT_NOTICE.md`. | Preserve the notice and the written permission record. Do not claim that the root Apache-2.0 license relicenses ASCENT. |
| DAG-Net provenance in `model/utils.py` | Not vendored as a separate project; provenance is retained in the inherited docstring | [`alexmonti19/dagnet`](https://github.com/alexmonti19/dagnet) | No explicit license file was visible on the verified upstream repository page. | Permission covering the inherited ASCENT utility must be retained, or replace the utility with independently authored code before `v1.0.0`. |
| TrajAirNet provenance in `model/utils.py` | Not vendored as a separate project | [`castacks/trajairnet`](https://github.com/castacks/trajairnet) | BSD-4-Clause source license | Retain attribution and the advertising clause if any derivative source is distributed. |
| EqMotion | No upstream source tree included; only adapters/protocol metadata | [`MediaBrain-SJTU/EqMotion`](https://github.com/MediaBrain-SJTU/EqMotion), commit `5aec2e0b61c511fa93a24138dd90da59a089084b` | MIT source-code license; locally trained weights are not covered by that source license | Fetch upstream separately under its terms; do not imply permission for weights. |
| TrajAir dataset | Not included | KiltHub DOI [`10.1184/R1/14866251.v1`](https://doi.org/10.1184/R1/14866251.v1) | CC BY 4.0 as recorded in the dataset registry | Provide the official download route and cite the DOI; do not commit the payload. |
| TartanAviation source | Not included; acquisition is delegated to the official script | [`castacks/TartanAviation`](https://github.com/castacks/TartanAviation), commit `4065f5bb11c3d8e557dcaf20a56469e6b0738714` | BSD-3-Clause for verified source; downloaded ADS-B payload terms are not explicit on the verified pages | Provide the official acquisition route only; do not redistribute payloads without custodian permission. |
| Python dependencies | Not vendored | PyPI/package upstreams | Individual upstream terms; versions are recorded in the SBOM and lock files | Rebuild the SBOM for every release. |
| Paper checkpoints | Not included | Separate model record | No redistribution term or DOI is asserted in this repository yet | Add the model record URL, DOI, model card, and terms before the final tag. |

## Decision rule

The technical-only audit records the ASCENT repository and pinned commit in the
model index and permits an initial source-only upload. This matrix remains the
separate rights review record. A formal `v1.0.0` archive still requires the
repository/software identifiers and model DOI; any rights approval is managed
outside the technical completeness result.
