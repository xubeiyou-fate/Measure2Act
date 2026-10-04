# Third-party license and provenance matrix

This matrix separates what is present in the source-only GitHub tree from
what is merely referenced. It is part of the release review and must be
updated if a source file, data asset, or checkpoint is added.

| Component | Repository content | Upstream record | Terms/status | Release action |
|---|---|---|---|---|
| Measure2Act original code and tests | Included | This repository | Apache-2.0 in the root `LICENSE` | Keep the root notice and file-level notices. |
| Workflow figure | Included | `docs/assets/measure2act_workflow.png` | Author-provided project figure; Apache-2.0 with the original project files | Confirm all authors own the figure before the final tag. |
| ASCENT architecture reference | Not vendored; independently reimplemented | [`a-pru/ascent`](https://github.com/a-pru/ascent) | Scientific and architectural reference only; no ASCENT source or official checkpoint is redistributed | Retain the independent-authorship notice and cite the reference. The root Apache-2.0 license applies only to original Measure2Act code. |
| DAG-Net/TrajAirNet concepts | Not vendored as separate projects | Upstream projects cited in source comments | Conceptual provenance only; no upstream source is redistributed | Keep attribution and do not describe the implementation as copied or official. |
| EqMotion | No upstream source tree included; only adapters/protocol metadata | [`MediaBrain-SJTU/EqMotion`](https://github.com/MediaBrain-SJTU/EqMotion), commit `5aec2e0b61c511fa93a24138dd90da59a089084b` | MIT source-code license; locally trained weights are not covered by that source license | Fetch upstream separately under its terms; do not imply permission for weights. |
| TrajAir dataset | Not included | KiltHub DOI [`10.1184/R1/14866251.v1`](https://doi.org/10.1184/R1/14866251.v1) | CC BY 4.0 as recorded in the dataset registry | Provide the official download route and cite the DOI; do not commit the payload. |
| TartanAviation source | Not included; acquisition is delegated to the official script | [`castacks/TartanAviation`](https://github.com/castacks/TartanAviation), commit `4065f5bb11c3d8e557dcaf20a56469e6b0738714` | BSD-3-Clause for verified source; downloaded ADS-B payload terms are not explicit on the verified pages | Provide the official acquisition route only; do not redistribute payloads without custodian permission. |
| Python dependencies | Not vendored | PyPI/package upstreams | Individual upstream terms; versions are recorded in the SBOM and lock files | Rebuild the SBOM for every release. |
| Paper checkpoints | Not included | Separate model record | Author-created weights are CC BY 4.0; the record DOI is an external release gate | Add the resolving model URL and DOI before the final tag. |

## Decision rule

The technical-only audit records the ASCENT repository and pinned commit in the
model index and permits an initial source-only upload. This matrix remains the
separate rights review record. A formal `v1.0.0` archive still requires the
repository/software identifiers and model DOI; any rights approval is managed
outside the technical completeness result.
