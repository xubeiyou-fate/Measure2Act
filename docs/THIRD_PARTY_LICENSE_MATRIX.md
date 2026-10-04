# Third-party license and provenance matrix

This matrix separates what is present in the Git-tracked source tree, what is
attached as a GitHub Release asset, and what is merely referenced. It must be
updated if a source file, data asset, or checkpoint is added.

| Component | Repository content | Upstream record | Terms/status | Release action |
|---|---|---|---|---|
| Measure2Act original code and tests | Included | This repository | Apache-2.0 in the root `LICENSE` | Keep the root notice and file-level notices. |
| Workflow figure | Included | `docs/assets/measure2act_workflow.png` | Author-provided project figure; Apache-2.0 with the original project files | Author release authorization recorded on 2026-10-04. |
| ASCENT architecture reference | Not vendored; independently reimplemented | [`a-pru/ascent`](https://github.com/a-pru/ascent) | Scientific and architectural reference only; no ASCENT source or official checkpoint is redistributed | Retain the independent-authorship notice and cite the reference. The root Apache-2.0 license applies only to original Measure2Act code. |
| DAG-Net/TrajAirNet concepts | Not vendored as separate projects | Upstream projects cited in source comments | Conceptual provenance only; no upstream source is redistributed | Keep attribution and do not describe the implementation as copied or official. |
| EqMotion | Independently authored Measure2Act adapters/protocol metadata are included; no upstream source tree is vendored | [`MediaBrain-SJTU/EqMotion`](https://github.com/MediaBrain-SJTU/EqMotion), commit `5aec2e0b61c511fa93a24138dd90da59a089084b` | Upstream source is MIT; the 10 author-trained adaptation weights are separately released under CC BY 4.0 by author authorization | Fetch upstream separately under its terms; retain its MIT notice and do not describe the adaptations as official EqMotion weights. |
| TrajAir dataset | Not included | KiltHub DOI [`10.1184/R1/14866251.v1`](https://doi.org/10.1184/R1/14866251.v1) | CC BY 4.0 as recorded in the dataset registry | Provide the official download route and cite the DOI; do not commit the payload. |
| TartanAviation source | Not included; acquisition is delegated to the official script | [`castacks/TartanAviation`](https://github.com/castacks/TartanAviation), commit `4065f5bb11c3d8e557dcaf20a56469e6b0738714` | BSD-3-Clause for verified source; downloaded ADS-B payload terms are not explicit on the verified pages | Provide the official acquisition route only; do not redistribute payloads without custodian permission. |
| Python dependencies | Not vendored | PyPI/package upstreams | Individual upstream terms; versions are recorded in the SBOM and lock files | Rebuild the SBOM for every release. |
| Paper checkpoints | Not in Git history; attached as `Measure2Act-models-v1.0.0.tar.gz` | [GitHub v1.0.0 Release](https://github.com/xubeiyou-fate/Measure2Act/releases/tag/v1.0.0) | All 70 author-created weights, including 10 EqMotion adaptations, are CC BY 4.0 by author authorization | Keep the archive checksum, model card, and 70-row index with the Release asset. A later DOI is optional. |

## Decision rule

The audit records the ASCENT repository and pinned commit in the model index
and keeps raw third-party datasets out of the release. The author authorization
for Apache-2.0 code and CC BY 4.0 weights is recorded in the release-control
matrix. No software or model DOI is claimed; the versioned GitHub Release is
the public distribution record.
