# Data Availability

## Submission-ready draft

TrajAir version 1 is publicly available from Carnegie Mellon University's
KiltHub at <https://doi.org/10.1184/R1/14866251.v1> under the CC BY 4.0
license.
TartanAviation ADS-B trajectory data were obtained through the official
TartanAviation project page and downloader at
<https://theairlab.org/tartanaviation/> and
<https://github.com/castacks/TartanAviation>, using repository commit
`4065f5bb11c3d8e557dcaf20a56469e6b0738714`. The TartanAviation source-code
repository is BSD-3-Clause licensed; no separate license for the downloaded
data payload was identified on the verified official pages, so the payload is
not redistributed here. Download and preprocessing entry points, source
versions and checksums, and the aggregate values underlying manuscript Tables
3-7 are available in the versioned Measure2Act GitHub release. The 70 fitted
model checkpoints are attached to the same release as a separate binary asset
under CC BY 4.0. A persistent DOI may be added later if an archive provider
returns one; no placeholder DOI is used. The GitHub release contains no
third-party raw or derived trajectory dataset.

## Repository and citation actions

- If a software or model DOI is later minted, add its resolving landing page
  to this statement and the repository metadata.
- Cite the TrajAir dataset DOI in the reference list, not only in this statement.
- Cite the TartanAviation paper/project and record the exact repository commit.
- Test all official download routes outside the author's authenticated session.
- The internal derived-case pool is excluded by policy. If a rights-cleared,
  privacy-reviewed evidence deposit is published, cite its distinct data DOI;
  otherwise do not claim that a Measure2Act data DOI exists.

Recommended dataset reference:

> Patrikar, J., Moon, B., Ghosh, S., Oh, J., & Scherer, S. (2021). *TrajAir: A
> General Aviation Trajectory Dataset* (Version 1) [Data set]. Carnegie Mellon
> University. https://doi.org/10.1184/R1/14866251.v1

## Missing information / risk flags

- TartanAviation data-payload licensing remains an upstream-rights uncertainty.
- No Measure2Act data DOI is claimed: reused datasets retain their official
  identifiers and the internal derived-case pool remains excluded.
- The aggregate CSV files support numerical inspection but are not substitutes
  for the third-party datasets and model weights needed for full reruns.

## 中文核对

- TrajAir 官方版本化数据 DOI 已核实为 `10.1184/R1/14866251.v1`，KiltHub 标注 CC BY 4.0。
- TartanAviation 的 Zenodo DOI `10.5281/zenodo.14699102` 是软件记录，不是数据 DOI。
- TartanAviation 仓库的 BSD-3-Clause 只可明确解释为源码许可证，不能自动覆盖下载的数据载荷。
- 当前 GitHub 策略不需要另建 Measure2Act 数据 DOI；只有正式发布内部完整证据包时才需要。
