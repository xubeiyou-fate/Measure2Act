# Paper–Repository Publication Audit

**Audit date:** 2026-10-04 (Asia/Shanghai)
**Manuscript checked:** `01_Manuscript.docx` (author workspace copy)
**Target journal:** *Aerospace Science and Technology* (AST)
**Reference layout:** [yuqinie98/PatchTST](https://github.com/yuqinie98/patchtst)

This audit separates technical integrity from public-release readiness. A
passing checksum or unit test does not create a DOI, grant redistribution
rights, or make the raw third-party data public.

## Current state

| Item | Finding | Status |
|---|---|---|
| Public repository | `https://github.com/xubeiyou-fate/Measure2Act`; v1.0.1 release built from the audited source branch | **PASS** |
| Source tree | Source/document paths only; no raw archive or checkpoint payload; boundary, CPU smoke, tests, and table-summary checks pass | **PASS** |
| Derived-data deposit | 4,323 files, approximately 188 MiB; 2,000 JSON/NPZ cases from 1,000 selected scenes; local archive verifier 8/8 and Tables 3–7 numeric checks 38/38 pass | **TECHNICAL PASS / NOT PUBLIC** |
| Model deposit | 223 files (221 manifest-covered plus manifest and report), approximately 497 MiB compressed; 70 checkpoints (60 core + 10 EqMotion); all 70 load and manifest/index checks pass | **PUBLIC RELEASE ASSET** |
| Persistent records | GitHub tag/release is public; no DOI is fabricated locally and a DOI archive remains optional | **PASS / DOI OPTIONAL** |
| Manuscript availability text | `01_Manuscript.docx` now names the official dataset routes, public repository, `v1.0.1` Release, Apache-2.0 code, and CC BY 4.0 weights | **PASS** |

## Paper-to-repository consistency

### Matches

- Title, author order, ORCID values, and the Measure2Act method name agree
  between the manuscript, `README.md`, and `CITATION.cff`.
- The point estimates in the public aggregate CSVs reproduce the manuscript
  Tables 3–7 (38/38 numeric checks). The repository correctly labels these as
  aggregate evidence rather than raw trajectories.
- The manuscript's data sources (TrajAir v1 and TartanAviation ADS-B) agree
  with `docs/DATASETS.md`, `docs/DATA_SOURCES.md`, and the source registry.
- The repository correctly keeps raw third-party archives outside GitHub and
  records their official acquisition routes and frozen versions.
- The five-trajectory support, 16 historical observations, 24 future positions,
  two airports, two regimes, and five seeds agree with the model card and the
  manuscript protocol table.

### Resolved checks and retained caveats

1. **Availability consistency.** The manuscript paragraph headed “Data
   availability” now points to the official upstream datasets, public source
   repository, and versioned 70-weight Release asset. The pre-edit manuscript
   is preserved locally as `01_Manuscript_before_GitHub_release.docx`.
2. **Model provenance wording.** The source, model index, model card, and
   notices describe an independently authored aircraft-forecasting
   implementation. A single upstream-reference notice records scientific
   attribution; no upstream source or official checkpoint is redistributed.
3. **Legacy role names.** The archive role names `ascent`, `decision_support`,
   and `predicted_risk` are retained only in the manifest and data card because
   they are part of the released artifact schema; they are not project branding.
   Use the following public aliases while retaining the old archive path as an
   `archive_role` for checksum compatibility:

   | Archive role | Public role name | Meaning |
   |---|---|---|
   | `ascent` | `source_forecaster` | source five-trajectory forecast and source mass |
   | `decision_support` | `replacement_forecaster` | replacement support and candidate-return branch |
   | `predicted_risk` | `target_risk_head` | candidate-specific displacement-risk estimate |
   | `eqmotion_target_support` | `eqmotion_fixed_support_control` | auxiliary prior-only fixed-support control |

4. **Historical experiment labels.** Directory names are now descriptive under
   `experiments/`, but internal files still expose C96/C99/C127/C129/C133/C134/
   C161/C162/C165 and `journal_extension`. These are internal protocol IDs, not
   paper method names. Either keep them explicitly under an “internal protocol
   identifier” field, or perform a compatibility-preserving rename and
   regenerate every affected hash. Do not silently rename them in a final
   release.
5. **Derived-case privacy wording.** The local data card says “pseudonymous,”
   while airport, date, scene and segment fields remain quasi-identifiers. Do
   not describe this pool as anonymous or de-identified until the authors'
   privacy/sensitivity review is complete. The pool is a derived inspection
   sample, not the raw TartanAviation dataset.

## Availability text matched to v1.0.1

### Data Availability

> TrajAir version 1 is available from Carnegie Mellon University's KiltHub
> (https://doi.org/10.1184/R1/14866251.v1). TartanAviation ADS-B data were
> obtained from the official project and downloader at
> https://theairlab.org/tartanaviation/ and
> https://github.com/castacks/TartanAviation, using commit
> `4065f5bb11c3d8e557dcaf20a56469e6b0738714`. The raw third-party archives are
> not redistributed in this project. Source versions, official access routes,
> preprocessing entry points, and the aggregate values underlying Tables 3-7
> are available in the Measure2Act v1.0.1 Release at
> `https://github.com/xubeiyou-fate/Measure2Act/releases/tag/v1.0.1`. All 70
> author-created model checkpoints, their configuration bindings, and SHA-256
> checksums are attached to that Release under CC BY 4.0. The release contains
> no third-party raw or derived trajectory dataset.

### Code Availability

> The independently authored Measure2Act implementation, tests, protocols,
> aggregate manuscript tables, and environment specifications are available
> under Apache-2.0 at `https://github.com/xubeiyou-fate/Measure2Act`, release
> `v1.0.1`. The Git repository does not contain raw third-party trajectory
> archives or checkpoint binaries; the weight archive is a Release asset.

## AST/Elsevier submission interpretation

Elsevier journals apply journal-specific research-data options; the live AST
submission-system selection must be checked when submitting. The public GitHub
Release supplies a resolving access route, while a later DOI remains useful
for long-term archival persistence. Use
the official [Elsevier research-data guidelines](https://www.elsevier.com/researcher/author/tools-and-resources/research-data/data-guidelines)
and the [AST journal page](https://www.sciencedirect.com/journal/aerospace-science-and-technology)
when completing the submission fields.

## Release gates

The release procedure is complete when all of these checks pass:

1. push the audited source commit and verify the remote `main` ref;
2. keep the derived case pool excluded unless upstream-rights and privacy review
   authorizes a separate evidence record;
3. publish the 70-weight model deposit under the prepared CC BY 4.0 terms;
4. run source, model, clean-clone, and table-rebuild checks against the
   exact release commit.

The correct description is **public v1.0.1 GitHub release with technically
verified model weights and official-link-only dataset access**. It is prepared
for AST submission and does not claim AST acceptance. No placeholder DOI is
presented as a citation.
