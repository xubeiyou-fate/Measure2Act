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
| Public repository | `https://github.com/xubeiyou-fate/Measure2Act`; local commit `8b5762c` exists, but the `main` ref has not yet been pushed because the GitHub token lacks the `workflow` scope | **BLOCKED** |
| Source tree | 405 staged source/document paths; no raw archive or checkpoint payload; code boundary, CPU smoke, six tests, and table-summary checks pass | **PASS** |
| Derived-data deposit | 4,323 files, approximately 188 MiB; 2,000 JSON/NPZ cases from 1,000 selected scenes; local archive verifier 8/8 and Tables 3–7 numeric checks 38/38 pass | **TECHNICAL PASS / NOT PUBLIC** |
| Model deposit | 222 files, approximately 575 MiB; 70 checkpoints (60 core + 10 EqMotion); all 70 load and manifest/index checks pass | **TECHNICAL PASS / NOT PUBLIC** |
| Persistent records | Software DOI, derived-data DOI, and model DOI/landing page are unresolved | **BLOCKED** |
| Manuscript availability text | The manuscript currently says processed outputs and code/checkpoints are not publicly deposited, which conflicts with the intended release plan | **MUST UPDATE** |

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

### Required corrections

1. **Availability contradiction.** The manuscript paragraph headed “Data
   availability” still says that processed outputs and reconstruction scripts
   are not in a public repository and that code/checkpoint tensors are not
   shareable. Replace it only after the GitHub commit and the separate data and
   model records are public and reviewer-accessible.
2. **Model provenance wording.** The current `docs/ASCENT_NOTICE.md`, source
   comments, and model index describe `model/` as a redistributed ASCENT
   implementation and use `ASCENT_official_implementation`. That is not the
   same statement as “architecture referenced from the ASCENT paper, code
   independently written for Measure2Act.” Choose one provenance path and use
   it consistently. Do not call independently authored weights “official
   ASCENT weights.”
3. **Role names.** The archive role names `ascent`, `decision_support`, and
   `predicted_risk` are technically traceable but not self-explanatory. Use the
   following public aliases while retaining the old archive path as an
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

## Ready-to-paste availability text after records resolve

Replace each bracketed field with the real landing page/DOI; do not submit the
brackets literally.

### Data Availability

> TrajAir version 1 is available from Carnegie Mellon University's KiltHub
> (https://doi.org/10.1184/R1/14866251.v1). TartanAviation ADS-B data were
> obtained from the official project and downloader at
> https://theairlab.org/tartanaviation/ and
> https://github.com/castacks/TartanAviation, using commit
> `4065f5bb11c3d8e557dcaf20a56469e6b0738714`. The raw third-party archives are
> not redistributed in this project. The derived Measure2Act evidence deposit
> (selected trajectory-derived cases, frozen evaluation outputs, split
> manifests, and deterministic table-rebuild inputs) is available at
> `[DATA_DOI_OR_LANDING_PAGE]`. The fitted model weights and their checksums,
> configuration records, and model card are available at
> `[MODEL_DOI_OR_LANDING_PAGE]`. Source code and data/model access instructions
> are available at `https://github.com/xubeiyou-fate/Measure2Act`.

### Code Availability

> The independently authored Measure2Act implementation, tests, protocols,
> aggregate manuscript tables, and environment specifications are available at
> `https://github.com/xubeiyou-fate/Measure2Act`, archived as
> `[SOFTWARE_DOI_OR_LANDING_PAGE]`. The GitHub repository does not contain raw
> third-party trajectory archives or model checkpoint payloads; those assets
> are linked through the Data Availability statement above.

## AST/Elsevier submission interpretation

Elsevier's research-data policy encourages deposit, citation and persistent
linking of data; its journal-specific data option must be selected in the AST
submission system. The GitHub URL alone is not a substitute for a persistent
data/model record when the submission form asks for research-data access. Use
the official [Elsevier research-data guidelines](https://www.elsevier.com/researcher/author/tools-and-resources/research-data/data-guidelines)
and the [AST journal page](https://www.sciencedirect.com/journal/aerospace-science-and-technology)
when completing the submission fields.

## Release gates

Before calling this project “complete” or tagging `v1.0.0`:

1. authorize the GitHub `workflow` scope and verify that `git ls-remote --heads
   origin main` returns commit `8b5762c` (or its final follow-up commit);
2. resolve the software archive DOI and update `README.md`, `CITATION.cff`,
   `docs/CODE_AVAILABILITY.md`, and the manuscript together;
3. deposit the derived evidence only after upstream-rights and privacy review,
   then resolve its DOI and update the Data Availability statement;
4. deposit the 70 model weights only after ownership/weight terms are approved,
   then resolve the model DOI and update `model_release.json` and the model
   card; and
5. run the source, data, model, clean-clone and table-rebuild checks against the
   exact release commit.

Until these gates are closed, the correct description is **source-only GitHub
release candidate with technically verified local data/model deposits**, not a
fully archived open reproduction package.
