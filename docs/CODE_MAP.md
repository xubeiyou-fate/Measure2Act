# Code map

This map provides a reader-facing index for the descriptive public package
paths. The paper protocols and their relative paths are updated together with
the package rename; their experiment semantics and evidence boundaries are
unchanged.

## Experiment packages

The public tree uses descriptive functional names under `experiments/`. Each
package contains its own protocol, implementation, tests, and evidence
helpers. Frozen evidence metadata may retain historical identifiers internally
so that old receipts remain auditable; those identifiers are not public package
names or GitHub labels.

| Directory | Function |
|---|---|
| `experiments/edfa_ascent/` | Encounter-relation graphs and factorized scene prediction. |
| `experiments/dive_ascent/` | Isolated mode geometry, gradient shielding, and deterministic expert birth. |
| `experiments/metric_exact/` | Exact metric optimization and score-isolated controls. |
| `experiments/joint_coupled/` | Joint-coupled dual-oracle geometry control. |
| `experiments/dual_expected_risk/` | Full-candidate dual expected-risk prediction. |
| `experiments/decision_regret/` | Native-K decision-regret objective. |
| `experiments/energy_predict_optimize/` | Target-free Energy predict-and-optimize probability inference. |
| `experiments/ascent_recomparison/` | Matched re-comparison against the archived reference baseline. |
| `experiments/tpmo_ascent/` | Transported-prior measure optimization (TPMO). |
| `experiments/mabpt_ascent/` | Mass-aware Bayesian permutation transport (MABPT). |
| `experiments/journal_extension/` | Registered controls and extensions. |

For normal use, readers should start with the stable `Measure2Act_*` packages.
The experiment packages are needed only when reproducing a specific frozen
paper protocol or audit. Their descriptive names are stable public paths.

## Public entry points

| Path | Role |
|---|---|
| `Measure2Act_probability_transfer/` | Stable finite-measure probability-transfer API and CPU smoke command |
| `Measure2Act_forecasting/` | Forecasting training and evaluation command-line facades |
| `measure2act_ast_tools/` | Final paper evaluation, aggregation, capacity, and runtime audits |

## Core method

| Path | Role |
|---|---|
| `mabpt/` | Gibbs correspondence, transported priors, Energy-KL refinement, metrics, and formal aggregators |
| `continuous_geometry/` | Geometry and retrieval primitives used by support matching |
| `causal_mode_generation/` | Candidate-mode diagnostics and winner alignment |
| `mode_state_query/` | Mode-state query components |
| `proper_set_ascent/` | Proper-set components retained for compatibility |
| `tail_query/` | Tail-query components |

## Forecast backbone and adapters

| Path | Role |
|---|---|
| `model/` | Independently authored aircraft-forecasting implementation |
| `airroute_stage_m/` | Air-route evaluation/model components retained by released entry points |
| `modern_baseline/` | Author-maintained EqMotion aviation adapters and protocols; upstream EqMotion source is not vendored |

## Frozen study implementations

| Path | Paper-facing role |
|---|---|
| `experiments/edfa_ascent/` | Historical encounter-relation experiment implementation |
| `experiments/dive_ascent/` | Historical mode-isolation experiment and locked-test implementation |
| `experiments/metric_exact/` | Exact metric and locked-analysis controls |
| `experiments/joint_coupled/` | Joint-coupled control |
| `experiments/dual_expected_risk/` | Dual expected-risk control |
| `experiments/decision_regret/` | Decision-regret control |
| `experiments/energy_predict_optimize/` | Energy prediction/optimization and probability solver |
| `experiments/ascent_recomparison/` | Archived reference re-comparison protocols |
| `experiments/tpmo_ascent/` | Frozen TPMO operator and aggregation protocol |
| `experiments/mabpt_ascent/` | Frozen mass-aware operator and aggregation protocol |
| `experiments/journal_extension/` | Final registered controls, fixed-support evaluations, and experiment orchestration |

Use the two `Measure2Act_*` packages for new integrations. The experiment
packages are paper-facing reconstruction tools, not the recommended API.

## Release support

| Path | Role |
|---|---|
| `docs/REPRODUCIBILITY.md` | Three-level reconstruction contract and asset paths |
| `docs/ASSET_BOUNDARY.md` | Code/data/model inclusion rules |
| `docs/MODEL_RELEASE.md` | External model-deposit contract and pre-release gates |
| `model_release.json` | Machine-readable external model-record descriptor |
| `results/RESULTS_MASTER.csv` | Small orientation index; not the authoritative paper evidence |
| `scripts/audit_code_release.py` | Rejects data, checkpoints, local paths, and oversized payloads |
| `scripts/audit_release_readiness.py` | Checks GitHub presentation and enforces final metadata on release tags |
| `tests/` | Data-free CPU tests for the public interfaces and source boundary |
| `sbom/` | Resolved dependency inventory |

## Why the tree is not flattened

Renaming or merging the frozen modules would change import paths recorded by
protocols and make it harder to compare the public source with archived
evidence. The repository therefore presents a stable facade and a conceptual
map while preserving the evidence-linked implementation paths.
