# GitHub project structure

The repository groups paper experiments under descriptive package names. Their
protocols, imports, and relative evidence paths were updated together, while
the experiment semantics and frozen data boundaries remain unchanged.

```text
Measure2Act/
├── README.md                         # project overview and quick start
├── LICENSE                            # original Measure2Act source terms
├── CITATION.cff                       # citation metadata; DOI added only if later minted
├── pyproject.toml                     # package metadata and CLI entry points
├── requirements.txt                   # broad runtime dependencies
├── requirements-lock.txt              # validated environment lock
├── requirements-dev.txt               # development/test dependencies
├── constraints/requirements-cpu.txt  # CPU installation constraint
├── environment.yml                    # Conda environment
├── Dockerfile                         # container build
├── MANIFEST.sha256                    # source-only integrity manifest
├── model_release.json                 # external model-record contract
│
├── .github/
│   ├── workflows/ci.yml               # install, tests, audits, wheel build
│   ├── ISSUE_TEMPLATE/                # issue forms
│   └── pull_request_template.md
│
├── Measure2Act_probability_transfer/  # stable public probability-transfer API
│   ├── operator.py                    # correspondence and Energy-KL operators
│   └── run.py                         # CPU smoke command
├── Measure2Act_forecasting/           # stable training/evaluation facades
│   ├── run_train.py                   # measure2act-train
│   ├── run_evaluate.py                # measure2act-evaluate
│   └── model.py                       # public forecasting imports
│
├── mabpt/                             # core probability-transfer implementation
├── continuous_geometry/               # support geometry and retrieval
├── causal_mode_generation/            # candidate-mode diagnostics
├── mode_state_query/                  # mode-state query components
├── proper_set_ascent/                 # proper-set ASCENT components
├── tail_query/                        # tail-query components
├── model/                             # independently authored ASCENT-inspired backbone
├── airroute_stage_m/                  # air-route model/evaluation components
├── modern_baseline/                   # EqMotion aviation adapters and protocols
│
├── experiments/                         # descriptive paper experiment packages
│   ├── edfa_ascent/                     # encounter-relation experiment
│   ├── dive_ascent/                     # mode-isolation experiment
│   ├── metric_exact/                    # exact metric controls
│   ├── joint_coupled/                   # joint-coupled control
│   ├── dual_expected_risk/              # dual-risk prediction
│   ├── decision_regret/                 # decision-regret objective
│   ├── energy_predict_optimize/         # Energy probability inference
│   ├── ascent_recomparison/             # matched ASCENT comparison
│   ├── tpmo_ascent/                     # TPMO operator
│   ├── mabpt_ascent/                    # MABPT operator
│   └── journal_extension/               # registered extensions
│
├── measure2act_ast_tools/             # final AST evaluation and aggregation
├── results/
│   ├── RESULTS_MASTER.csv             # orientation index
│   └── paper_tables/                  # small aggregate Tables 3–7 evidence
│
├── docs/
│   ├── PROJECT_STRUCTURE.md           # this structure guide
│   ├── CODE_MAP.md                    # conceptual code map
│   ├── REPRODUCIBILITY.md             # three-level reproduction contract
│   ├── DATASETS.md                    # official data access matrix
│   ├── DATA_SOURCES.md                # URLs, versions, checksums, commands
│   ├── DATA_AVAILABILITY.md            # manuscript-ready statement
│   ├── MODEL_RELEASE.md               # external model deposit contract
│   ├── ASCENT_NOTICE.md               # independent ASCENT-inspired boundary
│   ├── ASSET_BOUNDARY.md              # inclusion/exclusion rules
│   ├── THIRD_PARTY_LICENSE_MATRIX.md  # provenance and rights matrix
│   ├── CODE_AVAILABILITY.md            # code sharing statement
│   ├── GITHUB_UPLOAD.md               # push and release checklist
│   ├── RELEASE_AUDIT.md               # latest audit snapshot
│   ├── data_sources.json              # machine-readable data registry
│   ├── dataset_sources.csv            # asset URLs and checksums
│   └── assets/                        # author-provided workflow figure
│
├── scripts/
│   ├── audit_code_release.py          # rejects payloads and local paths
│   ├── audit_release_readiness.py    # GitHub layout and strict metadata gate
│   ├── build_manifest.py              # deterministic SHA256 manifest
│   ├── fetch_official_data.py         # official data acquisition wrapper
│   └── verify_paper_summaries.py      # Tables 3–7 checksum verification
├── tests/                             # data-free public tests
├── sbom/                              # CycloneDX dependency inventory
└── THIRD_PARTY_NOTICES.md             # dependency/provenance boundary
```

## Deliberately external assets

The GitHub tree does not contain raw TrajAir/TartanAviation data, processed
trajectory files, local caches, or model checkpoints. The official data routes
are documented in `docs/DATA_SOURCES.md`. The complete 70-checkpoint model
deposit is maintained separately under the model-record contract and is linked
through `model_release.json`; it must not be copied into Git history or Git
LFS.

## User-facing path order

For a new reader, use this order:

1. `README.md` and `Measure2Act_probability_transfer/` for the public API.
2. `docs/REPRODUCIBILITY.md` for source-only, aggregate-table, and full-run
   requirements.
3. `docs/DATASETS.md` and `docs/MODEL_RELEASE.md` for external assets.
4. `measure2act_ast_tools/` and `results/paper_tables/` for paper evidence.
5. The relevant package under `experiments/` only when reconstructing a
   specific frozen protocol.
