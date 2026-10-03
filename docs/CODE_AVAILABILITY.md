# Code availability

The source-only Measure2Act implementation, tests, paper-table summaries,
build metadata, and reproducibility scripts are intended to be released in a
public GitHub repository at
<https://github.com/xubeiyou-fate/Measure2Act>. This is the source-only code
record for the AST submission; it does not contain raw trajectory data or
model checkpoints.

The source tree intentionally excludes third-party raw trajectory files,
derived trajectory files, checkpoints, and local caches. Those assets are
obtained from the official dataset records and the separate model record as
described in [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) and
[MODEL_RELEASE.md](MODEL_RELEASE.md).

For a source-only upload, run:

```bash
python scripts/audit_code_release.py
python scripts/audit_release_readiness.py
python -m pytest -q
python scripts/verify_paper_summaries.py
```

The final paper statement must replace the pending software and model
identifiers after the archive DOI and model DOI are known.
