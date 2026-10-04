# Code availability

The Measure2Act implementation, tests, paper-table summaries, build metadata,
and reproducibility scripts are publicly released in the GitHub repository at
<https://github.com/xubeiyou-fate/Measure2Act>. This is the Git-tracked code
record for a manuscript prepared for submission to AST; it does not contain
raw trajectory data or checkpoint binaries in Git history. The source code is
released under Apache-2.0.

The source tree intentionally excludes third-party raw trajectory files,
derived trajectory files, checkpoints, and local caches. Those assets are
obtained from the official dataset records and the separate model record as
described in [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) and
[MODEL_RELEASE.md](MODEL_RELEASE.md).

To verify the Git-tracked release, run:

```bash
python scripts/audit_code_release.py
python scripts/audit_release_readiness.py
python -m pytest -q
python scripts/verify_paper_summaries.py
```

The 70 author-created checkpoints are attached to the v1.0.1 GitHub Release
under CC BY 4.0, including the EqMotion adaptations. A DOI may be added after
an external archive returns a real identifier; no placeholder DOI is used.
