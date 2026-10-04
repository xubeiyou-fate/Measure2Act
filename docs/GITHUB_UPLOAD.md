# GitHub upload checklist

This directory documents the public release. The GitHub tree does not contain
raw trajectory data or model checkpoints; the 70-weight archive is attached to
the `v1.0.1` GitHub Release. The canonical GitHub repository is
`https://github.com/xubeiyou-fate/Measure2Act`. Software and model DOI records
are optional follow-up archival objects; no DOI is fabricated locally.

```bash
cd /path/to/Measure2Act
python scripts/audit_code_release.py
python scripts/audit_release_readiness.py
python -m pytest -q
python scripts/build_manifest.py
git diff --check

git config user.name "Hang XU"
git config user.email "beiyou1234@mail.dlut.edu.cn"
git remote add origin https://github.com/xubeiyou-fate/Measure2Act.git
git add -A
git diff --cached --check
git commit -m "Initial public Measure2Act release"
git branch -M main
git push -u origin main
```

The `model/` package is an independently authored aircraft-forecasting
implementation; its provenance boundary is documented in
[ARCHITECTURE_REFERENCE_NOTICE.md](ARCHITECTURE_REFERENCE_NOTICE.md).
The root `LICENSE` applies only to original Measure2Act material. Datasets and
checkpoints are acquired from their official or separately archived records.

After changing the release asset, update its SHA256 and rerun the model and
source audits. The `v1.0.1` tag and GitHub release are the public records for
this submission package.
