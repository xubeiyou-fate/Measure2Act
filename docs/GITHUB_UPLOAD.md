# GitHub upload checklist

This directory is a source-only release candidate. It does not contain raw
trajectory data or model checkpoints. The canonical GitHub repository is
`https://github.com/xubeiyou-fate/Measure2Act`. Software, derived-evidence,
and model records are separate archival objects; their identifiers must be
copied here only after the external providers return real DOI landing pages.

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

The `model/` package is an ASCENT-inspired, independently authored
implementation; its provenance boundary is documented in
[ASCENT_NOTICE.md](ASCENT_NOTICE.md).
The root `LICENSE` applies only to original Measure2Act material. Datasets and
checkpoints are acquired from their official or separately archived records.

After the repository and DOI records exist, update `README.md`,
`CITATION.cff`, `model_release.json`, and `docs/DATA_AVAILABILITY.md` together,
run `python scripts/audit_release_readiness.py --strict`, and only then create
the `v1.0.0` tag and GitHub release.
