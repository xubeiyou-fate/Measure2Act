# Aggregate evidence for manuscript Tables 3-7

These small CSV files are the frozen paper-facing aggregate outputs permitted
in the public code repository. They contain no raw trajectory, per-flight
record, per-window record, actor identifier, model weight, or candidate array.

Run:

```bash
python scripts/verify_paper_summaries.py
```

The files preserve the displayed point estimates and aggregation labels. They
do not independently recreate manuscript bootstrap intervals and do not make
the third-party datasets or model checkpoints unnecessary for a full rerun.
