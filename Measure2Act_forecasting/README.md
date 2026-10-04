# Measure2Act forecasting workflow

This package is the training and evaluation entrypoint for the
Independently authored aircraft-forecasting implementation used to generate
fixed multimodal trajectory supports.

Data and checkpoints are not stored in the GitHub repository. Point the CLI to
the separately archived assets:

```bash
measure2act-train \
  --dataset_folder /path/to/data-deposit/dataset \
  --dataset_name tartan_kagc_processed_official \
  --output_folder /path/to/new/runs \
  --obs 11 --preds 120 --k 5

measure2act-evaluate \
  --dataset_folder /path/to/data-deposit/dataset \
  --dataset_name tartan_kagc_processed_official \
  --exp_folder /path/to/model-deposit/run-directory \
  --epoch 20
```

The generic CLI is provided for inspection and new runs. Paper reconstruction
must use the frozen protocol files and exact weight index documented in
`docs/REPRODUCIBILITY.md`; it must never overwrite the archived evidence.
