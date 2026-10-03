#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m Measure2Act_forecasting.run_train \
  --dataset_folder /dataset/ \
  --dataset_name 7days1_trajair_reconstructed \
  --obs 11 --preds 120 --k 5
