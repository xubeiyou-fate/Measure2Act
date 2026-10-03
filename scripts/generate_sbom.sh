#!/usr/bin/env sh
set -eu

python -m pip install 'cyclonedx-bom>=4,<7'
cyclonedx-py requirements requirements-lock.txt \
  --pyproject pyproject.toml \
  --mc-type library \
  --output-reproducible \
  --output-format JSON \
  --output-file sbom/python-environment.cdx.json
