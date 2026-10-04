# Software bill of materials

`python-environment.cdx.json` is generated from the fully resolved Python 3.11
CPU lock. Run `scripts/generate_sbom.sh` after any lock change and commit the
regenerated JSON in the v1.0.0 public release.

The SBOM inventories Python dependencies; it does not replace the separate
code, third-party, data, and model license matrices.
