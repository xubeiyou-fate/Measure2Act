## Summary

Describe the behavior or documentation change and why it is needed.

## Validation

- [ ] `python scripts/audit_code_release.py`
- [ ] `python scripts/audit_release_readiness.py`
- [ ] `python -m pytest -q`
- [ ] `python -m Measure2Act_probability_transfer.run --smoke`
- [ ] Documentation and tests were updated where needed.
- [ ] No data, weights, restricted records, credentials, or private links were added.
- [ ] Frozen protocol paths and evidence mappings remain valid, or a new version is documented.
