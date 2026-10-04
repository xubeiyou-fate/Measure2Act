# Measure2Act probability transfer

This package is the stable facade for exact finite-permutation transport and
the Energy-KL probability projection used in the paper. It also exposes the
registered TPMO and mass-aware control implementations.

Run the CPU-only, data-free check after installation:

```bash
python -m Measure2Act_probability_transfer.run --smoke
```

The canonical operators are in `mabpt/operator.py`; frozen historical/control
implementations remain under compatibility paths so the released protocol
hashes can be interpreted.
Formal evaluation additionally requires the external data/evidence and model
records listed in the top-level README.
