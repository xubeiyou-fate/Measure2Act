# ASCENT attribution and source boundary

Measure2Act uses the ASCENT aircraft-trajectory forecasting implementation as
the forecasting backbone and comparator described in the manuscript. ASCENT is
an upstream, publicly released research project:

- Repository: <https://github.com/a-pru/ascent>
- Pinned source commit used for this release: `814e0a18a8a7500dfb0498ab2ee873d022874e8a`
- Paper: Prutsch et al., *ASCENT: Transformer-Based Aircraft Trajectory
  Prediction in Non-Towered Terminal Airspace* (ICRA 2026)

The release owner has confirmed permission to redistribute the ASCENT source as
part of this reproducibility package. The upstream repository remains the
authoritative source for its copyright, attribution, and licensing terms. The
root `LICENSE` applies only to original Measure2Act material; it does not
relicense ASCENT source, ASCENT checkpoints, third-party datasets, or external
baseline implementations.

The ASCENT-related files in this repository are retained for reproducibility
and are clearly identified by the `model/` package and the experiment modules
listed in [CODE_MAP.md](CODE_MAP.md). Measure2Act-specific probability-transfer
operators and evaluation code are original project components and are covered
by the root license unless a file-level notice says otherwise.

No ASCENT checkpoint is stored in this Git repository. Checkpoints are
published separately with their own model card, checksums, and terms.
