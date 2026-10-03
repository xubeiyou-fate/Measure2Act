"""Frozen metric gates for DIVE-ASCENT."""

from __future__ import annotations

import math


def gain(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / max(abs(baseline), 1e-12)


def compare(
    baseline: dict[str, object],
    candidate: dict[str, object],
    gates: dict[str, float],
) -> dict[str, object]:
    gains = {
        "minade": gain(float(baseline["minade"]), float(candidate["minade"])),
        "minfde": gain(float(baseline["minfde"]), float(candidate["minfde"])),
        "p95_fde": gain(float(baseline["minfde_p95"]), float(candidate["minfde_p95"])),
        "energy": gain(float(baseline["energy_score"]), float(candidate["energy_score"])),
        "tail_minfde": gain(float(baseline["tail_minfde"]), float(candidate["tail_minfde"])),
    }
    baseline_modes = float(baseline["winner_distribution"]["effective_modes"])
    candidate_modes = float(candidate["winner_distribution"]["effective_modes"])
    retention = candidate_modes / max(baseline_modes, 1e-12)
    fractions = [float(value) for value in candidate["winner_distribution"]["fractions"]]
    finite = all(math.isfinite(value) for value in gains.values()) and all(
        math.isfinite(value) for value in fractions
    )
    checks = {
        "all_metrics_finite": finite,
        "primary_minfde_gain": gains["minfde"] >= gates["primary_minfde_relative_gain_minimum"],
        "primary_minade_gain": gains["minade"] >= gates["primary_minade_relative_gain_minimum"],
        "primary_p95_nonworse": gains["p95_fde"] >= gates["primary_p95_relative_gain_minimum"],
        "primary_energy_nonworse": gains["energy"] >= gates["primary_energy_relative_gain_minimum"],
        "primary_tail_nonworse": gains["tail_minfde"] >= gates["primary_tail_relative_gain_minimum"],
        "effective_mode_retention": retention >= gates["effective_mode_retention_minimum"],
        "minimum_winner_fraction": min(fractions) >= gates["minimum_winner_fraction_minimum"],
    }
    return {
        "gains": gains,
        "baseline_effective_modes": baseline_modes,
        "candidate_effective_modes": candidate_modes,
        "effective_mode_retention": retention,
        "minimum_winner_fraction": min(fractions),
        "gates": checks,
        "passed": all(checks.values()),
    }


__all__ = ["compare", "gain"]
