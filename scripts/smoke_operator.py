"""Run a small data-free check of the paper's probability operators."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

# Allow this documented script to run both as a file and as an installed module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.energy_predict_optimize.solver import energy_optimal_probabilities
from experiments.tpmo_ascent.operator import transported_prior, tpmo_probabilities
from experiments.mabpt_ascent.operator import mass_aware_transported_prior


def main() -> None:
    torch.manual_seed(0)
    support_cost = torch.rand(1, 5, 5, dtype=torch.float64)
    baseline = torch.softmax(torch.randn(1, 5, dtype=torch.float64), dim=1)
    c162 = transported_prior(baseline, support_cost)["transported"]
    c165 = mass_aware_transported_prior(baseline, support_cost)["transported"]
    target = torch.rand(1, 5, dtype=torch.float64)
    pairwise = torch.rand(1, 5, 5, dtype=torch.float64)
    pairwise = 0.5 * (pairwise + pairwise.transpose(1, 2))
    pairwise[:, torch.arange(5), torch.arange(5)] = 0.0
    refined, diagnostics = tpmo_probabilities(c162, target, pairwise)
    energy = energy_optimal_probabilities(target, pairwise)
    for name, value in {"c162": c162, "c165": c165, "tpmo": refined, "energy": energy}.items():
        assert torch.isfinite(value).all(), name
        assert torch.allclose(value.sum(dim=1), torch.ones(1, dtype=value.dtype), atol=1e-6), name
    print(json.dumps({
        "status": "ok",
        "modes": 5,
        "tpmo_convergence_iteration": diagnostics["convergence_iteration"].tolist(),
    }))


if __name__ == "__main__":
    main()
