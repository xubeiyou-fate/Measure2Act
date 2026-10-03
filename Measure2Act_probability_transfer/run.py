"""Data-free smoke runner for the public probability-transfer facade."""

from __future__ import annotations

import argparse
import json

import torch

from .operator import energy_kl_projection, exact_gibbs_transport, pairwise_trajectory_distance, support_cost


def smoke() -> dict[str, float | int | str]:
    torch.manual_seed(0)
    source = torch.randn(2, 5, 8, 3, dtype=torch.float64)
    target = torch.randn(2, 5, 8, 3, dtype=torch.float64)
    source_mass = torch.softmax(torch.randn(2, 5, dtype=torch.float64), dim=1)
    cost = support_cost(source, target)
    transported = exact_gibbs_transport(source_mass, cost, mass_weighted=True)["transported"]
    pairwise = pairwise_trajectory_distance(target)
    probabilities, diagnostics = energy_kl_projection(
        transported,
        cost.diagonal(dim1=1, dim2=2),
        pairwise,
    )
    return {
        "status": "ok",
        "batch": int(probabilities.shape[0]),
        "modes": int(probabilities.shape[1]),
        "simplex_error": float((probabilities.sum(dim=1) - 1.0).abs().max()),
        "objective_gain_mean": float(diagnostics["objective_gain"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run the data-free operator smoke test")
    args = parser.parse_args()
    if not args.smoke:
        parser.error("pass --smoke")
    print(json.dumps(smoke(), sort_keys=True))


if __name__ == "__main__":
    main()
