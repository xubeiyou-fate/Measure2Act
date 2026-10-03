"""Frozen C165 boundaries based on immutable C162 inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from experiments.tpmo_ascent.protocol import atomic_json
from experiments.tpmo_ascent.protocol import load_protocol as load_c162_protocol
from experiments.tpmo_ascent.protocol import sha256


@dataclass(frozen=True)
class Protocol:
    repository_root: Path
    path: Path
    payload: dict[str, object]

    def base(self):
        return load_c162_protocol()

    def assert_boundaries(self) -> None:
        self.base().assert_boundaries()
        observed = sha256(self.repository_root / str(self.payload["base_protocol"]))
        if observed != str(self.payload["base_protocol_sha256"]):
            raise RuntimeError("C165 base protocol hash mismatch")
        algorithm = self.payload["algorithm"]
        if algorithm["target_in_validation_probability_forward"] is not False:
            raise RuntimeError("C165 validation forward may not read target")
        if "sum_i q_B0[i] * cost[i, permutation(i)]" not in str(algorithm["assignment_cost"]):
            raise RuntimeError("C165 mass-aware assignment cost is not frozen")
        forbidden = set(self.payload["forbidden_mechanisms"])
        required = {
            "trajectory_or_control_residual",
            "learned_gate_router_or_mixture_of_experts",
            "temperature_or_assignment_scale_search",
            "score_weight_or_regularization_search",
            "candidate_expansion_selection_reranking_or_nms",
        }
        if not required.issubset(forbidden):
            raise RuntimeError("C165 forbidden-mechanism boundary is incomplete")


def load_protocol() -> Protocol:
    path = Path(__file__).with_name("protocol.json").resolve()
    return Protocol(path.parents[1], path, json.loads(path.read_text(encoding="utf-8")))


__all__ = ["Protocol", "atomic_json", "load_protocol", "sha256"]
