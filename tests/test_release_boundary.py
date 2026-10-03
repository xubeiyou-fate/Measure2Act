from __future__ import annotations

from pathlib import Path

from scripts.audit_code_release import audit


def test_code_release_contains_no_data_or_model_payloads() -> None:
    root = Path(__file__).resolve().parents[1]
    assert audit(root) == []
