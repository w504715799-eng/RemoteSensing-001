"""Frozen scientific-policy behavior for the one-time Phase 2B3-C evaluation."""

from __future__ import annotations

import importlib

import pytest


def _policy():
    return importlib.import_module("trustsr.evaluation.phase2b3c_policy")


def test_accepts_only_the_preregistered_phase2b3c_policy() -> None:
    policy = _policy()

    assert policy.require_phase2b3c_policy(
        alpha=0.05,
        minimum_coverage=0.10,
        delta=0.05,
        grid_size=20,
        threshold=7.970395366024563e-06,
    ) == (0.05, 0.10, 0.05, 20, 7.970395366024563e-06)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("alpha", 0.0500001),
        ("alpha", True),
        ("minimum_coverage", 0.100001),
        ("minimum_coverage", False),
        ("delta", 0.050001),
        ("delta", "0.05"),
        ("grid_size", 19),
        ("grid_size", True),
        ("threshold", 7.970395366025563e-06),
        ("threshold", "7.970395366024563e-06"),
    ),
)
def test_rejects_any_policy_change_or_non_builtin_type(field: str, value: object) -> None:
    policy = _policy()
    arguments: dict[str, object] = {
        "alpha": 0.05,
        "minimum_coverage": 0.10,
        "delta": 0.05,
        "grid_size": 20,
        "threshold": 7.970395366024563e-06,
    }
    arguments[field] = value

    with pytest.raises(ValueError, match="approved Phase 2B3-C"):
        policy.require_phase2b3c_policy(**arguments)
