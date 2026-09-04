"""Frozen operating-point behavior for formal Phase 2B3-B evidence."""

from __future__ import annotations

import importlib

import pytest


def _policy():
    return importlib.import_module("trustsr.evaluation.phase2b3b_policy")


def test_accepts_only_the_preregistered_operating_point() -> None:
    policy = _policy()

    assert policy.require_approved_operating_point(0.05, 0.10) == (0.05, 0.10)


@pytest.mark.parametrize(
    ("alpha", "minimum_coverage"),
    (
        (0.0500001, 0.10),
        (0.05, 0.100001),
        (True, 0.10),
        (0.05, False),
        ("0.05", 0.10),
        (0.05, "0.10"),
    ),
)
def test_rejects_every_other_or_non_float_operating_point(
    alpha: object, minimum_coverage: object
) -> None:
    policy = _policy()

    with pytest.raises(ValueError, match="approved Phase 2B3-B"):
        policy.require_approved_operating_point(alpha, minimum_coverage)
