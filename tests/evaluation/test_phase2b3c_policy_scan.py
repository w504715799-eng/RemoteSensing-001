"""Git-safe Phase 2B3-C publication allowlist and leakage policy."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from trustsr.evaluation.phase2b3c_policy_scan import (
    phase2b3c_publication_policy_violations,
)
from trustsr.jsonio import canonical_json

_ALLOWED = {
    "sen2naipv2-internal-test-access-authorization-v1.json": (
        "trustsr.phase2b3c-internal-test-access-authorization.v1"
    ),
    "sen2naipv2-internal-test-evaluation-v1.json": (
        "trustsr.phase2b3c-evaluation.v1"
    ),
    "sen2naipv2-internal-test-evaluation-cache-audit-v1.json": (
        "trustsr.phase2b3c-evaluation-cache-audit.v1"
    ),
    "sen2naipv2-internal-test-evaluation-acceptance-v1.json": (
        "trustsr.phase2b3c-evaluation-acceptance.v1"
    ),
}


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init")
    return root


def _stage(root: Path, name: str, document: object) -> Path:
    path = root / "artifacts" / "phase2b3c" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = document if type(document) is bytes else canonical_json(document)
    path.write_bytes(payload)
    _git(root, "add", "-f", str(path.relative_to(root)))
    return path


def test_exact_four_canonical_host_free_documents_are_allowed(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    for name, schema in _ALLOWED.items():
        _stage(root, name, {"schema": schema, "digest": "0" * 64})

    assert phase2b3c_publication_policy_violations(root) == ()


@pytest.mark.parametrize(
    "name",
    (
        "prediction-cache.pt",
        "test-manifest.json",
        "model.pth",
        "phase2b3c-bundle-manifest.json",
        "unexpected.json",
    ),
)
def test_rejects_every_nonallowlisted_phase2b3c_artifact(
    tmp_path: Path, name: str
) -> None:
    root = _repository(tmp_path)
    _stage(root, name, {"schema": "unexpected"})

    violations = phase2b3c_publication_policy_violations(root)

    assert any(name in violation and "allowlist" in violation for violation in violations)


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("cache_path", "/srv/internal/test.pt"),
        ("hostname", "gpu-worker-7"),
        ("endpoint", "https://example.invalid/api"),
        ("credential", "opaque-secret"),
        ("access_token", "token-value"),
        ("raw_timestamp", "2026-09-04T01:02:03Z"),
        ("gpu_uuid", "GPU-deadbeef"),
    ),
)
def test_rejects_operational_or_secret_fields(
    tmp_path: Path, key: str, value: str
) -> None:
    root = _repository(tmp_path)
    name, schema = next(iter(_ALLOWED.items()))
    _stage(root, name, {"schema": schema, key: value})

    violations = phase2b3c_publication_policy_violations(root)

    assert any(key in violation for violation in violations)


@pytest.mark.parametrize(
    "metric",
    ("loss", "coverage", "trusted_pixels", "total_pixels", "risk_ucb"),
)
def test_rejects_per_sample_numerical_metrics(tmp_path: Path, metric: str) -> None:
    root = _repository(tmp_path)
    name = "sen2naipv2-internal-test-evaluation-cache-audit-v1.json"
    _stage(
        root,
        name,
        {
            "schema": _ALLOWED[name],
            "samples": [{"sample_id": "test-000", metric: 0.25}],
        },
    )

    violations = phase2b3c_publication_policy_violations(root)

    assert any(metric in violation and "per-sample" in violation for violation in violations)


def test_rejects_noncanonical_bytes_and_a_symlink(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    name, schema = next(iter(_ALLOWED.items()))
    _stage(root, name, b'{"schema": "' + schema.encode() + b'"}\n')
    link = root / "artifacts" / "phase2b3c" / next(iter(tuple(_ALLOWED)[1:]))
    link.symlink_to(root / "outside")
    _git(root, "add", "-f", str(link.relative_to(root)))

    violations = phase2b3c_publication_policy_violations(root)

    assert any("canonical" in violation for violation in violations)
    assert any("symlink" in violation for violation in violations)


def test_current_repository_phase2b3c_publication_is_policy_clean() -> None:
    root = Path(__file__).parents[2]

    assert phase2b3c_publication_policy_violations(root) == ()
