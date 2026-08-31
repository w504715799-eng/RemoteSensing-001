"""Deterministic Phase 2B2-B development smoke evaluation."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch

from trustsr.artifacts.predictions import PredictionCache, build_identity
from trustsr.data.crosssensor_pairs import POST_MANIFEST_SHA256
from trustsr.evaluation import crosssensor_smoke
from trustsr.evaluation.crosssensor_smoke import (
    EXPERIMENT_SCHEMA,
    INPUT_AUDIT_SHA256,
    build_cache_provenance,
    cache_entry_evidence,
    snapshot_cache_files,
)


def _lr(value: float = 0.25) -> torch.Tensor:
    return torch.full((4, 2, 3), value, dtype=torch.float32)


def _prediction(value: float = 0.5) -> torch.Tensor:
    return torch.full((4, 8, 12), value, dtype=torch.float32)


def _stored_identity(tmp_path: Path):
    provenance = build_cache_provenance({"name": "bicubic-x4", "scale": 4})
    identity = build_identity(provenance, "source", "sample-0", _lr())
    cache = PredictionCache(tmp_path)
    cache.put(identity, _prediction())
    return identity


def test_build_cache_provenance_binds_experiment_and_upstream() -> None:
    provenance = build_cache_provenance(
        {"name": "bicubic-x4", "scale": 4, "implementation_schema_version": 1}
    )

    assert provenance == {
        "name": "bicubic-x4",
        "scale": 4,
        "implementation_schema_version": 1,
        "experiment_schema": "trustsr.phase2b2b-development-smoke.v1",
        "post_manifest_sha256": POST_MANIFEST_SHA256,
        "input_audit_sha256": (
            "fceb2ec04680ddf46bf4d0ed5a4a93edd33d58a09fc176d936bdef783114b44b"
        ),
    }


def test_context_change_changes_prediction_identity(monkeypatch) -> None:
    model = {"name": "bicubic-x4", "scale": 4}
    first = build_identity(build_cache_provenance(model), "source", "sample-0", _lr())
    monkeypatch.setattr(crosssensor_smoke, "INPUT_AUDIT_SHA256", "e" * 64)
    second = build_identity(build_cache_provenance(model), "source", "sample-0", _lr())

    assert first.key != second.key


def test_cache_provenance_rejects_reserved_context_collision() -> None:
    for key, value in (
        ("experiment_schema", EXPERIMENT_SCHEMA),
        ("post_manifest_sha256", POST_MANIFEST_SHA256),
        ("input_audit_sha256", INPUT_AUDIT_SHA256),
    ):
        try:
            build_cache_provenance({"name": "bicubic-x4", "scale": 4, key: value})
        except ValueError as exc:
            assert "reserved" in str(exc)
        else:
            raise AssertionError(f"reserved key was accepted: {key}")


def test_cache_entry_evidence_hashes_both_named_files(tmp_path: Path) -> None:
    identity = _stored_identity(tmp_path)

    evidence = cache_entry_evidence(tmp_path, identity)

    assert evidence["cache_key"] == identity.key
    assert evidence["lr"] == identity.as_dict()["lr"]
    assert evidence["prediction_sha256"] == hashlib.sha256(
        _prediction().numpy().tobytes()
    ).hexdigest()
    assert [item["filename"] for item in evidence["files"]] == [
        f"{identity.key}.json",
        f"{identity.key}.safetensors",
    ]
    for item in evidence["files"]:
        path = tmp_path / item["filename"]
        assert item["size_bytes"] == path.stat().st_size
        assert item["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_snapshot_cache_files_observes_size_mtime_and_digest_change(tmp_path: Path) -> None:
    identity = _stored_identity(tmp_path)
    before = snapshot_cache_files(tmp_path, [identity])
    tensor_path = tmp_path / f"{identity.key}.safetensors"
    original = tensor_path.read_bytes()
    tensor_path.write_bytes(original + b"damage")
    stat = tensor_path.stat()
    os.utime(tensor_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    after = snapshot_cache_files(tmp_path, [identity])

    assert before != after
