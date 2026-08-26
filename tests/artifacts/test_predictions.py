import json

import pytest
import torch

from trustsr.artifacts.predictions import (
    CacheIntegrityError,
    PredictionCache,
    build_identity,
)


def _identity(lr=None):
    lr = torch.zeros((4, 2, 3), dtype=torch.float32) if lr is None else lr
    return build_identity({"scale": 4, "name": "demo"}, "scene-a", "sample-1", lr)


def test_identity_is_order_independent_and_sensitive_to_inputs():
    lr = torch.zeros((4, 2, 3), dtype=torch.float32)
    one = build_identity({"scale": 4, "name": "demo"}, "s", "id", lr)
    two = build_identity({"name": "demo", "scale": 4}, "s", "id", lr)
    assert one.key == two.key
    assert one.key != build_identity({"scale": 4, "name": "other"}, "s", "id", lr).key
    assert one.key != build_identity({"scale": 4, "name": "demo"}, "t", "id", lr).key
    assert one.key != build_identity({"scale": 4, "name": "demo"}, "s", "other", lr).key
    assert one.key != _identity(torch.ones_like(lr)).key
    assert one.key != _identity(torch.zeros((4, 3, 2))).key
    assert one.key != _identity(lr.to(torch.float64)).key


def test_round_trip_and_miss(tmp_path):
    cache = PredictionCache(tmp_path)
    identity = _identity()
    assert cache.get(identity) is None
    prediction = torch.full((4, 8, 12), 0.25, dtype=torch.float32)
    assert cache.put(identity, prediction) == identity.key
    loaded = cache.get(identity)
    assert loaded is not None and loaded.dtype == torch.float32
    assert torch.equal(loaded, prediction)
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob("*.pt"))


@pytest.mark.parametrize(
    "prediction",
    [
        torch.zeros((4, 8, 8)),
        torch.zeros((4, 8, 12), dtype=torch.float64),
        torch.full((4, 8, 12), float("nan")),
        torch.full((4, 8, 12), 2.0),
    ],
)
def test_output_contract_rejected(tmp_path, prediction):
    with pytest.raises(ValueError):
        PredictionCache(tmp_path).put(_identity(), prediction)


def test_metadata_tampering_is_integrity_error(tmp_path):
    cache = PredictionCache(tmp_path)
    identity = _identity()
    cache.put(identity, torch.zeros((4, 8, 12)))
    path = tmp_path / f"{identity.key}.json"
    metadata = json.loads(path.read_text())
    metadata["cache_key"] = "0" * 64
    path.write_text(json.dumps(metadata))
    with pytest.raises(CacheIntegrityError):
        cache.get(identity)


def test_tensor_only_is_miss(tmp_path):
    cache = PredictionCache(tmp_path)
    identity = _identity()
    (tmp_path / f"{identity.key}.safetensors").write_bytes(b"orphan")
    assert cache.get(identity) is None
