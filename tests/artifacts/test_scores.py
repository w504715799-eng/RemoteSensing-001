import hashlib
import json
import multiprocessing
import os
from dataclasses import replace
from pathlib import Path
from threading import BrokenBarrierError

import pytest
import torch
from safetensors.torch import save_file

from trustsr.artifacts.scores import (
    CacheIntegrityError,
    ScoreCache,
    ScoreIdentity,
    score_entry_evidence,
    tensor_sha256,
)


def _identity() -> ScoreIdentity:
    return ScoreIdentity(
        score_name="lr_reprojection_l1",
        score_schema_version=1,
        sample_id="development-0",
        input_sha256s=("a" * 64, "b" * 64),
        operator_parameters={"scale": 4},
    )


def _score() -> torch.Tensor:
    return torch.arange(12, dtype=torch.float64).reshape(3, 4)


def _paths(root: Path, identity: ScoreIdentity) -> tuple[Path, Path]:
    return root / f"{identity.key}.safetensors", root / f"{identity.key}.json"


def _concurrent_put(root: Path, identity: ScoreIdentity, score: torch.Tensor) -> None:
    ScoreCache(root).put(identity, score)


def test_score_identity_changes_for_input_order_or_operator() -> None:
    original = _identity()
    reversed_inputs = replace(original, input_sha256s=tuple(reversed(original.input_sha256s)))
    changed_scale = replace(original, operator_parameters={"scale": 2})
    assert len({original.key, reversed_inputs.key, changed_scale.key}) == 3


def test_score_identity_defensively_copies_scalar_parameters() -> None:
    parameters = {"scale": 4}
    identity = replace(_identity(), operator_parameters=parameters)
    parameters["scale"] = 2
    assert identity.operator_parameters["scale"] == 4
    with pytest.raises(TypeError):
        replace(_identity(), operator_parameters={"nested": {"scale": 4}})


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("score_name", "", ValueError),
        ("score_name", 1, TypeError),
        ("score_schema_version", 0, ValueError),
        ("score_schema_version", True, TypeError),
        ("sample_id", "", ValueError),
        ("sample_id", None, TypeError),
        ("input_sha256s", ["a" * 64], TypeError),
        ("input_sha256s", ("A" * 64,), ValueError),
        ("input_sha256s", (), ValueError),
    ],
)
def test_score_identity_rejects_invalid_fields(
    field: str, value: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        replace(_identity(), **{field: value})


def test_score_cache_round_trip_is_detached_float64(tmp_path: Path) -> None:
    score = _score()
    cache = ScoreCache(tmp_path)
    cache.put(_identity(), score)
    loaded = cache.get(_identity())
    assert loaded is not None
    assert loaded.dtype == torch.float64
    assert loaded.device.type == "cpu"
    assert loaded.is_contiguous()
    assert torch.equal(loaded, score)
    assert loaded.data_ptr() != score.data_ptr()


@pytest.mark.parametrize(
    "score",
    [
        torch.zeros((3, 4), dtype=torch.float32),
        torch.full((3, 4), -1.0, dtype=torch.float64),
        torch.full((3, 4), float("nan"), dtype=torch.float64),
        torch.full((3, 4), float("inf"), dtype=torch.float64),
        torch.zeros(12, dtype=torch.float64),
        torch.zeros((1, 3, 4), dtype=torch.float64),
        torch.empty((0, 4), dtype=torch.float64),
    ],
)
def test_score_cache_rejects_invalid_score_contract(tmp_path: Path, score: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        ScoreCache(tmp_path).put(_identity(), score)


def test_score_cache_rejects_symlink_entries(tmp_path: Path) -> None:
    cache = ScoreCache(tmp_path)
    identity = _identity()
    cache.put(identity, _score())
    tensor_path, metadata_path = _paths(tmp_path, identity)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(tensor_path.read_bytes())
    tensor_path.unlink()
    tensor_path.symlink_to(replacement)
    with pytest.raises(CacheIntegrityError, match="non-symlink"):
        cache.get(identity)
    assert metadata_path.exists()


@pytest.mark.parametrize("missing_suffix", [".json", ".safetensors"])
def test_score_cache_rejects_one_sided_entries(tmp_path: Path, missing_suffix: str) -> None:
    cache = ScoreCache(tmp_path)
    identity = _identity()
    cache.put(identity, _score())
    (tmp_path / f"{identity.key}{missing_suffix}").unlink()
    with pytest.raises(CacheIntegrityError, match="without"):
        cache.get(identity)


def test_score_cache_rejects_noncanonical_or_altered_json(tmp_path: Path) -> None:
    cache = ScoreCache(tmp_path)
    identity = _identity()
    cache.put(identity, _score())
    _, metadata_path = _paths(tmp_path, identity)
    metadata_path.write_bytes(metadata_path.read_bytes() + b"\n")
    with pytest.raises(CacheIntegrityError, match="invalid"):
        cache.get(identity)


def test_score_cache_rejects_altered_tensor_bytes(tmp_path: Path) -> None:
    cache = ScoreCache(tmp_path)
    identity = _identity()
    cache.put(identity, _score())
    tensor_path, _ = _paths(tmp_path, identity)
    content = bytearray(tensor_path.read_bytes())
    content[-1] ^= 1
    tensor_path.write_bytes(content)
    with pytest.raises(CacheIntegrityError):
        cache.get(identity)


def test_score_cache_rejects_wrong_tensor_digest(tmp_path: Path) -> None:
    cache = ScoreCache(tmp_path)
    identity = _identity()
    cache.put(identity, _score())
    _, metadata_path = _paths(tmp_path, identity)
    metadata = json.loads(metadata_path.read_text())
    metadata["score"]["sha256"] = "0" * 64
    metadata_path.write_bytes(json.dumps(metadata, separators=(",", ":")).encode())
    with pytest.raises(CacheIntegrityError):
        cache.get(identity)


def test_score_cache_rejects_unexpected_tensor_key(tmp_path: Path) -> None:
    cache = ScoreCache(tmp_path)
    identity = _identity()
    cache.put(identity, _score())
    tensor_path, _ = _paths(tmp_path, identity)
    save_file({"score": _score(), "unexpected": _score()}, str(tensor_path))
    with pytest.raises(CacheIntegrityError):
        cache.get(identity)


def test_score_cache_cleans_temporary_files_after_commit(tmp_path: Path) -> None:
    ScoreCache(tmp_path).put(_identity(), _score())
    assert not list(tmp_path.glob(".*.tmp"))


def test_score_cache_never_overwrites_valid_entry(tmp_path: Path) -> None:
    cache = ScoreCache(tmp_path)
    identity = _identity()
    cache.put(identity, _score())
    tensor_path, metadata_path = _paths(tmp_path, identity)
    original_bytes = (tensor_path.read_bytes(), metadata_path.read_bytes())
    cache.put(identity, torch.full((3, 4), 9.0, dtype=torch.float64))
    assert (tensor_path.read_bytes(), metadata_path.read_bytes()) == original_bytes
    assert torch.equal(cache.get(identity), _score())


@pytest.mark.filterwarnings("ignore:This process.*multi-threaded.*:DeprecationWarning")
def test_concurrent_score_writers_preserve_first_committed_valid_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _identity()
    first_score = _score()
    second_score = torch.full((3, 4), 9.0, dtype=torch.float64)
    context = multiprocessing.get_context("fork")
    both_writers_ready = context.Barrier(2)
    first_committed = context.Event()
    original_commit = ScoreCache._atomic_commit

    def coordinated_commit(
        cache: ScoreCache, committed_identity: ScoreIdentity, score: torch.Tensor
    ) -> str:
        if torch.equal(score, first_score):
            try:
                both_writers_ready.wait(timeout=1)
            except BrokenBarrierError:
                pass
            result = original_commit(cache, committed_identity, score)
            first_committed.set()
            return result
        both_writers_ready.wait(timeout=1)
        assert first_committed.wait(timeout=1)
        return original_commit(cache, committed_identity, score)

    monkeypatch.setattr(ScoreCache, "_atomic_commit", coordinated_commit)
    first_writer = context.Process(target=_concurrent_put, args=(tmp_path, identity, first_score))
    second_writer = context.Process(target=_concurrent_put, args=(tmp_path, identity, second_score))
    first_writer.start()
    second_writer.start()
    first_writer.join(timeout=5)
    second_writer.join(timeout=5)

    assert not first_writer.is_alive()
    assert not second_writer.is_alive()
    assert first_writer.exitcode == 0
    assert second_writer.exitcode == 0
    assert torch.equal(ScoreCache(tmp_path).get(identity), first_score)


def test_score_entry_evidence_is_verified_and_hashes_exact_files(tmp_path: Path) -> None:
    cache = ScoreCache(tmp_path)
    identity = _identity()
    cache.put(identity, _score())
    tensor_path, metadata_path = _paths(tmp_path, identity)
    evidence = score_entry_evidence(tmp_path, identity)
    assert evidence == {
        "json": {
            "filename": metadata_path.name,
            "sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        },
        "safetensors": {
            "filename": tensor_path.name,
            "sha256": hashlib.sha256(tensor_path.read_bytes()).hexdigest(),
        },
    }
    assert tensor_sha256(_score()) == hashlib.sha256(_score().numpy().tobytes()).hexdigest()


def test_score_entry_evidence_refuses_invalid_entry(tmp_path: Path) -> None:
    cache = ScoreCache(tmp_path)
    identity = _identity()
    cache.put(identity, _score())
    tensor_path, _ = _paths(tmp_path, identity)
    os.truncate(tensor_path, 3)
    with pytest.raises(CacheIntegrityError):
        score_entry_evidence(tmp_path, identity)
