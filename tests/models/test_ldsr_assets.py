import hashlib
from pathlib import Path

import pytest

from trustsr.models.ldsr_assets import (
    CHECKPOINT_NAME,
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE,
    CHECKPOINT_URL,
    CONFIG_RELATIVE_PATH,
    CONFIG_SHA256,
    CONFIG_SIZE,
    AssetIntegrityError,
    download_verified_checkpoint,
    file_sha256,
    verify_asset,
    verify_packaged_config,
)


class FakeResponse:
    def __init__(self, payload: bytes, error: Exception | None = None):
        self.payload, self.error = payload, error

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size):
        if self.error:
            raise self.error
        chunk, self.payload = self.payload[:size], self.payload[size:]
        return chunk


class FakeOpener:
    def __init__(self, payload: bytes, error: Exception | None = None):
        self.payload, self.error, self.calls = payload, error, 0

    def __call__(self, url):
        self.calls += 1
        return FakeResponse(self.payload, self.error)


def test_download_verifies_then_atomically_commits(tmp_path):
    payload = b"verified checkpoint"
    asset = download_verified_checkpoint(
        tmp_path,
        opener=FakeOpener(payload),
        expected_size=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )
    assert asset.path == tmp_path / CHECKPOINT_NAME
    assert asset.path.read_bytes() == payload
    assert not list(tmp_path.glob(f".{CHECKPOINT_NAME}.*.tmp"))


def test_download_hash_failure_never_commits_final_file(tmp_path):
    with pytest.raises(AssetIntegrityError, match="SHA-256"):
        download_verified_checkpoint(
            tmp_path, opener=FakeOpener(b"wrong"), expected_size=5, expected_sha256="0" * 64
        )
    assert not (tmp_path / CHECKPOINT_NAME).exists()


def test_missing_and_size_mismatch_fail(tmp_path):
    with pytest.raises(AssetIntegrityError, match="does not exist"):
        verify_asset(tmp_path / "missing", expected_size=1, expected_sha256="0" * 64)
    path = tmp_path / "asset"
    path.write_bytes(b"abc")
    with pytest.raises(AssetIntegrityError, match="size"):
        verify_asset(path, expected_size=4, expected_sha256="0" * 64)


def test_existing_valid_file_reused_without_network(tmp_path):
    path = tmp_path / CHECKPOINT_NAME
    payload = b"cached"
    path.write_bytes(payload)
    opener = FakeOpener(b"network")
    asset = download_verified_checkpoint(
        tmp_path,
        opener=opener,
        expected_size=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )
    assert asset.path == path
    assert opener.calls == 0


def test_existing_invalid_file_fails_without_redownload(tmp_path):
    path = tmp_path / CHECKPOINT_NAME
    path.write_bytes(b"invalid")
    opener = FakeOpener(b"valid")
    with pytest.raises(AssetIntegrityError):
        download_verified_checkpoint(
            tmp_path, opener=opener, expected_size=5, expected_sha256="0" * 64
        )
    assert opener.calls == 0
    assert path.read_bytes() == b"invalid"


def test_read_error_cleans_temporary_file(tmp_path):
    with pytest.raises(OSError, match="read failed"):
        download_verified_checkpoint(
            tmp_path,
            opener=FakeOpener(b"partial", OSError("read failed")),
            expected_size=7,
            expected_sha256="0" * 64,
        )
    assert not list(tmp_path.glob(f".{CHECKPOINT_NAME}.*.tmp"))
    assert not (tmp_path / CHECKPOINT_NAME).exists()


def test_fsync_error_cleans_temporary_file(tmp_path, monkeypatch):
    def fail_fsync(_fd):
        raise OSError("fsync failed")

    monkeypatch.setattr("trustsr.models.ldsr_assets.os.fsync", fail_fsync)
    with pytest.raises(OSError, match="fsync failed"):
        download_verified_checkpoint(
            tmp_path,
            opener=FakeOpener(b"payload"),
            expected_size=7,
            expected_sha256=hashlib.sha256(b"payload").hexdigest(),
        )
    assert not list(tmp_path.glob(f".{CHECKPOINT_NAME}.*.tmp"))
    assert not (tmp_path / CHECKPOINT_NAME).exists()


def test_config_is_confined_and_verified(tmp_path):
    root = tmp_path / "package"
    config = root / CONFIG_RELATIVE_PATH
    config.parent.mkdir(parents=True)
    config.write_bytes(b"config" * 248 + b"xxx")
    with pytest.raises(AssetIntegrityError, match="size"):
        verify_packaged_config(root)
    config.write_bytes(b"x" * CONFIG_SIZE)
    with pytest.raises(AssetIntegrityError, match="SHA-256"):
        verify_packaged_config(root)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "configs").rename(root / "configs-real")
    (root / "configs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        verify_packaged_config(root)


def test_file_sha256_and_constants():
    assert file_sha256(Path(__file__))
    assert CHECKPOINT_NAME == "opensr-ldsrs2_v1_0_0.ckpt"
    assert CHECKPOINT_URL == (
        "https://huggingface.co/simon-donike/RS-SR-LTDF/resolve/main/"
        "opensr-ldsrs2_v1_0_0.ckpt"
    )
    assert CHECKPOINT_SIZE == 1_130_715_795
    assert CHECKPOINT_SHA256 == "e2621e3912eb7c14867c3d20c9029607ba941be8e166dc09621860fcac27dc3a"
    assert CONFIG_RELATIVE_PATH == Path("configs/config_10m.yaml")
    assert CONFIG_SIZE == 1_487
    assert CONFIG_SHA256 == "ac76685d354bfec32e3e0641aef574bedd7d650402c97dbd0ade86304e69ca6f"
