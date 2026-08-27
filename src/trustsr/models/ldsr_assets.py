"""Supply-chain checks and atomic acquisition of LDSR-S2 assets."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from urllib.request import urlopen

CHECKPOINT_NAME = "opensr-ldsrs2_v1_0_0.ckpt"
CHECKPOINT_URL = (
    "https://huggingface.co/simon-donike/RS-SR-LTDF/resolve/main/" + CHECKPOINT_NAME
)
CHECKPOINT_SIZE = 1_130_715_795
CHECKPOINT_SHA256 = "e2621e3912eb7c14867c3d20c9029607ba941be8e166dc09621860fcac27dc3a"
CONFIG_RELATIVE_PATH = Path("configs/config_10m.yaml")
CONFIG_SIZE = 1_487
CONFIG_SHA256 = "ac76685d354bfec32e3e0641aef574bedd7d650402c97dbd0ade86304e69ca6f"


class AssetIntegrityError(RuntimeError):
    """Raised when an asset is absent or differs from its frozen specification."""


@dataclass(frozen=True)
class VerifiedAsset:
    path: Path
    size: int
    sha256: str


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify_asset(
    path: Path | str, *, expected_size: int, expected_sha256: str
) -> VerifiedAsset:
    asset_path = Path(path)
    try:
        size = 0
        digest = hashlib.sha256()
        with asset_path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                size += len(block)
                digest.update(block)
    except OSError as exc:
        raise AssetIntegrityError(f"asset does not exist or cannot be read: {asset_path}") from exc
    actual_sha256 = digest.hexdigest()
    if size != expected_size:
        raise AssetIntegrityError(f"asset size {size} does not match expected size {expected_size}")
    if actual_sha256 != expected_sha256:
        raise AssetIntegrityError(
            f"asset SHA-256 {actual_sha256} does not match expected SHA-256 {expected_sha256}"
        )
    return VerifiedAsset(asset_path, size, actual_sha256)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def download_verified_checkpoint(
    model_dir: Path | str,
    *,
    opener: Callable[..., AbstractContextManager[BinaryIO]] = urlopen,
    url: str = CHECKPOINT_URL,
    expected_size: int = CHECKPOINT_SIZE,
    expected_sha256: str = CHECKPOINT_SHA256,
) -> VerifiedAsset:
    model_root = Path(model_dir).resolve()
    model_root.mkdir(parents=True, exist_ok=True)
    final_path = (model_root / CHECKPOINT_NAME).resolve()
    if final_path.parent != model_root:
        raise ValueError("checkpoint path escapes model directory")
    if final_path.exists():
        return verify_asset(
            final_path, expected_size=expected_size, expected_sha256=expected_sha256
        )

    temporary_path: Path | None = None
    try:
        size = 0
        digest = hashlib.sha256()
        with opener(url) as response:
            fd, temporary_name = tempfile.mkstemp(
                dir=model_root, prefix=f".{CHECKPOINT_NAME}.", suffix=".tmp"
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(fd, "wb") as output:
                while block := response.read(1024 * 1024):
                    size += len(block)
                    digest.update(block)
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
        actual_sha256 = digest.hexdigest()
        if size != expected_size:
            raise AssetIntegrityError(
                f"asset size {size} does not match expected size {expected_size}"
            )
        if actual_sha256 != expected_sha256:
            raise AssetIntegrityError(
                f"asset SHA-256 {actual_sha256} does not match expected SHA-256 {expected_sha256}"
            )
        os.replace(temporary_path, final_path)
        temporary_path = None
        _fsync_directory(model_root)
        return VerifiedAsset(final_path, size, actual_sha256)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def verify_packaged_config(package_root: Path | str) -> VerifiedAsset:
    package_path = Path(package_root).resolve()
    config_path = (package_path / CONFIG_RELATIVE_PATH).resolve()
    try:
        config_path.relative_to(package_path)
    except ValueError:
        raise ValueError("config path escapes package root") from None
    return verify_asset(config_path, expected_size=CONFIG_SIZE, expected_sha256=CONFIG_SHA256)
