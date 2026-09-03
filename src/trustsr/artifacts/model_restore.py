"""Verified unprivileged model-tree restore for Phase 2B3-A."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from trustsr.jsonio import canonical_json

_HASH_CHUNK_SIZE = 1024 * 1024
_RENAME_NOREPLACE = 1


class ModelRestoreError(RuntimeError):
    """A model source or copy violates the restore contract."""


@dataclass(frozen=True)
class ModelCopyResult:
    """Digest identities for two verified model-tree copies."""

    sen2srlite_inventory_sha256: str
    ldsr_inventory_sha256: str
    mode: str = "copy"

    def as_dict(self) -> dict[str, str]:
        return {
            "ldsr_inventory_sha256": self.ldsr_inventory_sha256,
            "mode": self.mode,
            "sen2srlite_inventory_sha256": self.sen2srlite_inventory_sha256,
            "status": "models-restored",
        }


def _require_directory(path: Path, description: str) -> os.stat_result:
    try:
        source = path.lstat()
    except OSError as exc:
        raise ModelRestoreError(f"{description} is unavailable") from exc
    if stat.S_ISLNK(source.st_mode) or not stat.S_ISDIR(source.st_mode):
        raise ModelRestoreError(f"{description} must be a non-symlink directory")
    return source


def _safe_relative(relative: PurePosixPath) -> str:
    value = relative.as_posix()
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ModelRestoreError("model entry name is not valid UTF-8") from exc
    if value != "." and (relative.is_absolute() or ".." in relative.parts or not relative.parts):
        raise ModelRestoreError("model entry path is unsafe")
    return value


def _directory_changed(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
    )


def _file_entry(
    directory_descriptor: int,
    name: str,
    relative: PurePosixPath,
    source: os.stat_result,
) -> dict[str, object]:
    if source.st_nlink != 1:
        raise ModelRestoreError("model source contains a hard link")
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory_descriptor,
        )
    except OSError as exc:
        raise ModelRestoreError("model source file could not be opened safely") from exc
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_dev != source.st_dev
                or opened.st_ino != source.st_ino
                or opened.st_size != source.st_size
                or opened.st_nlink != 1
            ):
                raise ModelRestoreError("model source file changed while being opened")
            for block in iter(lambda: stream.read(_HASH_CHUNK_SIZE), b""):
                digest.update(block)
            current = os.fstat(stream.fileno())
            if (
                current.st_size != opened.st_size
                or current.st_mtime_ns != opened.st_mtime_ns
                or current.st_ctime_ns != opened.st_ctime_ns
                or current.st_nlink != 1
            ):
                raise ModelRestoreError("model source file changed while being read")
    except OSError as exc:
        raise ModelRestoreError("model source file could not be read") from exc
    return {
        "mode": stat.S_IMODE(source.st_mode),
        "path": _safe_relative(relative),
        "sha256": digest.hexdigest(),
        "size_bytes": source.st_size,
        "type": "file",
    }


def _inventory_directory(
    directory_descriptor: int,
    relative: PurePosixPath,
    inventory: list[dict[str, object]],
) -> None:
    before = os.fstat(directory_descriptor)
    if not stat.S_ISDIR(before.st_mode):
        raise ModelRestoreError("model source permits only directories and regular files")
    inventory.append(
        {
            "mode": stat.S_IMODE(before.st_mode),
            "path": _safe_relative(relative),
            "type": "directory",
        }
    )
    try:
        with os.scandir(directory_descriptor) as entries:
            names = sorted((entry.name for entry in entries), key=lambda item: item.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ModelRestoreError("model entry name is not valid UTF-8") from exc
    except OSError as exc:
        raise ModelRestoreError("model source directory could not be scanned") from exc
    for name in names:
        child_relative = relative / name if relative.as_posix() != "." else PurePosixPath(name)
        _safe_relative(child_relative)
        try:
            source = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        except OSError as exc:
            raise ModelRestoreError("model source entry is unavailable") from exc
        if stat.S_ISLNK(source.st_mode):
            raise ModelRestoreError("model source contains a symlink")
        if stat.S_ISDIR(source.st_mode):
            try:
                child = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=directory_descriptor,
                )
            except OSError as exc:
                raise ModelRestoreError(
                    "model source directory could not be opened safely"
                ) from exc
            try:
                opened = os.fstat(child)
                if opened.st_dev != source.st_dev or opened.st_ino != source.st_ino:
                    raise ModelRestoreError("model source directory changed while being opened")
                _inventory_directory(child, child_relative, inventory)
            finally:
                os.close(child)
        elif stat.S_ISREG(source.st_mode):
            inventory.append(_file_entry(directory_descriptor, name, child_relative, source))
        else:
            raise ModelRestoreError("model source permits only directories and regular files")
    if _directory_changed(before, os.fstat(directory_descriptor)):
        raise ModelRestoreError("model source directory changed while being scanned")


def _inventory(path: Path) -> tuple[list[dict[str, object]], str]:
    _require_directory(path, "model tree")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
    except OSError as exc:
        raise ModelRestoreError("model tree could not be opened safely") from exc
    entries: list[dict[str, object]] = []
    try:
        _inventory_directory(descriptor, PurePosixPath("."), entries)
    finally:
        os.close(descriptor)
    digest = hashlib.sha256(canonical_json({"entries": entries})).hexdigest()
    return entries, digest


def _copy_regular_file(
    source_directory: int,
    destination_directory: int,
    name: str,
    source: os.stat_result,
    copy_file: Callable[[BinaryIO, BinaryIO], None],
) -> None:
    if source.st_nlink != 1:
        raise ModelRestoreError("model source contains a hard link")
    try:
        source_descriptor = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=source_directory,
        )
        destination_descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=destination_directory,
        )
    except OSError as exc:
        if "source_descriptor" in locals():
            os.close(source_descriptor)
        raise ModelRestoreError("model file copy could not be opened safely") from exc
    try:
        with (
            os.fdopen(source_descriptor, "rb") as source_stream,
            os.fdopen(destination_descriptor, "wb") as destination_stream,
        ):
            opened = os.fstat(source_stream.fileno())
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_dev != source.st_dev
                or opened.st_ino != source.st_ino
                or opened.st_size != source.st_size
                or opened.st_nlink != 1
            ):
                raise ModelRestoreError("model source file changed while being opened")
            copy_file(source_stream, destination_stream)
            destination_stream.flush()
            os.fchmod(destination_stream.fileno(), stat.S_IMODE(opened.st_mode))
            os.fsync(destination_stream.fileno())
            current = os.fstat(source_stream.fileno())
            if (
                current.st_size != opened.st_size
                or current.st_mtime_ns != opened.st_mtime_ns
                or current.st_ctime_ns != opened.st_ctime_ns
                or current.st_nlink != 1
            ):
                raise ModelRestoreError("model source file changed while being copied")
    except ModelRestoreError:
        raise
    except OSError as exc:
        raise ModelRestoreError("model file copy failed") from exc


def _copy_directory(
    source_descriptor: int,
    destination_descriptor: int,
    copy_file: Callable[[BinaryIO, BinaryIO], None],
) -> None:
    before = os.fstat(source_descriptor)
    try:
        with os.scandir(source_descriptor) as entries:
            names = sorted((entry.name for entry in entries), key=lambda item: item.encode("utf-8"))
    except (OSError, UnicodeEncodeError) as exc:
        raise ModelRestoreError("model source directory could not be scanned") from exc
    for name in names:
        try:
            source = os.stat(name, dir_fd=source_descriptor, follow_symlinks=False)
        except OSError as exc:
            raise ModelRestoreError("model source entry is unavailable") from exc
        if stat.S_ISDIR(source.st_mode):
            child_source: int | None = None
            try:
                child_source = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=source_descriptor,
                )
                os.mkdir(name, mode=0o700, dir_fd=destination_descriptor)
                child_destination = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=destination_descriptor,
                )
            except OSError as exc:
                if child_source is not None:
                    os.close(child_source)
                raise ModelRestoreError("model directory copy could not be created safely") from exc
            try:
                opened = os.fstat(child_source)
                if opened.st_dev != source.st_dev or opened.st_ino != source.st_ino:
                    raise ModelRestoreError("model source directory changed while being opened")
                _copy_directory(child_source, child_destination, copy_file)
                os.fchmod(child_destination, stat.S_IMODE(opened.st_mode))
                os.fsync(child_destination)
            finally:
                os.close(child_destination)
                os.close(child_source)
        elif stat.S_ISREG(source.st_mode):
            _copy_regular_file(source_descriptor, destination_descriptor, name, source, copy_file)
        elif stat.S_ISLNK(source.st_mode):
            raise ModelRestoreError("model source contains a symlink")
        else:
            raise ModelRestoreError("model source permits only directories and regular files")
    if _directory_changed(before, os.fstat(source_descriptor)):
        raise ModelRestoreError("model source directory changed while being copied")


def _copy_tree(
    source: Path,
    destination: Path,
    copy_file: Callable[[BinaryIO, BinaryIO], None],
) -> None:
    source_mode = stat.S_IMODE(_require_directory(source, "model source").st_mode)
    destination.mkdir(mode=0o700)
    source_descriptor = os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    destination_descriptor = os.open(
        destination, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    )
    try:
        _copy_directory(source_descriptor, destination_descriptor, copy_file)
        os.fchmod(destination_descriptor, source_mode)
        os.fsync(destination_descriptor)
    finally:
        os.close(destination_descriptor)
        os.close(source_descriptor)


def _locked_inventory(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{**entry, "mode": int(entry["mode"]) & ~0o222} for entry in entries]


def _remove_write_permissions(path: Path, source_entries: list[dict[str, object]]) -> None:
    entries = _locked_inventory(source_entries)
    for entry in reversed(entries):
        relative = Path(str(entry["path"]))
        target = path if relative == Path(".") else path / relative
        os.chmod(target, int(entry["mode"]), follow_symlinks=False)
    observed, _ = _inventory(path)
    if observed != entries:
        raise ModelRestoreError("copied model inventory is not verified and non-writable")


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as exc:
        raise ModelRestoreError("renameat2 is unsupported; model restore fails closed") from exc
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise ModelRestoreError("model restore destination already exists")
    if error == errno.ENOSYS:
        raise ModelRestoreError("renameat2 is unsupported; model restore fails closed")
    raise ModelRestoreError(
        f"model restore destination could not be published: {os.strerror(error)}"
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_private_tree(path: Path) -> None:
    if not path.exists():
        return
    for root, directories, files in os.walk(path, topdown=False, followlinks=False):
        root_path = Path(root)
        for name in files:
            try:
                os.chmod(root_path / name, 0o600, follow_symlinks=False)
            except OSError:
                pass
        for name in directories:
            try:
                os.chmod(root_path / name, 0o700, follow_symlinks=False)
            except OSError:
                pass
        try:
            os.chmod(root_path, 0o700, follow_symlinks=False)
        except OSError:
            pass
    shutil.rmtree(path, ignore_errors=True)


def copy_model_trees(
    destination: Path,
    *,
    sen2srlite_source: Path,
    ldsr_source: Path,
    copy_file: Callable[[BinaryIO, BinaryIO], None] = shutil.copyfileobj,
) -> ModelCopyResult:
    """Copy, verify, make non-writable, and atomically publish both model trees."""
    destination = Path(destination)
    sen2srlite_source = Path(sen2srlite_source)
    ldsr_source = Path(ldsr_source)
    if destination.exists() or destination.is_symlink():
        raise ModelRestoreError("model restore destination already exists")
    destination.parent.mkdir(mode=0o700, exist_ok=True)
    _require_directory(destination.parent, "model restore parent")
    sen2_before, sen2_digest = _inventory(sen2srlite_source)
    ldsr_before, ldsr_digest = _inventory(ldsr_source)
    staging_directory = Path(
        tempfile.mkdtemp(prefix=".phase2b3a-model-copy.", dir=destination.parent)
    )
    staged_destination = staging_directory / destination.name
    published = False
    try:
        staged_destination.mkdir(mode=0o700)
        _copy_tree(
            sen2srlite_source,
            staged_destination / "sen2srlite",
            copy_file,
        )
        _copy_tree(
            ldsr_source,
            staged_destination / "ldsr-s2",
            copy_file,
        )
        sen2_after, _ = _inventory(sen2srlite_source)
        ldsr_after, _ = _inventory(ldsr_source)
        if sen2_before != sen2_after:
            raise ModelRestoreError("SEN2SRLite model inventory changed during copy")
        if ldsr_before != ldsr_after:
            raise ModelRestoreError("LDSR model inventory changed during copy")
        _remove_write_permissions(staged_destination / "sen2srlite", sen2_before)
        _remove_write_permissions(staged_destination / "ldsr-s2", ldsr_before)
        _fsync_directory(staged_destination)
        _rename_noreplace(staged_destination, destination)
        published = True
        os.chmod(destination, 0o500, follow_symlinks=False)
        _fsync_directory(destination)
        _fsync_directory(destination.parent)
        return ModelCopyResult(
            sen2srlite_inventory_sha256=sen2_digest,
            ldsr_inventory_sha256=ldsr_digest,
        )
    except ModelRestoreError:
        raise
    except OSError as exc:
        raise ModelRestoreError("model trees could not be restored") from exc
    finally:
        if published:
            try:
                locked = stat.S_IMODE(destination.stat().st_mode) == 0o500
            except OSError:
                locked = False
            if not locked:
                _remove_private_tree(destination)
        _remove_private_tree(staging_directory)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("sen2srlite_source", type=Path)
    parser.add_argument("ldsr_source", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = copy_model_trees(
            args.destination,
            sen2srlite_source=args.sen2srlite_source,
            ldsr_source=args.ldsr_source,
        )
    except ModelRestoreError as exc:
        print(f"model restore failed: {exc}", file=sys.stderr)
        return 2
    print(canonical_json(result.as_dict()).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
