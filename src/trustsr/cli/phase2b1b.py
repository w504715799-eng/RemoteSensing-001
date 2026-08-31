"""Run restartable Phase 2B1B research-subset stages in explicit cloud storage."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from trustsr.data.crosssensor_manifest import (
    SOURCE_OBJECT_NAME,
    SOURCE_OBJECT_SHA256,
    SOURCE_OBJECT_SIZE_BYTES,
    SOURCE_REVISION,
    ManifestArtifact,
    load_manifest,
)
from trustsr.data.crosssensor_source import (
    require_cloud_confirmation,
    require_crosssensor_object,
    source_paths,
    verify_crosssensor,
)
from trustsr.data.provenance import DatasetSource, LfsObject, load_dataset_source
from trustsr.data.subset_manifest import (
    BASE_MANIFEST_SHA256,
    load_subset_manifest,
    select_from_base_manifest,
    validate_subset_against_base,
    write_subset_manifest,
)
from trustsr.jsonio import atomic_write_bytes, canonical_json

_SOURCE_REPOSITORY = "tacofoundation/SEN2NAIPv2"
_SOURCE_LICENSE = "cc0-1.0"


def build_parser() -> argparse.ArgumentParser:
    """Build the three-stage Phase 2B1B command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for stage in ("select", "extract", "audit"):
        command = subparsers.add_parser(stage)
        command.add_argument("--source", type=Path, required=True)
        command.add_argument("--storage-root", type=Path, required=True)
        command.add_argument("--confirm-cloud-storage", action="store_true")
        if stage == "select":
            command.add_argument("--base-manifest", type=Path, required=True)
        else:
            command.add_argument("--selection-manifest", type=Path, required=True)
    return parser


def _load_frozen_source(source_path: Path) -> tuple[DatasetSource, LfsObject]:
    source = load_dataset_source(source_path)
    object_spec = require_crosssensor_object(source)
    if (
        source.repository != _SOURCE_REPOSITORY
        or source.revision != SOURCE_REVISION
        or source.license_claim != _SOURCE_LICENSE
        or object_spec.path != SOURCE_OBJECT_NAME
        or object_spec.sha256 != SOURCE_OBJECT_SHA256
        or object_spec.size_bytes != SOURCE_OBJECT_SIZE_BYTES
    ):
        raise ValueError("source does not match the frozen crosssensor source")
    return source, object_spec


def _phase_root(storage_root: Path) -> Path:
    return storage_root / "trustsr" / "phase2b1b"


def _require_confined(storage_root: Path, candidate: Path) -> Path:
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(storage_root)
    except ValueError:
        raise ValueError("derived Phase 2B1B path escapes storage_root") from None
    return resolved


def _load_base_manifest(
    storage_root: Path, manifest_path: Path
) -> tuple[dict[str, object], ...]:
    if not isinstance(manifest_path, Path):
        raise TypeError("base_manifest_path must be a pathlib.Path")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("base manifest must be the frozen Phase 2B1A manifest")
    resolved = manifest_path.resolve(strict=True)
    _require_confined(storage_root, resolved)
    expected = (
        storage_root
        / "trustsr"
        / "phase2b1a"
        / "manifests"
        / BASE_MANIFEST_SHA256
        / "samples.jsonl"
    ).resolve(strict=False)
    if resolved != expected:
        raise ValueError("base manifest must be the frozen Phase 2B1A manifest")
    return load_manifest(resolved, expected_sha256=BASE_MANIFEST_SHA256)


def _commit_selection(
    selection_root: Path, candidate: ManifestArtifact
) -> tuple[Path, bool]:
    destination_directory = selection_root / candidate.sha256
    destination = destination_directory / "samples.jsonl"
    payload = candidate.path.read_bytes()
    if destination_directory.exists() or destination_directory.is_symlink():
        if destination_directory.is_symlink() or not destination_directory.is_dir():
            raise ValueError("existing selection digest directory is invalid")
        if tuple(destination_directory.iterdir()) != (destination,):
            raise ValueError("existing selection digest directory is incomplete or invalid")
        if destination.is_symlink() or not destination.is_file():
            raise ValueError("existing digest-addressed selection is invalid")
        load_subset_manifest(destination, expected_sha256=candidate.sha256)
        if destination.read_bytes() != payload:
            raise ValueError("existing digest-addressed selection has different bytes")
        return destination, True

    destination_directory.mkdir()
    atomic_write_bytes(destination, payload)
    load_subset_manifest(destination, expected_sha256=candidate.sha256)
    return destination, False


def run_select(
    source_path: Path,
    storage_root: Path,
    base_manifest_path: Path,
    *,
    confirmed_cloud_storage: bool,
) -> dict[str, object]:
    """Select and commit the canonical all-null Phase 2B1B sidecar."""
    root = require_cloud_confirmation(storage_root, confirmed_cloud_storage)
    _, object_spec = _load_frozen_source(source_path)
    verified = verify_crosssensor(source_paths(root, object_spec).final, object_spec)
    if (
        verified.size_bytes != SOURCE_OBJECT_SIZE_BYTES
        or verified.sha256 != SOURCE_OBJECT_SHA256
    ):
        raise ValueError("verified source does not match the frozen crosssensor object")
    base_records = _load_base_manifest(root, base_manifest_path)
    choices = select_from_base_manifest(base_records)
    if len(choices) != 360:
        raise ValueError("select stage must choose exactly 360 pairs")

    selection_root = _phase_root(root) / "selections"
    _require_confined(root, selection_root)
    selection_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".candidate-", dir=selection_root) as temporary:
        candidate_path = Path(temporary) / "samples.jsonl"
        artifact = write_subset_manifest(candidate_path, base_records, choices, {})
        records = load_subset_manifest(candidate_path, expected_sha256=artifact.sha256)
        validate_subset_against_base(records, base_records)
        _, reused = _commit_selection(selection_root, artifact)
    return {
        "stage": "select",
        "digests": {
            "base_manifest_sha256": BASE_MANIFEST_SHA256,
            "selection_manifest_sha256": artifact.sha256,
            "source_sha256": verified.sha256,
        },
        "counts": {"subset_pairs": 360, "subset_geotiffs": 0},
        "reused": reused,
    }


def main(argv: list[str] | None = None) -> int:
    """Dispatch an implemented Phase 2B1B stage and emit canonical JSON."""
    args = build_parser().parse_args(argv)
    if args.stage != "select":
        raise ValueError(f"Phase 2B1B stage is not implemented yet: {args.stage}")
    payload = run_select(
        args.source,
        args.storage_root,
        args.base_manifest,
        confirmed_cloud_storage=args.confirm_cloud_storage,
    )
    print(canonical_json(payload).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
