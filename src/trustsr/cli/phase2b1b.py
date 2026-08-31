"""Run restartable Phase 2B1B research-subset stages in explicit cloud storage."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import cast

from trustsr.data.crosssensor_manifest import (
    SOURCE_OBJECT_NAME,
    SOURCE_OBJECT_SHA256,
    SOURCE_OBJECT_SIZE_BYTES,
    SOURCE_REVISION,
    ExtractedAsset,
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
from trustsr.data.taco_v1_adapter import extract_pair
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


def _load_frozen_base_from_storage(
    storage_root: Path,
) -> tuple[dict[str, object], ...]:
    path = (
        storage_root
        / "trustsr"
        / "phase2b1a"
        / "manifests"
        / BASE_MANIFEST_SHA256
        / "samples.jsonl"
    )
    return _load_base_manifest(storage_root, path)


def _load_digest_selection(
    storage_root: Path, manifest_path: Path
) -> tuple[str, tuple[dict[str, object], ...]]:
    if not isinstance(manifest_path, Path):
        raise TypeError("selection_manifest_path must be a pathlib.Path")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("selection manifest must be an existing regular file")
    resolved = manifest_path.resolve(strict=True)
    _require_confined(storage_root, resolved)
    selection_root = (_phase_root(storage_root) / "selections").resolve(strict=True)
    if resolved.name != "samples.jsonl" or resolved.parent.parent != selection_root:
        raise ValueError("selection manifest must use the digest-addressed stage layout")
    actual_sha256 = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if resolved.parent.name != actual_sha256:
        raise ValueError("selection manifest parent must equal its actual SHA-256")
    return actual_sha256, load_subset_manifest(
        resolved, expected_sha256=actual_sha256
    )


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


def _require_safe_component(sample_id: str) -> None:
    if (
        type(sample_id) is not str
        or not sample_id
        or "\x00" in sample_id
        or PurePosixPath(sample_id).parts != (sample_id,)
        or PureWindowsPath(sample_id).parts != (sample_id,)
        or sample_id in {".", ".."}
    ):
        raise ValueError("selection sample_id must be a safe path component")


def _require_absent_or_complete_pair(output_root: Path) -> None:
    if not output_root.exists() and not output_root.is_symlink():
        return
    if output_root.is_symlink() or not output_root.is_dir():
        raise ValueError("existing research-subset pair is partial or invalid")
    entries = tuple(output_root.iterdir())
    if {entry.name for entry in entries} != {"lr.tif", "hr.tif"} or any(
        entry.is_symlink() or not entry.is_file() for entry in entries
    ):
        raise ValueError("existing research-subset pair is partial or invalid")


def _hash_file(path: Path) -> tuple[int, str]:
    size_bytes = 0
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            size_bytes += len(block)
            digest.update(block)
    return size_bytes, digest.hexdigest()


def _require_matching_post_selection(
    input_records: Sequence[Mapping[str, object]],
    candidate_records: Sequence[Mapping[str, object]],
) -> None:
    if len(input_records) != len(candidate_records):
        raise ValueError("post-extraction sidecar does not match its input sidecar")
    asset_fields = {"lr_asset", "hr_asset"}
    for input_record, candidate_record in zip(
        input_records, candidate_records, strict=True
    ):
        if any(
            candidate_record[field] != value
            for field, value in input_record.items()
            if field not in asset_fields
        ):
            raise ValueError("post-extraction sidecar does not match its input sidecar")
        if candidate_record["lr_asset"] is None or candidate_record["hr_asset"] is None:
            raise ValueError("post-extraction sidecar must contain every asset pair")


def _verify_post_selection_assets(
    storage_root: Path, records: Sequence[Mapping[str, object]]
) -> None:
    phase_root = _phase_root(storage_root)
    asset_count = 0
    for record in records:
        sample_id = cast(str, record["sample_id"])
        split = cast(str, record["split"])
        _require_safe_component(sample_id)
        for kind in ("lr", "hr"):
            asset = cast(Mapping[str, object], record[f"{kind}_asset"])
            expected = PurePosixPath(
                "subset-v1", split, sample_id, f"{kind}.tif"
            ).as_posix()
            if asset["relative_path"] != expected:
                raise ValueError("asset relative_path must use the exact Phase 2B1B layout")
            path = phase_root / expected
            _require_confined(storage_root, path)
            if path.is_symlink() or not path.is_file():
                raise ValueError("post-extraction sidecar requires all 720 regular assets")
            if _hash_file(path) != (asset["size_bytes"], asset["sha256"]):
                raise ValueError("asset bytes do not match the post-extraction sidecar")
            asset_count += 1
    if asset_count != 720:
        raise ValueError("post-extraction sidecar requires all 720 regular assets")


def _find_reusable_post_selection(
    storage_root: Path,
    input_manifest_sha256: str,
    input_records: Sequence[Mapping[str, object]],
) -> ManifestArtifact | None:
    selection_root = _phase_root(storage_root) / "selections"
    matches: list[ManifestArtifact] = []
    for directory in sorted(selection_root.iterdir(), key=lambda path: path.name):
        if directory.name == input_manifest_sha256:
            continue
        candidate = directory / "samples.jsonl"
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("selection tree contains an invalid digest directory")
        if tuple(directory.iterdir()) != (candidate,):
            raise ValueError("selection digest directory is incomplete or invalid")
        digest, records = _load_digest_selection(storage_root, candidate)
        _require_matching_post_selection(input_records, records)
        _verify_post_selection_assets(storage_root, records)
        matches.append(ManifestArtifact(candidate, candidate.stat().st_size, digest))
    if len(matches) > 1:
        raise ValueError("multiple matching post-extraction sidecars are ambiguous")
    return matches[0] if matches else None


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


def run_extract(
    source_path: Path,
    storage_root: Path,
    selection_manifest_path: Path,
    *,
    confirmed_cloud_storage: bool,
) -> dict[str, object]:
    """Extract all 360 independent pairs and commit an all-assets sidecar."""
    root = require_cloud_confirmation(storage_root, confirmed_cloud_storage)
    source, object_spec = _load_frozen_source(source_path)
    verified = verify_crosssensor(source_paths(root, object_spec).final, object_spec)
    if (
        verified.size_bytes != SOURCE_OBJECT_SIZE_BYTES
        or verified.sha256 != SOURCE_OBJECT_SHA256
    ):
        raise ValueError("verified source does not match the frozen crosssensor object")
    input_digest, records = _load_digest_selection(root, selection_manifest_path)
    if any(
        record["lr_asset"] is not None or record["hr_asset"] is not None
        for record in records
    ):
        raise ValueError("extract requires an all-null pre-extraction sidecar")
    base_records = _load_frozen_base_from_storage(root)
    choices = validate_subset_against_base(records, base_records)
    if len(choices) != 360:
        raise ValueError("extract stage requires exactly 360 deterministic choices")
    records_by_id = {cast(str, record["sample_id"]): record for record in records}
    outputs = {
        choice.sample_id: _phase_root(root)
        / "subset-v1"
        / choice.split
        / choice.sample_id
        for choice in choices
    }
    for sample_id, output_root in outputs.items():
        _require_safe_component(sample_id)
        _require_confined(root, output_root)
        _require_absent_or_complete_pair(output_root)

    reusable = _find_reusable_post_selection(root, input_digest, records)
    if reusable is not None:
        artifact = reusable
        reused = True
    else:
        assets: dict[str, tuple[ExtractedAsset, ExtractedAsset]] = {}
        for choice in choices:
            record = records_by_id[choice.sample_id]
            lr_asset, hr_asset = extract_pair(
                verified.path,
                cast(int, record["source_index"]),
                outputs[choice.sample_id],
                source.bands,
            )
            if lr_asset.relative_path != "lr.tif" or hr_asset.relative_path != "hr.tif":
                raise ValueError("extractor must return only lr.tif and hr.tif")
            if lr_asset.time_start != record["lr_time_start"]:
                raise ValueError("extracted LR time_start must equal the sidecar")
            if hr_asset.time_start != record["hr_time_start"]:
                raise ValueError("extracted HR time_start must equal the sidecar")
            prefix = PurePosixPath("subset-v1", choice.split, choice.sample_id)
            assets[choice.sample_id] = (
                replace(lr_asset, relative_path=(prefix / "lr.tif").as_posix()),
                replace(hr_asset, relative_path=(prefix / "hr.tif").as_posix()),
            )

        selection_root = _phase_root(root) / "selections"
        with tempfile.TemporaryDirectory(
            prefix=".candidate-", dir=selection_root
        ) as temporary:
            candidate_path = Path(temporary) / "samples.jsonl"
            artifact = write_subset_manifest(
                candidate_path, base_records, choices, assets
            )
            post_records = load_subset_manifest(
                candidate_path, expected_sha256=artifact.sha256
            )
            validate_subset_against_base(post_records, base_records)
            _require_matching_post_selection(records, post_records)
            _verify_post_selection_assets(root, post_records)
            _, reused = _commit_selection(selection_root, artifact)
    return {
        "stage": "extract",
        "digests": {
            "base_manifest_sha256": BASE_MANIFEST_SHA256,
            "input_selection_manifest_sha256": input_digest,
            "selection_manifest_sha256": artifact.sha256,
            "source_sha256": verified.sha256,
        },
        "counts": {"subset_pairs": 360, "subset_geotiffs": 720},
        "reused": reused,
    }


def main(argv: list[str] | None = None) -> int:
    """Dispatch an implemented Phase 2B1B stage and emit canonical JSON."""
    args = build_parser().parse_args(argv)
    if args.stage == "select":
        payload = run_select(
            args.source,
            args.storage_root,
            args.base_manifest,
            confirmed_cloud_storage=args.confirm_cloud_storage,
        )
    elif args.stage == "extract":
        payload = run_extract(
            args.source,
            args.storage_root,
            args.selection_manifest,
            confirmed_cloud_storage=args.confirm_cloud_storage,
        )
    else:
        raise ValueError(f"Phase 2B1B stage is not implemented yet: {args.stage}")
    print(canonical_json(payload).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
