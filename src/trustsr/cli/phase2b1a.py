"""Run restartable Phase 2B1A crosssensor data stages in explicit cloud storage."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import cast

from trustsr.data.crosssensor_manifest import (
    PRODUCTION_EXPECTED_COUNTS,
    SOURCE_OBJECT_NAME,
    SOURCE_OBJECT_SHA256,
    SOURCE_OBJECT_SIZE_BYTES,
    SOURCE_REVISION,
    ExtractedAsset,
    ManifestArtifact,
    build_audit,
    load_manifest,
    write_manifest,
)
from trustsr.data.crosssensor_schema import CrosssensorSample, normalize_top_level
from trustsr.data.crosssensor_source import (
    VerifiedSourceObject,
    acquire_crosssensor,
    require_cloud_confirmation,
    require_crosssensor_object,
    source_paths,
    verify_crosssensor,
)
from trustsr.data.pilot_sampling import select_pilot
from trustsr.data.provenance import DatasetSource, LfsObject, load_dataset_source
from trustsr.data.spatial_split import (
    AssignedSample,
    assign_spatial_splits,
    minimum_cross_split_distances,
)
from trustsr.data.taco_v1_adapter import extract_pair, load_crosssensor_metadata
from trustsr.jsonio import atomic_write_bytes, canonical_json

_SOURCE_REPOSITORY = "tacofoundation/SEN2NAIPv2"
_SOURCE_LICENSE = "cc0-1.0"
def build_parser() -> argparse.ArgumentParser:
    """Build the four-stage Phase 2B1A command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for stage in ("download", "manifest", "pilot", "audit"):
        subparser = subparsers.add_parser(stage)
        subparser.add_argument("--source", type=Path, required=True)
        subparser.add_argument("--storage-root", type=Path, required=True)
        subparser.add_argument("--confirm-cloud-storage", action="store_true")
        if stage == "download":
            subparser.add_argument("--transport-url", required=True)
        if stage in {"pilot", "audit"}:
            subparser.add_argument("--manifest", type=Path, required=True)
    return parser


def run_download(
    source_path: Path,
    storage_root: Path,
    transport_url: str,
    *,
    confirmed_cloud_storage: bool,
) -> dict[str, object]:
    """Run the source-acquisition stage."""
    source, object_spec = _load_frozen_source(source_path)
    final_path = source_paths(storage_root, object_spec).final
    reused = final_path.exists() or final_path.is_symlink()
    acquired = acquire_crosssensor(
        source,
        storage_root,
        transport_url,
        confirmed_cloud_storage=confirmed_cloud_storage,
    )
    return {
        "stage": "download",
        "digests": {"source_sha256": acquired.sha256},
        "counts": {"source_bytes": acquired.size_bytes},
        "reused": reused,
    }


def run_manifest(
    source_path: Path,
    storage_root: Path,
    *,
    confirmed_cloud_storage: bool,
) -> dict[str, object]:
    """Run the pre-extraction manifest stage."""
    root = require_cloud_confirmation(storage_root, confirmed_cloud_storage)
    source, object_spec = _load_frozen_source(source_path)
    taco_path = source_paths(root, object_spec).final
    verified = verify_crosssensor(taco_path, object_spec)
    records, acquisition_times = load_crosssensor_metadata(verified.path)
    _require_top_level_shape(records)
    samples = normalize_top_level(
        records, acquisition_times=acquisition_times, expected_count=8_000
    )
    assignments = assign_spatial_splits(samples)
    minimum_distances = minimum_cross_split_distances(assignments)
    choices = select_pilot(assignments)
    if len(choices) != 36:
        raise ValueError("manifest stage must select exactly 36 pilot samples")

    manifest_root = _phase_root(root) / "manifests"
    _require_confined(root, manifest_root)
    manifest_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".candidate-", dir=manifest_root) as temporary:
        candidate_path = Path(temporary) / "samples.jsonl"
        artifact = write_manifest(candidate_path, assignments, choices, {})
        manifest_records = load_manifest(candidate_path, expected_sha256=artifact.sha256)
        audit = build_audit(
            manifest_records,
            manifest_sha256=artifact.sha256,
            minimum_distances=minimum_distances,
            expected=PRODUCTION_EXPECTED_COUNTS,
        )
        _require_audit_source_identity(audit, verified)
        if audit["pilot_pair_count"] != 36 or audit["pilot_geotiff_count"] != 0:
            raise ValueError("pre-extraction manifest must contain 36 null-asset pilot pairs")
        _, reused = _commit_manifest(manifest_root, artifact)

    return {
        "stage": "manifest",
        "digests": {
            "manifest_sha256": artifact.sha256,
            "source_sha256": verified.sha256,
        },
        "counts": {
            "samples": audit["sample_count"],
            "components": audit["component_count"],
            "pilot_pairs": audit["pilot_pair_count"],
        },
        "reused": reused,
    }


def run_pilot(
    source_path: Path,
    storage_root: Path,
    manifest_path: Path,
    *,
    confirmed_cloud_storage: bool,
) -> dict[str, object]:
    """Run the 36-pair extraction stage."""
    root = require_cloud_confirmation(storage_root, confirmed_cloud_storage)
    source, object_spec = _load_frozen_source(source_path)
    taco_path = source_paths(root, object_spec).final
    verified = verify_crosssensor(taco_path, object_spec)
    input_digest, records = _load_digest_manifest(root, manifest_path)
    if any(record["lr_asset"] is not None or record["hr_asset"] is not None for record in records):
        raise ValueError("pilot input must be a pre-extraction manifest with null assets")
    selected_records = {
        cast(str, record["sample_id"]): record
        for record in records
        if record["pilot"] is not None
    }
    if len(selected_records) != 36:
        raise ValueError("pilot manifest must select exactly 36 samples")
    for sample_id in selected_records:
        _require_safe_component(sample_id)

    assignments = _assignments_from_records(records)
    choices = select_pilot(assignments)
    if len(choices) != 36 or {choice.sample_id for choice in choices} != set(selected_records):
        raise ValueError("pilot manifest must contain exactly 36 deterministic choices")
    pre_audit = build_audit(
        records,
        manifest_sha256=input_digest,
        minimum_distances=minimum_cross_split_distances(assignments),
        expected=PRODUCTION_EXPECTED_COUNTS,
    )
    _require_audit_source_identity(pre_audit, verified)
    if pre_audit["pilot_pair_count"] != 36 or pre_audit["pilot_geotiff_count"] != 0:
        raise ValueError("pilot input must audit as 36 null-asset pilot pairs")

    phase_root = _phase_root(root)
    outputs = {
        choice.sample_id: phase_root / "pilot-v1" / choice.split / choice.sample_id
        for choice in choices
    }
    for output_root in outputs.values():
        _require_confined(root, output_root)
        _require_absent_or_complete_pair(output_root)

    manifest_root = phase_root / "manifests"
    reusable = _find_reusable_post_manifest(root, input_digest, records)
    if reusable is not None:
        artifact = reusable
        reused = True
    else:
        assets: dict[str, tuple[ExtractedAsset, ExtractedAsset]] = {}
        for choice in choices:
            record = selected_records[choice.sample_id]
            output_root = outputs[choice.sample_id]
            lr_asset, hr_asset = extract_pair(
                verified.path,
                cast(int, record["source_index"]),
                output_root,
                source.bands,
            )
            if lr_asset.relative_path != "lr.tif" or hr_asset.relative_path != "hr.tif":
                raise ValueError("extractor must return only lr.tif and hr.tif asset paths")
            if lr_asset.time_start != record["lr_time_start"]:
                raise ValueError("extracted LR time_start must equal manifest lr_time_start")
            if hr_asset.time_start != record["hr_time_start"]:
                raise ValueError("extracted HR time_start must equal manifest hr_time_start")
            prefix = PurePosixPath("pilot-v1", choice.split, choice.sample_id)
            assets[choice.sample_id] = (
                replace(lr_asset, relative_path=(prefix / "lr.tif").as_posix()),
                replace(hr_asset, relative_path=(prefix / "hr.tif").as_posix()),
            )

        with tempfile.TemporaryDirectory(prefix=".candidate-", dir=manifest_root) as temporary:
            candidate_path = Path(temporary) / "samples.jsonl"
            artifact = write_manifest(candidate_path, assignments, choices, assets)
            _, reused = _commit_manifest(manifest_root, artifact)

    return {
        "stage": "pilot",
        "digests": {
            "input_manifest_sha256": input_digest,
            "manifest_sha256": artifact.sha256,
            "source_sha256": verified.sha256,
        },
        "counts": {
            "samples": len(records),
            "pilot_pairs": len(choices),
            "pilot_geotiffs": len(choices) * 2,
        },
        "reused": reused,
    }


def run_audit(
    source_path: Path,
    storage_root: Path,
    manifest_path: Path,
    *,
    confirmed_cloud_storage: bool,
) -> dict[str, object]:
    """Run the post-extraction audit stage."""
    root = require_cloud_confirmation(storage_root, confirmed_cloud_storage)
    _, object_spec = _load_frozen_source(source_path)
    taco_path = source_paths(root, object_spec).final
    verified = verify_crosssensor(taco_path, object_spec)
    manifest_digest, records = _load_digest_manifest(root, manifest_path)
    selected_records = tuple(record for record in records if record["pilot"] is not None)
    if len(selected_records) != 36 or any(
        record["lr_asset"] is None or record["hr_asset"] is None
        for record in selected_records
    ):
        raise ValueError("audit requires a post-extraction manifest with all 72 GeoTIFF assets")

    phase_root = _phase_root(root)
    asset_files: list[tuple[Path, Mapping[str, object]]] = []
    for record in selected_records:
        sample_id = cast(str, record["sample_id"])
        split = cast(str, record["split"])
        _require_safe_component(sample_id)
        for kind in ("lr", "hr"):
            asset = cast(Mapping[str, object], record[f"{kind}_asset"])
            expected_relative = PurePosixPath(
                "pilot-v1", split, sample_id, f"{kind}.tif"
            ).as_posix()
            if asset["relative_path"] != expected_relative:
                raise ValueError("asset relative_path must use the exact pilot sample layout")
            asset_path = phase_root / expected_relative
            _require_confined(root, asset_path)
            asset_files.append((asset_path, asset))
    if len(asset_files) != 72 or any(
        path.is_symlink() or not path.is_file() for path, _ in asset_files
    ):
        raise ValueError("audit requires all 72 confined GeoTIFF files")

    for path, asset in asset_files:
        size_bytes, sha256 = _hash_file(path)
        if size_bytes != asset["size_bytes"]:
            raise ValueError("asset byte size does not match the post-extraction manifest")
        if sha256 != asset["sha256"]:
            raise ValueError("asset SHA-256 does not match the post-extraction manifest")

    assignments = _assignments_from_records(records)
    minimum_distances = minimum_cross_split_distances(assignments)
    audit = build_audit(
        records,
        manifest_sha256=manifest_digest,
        minimum_distances=minimum_distances,
        expected=PRODUCTION_EXPECTED_COUNTS,
    )
    _require_audit_source_identity(audit, verified)
    if audit["pilot_pair_count"] != 36 or audit["pilot_geotiff_count"] != 72:
        raise ValueError("audit must report exactly 36 pilot pairs and 72 GeoTIFF files")
    audit_payload = canonical_json(audit)
    audit_root = phase_root / "audits" / manifest_digest
    _require_confined(root, audit_root)
    _, reused = _commit_audit(phase_root, manifest_digest, audit_payload)
    return {
        "stage": "audit",
        "digests": {
            "audit_sha256": hashlib.sha256(audit_payload).hexdigest(),
            "manifest_sha256": manifest_digest,
            "source_sha256": verified.sha256,
        },
        "counts": {
            "samples": audit["sample_count"],
            "components": audit["component_count"],
            "pilot_pairs": audit["pilot_pair_count"],
            "pilot_geotiffs": audit["pilot_geotiff_count"],
        },
        "reused": reused,
    }


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


def _require_audit_source_identity(
    audit: Mapping[str, object], verified: VerifiedSourceObject
) -> None:
    if (
        audit.get("source_revision") != SOURCE_REVISION
        or audit.get("source_object_name") != SOURCE_OBJECT_NAME
        or audit.get("source_object_size_bytes") != SOURCE_OBJECT_SIZE_BYTES
        or audit.get("source_object_sha256") != SOURCE_OBJECT_SHA256
        or verified.size_bytes != SOURCE_OBJECT_SIZE_BYTES
        or verified.sha256 != SOURCE_OBJECT_SHA256
    ):
        raise ValueError("audit source identity does not match the frozen source")


def _phase_root(storage_root: Path) -> Path:
    return storage_root / "trustsr" / "phase2b1a"


def _require_top_level_shape(records: Sequence[Mapping[str, object]]) -> None:
    if not records:
        return
    columns = set(records[0])
    if len(columns) != 26 or any(set(record) != columns for record in records):
        raise ValueError("crosssensor table must contain exactly 26 top-level columns")


def _commit_manifest(
    manifest_root: Path, candidate: ManifestArtifact
) -> tuple[Path, bool]:
    destination_directory = manifest_root / candidate.sha256
    destination = destination_directory / "samples.jsonl"
    candidate_payload = candidate.path.read_bytes()
    if destination_directory.exists() or destination_directory.is_symlink():
        if destination_directory.is_symlink() or not destination_directory.is_dir():
            raise ValueError("existing manifest digest directory is invalid")
        if tuple(destination_directory.iterdir()) != (destination,):
            raise ValueError("existing manifest digest directory is incomplete or invalid")
        if destination.is_symlink() or not destination.is_file():
            raise ValueError("existing digest-addressed manifest is invalid")
        load_manifest(destination, expected_sha256=candidate.sha256)
        if destination.read_bytes() != candidate_payload:
            raise ValueError("existing digest-addressed manifest has different bytes")
        return destination, True

    destination_directory.mkdir()
    atomic_write_bytes(destination, candidate_payload)
    load_manifest(destination, expected_sha256=candidate.sha256)
    return destination, False


def _load_digest_manifest(
    storage_root: Path, manifest_path: Path
) -> tuple[str, tuple[dict[str, object], ...]]:
    if not isinstance(manifest_path, Path):
        raise TypeError("manifest_path must be a pathlib.Path")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("manifest must be an existing regular file")
    resolved = manifest_path.resolve(strict=True)
    _require_confined(storage_root, resolved)
    manifest_root = (_phase_root(storage_root) / "manifests").resolve(strict=True)
    if resolved.name != "samples.jsonl" or resolved.parent.parent != manifest_root:
        raise ValueError("manifest must use the confined digest-addressed stage layout")
    payload = resolved.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if resolved.parent.name != actual_sha256:
        raise ValueError("manifest parent must equal its actual SHA-256")
    return actual_sha256, load_manifest(resolved, expected_sha256=actual_sha256)


def _find_reusable_post_manifest(
    storage_root: Path,
    input_manifest_sha256: str,
    input_records: Sequence[Mapping[str, object]],
) -> ManifestArtifact | None:
    manifest_root = _phase_root(storage_root) / "manifests"
    matches: list[ManifestArtifact] = []
    for directory in sorted(manifest_root.iterdir(), key=lambda path: path.name):
        if directory.name == input_manifest_sha256:
            continue
        candidate = directory / "samples.jsonl"
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("existing manifest tree contains an invalid digest directory")
        if tuple(directory.iterdir()) != (candidate,):
            raise ValueError("existing manifest digest directory is incomplete or invalid")
        digest, records = _load_digest_manifest(storage_root, candidate)
        _require_matching_post_manifest(input_records, records)
        _verify_post_manifest_assets(storage_root, records)
        matches.append(ManifestArtifact(candidate, candidate.stat().st_size, digest))
    if len(matches) > 1:
        raise ValueError("multiple matching post-extraction manifests are ambiguous")
    return matches[0] if matches else None


def _require_matching_post_manifest(
    input_records: Sequence[Mapping[str, object]],
    candidate_records: Sequence[Mapping[str, object]],
) -> None:
    if len(input_records) != len(candidate_records):
        raise ValueError("post-extraction manifest does not match its input manifest")
    asset_fields = {"lr_asset", "hr_asset"}
    for input_record, candidate_record in zip(
        input_records, candidate_records, strict=True
    ):
        if any(
            candidate_record[field] != value
            for field, value in input_record.items()
            if field not in asset_fields
        ):
            raise ValueError("post-extraction manifest does not match its input manifest")
        selected = input_record["pilot"] is not None
        if selected != (
            candidate_record["lr_asset"] is not None
            and candidate_record["hr_asset"] is not None
        ):
            raise ValueError(
                "post-extraction manifest must contain every deterministic asset pair"
            )


def _verify_post_manifest_assets(
    storage_root: Path, records: Sequence[Mapping[str, object]]
) -> None:
    phase_root = _phase_root(storage_root)
    asset_count = 0
    for record in records:
        if record["pilot"] is None:
            continue
        sample_id = cast(str, record["sample_id"])
        split = cast(str, record["split"])
        _require_safe_component(sample_id)
        for kind in ("lr", "hr"):
            asset = cast(Mapping[str, object], record[f"{kind}_asset"])
            expected_relative = PurePosixPath(
                "pilot-v1", split, sample_id, f"{kind}.tif"
            ).as_posix()
            if asset["relative_path"] != expected_relative:
                raise ValueError("asset relative_path must use the exact pilot sample layout")
            asset_path = phase_root / expected_relative
            _require_confined(storage_root, asset_path)
            if asset_path.is_symlink() or not asset_path.is_file():
                raise ValueError("reusable pilot manifest requires all 72 confined GeoTIFF files")
            size_bytes, sha256 = _hash_file(asset_path)
            if size_bytes != asset["size_bytes"]:
                raise ValueError("asset byte size does not match the post-extraction manifest")
            if sha256 != asset["sha256"]:
                raise ValueError("asset SHA-256 does not match the post-extraction manifest")
            asset_count += 1
    if asset_count != 72:
        raise ValueError("reusable pilot manifest requires all 72 confined GeoTIFF files")


def _assignments_from_records(
    records: Sequence[Mapping[str, object]],
) -> tuple[AssignedSample, ...]:
    assignments: list[AssignedSample] = []
    for record in records:
        centroid = cast(Mapping[str, object], record["centroid"])
        admin = cast(Mapping[str, object], record["admin"])
        sample = CrosssensorSample(
            source_index=cast(int, record["source_index"]),
            sample_id=cast(str, record["sample_id"]),
            longitude=cast(float, centroid["longitude"]),
            latitude=cast(float, centroid["latitude"]),
            crs=cast(str, record["crs"]),
            geotransform=tuple(cast(Sequence[float], record["geotransform"])),  # type: ignore[arg-type]
            raster_shape=tuple(cast(Sequence[int], record["raster_shape"])),  # type: ignore[arg-type]
            time_start=cast(str, record["time_start"]),
            lr_time_start=cast(str, record["lr_time_start"]),
            hr_time_start=cast(str, record["hr_time_start"]),
            admin0=cast(str | None, admin["admin0"]),
            admin1=cast(str | None, admin["admin1"]),
            admin2=cast(str | None, admin["admin2"]),
            days_between=cast(int, record["days_between"]),
            correlation=cast(float, record["correlation"]),
            scale_factor=cast(int, record["scale_factor"]),
        )
        assignments.append(
            AssignedSample(
                sample=sample,
                spatial_group_id=cast(str, record["spatial_group_id"]),
                split=cast(str, record["split"]),  # type: ignore[arg-type]
            )
        )
    return tuple(assignments)


def _require_safe_component(sample_id: str) -> None:
    if (
        type(sample_id) is not str
        or not sample_id
        or "\x00" in sample_id
        or PurePosixPath(sample_id).parts != (sample_id,)
        or PureWindowsPath(sample_id).parts != (sample_id,)
        or sample_id in {".", ".."}
    ):
        raise ValueError("pilot sample_id must be a safe path component")


def _require_confined(storage_root: Path, candidate: Path) -> Path:
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(storage_root)
    except ValueError:
        raise ValueError("derived stage path escapes storage_root") from None
    return resolved


def _require_absent_or_complete_pair(output_root: Path) -> None:
    if not output_root.exists() and not output_root.is_symlink():
        return
    if output_root.is_symlink() or not output_root.is_dir():
        raise ValueError("existing pilot pair is partial or invalid")
    expected = {"lr.tif", "hr.tif"}
    entries = tuple(output_root.iterdir())
    if {entry.name for entry in entries} != expected or any(
        entry.is_symlink() or not entry.is_file() for entry in entries
    ):
        raise ValueError("existing pilot pair is partial or invalid")


def _hash_file(path: Path) -> tuple[int, str]:
    size_bytes = 0
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            size_bytes += len(block)
            digest.update(block)
    return size_bytes, digest.hexdigest()


def _commit_audit(
    phase_root: Path, manifest_sha256: str, payload: bytes
) -> tuple[Path, bool]:
    audit_directory = phase_root / "audits" / manifest_sha256
    audit_path = audit_directory / "phase2b1a-audit.json"
    if audit_directory.exists() or audit_directory.is_symlink():
        if audit_directory.is_symlink() or not audit_directory.is_dir():
            raise ValueError("existing audit digest directory is invalid")
        if tuple(audit_directory.iterdir()) != (audit_path,):
            raise ValueError("existing audit digest directory is incomplete or invalid")
        if audit_path.is_symlink() or not audit_path.is_file():
            raise ValueError("existing audit is invalid")
        if audit_path.read_bytes() != payload:
            raise ValueError("existing audit has different bytes")
        return audit_path, True

    audit_directory.parent.mkdir(parents=True, exist_ok=True)
    audit_directory.mkdir()
    atomic_write_bytes(audit_path, payload)
    return audit_path, False


def main(argv: list[str] | None = None) -> int:
    """Dispatch one stage and emit one host-free canonical JSON summary."""
    args = build_parser().parse_args(argv)
    common = (args.source, args.storage_root)
    if args.stage == "download":
        payload = run_download(
            *common,
            args.transport_url,
            confirmed_cloud_storage=args.confirm_cloud_storage,
        )
    elif args.stage == "manifest":
        payload = run_manifest(
            *common,
            confirmed_cloud_storage=args.confirm_cloud_storage,
        )
    elif args.stage == "pilot":
        payload = run_pilot(
            *common,
            args.manifest,
            confirmed_cloud_storage=args.confirm_cloud_storage,
        )
    else:
        payload = run_audit(
            *common,
            args.manifest,
            confirmed_cloud_storage=args.confirm_cloud_storage,
        )
    print(canonical_json(payload).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
