"""Audit frozen Phase 2B2-A crosssensor model inputs without model inference."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path

from trustsr.data.crosssensor_pairs import (
    load_crosssensor_pair,
    load_crosssensor_records,
    select_input_smoke_records,
)
from trustsr.data.crosssensor_source import require_cloud_confirmation
from trustsr.data.input_audit import build_input_audit
from trustsr.jsonio import atomic_write_bytes, canonical_json


def build_parser() -> argparse.ArgumentParser:
    """Build the single-stage Phase 2B2-A parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    command = subparsers.add_parser("audit-inputs")
    command.add_argument("--storage-root", type=Path, required=True)
    command.add_argument("--selection-manifest", type=Path, required=True)
    command.add_argument("--selection-manifest-sha256", required=True)
    command.add_argument("--confirm-cloud-storage", action="store_true")
    return parser


def _commit_audit(
    storage_root: Path,
    manifest_sha256: str,
    payload: bytes,
) -> tuple[Path, bool]:
    phase_root = storage_root / "trustsr" / "phase2b2a"
    audit_parent = phase_root / "input-audits"
    audit_directory = audit_parent / manifest_sha256
    audit_path = audit_directory / "phase2b2a-input-audit.json"
    if audit_directory.exists() or audit_directory.is_symlink():
        if audit_directory.is_symlink() or not audit_directory.is_dir():
            raise ValueError("existing Phase 2B2-A audit directory is invalid")
        entries = tuple(audit_directory.iterdir())
        if entries != (audit_path,):
            raise ValueError("existing Phase 2B2-A audit contains unexpected files")
        if audit_path.is_symlink() or not audit_path.is_file():
            raise ValueError("existing Phase 2B2-A audit is not a regular file")
        if audit_path.read_bytes() != payload:
            raise ValueError("existing Phase 2B2-A audit has different bytes")
        return audit_path, True

    audit_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".input-audit-", dir=phase_root) as temporary:
        staged_directory = Path(temporary) / manifest_sha256
        staged_directory.mkdir()
        atomic_write_bytes(staged_directory / audit_path.name, payload)
        staged_directory.rename(audit_directory)
    return audit_path, False


def run_audit_inputs(
    storage_root: Path,
    selection_manifest_path: Path,
    selection_manifest_sha256: str,
    *,
    confirmed_cloud_storage: bool,
) -> dict[str, object]:
    """Load the 12-pair smoke set twice and commit its canonical audit."""

    root = require_cloud_confirmation(storage_root, confirmed_cloud_storage)
    records = load_crosssensor_records(
        root,
        selection_manifest_path,
        expected_sha256=selection_manifest_sha256,
    )
    selected = select_input_smoke_records(records)
    first = tuple(
        load_crosssensor_pair(root, record, manifest_sha256=selection_manifest_sha256)
        for record in selected
    )
    second = tuple(
        load_crosssensor_pair(root, record, manifest_sha256=selection_manifest_sha256)
        for record in selected
    )
    audit = build_input_audit(first, second)
    payload = canonical_json(audit)
    _, reused = _commit_audit(root, selection_manifest_sha256, payload)
    return {
        "stage": "audit-inputs",
        "digests": {
            "selection_manifest_sha256": selection_manifest_sha256,
            "audit_sha256": hashlib.sha256(payload).hexdigest(),
        },
        "counts": {"smoke_pairs": 12, "smoke_geotiffs": 24},
        "reused": reused,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the CPU input audit and emit one canonical JSON object."""

    args = build_parser().parse_args(argv)
    payload = run_audit_inputs(
        args.storage_root,
        args.selection_manifest,
        args.selection_manifest_sha256,
        confirmed_cloud_storage=args.confirm_cloud_storage,
    )
    print(canonical_json(payload).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
