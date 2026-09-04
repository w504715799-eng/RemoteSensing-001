"""Independently verify Phase 2B3-B and publish its acceptance decision."""

from __future__ import annotations

import argparse
import sys
from contextlib import redirect_stdout
from pathlib import Path

from trustsr.evaluation.phase2b3b_acceptance import (
    run_independent_phase2b3b_verification,
)
from trustsr.jsonio import canonical_json


def build_parser() -> argparse.ArgumentParser:
    """Build the fixed independent acceptance verifier parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--confirm-persistent-storage", action="store_true")
    return parser


def run_verify(args: argparse.Namespace) -> dict[str, object]:
    """Return the host-free final independent verification receipt."""

    receipt = run_independent_phase2b3b_verification(
        bundle_dir=args.bundle,
        project_root=args.project_root,
        evidence_dir=args.evidence_dir,
        storage_root=args.storage_root,
        manifest_path=args.manifest,
        confirmed_persistent_storage=args.confirm_persistent_storage,
    )
    result = receipt.as_dict()
    if type(result) is not dict:
        raise TypeError("independent verifier returned an invalid receipt")
    return result


def main(argv: list[str] | None = None) -> int:
    """Verify acceptance, publish evidence, and emit one canonical JSON document."""

    args = build_parser().parse_args(argv)
    with redirect_stdout(sys.stderr):
        result = run_verify(args)
        payload = canonical_json(result)
    print(payload.decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
