"""Verify Phase 2B3-B candidate metadata only; cannot authorize acceptance."""

from __future__ import annotations

import argparse
import sys
from contextlib import redirect_stdout
from pathlib import Path

from trustsr.evaluation.phase2b3b_bundle_verify import (
    SCHEMA as BUNDLE_VERIFICATION_SCHEMA,
)
from trustsr.evaluation.phase2b3b_bundle_verify import (
    VerifiedPhase2B3BBundle,
    verify_phase2b3b_bundle,
)
from trustsr.jsonio import canonical_json

SCHEMA = "trustsr.phase2b3b-candidate-metadata-verification-cli.v1"


def build_parser() -> argparse.ArgumentParser:
    """Build the candidate-only metadata verifier parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def run_verify(args: argparse.Namespace) -> dict[str, object]:
    """Return a host-free projection of one verified candidate bundle receipt."""

    receipt = verify_phase2b3b_bundle(
        args.bundle,
        project_root=args.project_root,
        evidence_dir=args.evidence_dir,
        storage_root=args.storage_root,
        manifest_path=args.manifest,
    )
    if type(receipt) is not VerifiedPhase2B3BBundle:
        raise TypeError("candidate bundle verifier returned an invalid receipt")
    if (
        receipt.schema != BUNDLE_VERIFICATION_SCHEMA
        or receipt.verification_scope != "metadata_consistency_only"
        or receipt.cache_computation_verified is not False
    ):
        raise ValueError("candidate bundle verifier returned an invalid verification scope")
    return {
        "schema": SCHEMA,
        "verification_scope": "metadata_consistency_only",
        "cache_computation_verified": False,
        "acceptance_authorized": False,
        "manifest_sha256": receipt.manifest_sha256,
        "result_sha256": receipt.result_sha256,
        "cache_audit_sha256": receipt.cache_audit_sha256,
        "runtime_manifest_sha256": receipt.runtime_manifest_sha256,
        "replay_sha256": receipt.replay_sha256,
        "producer_revision": receipt.producer_revision,
        "ordered_sample_ids_sha256": receipt.ordered_sample_ids_sha256,
        "ordered_membership_sha256": receipt.ordered_membership_sha256,
        "input_receipt_sha256": receipt.input_receipt_sha256,
        "ordered_inputs_sha256": receipt.ordered_inputs_sha256,
        "map_evidence_sha256": receipt.map_evidence_sha256,
        "radiometry_aggregate_sha256": receipt.radiometry_aggregate_sha256,
        "phase_decision": receipt.phase_decision,
    }


def main(argv: list[str] | None = None) -> int:
    """Verify candidate metadata and emit exactly one canonical JSON document."""

    args = build_parser().parse_args(argv)
    with redirect_stdout(sys.stderr):
        result = run_verify(args)
        payload = canonical_json(result)
    print(payload.decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
