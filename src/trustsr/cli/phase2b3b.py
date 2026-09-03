"""Run the metadata-only Phase 2B3-B preflight gate."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from contextlib import redirect_stdout
from pathlib import Path

from trustsr.evaluation.phase2b3b_preflight import load_phase2b3b_preflight
from trustsr.evaluation.phase2b3b_revision import (
    Phase2B3BRevision,
    verify_phase2b3b_revision,
)
from trustsr.jsonio import canonical_json


def build_parser() -> argparse.ArgumentParser:
    """Build the intentionally narrow Phase 2B3-B command parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--project-root", type=Path, required=True)
    preflight.add_argument("--evidence-dir", type=Path, required=True)
    preflight.add_argument("--storage-root", type=Path, required=True)
    preflight.add_argument("--manifest", type=Path, required=True)
    preflight.set_defaults(handler=run_preflight)
    return parser


def _json_native(value: object) -> object:
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("Phase 2B3-B output mapping keys must be strings")
        return {key: _json_native(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return [_json_native(item) for item in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TypeError("Phase 2B3-B output contains a non-JSON value")


def run_preflight(args: argparse.Namespace) -> dict[str, object]:
    """Run revision validation before loading evidence or manifest metadata."""

    revision = verify_phase2b3b_revision(args.project_root)
    if not isinstance(revision, Phase2B3BRevision):
        raise TypeError("Phase 2B3-B revision gate returned an invalid identity")
    preflight = load_phase2b3b_preflight(
        args.evidence_dir,
        args.storage_root,
        args.manifest,
    )
    if not isinstance(preflight, Mapping):
        raise TypeError("Phase 2B3-B preflight returned an invalid receipt")
    result = _json_native(
        {
            "schema": "trustsr.phase2b3b-cli-preflight.v1",
            "revision": {
                "branch": revision.branch,
                "head_revision": revision.head_revision,
                "calculation_revision": revision.calculation_revision,
                "evidence_publication": revision.evidence_publication,
            },
            "preflight": preflight,
        }
    )
    if type(result) is not dict:
        raise AssertionError("Phase 2B3-B CLI output must be a JSON object")
    return result


def main(argv: list[str] | None = None) -> int:
    """Run one safe preflight and emit exactly one canonical JSON document."""

    args = build_parser().parse_args(argv)
    with redirect_stdout(sys.stderr):
        result = args.handler(args)
        payload = canonical_json(result)
    print(payload.decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
