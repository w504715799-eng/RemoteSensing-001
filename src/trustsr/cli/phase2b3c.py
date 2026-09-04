"""Run fixed Phase 2B3-C preflight, evaluation, and inference-free replay."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from contextlib import redirect_stdout
from pathlib import Path

from trustsr.evaluation.phase2b3c_workflow import (
    run_formal_evaluation,
    run_formal_evaluation_replay,
    run_metadata_preflight,
)
from trustsr.jsonio import canonical_json


def build_parser() -> argparse.ArgumentParser:
    """Build the intentionally narrow one-time evaluation parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    for name, handler in (
        ("preflight", run_preflight),
        ("evaluate", run_evaluate),
        ("evaluation-replay", run_evaluation_replay),
    ):
        child = commands.add_parser(name)
        child.add_argument("--project-root", type=Path, required=True)
        child.add_argument("--evidence-dir", type=Path, required=True)
        child.add_argument("--storage-root", type=Path, required=True)
        child.add_argument("--manifest", type=Path, required=True)
        child.add_argument("--confirm-persistent-storage", action="store_true")
        child.add_argument(
            "--access-permit", type=Path, required=name != "preflight"
        )
        if name == "evaluate":
            child.add_argument(
                "--ldsr-model-dir",
                type=Path,
                help=(
                    "supply only after the exact cache probe reports missing entries "
                    "and separate GPU authorization has been obtained"
                ),
            )
        child.set_defaults(handler=handler)
    return parser


def _json_native(value: object) -> object:
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("Phase 2B3-C output mapping keys must be strings")
        return {key: _json_native(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return [_json_native(item) for item in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TypeError("Phase 2B3-C output contains a non-JSON value")


def run_preflight(args: argparse.Namespace) -> dict[str, object]:
    """Run metadata-only readiness without a permit or protected-data access."""

    return run_metadata_preflight(
        project_root=args.project_root,
        evidence_dir=args.evidence_dir,
        storage_root=args.storage_root,
        manifest_path=args.manifest,
        confirmed_persistent_storage=args.confirm_persistent_storage,
        access_permit_path=args.access_permit,
    )


def run_evaluate(args: argparse.Namespace) -> dict[str, object]:
    """Run the reviewed one-time evaluation."""

    return run_formal_evaluation(
        project_root=args.project_root,
        evidence_dir=args.evidence_dir,
        storage_root=args.storage_root,
        manifest_path=args.manifest,
        access_permit_path=args.access_permit,
        ldsr_model_dir=args.ldsr_model_dir,
        confirmed_persistent_storage=args.confirm_persistent_storage,
    ).as_dict()


def run_evaluation_replay(args: argparse.Namespace) -> dict[str, object]:
    """Replay verified caches without a model construction path."""

    return run_formal_evaluation_replay(
        project_root=args.project_root,
        evidence_dir=args.evidence_dir,
        storage_root=args.storage_root,
        manifest_path=args.manifest,
        access_permit_path=args.access_permit,
        confirmed_persistent_storage=args.confirm_persistent_storage,
    ).as_dict()


def main(argv: list[str] | None = None) -> int:
    """Emit exactly one canonical JSON document on successful completion."""

    args = build_parser().parse_args(argv)
    with redirect_stdout(sys.stderr):
        result = _json_native(args.handler(args))
        payload = canonical_json(result)
    print(payload.decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
