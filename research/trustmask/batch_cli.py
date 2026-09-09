"""Freeze, preflight and resume the new public-source real-data study."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREVIEW = ROOT / "research/evidence/crosssensor-public-source-partition-preview-v1.json"
DEFAULT_ASSIGNMENTS = ROOT / "artifacts/progressive-partition-preview/assignments.json"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def load_assignments(path):
    raw = Path(path).read_bytes()
    preview = json.loads(PREVIEW.read_bytes())
    if digest(raw) != preview["assignments_sha256"]:
        raise ValueError("assignment digest differs from the accepted partition")
    rows = json.loads(raw)
    members, groups = set(), set()
    counts = dict.fromkeys(preview["member_counts"], 0)
    group_counts = dict.fromkeys(counts, 0)
    for row in rows:
        group, role, names = row["group_id"], row["role"], row["members"]
        if role not in counts or not names or group in groups:
            raise ValueError("invalid assignment group or role")
        if len(set(names)) != len(names) or members.intersection(names):
            raise ValueError("duplicate assignment member")
        if digest("\n".join(sorted(names)).encode()) != group:
            raise ValueError("assignment group identity mismatch")
        members.update(names)
        groups.add(group)
        counts[role] += len(names)
        group_counts[role] += 1
    if counts != preview["member_counts"] or any(
        group_counts[r] != preview["group_counts"][r] for r in counts
    ):
        raise ValueError("assignment counts differ from accepted partition")
    return rows


def implementation_hashes():
    paths = sorted(
        list((ROOT / "research/trustmask").glob("*.py"))
        + list((ROOT / "src/trustsr").rglob("*.py"))
    )
    return {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in paths}


def protocol_definition(assignments):
    from research.trustmask.real_data import binding_metadata

    load_assignments(assignments)
    preview = json.loads(PREVIEW.read_bytes())
    return dict(
        schema="trustmask-real-batch-protocol-v1",
        study_id="progressive-public-crosssensor-v1",
        alpha=0.05,
        assignments_sha256=preview["assignments_sha256"],
        partition_evidence_sha256=digest(PREVIEW.read_bytes()),
        implementation_sha256=implementation_hashes(),
        source_model_support=binding_metadata(),
        split_frozen=True,
        per_member_s2_mapping_required=False,
        historical_terminal_ledger_reused=False,
        independence_certified=False,
        statistical_basis="exchangeable_calibration_groups_and_independent_test_groups_assumed",
        terminal_policy="same_binding_crash_resume_only_completed_receipts_never_reinferred",
        storage_policy="one_roi_in_memory_compact_statistics_only",
    )


def prepare_protocol(path, assignments=DEFAULT_ASSIGNMENTS):
    payload = protocol_definition(assignments)
    payload["protocol_sha256"] = digest(canonical(payload))
    with Path(path).open("x") as stream:
        stream.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    return payload


def verify_protocol(path, assignments=DEFAULT_ASSIGNMENTS):
    value = json.loads(Path(path).read_bytes())
    supplied = value.pop("protocol_sha256", None)
    if supplied != digest(canonical(value)) or value != protocol_definition(assignments):
        raise ValueError("protocol or implementation differs from the frozen definition")
    value["protocol_sha256"] = supplied
    return value, load_assignments(assignments)


def runtime_binding():
    names = ("numpy", "torch", "rasterio", "pyarrow", "opensr-model", "omegaconf")
    return {name: importlib.metadata.version(name) for name in names}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "preflight", "run"):
        child = sub.add_parser(command)
        child.add_argument("--protocol", type=Path, required=True)
        child.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
        if command != "prepare":
            child.add_argument("--taco", type=Path, required=True)
            child.add_argument("--model-dir", type=Path, required=True)
            child.add_argument("--device", default="cuda:0")
        if command == "run":
            child.add_argument("--study-dir", type=Path, required=True)
            child.add_argument("--max-rois", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_protocol(args.protocol, args.assignments)
        print(json.dumps({"protocol_sha256": result["protocol_sha256"], "pixel_access": False}))
        return
    protocol, assignments = verify_protocol(args.protocol, args.assignments)
    if args.command == "run" and args.max_rois <= 0:
        parser.error("--max-rois must be positive")
    from research.trustmask.real_data import LocalTacoProvider

    member_groups = {member: row["group_id"] for row in assignments for member in row["members"]}
    provider = LocalTacoProvider(args.taco, member_groups, args.model_dir, device=args.device)
    try:
        preflight = provider.preflight()
        if args.command == "preflight":
            print(json.dumps(preflight, sort_keys=True))
            return
        from research.trustmask.batch import run_batch

        result = run_batch(
            args.study_dir,
            assignments,
            provider,
            binding=dict(
                protocol=protocol,
                runtime=runtime_binding(),
                device=args.device,
                source_file=preflight["source_file_fingerprint"],
                evidence_kind="real_public_source_selected_assets",
            ),
            max_rois=args.max_rois,
            alpha=protocol["alpha"],
        )
        # Detailed group observations stay in the durable report, not console output.
        print(
            json.dumps(
                {key: value for key, value in result.items() if key != "result"}, sort_keys=True
            )
        )
    finally:
        provider.close()


if __name__ == "__main__":
    main()
