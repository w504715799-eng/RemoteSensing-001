"""Offline geographic partition preview under the user-approved public-source basis."""

import argparse
import hashlib
import json
from pathlib import Path

from research.trustmask.audit_catalog import historical_ids
from research.trustmask.grouping import audit_groups, check_partition

MEMBER_SHA256 = "30013c7285c0aa91b1f4de48fd1734db314a81b71a8da3809294d3ee919d9cd6"
COUNTS = dict(
    scale_fit=256,
    development_calibration=512,
    development_validation=845,
    calibration=1500,
    test=3045,
)
DOMAIN = "trustmask-partition-preview-v1\n3407\n"


def assign_roles(audit, counts):
    if any(type(n) is not int or n <= 0 for n in counts.values()) or sum(counts.values()) != len(
        audit["eligible_groups"]
    ):
        raise ValueError("group count allocation mismatch")
    if set(counts) != set(COUNTS):
        raise ValueError("all five roles required")
    groups = sorted(
        audit["eligible_groups"],
        key=lambda g: (
            hashlib.sha256((DOMAIN + g["group_id"]).encode()).hexdigest(),
            g["group_id"],
        ),
    )
    result = {}
    offset = 0
    for role in COUNTS:
        for group in groups[offset : offset + counts[role]]:
            result[group["group_id"]] = role
        offset += counts[role]
    check_partition(audit, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new ignored artifact directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    raw = (root / "artifacts/datasets/progressive-crosssensor-members-v1.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != MEMBER_SHA256:
        raise ValueError("member projection digest mismatch")
    rows = [
        dict(id=name, lon=lon, lat=lat, source_ids=["naip:" + name.split("__", 1)[1]])
        for name, lon, lat in json.loads(raw)["rows"]
    ]
    seen, history = historical_ids(root)
    audit = audit_groups(rows, seen, radius_km=5.0)
    assignments = assign_roles(audit, COUNTS)
    members = dict.fromkeys(COUNTS, 0)
    records = []
    for group in audit["eligible_groups"]:
        role = assignments[group["group_id"]]
        members[role] += len(group["members"])
        records.append(dict(**group, role=role))
    payload = (json.dumps(records, sort_keys=True, separators=(",", ":")) + "\n").encode()
    summary = dict(
        schema="trustmask-public-source-partition-preview-v1",
        source_basis="official_public_release_accepted_by_user_2026-09-09",
        source_url="https://huggingface.co/datasets/tacofoundation/SEN2NAIPv2",
        source_revision="c370504201072fdb1dd388013ab8c0fc7d00a57e",
        source_object="sen2naipv2-crosssensor.taco",
        member_projection_sha256=MEMBER_SHA256,
        grouping="shared_naip_id_or_centroid_distance_5km_connected_components",
        ranking_domain=DOMAIN,
        history=history,
        historical_member_count=len(seen),
        excluded_member_count=len(audit["excluded_members"]),
        eligible_member_count=sum(members.values()),
        group_counts=check_partition(audit, assignments),
        member_counts=members,
        assignments_sha256=hashlib.sha256(payload).hexdigest(),
        per_member_s2_mapping_required=False,
        independence_certified=False,
        inferential_basis="conditional_on_group_independence_not_established_by_public_availability",
        split_frozen=False,
        pixel_access_authorized=False,
        gpu_used=False,
    )
    args.output.mkdir(exist_ok=False)
    (args.output / "assignments.json").write_bytes(payload)
    (args.output / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    print(
        json.dumps(
            {
                "group_counts": summary["group_counts"],
                "member_counts": members,
                "excluded_members": summary["excluded_member_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
