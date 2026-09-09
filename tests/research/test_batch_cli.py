import json
from pathlib import Path

import pytest

from research.trustmask.batch_cli import load_assignments, prepare_protocol, verify_protocol

ROOT = Path(__file__).resolve().parents[2]
ASSIGNMENTS = ROOT / "artifacts/progressive-partition-preview/assignments.json"


@pytest.mark.skipif(
    not ASSIGNMENTS.exists(), reason="full real assignment is an ignored local artifact"
)
def test_exact_partition_required(tmp_path):
    rows = load_assignments(ASSIGNMENTS)
    assert sum(len(row["members"]) for row in rows) == 7477
    bad = tmp_path / "assignments.json"
    bad.write_text(json.dumps(rows[:-1]))
    with pytest.raises(ValueError, match="assignment"):
        load_assignments(bad)


@pytest.mark.skipif(
    not ASSIGNMENTS.exists(), reason="full real assignment is an ignored local artifact"
)
def test_protocol_binds_implementation_and_assignments(tmp_path):
    protocol = tmp_path / "protocol.json"
    prepare_protocol(protocol, ASSIGNMENTS)
    original = protocol.read_bytes()
    data, rows = verify_protocol(protocol, ASSIGNMENTS)
    assert data["split_frozen"] is True
    assert data["per_member_s2_mapping_required"] is False
    assert len(rows) == 6158
    with pytest.raises(FileExistsError):
        prepare_protocol(protocol, ASSIGNMENTS)
    value = json.loads(original)
    value["alpha"] = 0.1
    protocol.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="protocol"):
        verify_protocol(protocol, ASSIGNMENTS)
    protocol.write_bytes(original)
    data = json.loads(original)
    key = next(iter(data["implementation_sha256"]))
    data["implementation_sha256"][key] = "0" * 64
    protocol.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="protocol|implementation"):
        verify_protocol(protocol, ASSIGNMENTS)


def test_cli_connects_provider_to_resumable_engine_without_reinference(
    tmp_path, monkeypatch, capsys
):
    import numpy as np

    from research.trustmask import batch_cli, real_data
    from research.trustmask.pipeline import ROLES, Features, Observation

    assignments = [dict(role=role, group_id=role, members=[role]) for role in ROLES]
    monkeypatch.setattr(batch_cli, "verify_protocol", lambda *a: ({"alpha": 0.05}, assignments))
    monkeypatch.setattr(batch_cli, "runtime_binding", lambda: {"synthetic": "1"})
    calls, closed = [], []

    class FakeProvider:
        def __init__(self, source, member_groups, model_dir, device):
            assert member_groups == {r: r for r in ROLES}

        def preflight(self):
            return {"source_file_fingerprint": {"synthetic": True}}

        def __call__(self, sample, group):
            calls.append(sample)
            return Observation(
                sample,
                group,
                Features(*(np.full((4, 4), 0.1) for _ in range(4))),
                np.full((4, 4), 0.01),
            )

        def close(self):
            closed.append(True)

    monkeypatch.setattr(real_data, "LocalTacoProvider", FakeProvider)
    arguments = [
        "run",
        "--protocol",
        "ignored",
        "--taco",
        "synthetic",
        "--model-dir",
        "ignored",
        "--study-dir",
        str(tmp_path / "study"),
        "--max-rois",
        "2",
    ]
    for _ in range(4):
        batch_cli.main(arguments)
    assert calls == list(ROLES)
    assert len(closed) == 4
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["status"] == "complete"


def test_invalid_batch_budget_precedes_source_access(monkeypatch):
    from research.trustmask import batch_cli, real_data

    monkeypatch.setattr(batch_cli, "verify_protocol", lambda *a: ({"alpha": 0.05}, []))
    monkeypatch.setattr(
        real_data, "LocalTacoProvider", lambda *a, **k: pytest.fail("source opened")
    )
    with pytest.raises(SystemExit):
        batch_cli.main(
            [
                "run",
                "--protocol",
                "ignored",
                "--taco",
                "ignored",
                "--model-dir",
                "ignored",
                "--study-dir",
                "ignored",
                "--max-rois",
                "0",
            ]
        )


def test_self_contained_protocol_roundtrip_and_code_change(tmp_path, monkeypatch):
    from research.trustmask import batch_cli
    from research.trustmask.pipeline import ROLES

    rows = [
        dict(group_id=batch_cli.digest(role.encode()), members=[role], role=role) for role in ROLES
    ]
    assignments = tmp_path / "members.json"
    assignments.write_bytes(batch_cli.canonical(rows))
    preview = tmp_path / "preview.json"
    preview.write_text(
        json.dumps(
            dict(
                assignments_sha256=batch_cli.digest(assignments.read_bytes()),
                member_counts=dict.fromkeys(ROLES, 1),
                group_counts=dict.fromkeys(ROLES, 1),
            )
        )
    )
    monkeypatch.setattr(batch_cli, "PREVIEW", preview)
    protocol = tmp_path / "protocol.json"
    batch_cli.prepare_protocol(protocol, assignments)
    assert batch_cli.verify_protocol(protocol, assignments)[1] == rows
    monkeypatch.setattr(batch_cli, "implementation_hashes", lambda: {"changed.py": "0" * 64})
    with pytest.raises(ValueError, match="implementation"):
        batch_cli.verify_protocol(protocol, assignments)
