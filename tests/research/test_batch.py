import numpy as np
import pytest

from research.trustmask.pipeline import ROLES, Features, Observation, run_study


def fixture():
    rng = np.random.default_rng(29)
    partitions = {r: [] for r in ROLES}
    assignments = []
    for role in ROLES:
        for g in range(3):
            group = f"{role}-{g}"
            members = [f"{group}-{i}" for i in range(1 + (g == 0))]
            assignments.append(dict(group_id=group, members=members, role=role))
            for member in members:
                partitions[role].append(
                    Observation(
                        member,
                        group,
                        Features(*(rng.random((3, 4)) for _ in range(4))),
                        rng.random((3, 4)) * 0.1,
                    )
                )
    rows = {o.sample_id: o for rr in partitions.values() for o in rr}
    return partitions, assignments, rows


@pytest.mark.parametrize("alpha", [0.05, 0.4])
def test_resume_parity_and_stage_order(tmp_path, alpha):
    from research.trustmask.batch import run_batch

    partitions, assignments, rows = fixture()
    calls = []

    def provider(s, g):
        calls.append(s)
        if s.startswith("test-"):
            assert (tmp_path / "terminal_test_started.json").exists()
            assert (tmp_path / "frozen_methods.json").exists()
        return rows[s]

    while True:
        status = run_batch(
            tmp_path, assignments, provider, binding={"kind": "synthetic"}, max_rois=3, alpha=alpha
        )
        assert status["processed_rois"] <= 3
        if status["status"] == "complete":
            break
    assert calls == [o.sample_id for role in ROLES for o in partitions[role]]
    assert status["result"] == run_study(partitions, alpha=alpha)
    again = run_batch(
        tmp_path,
        assignments,
        lambda *_: pytest.fail("provider called"),
        binding={"kind": "synthetic"},
        max_rois=1,
        alpha=alpha,
    )
    assert again["result"] == status["result"]
    with pytest.raises(ValueError, match="binding"):
        run_batch(
            tmp_path, assignments, provider, binding={"kind": "changed"}, max_rois=1, alpha=alpha
        )
    receipt = next((tmp_path / "receipts").glob("*.json"))
    receipt.write_text("{}")
    with pytest.raises(ValueError, match="corrupt"):
        run_batch(
            tmp_path, assignments, provider, binding={"kind": "synthetic"}, max_rois=1, alpha=alpha
        )


def test_interrupted_test_retries_only_same_binding_and_requires_ledger(tmp_path):
    from research.trustmask.batch import run_batch

    _, assignments, rows = fixture()

    def interrupted(s, g):
        if s.startswith("test-"):
            raise RuntimeError("interrupted")
        return rows[s]

    with pytest.raises(RuntimeError, match="interrupted"):
        run_batch(
            tmp_path,
            assignments,
            interrupted,
            binding={"kind": "synthetic"},
            max_rois=100,
            alpha=0.4,
        )
    assert (tmp_path / "terminal_test_started.json").exists()
    (tmp_path / "frozen_methods.json").unlink()
    with pytest.raises(ValueError, match="corrupt"):
        run_batch(
            tmp_path,
            assignments,
            lambda s, g: rows[s],
            binding={"kind": "synthetic"},
            max_rois=100,
            alpha=0.4,
        )


def test_exclusive_writer_and_receipt_gap(tmp_path):
    import fcntl

    from research.trustmask.batch import run_batch

    _, assignments, rows = fixture()
    with (tmp_path / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="active writer"):
            run_batch(
                tmp_path,
                assignments,
                lambda s, g: rows[s],
                binding={"kind": "synthetic"},
                max_rois=2,
            )
    run_batch(
        tmp_path, assignments, lambda s, g: rows[s], binding={"kind": "synthetic"}, max_rois=2
    )
    (tmp_path / "receipts" / "00000000.json").unlink()
    with pytest.raises(ValueError, match="corrupt"):
        run_batch(
            tmp_path, assignments, lambda s, g: rows[s], binding={"kind": "synthetic"}, max_rois=2
        )


@pytest.mark.parametrize("started", [False, True])
def test_frozen_prefix_cannot_reenter_calibration(tmp_path, started):
    from research.trustmask.batch import run_batch

    _, assignments, rows = fixture()

    def provider(s, g):
        if s.startswith("test-"):
            raise RuntimeError("interrupted")
        return rows[s]

    with pytest.raises(RuntimeError):
        run_batch(tmp_path, assignments, provider, binding={"kind": "synthetic"}, max_rois=100)
    if not started:
        (tmp_path / "terminal_test_started.json").unlink()
    (tmp_path / "receipts" / "00000015.json").unlink()
    with pytest.raises(ValueError, match="corrupt"):
        run_batch(
            tmp_path,
            assignments,
            lambda *_: pytest.fail("reentered provider"),
            binding={"kind": "synthetic"},
            max_rois=1,
        )


def test_expected_interrupted_temp_is_discarded(tmp_path):
    from research.trustmask.batch import run_batch

    _, assignments, rows = fixture()

    def provider(s, g):
        return rows[s]

    run_batch(tmp_path, assignments, provider, binding={"kind": "synthetic"}, max_rois=1)
    temp = tmp_path / "receipts" / "00000001.tmp"
    temp.write_text("{")
    result = run_batch(tmp_path, assignments, provider, binding={"kind": "synthetic"}, max_rois=1)
    assert result["completed_rois"] == 2
    assert not temp.exists()
    (tmp_path / "receipts" / "00000009.tmp").write_text("{")
    with pytest.raises(ValueError, match="corrupt"):
        run_batch(tmp_path, assignments, provider, binding={"kind": "synthetic"}, max_rois=1)


def test_provider_provenance_and_real_evidence_are_retained(tmp_path):
    import json

    from research.trustmask.batch import run_batch

    _, assignments, rows = fixture()

    def provider(s, g):
        provider.last_provenance = {"sample_id": s, "digest": "verified"}
        return rows[s]

    result = run_batch(
        tmp_path,
        assignments,
        provider,
        binding={"evidence_kind": "authenticated_real_inputs"},
        max_rois=100,
    )
    receipt = json.loads((tmp_path / "receipts" / "00000000.json").read_text())["payload"]
    assert receipt["provenance"] == {"sample_id": "scale_fit-0-0", "digest": "verified"}
    assert result["result"]["evidence_kind"] == "authenticated_real_inputs"
    assert result["result"]["binding_sha256"] == result["binding_sha256"]
