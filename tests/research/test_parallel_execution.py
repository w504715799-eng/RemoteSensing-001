from concurrent.futures import Future

import pytest

from research.operations.parallel_batches import OrderedProvider


def test_ordered_bounded_and_role_barrier():
    expected = [("scale_fit", "g", "a"), ("scale_fit", "h", "b"), ("test", "t", "c")]
    sent, guarded = [], []

    def submit(sample, group):
        sent.append(sample)
        future = Future()
        future.set_result((sample, {"sample_id": sample}))
        return future

    provider = OrderedProvider(expected, submit, workers=2, guard=guarded.append)
    assert sent == []
    assert provider("a", "g") == "a"
    assert sent == ["a", "b"]
    assert provider.last_provenance == {"sample_id": "a"}
    assert provider("b", "h") == "b"
    assert sent == ["a", "b"]
    assert provider("c", "t") == "c"
    assert sent == ["a", "b", "c"]
    assert guarded == ["scale_fit", "test"]


def test_wrong_identity_or_failed_guard_never_submits():
    def submit(*args):
        pytest.fail("unexpected job")

    def guard(role):
        raise ValueError("missing terminal ledger")

    provider = OrderedProvider([("test", "g", "a")], submit, workers=2, guard=guard)
    with pytest.raises(ValueError, match="order"):
        provider("b", "g")
    with pytest.raises(ValueError, match="terminal"):
        provider("a", "g")


def test_prefetch_respects_exact_remaining_budget():
    sent = []

    def submit(sample, group):
        sent.append(sample)
        f = Future()
        f.set_result((sample, {}))
        return f

    p = OrderedProvider([("calibration", "g", "last")], submit, workers=2, guard=lambda _: None)
    assert p("last", "g") == "last"
    assert sent == ["last"]
    with pytest.raises(ValueError, match="order"):
        p("extra", "g")


def test_parallel_resume_equals_reference_and_test_guard(tmp_path):
    import numpy as np

    from research.operations.parallel_batches import terminal_guard
    from research.trustmask.batch import _digest, _read, run_batch
    from research.trustmask.pipeline import ROLES, Features, Observation, run_study

    partitions, assignments, rows, order = {}, [], {}, []
    for role in ROLES:
        group = role + "-group"
        members = [role + str(i) for i in range(3)]
        assignments.append(dict(role=role, group_id=group, members=members))
        partitions[role] = []
        for sample in members:
            row = Observation(
                sample,
                group,
                Features(*(np.full((4, 4), 0.1) for _ in range(4))),
                np.full((4, 4), 0.01),
            )
            rows[sample] = row
            partitions[role].append(row)
            order.append((role, group, sample))
    binding = {"kind": "synthetic"}
    first = run_batch(tmp_path, assignments, lambda s, g: rows[s], binding=binding, max_rois=2)
    assert first["completed_rois"] == 2
    calls = []

    def submit(s, g):
        calls.append(s)
        if s.startswith("test"):
            assert (tmp_path / "frozen_methods.json").exists()
            assert (tmp_path / "terminal_test_started.json").exists()
        f = Future()
        f.set_result((rows[s], None))
        return f

    journal_digest = _digest(_read(tmp_path / "journal.json"))
    provider = OrderedProvider(
        order[2:],
        submit,
        workers=2,
        guard=lambda role: terminal_guard(tmp_path, journal_digest, role),
    )
    result = run_batch(tmp_path, assignments, provider, binding=binding, max_rois=999)
    assert result["status"] == "complete"
    assert calls == [s for _, _, s in order[2:]]
    assert result["result"] == run_study(partitions)


def test_terminal_guard_rejects_mismatched_frozen_artifact(tmp_path):
    from research.operations.parallel_batches import terminal_guard
    from research.trustmask.batch import _write

    _write(tmp_path / "frozen_methods.json", {"method": "a"})
    _write(tmp_path / "terminal_test_started.json", {"binding_sha256": "wrong"})
    with pytest.raises(ValueError, match="ledger"):
        terminal_guard(tmp_path, "expected", "test")
