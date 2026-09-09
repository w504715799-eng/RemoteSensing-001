import pytest

from research.trustmask.grouping import audit_groups, check_partition
from research.trustmask.partition_preview import assign_roles


def test_partition_stable_under_input_order_and_excludes_history():
    rows = [dict(id=str(i), lon=float(i), lat=0.0, source_ids=[str(i)]) for i in range(7)]
    audit = audit_groups(rows, {"0"}, radius_km=5)
    counts = dict(
        scale_fit=1, development_calibration=1, development_validation=1, calibration=1, test=2
    )
    first = assign_roles(audit, counts)
    assert check_partition(audit, first)["test"] == 2
    assert "0" in audit["excluded_members"]
    reverse = audit_groups(list(reversed(rows)), {"0"}, radius_km=5)
    assert first == assign_roles(reverse, counts)
    assert len(first) == 6


def test_insufficient_groups_rejected():
    audit = audit_groups([], set())
    with pytest.raises(ValueError, match="count"):
        assign_roles(audit, {"scale_fit": 1})
