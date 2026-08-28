from pathlib import Path

import pytest

from trustsr.jsonio import atomic_write_bytes, canonical_json


def test_canonical_json_is_sorted_compact_utf8_and_rejects_nan() -> None:
    assert canonical_json({"z": 1, "name": "区域", "a": [2, 1]}) == (
        '{"a":[2,1],"name":"区域","z":1}'.encode()
    )
    with pytest.raises(ValueError, match="canonical JSON"):
        canonical_json({"value": float("nan")})


def test_atomic_write_replaces_complete_payload_and_leaves_no_temporary_file(
    tmp_path: Path,
) -> None:
    target = tmp_path / "nested" / "record.json"
    atomic_write_bytes(target, b"first")
    atomic_write_bytes(target, b"second")
    assert target.read_bytes() == b"second"
    assert list(target.parent.glob(".record.json.*.tmp")) == []
