import hashlib
import json

import pytest

from research.trustmask.recover_manifest import project


def test_digest_checked_before_parsing():
    with pytest.raises(ValueError, match="digest"):
        project(b"not json", "0" * 64, 1)


def test_projection_excludes_observations_and_preserves_numeric_tokens():
    record = dict(
        sample_id="sample",
        source_index=0,
        source={},
        centroid={},
        crs="x",
        geotransform=[1.25],
        raster_shape=[520],
        time_start="t",
        lr_time_start="l",
        hr_time_start="h",
        days_between=2,
    )
    record.update(
        correlation=0.987,
        lr_asset={"path": "/never/open", "min": 42},
        split="internal_test",
        pilot={"correlation_bin": 9},
    )
    raw = (json.dumps(record) + "\n").encode()
    output = project(raw, hashlib.sha256(raw).hexdigest(), 1)
    row = json.loads(output)
    assert row["source_index"] == "0"
    assert row["geotransform"] == ["1.25"]
    assert set(row).isdisjoint({"correlation", "lr_asset", "split", "pilot"})
    assert b"/never/open" not in output


def test_wrong_count_rejected():
    raw = b"{}\n"
    with pytest.raises(ValueError, match="count"):
        project(raw, hashlib.sha256(raw).hexdigest(), 8000)


def test_cli_rejects_untrusted_manifest_without_creating_output(tmp_path, monkeypatch):
    from research.trustmask.recover_manifest import main

    manifest = tmp_path / "samples.jsonl"
    manifest.write_bytes(b"not json")
    output = tmp_path / "export"
    monkeypatch.setattr(
        "sys.argv", ["recover", "--manifest", str(manifest), "--output", str(output)]
    )
    with pytest.raises(ValueError, match="digest"):
        main()
    assert not output.exists()


def test_cli_rejects_symlink(tmp_path, monkeypatch):
    from research.trustmask.recover_manifest import main

    manifest = tmp_path / "samples.jsonl"
    manifest.symlink_to(tmp_path / "absent")
    output = tmp_path / "export"
    monkeypatch.setattr(
        "sys.argv", ["recover", "--manifest", str(manifest), "--output", str(output)]
    )
    with pytest.raises(OSError):
        main()
    assert not output.exists()
