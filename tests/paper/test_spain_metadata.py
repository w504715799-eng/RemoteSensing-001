import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "spain_metadata", ROOT / "scripts/paper/audit_spain_metadata.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def raw_csv(roi="ROI_00001"):
    return (
        "roi,lr_file,hr_file,lr_gee_id,crs,affine,reflectance,spectral,spatial\n"
        'ROI_00002,lr2,HR__ROI_00002__source,scene,EPSG:32630,"2.5,0,10,0,-2.5,20",SECRET,SECRET,SECRET\n'
        f'{roi},lr1,HR__{roi}__source,scene,EPSG:32630,"2.5,0,30,0,-2.5,40",SECRET,SECRET,SECRET\n'
    ).encode()


def test_projection_sorts_members_and_never_exports_quality_fields():
    raw = raw_csv()
    result = audit.project_metadata(
        raw, expected_sha=hashlib.sha256(raw).hexdigest(), expected_count=2
    )
    assert [r["roi"] for r in result["members"]] == ["ROI_00001", "ROI_00002"]
    assert result["hr_source_counts"] == {"source": 2}
    assert result["lr_scene_count"] == 1
    assert result["members"][0]["affine"] == [2.5, 0.0, 30.0, 0.0, -2.5, 40.0]
    assert "SECRET" not in str(result)
    assert set(result["members"][0]) == {"roi", "lr_file", "hr_file", "lr_gee_id", "crs", "affine"}


def test_modified_metadata_is_rejected_before_projection():
    raw = raw_csv()
    with pytest.raises(ValueError, match="SHA-256"):
        audit.project_metadata(
            raw + b"\n", expected_sha=hashlib.sha256(raw).hexdigest(), expected_count=2
        )


@pytest.mark.parametrize("fault", ["duplicate", "count", "nonfinite", "missing_column"])
def test_invalid_member_metadata_cannot_be_used_as_a_manifest(fault):
    raw = raw_csv("ROI_00002") if fault == "duplicate" else raw_csv()
    if fault == "nonfinite":
        raw = raw.replace(b"2.5,0,10", b"nan,0,10")
    if fault == "missing_column":
        raw = raw.replace(b"lr_gee_id,", b"wrong_column,")
    with pytest.raises(ValueError):
        audit.project_metadata(
            raw,
            expected_sha=hashlib.sha256(raw).hexdigest(),
            expected_count=3 if fault == "count" else 2,
        )
