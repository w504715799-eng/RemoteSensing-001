"""Offline tests for the deliberately isolated legacy TACO v1 adapter."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from affine import Affine
from rasterio.io import MemoryFile

import trustsr.data.taco_v1_adapter as taco_v1_adapter
from trustsr.data.crosssensor_schema import AcquisitionTimes
from trustsr.jsonio import atomic_write_bytes

_BANDS = ("B04", "B03", "B02", "B08")
_LR_TIME = "2020-01-02T10:00:00Z"
_HR_TIME = "2020-01-03T10:00:00Z"
_TRANSFORM = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 400000.0)


class _Rows:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.rows[index]


class _Table:
    def __init__(self, rows: list[dict[str, object]], assets: list[bytes] | None = None) -> None:
        self.rows = rows
        self.assets = assets
        self.iloc = _Rows(rows)
        self.read_calls: list[int] = []

    def __len__(self) -> int:
        return len(self.rows)

    def to_dict(self, *, orient: str) -> list[dict[str, object]]:
        assert orient == "records"
        return self.rows

    def read(self, index: int) -> _Table | bytes:
        self.read_calls.append(index)
        if self.assets is None:
            return _NESTED
        return self.assets[index]


_NESTED: _Table


class _Reader:
    def __init__(self, metadata: dict[str, object], top: _Table) -> None:
        self.metadata = metadata
        self.top = top
        self.load_metadata_calls: list[str] = []
        self.load_calls: list[str] = []

    def load_metadata(self, path: str) -> dict[str, object]:
        self.load_metadata_calls.append(path)
        return self.metadata

    def load(self, path: str) -> _Table:
        self.load_calls.append(path)
        return self.top


def _geotiff_bytes(
    *,
    width: int,
    height: int,
    value: int | float,
    dtype: str = "uint16",
    crs: str = "EPSG:32618",
    transform: Affine = _TRANSFORM,
    descriptions: tuple[str | None, ...] | None = None,
) -> bytes:
    if dtype == "float32" and isinstance(value, float) and np.isnan(value):
        values = np.full((4, height, width), value, dtype=np.float32)
    else:
        values = np.full((4, height, width), value, dtype=dtype)
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            width=width,
            height=height,
            count=4,
            dtype=dtype,
            crs=crs,
            transform=transform,
        ) as dataset:
            dataset.write(values)
            if descriptions is not None:
                for band, description in enumerate(descriptions, start=1):
                    dataset.set_band_description(band, description)
        return memory.read()


def _three_band_geotiff_bytes() -> bytes:
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            width=130,
            height=130,
            count=3,
            dtype="uint16",
            crs="EPSG:32618",
            transform=_TRANSFORM,
        ) as dataset:
            dataset.write(np.full((3, 130, 130), 100, dtype=np.uint16))
        return memory.read()


def _vrt_bytes() -> bytes:
    band_template = (
        '  <VRTRasterBand dataType="UInt16" band="{band}">'
        "<NoDataValue>0</NoDataValue>"
        "</VRTRasterBand>"
    )
    bands = "\n".join(
        band_template.format(band=band) for band in range(1, 5)
    )
    return (
        "<VRTDataset rasterXSize=\"130\" rasterYSize=\"130\">\n"
        "  <SRS>EPSG:32618</SRS>\n"
        "  <GeoTransform>500000,10,0,400000,0,-10</GeoTransform>\n"
        f"{bands}\n"
        "</VRTDataset>\n"
    ).encode()


def _install_reader(
    monkeypatch: pytest.MonkeyPatch,
    *,
    metadata: dict[str, object],
    assets: list[bytes],
    rows: list[dict[str, object]] | None = None,
) -> _Reader:
    global _NESTED
    if rows is None:
        rows = [
            {"stac:time_start": _LR_TIME if index == 0 else _HR_TIME}
            for index in range(len(assets))
        ]
    _NESTED = _Table(rows, assets)
    top = _Table([{"tortilla:id": "sample-0"}])
    reader = _Reader(metadata, top)
    monkeypatch.setitem(sys.modules, "tacoreader", reader)
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.4.5")
    return reader


def test_importing_adapter_does_not_import_the_optional_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "tacoreader", raising=False)
    monkeypatch.delitem(sys.modules, "trustsr.data.taco_v1_adapter", raising=False)

    importlib.import_module("trustsr.data.taco_v1_adapter")

    assert "tacoreader" not in sys.modules


def test_require_tacoreader_v1_accepts_only_the_pinned_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "tacoreader", reader)
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.4.5")

    assert taco_v1_adapter.require_tacoreader_v1() is reader


def test_require_tacoreader_v1_explains_the_cloud_bootstrap_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "tacoreader", raising=False)

    def _missing(_: str) -> str:
        raise importlib.metadata.PackageNotFoundError("tacoreader")

    monkeypatch.setattr(importlib.metadata, "version", _missing)

    with pytest.raises(RuntimeError, match="cloud.*bootstrap"):
        taco_v1_adapter.require_tacoreader_v1()


def test_require_tacoreader_v1_rejects_a_new_reader_even_with_v1_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "tacoreader", SimpleNamespace(v1=SimpleNamespace()))
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "2.4.21")

    with pytest.raises(RuntimeError, match=r"0\.4\.5"):
        taco_v1_adapter.require_tacoreader_v1()


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({}, "collection.*version"),
        ({"taco_version": "0.4.0", "version": "0.4.1"}, "ambiguous"),
        ({"taco_version": "0.5.0"}, r"0\.4\.0"),
    ],
)
def test_load_top_level_records_rejects_invalid_legacy_collection_versions(
    monkeypatch: pytest.MonkeyPatch, metadata: dict[str, object], message: str
) -> None:
    reader = _install_reader(monkeypatch, metadata=metadata, assets=[])

    with pytest.raises(ValueError, match=message):
        taco_v1_adapter.load_top_level_records(Path("/cloud/source.taco"))

    assert reader.load_metadata_calls == ["/cloud/source.taco"]
    assert reader.load_calls == []


@pytest.mark.parametrize(
    "metadata",
    [
        {"taco_version": "0.4.0"},
        {"taco_version": "0.4.0", "version": "0.4.0"},
    ],
)
def test_load_top_level_records_accepts_the_exact_legacy_collection_version(
    monkeypatch: pytest.MonkeyPatch, metadata: dict[str, object]
) -> None:
    reader = _install_reader(monkeypatch, metadata=metadata, assets=[])

    assert taco_v1_adapter.load_top_level_records(Path("/cloud/source.taco")) == (
        {"tortilla:id": "sample-0"},
    )
    assert reader.load_calls == ["/cloud/source.taco"]


def test_load_crosssensor_metadata_reads_all_nested_rows_without_reading_pixels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _install_reader(
        monkeypatch,
        metadata={"taco_version": "0.4.0"},
        assets=[b"pixel-read-must-not-happen", b"pixel-read-must-not-happen"],
        rows=[{"stac:time_start": _LR_TIME}, {"stac:time_start": _HR_TIME}],
    )

    records, acquisition_times = taco_v1_adapter.load_crosssensor_metadata(
        Path("/cloud/source.taco")
    )

    assert records == ({"tortilla:id": "sample-0"},)
    assert acquisition_times == (AcquisitionTimes(_LR_TIME, _HR_TIME),)
    assert reader.load_metadata_calls == ["/cloud/source.taco"]
    assert reader.load_calls == ["/cloud/source.taco"]
    assert reader.top.read_calls == [0]
    assert _NESTED.read_calls == []


def test_load_crosssensor_metadata_normalizes_legacy_epoch_seconds_to_rfc3339(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _install_reader(
        monkeypatch,
        metadata={"taco_version": "0.4.0"},
        assets=[b"pixel-read-must-not-happen", b"pixel-read-must-not-happen"],
        rows=[
            {"stac:time_start": 1_577_959_200.0},
            {"stac:time_start": 1_578_045_600.0},
        ],
    )
    reader.top.rows[0]["stac:time_start"] = 1_577_959_200.0

    records, acquisition_times = taco_v1_adapter.load_crosssensor_metadata(
        Path("/cloud/source.taco")
    )

    assert records[0]["stac:time_start"] == _LR_TIME
    assert acquisition_times == (AcquisitionTimes(_LR_TIME, _HR_TIME),)
    assert _NESTED.read_calls == []


def test_load_crosssensor_metadata_rejects_nested_row_count_without_reading_pixels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_reader(
        monkeypatch,
        metadata={"taco_version": "0.4.0"},
        assets=[b"pixel-read-must-not-happen"],
        rows=[{"stac:time_start": _LR_TIME}],
    )

    with pytest.raises(ValueError, match="exactly two nested metadata rows"):
        taco_v1_adapter.load_crosssensor_metadata(Path("/cloud/source.taco"))

    assert _NESTED.read_calls == []


@pytest.mark.parametrize(
    "rows",
    [
        [{}, {"stac:time_start": _HR_TIME}],
        [{"stac:time_start": _LR_TIME}, {"stac:time_start": ""}],
    ],
)
def test_load_crosssensor_metadata_rejects_missing_nested_time_without_reading_pixels(
    monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, object]]
) -> None:
    _install_reader(
        monkeypatch,
        metadata={"taco_version": "0.4.0"},
        assets=[b"pixel-read-must-not-happen", b"pixel-read-must-not-happen"],
        rows=rows,
    )

    with pytest.raises(ValueError, match="nested asset stac:time_start"):
        taco_v1_adapter.load_crosssensor_metadata(Path("/cloud/source.taco"))

    assert _NESTED.read_calls == []


def test_extract_pair_writes_raw_geotiffs_and_records_their_source_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lr_bytes = _geotiff_bytes(width=130, height=130, value=100)
    hr_bytes = _geotiff_bytes(
        width=520,
        height=520,
        value=120,
        transform=Affine(2.5, 0.0, 500000.0, 0.0, -2.5, 400000.0),
    )
    reader = _install_reader(
        monkeypatch,
        metadata={"taco_version": "0.4.0"},
        assets=[hr_bytes, lr_bytes],
        rows=[{"stac:time_start": _HR_TIME}, {"stac:time_start": _LR_TIME}],
    )

    lr_asset, hr_asset = taco_v1_adapter.extract_pair(
        Path("/cloud/source.taco"), 7, tmp_path / "sample", _BANDS
    )

    assert reader.load_metadata_calls == ["/cloud/source.taco"]
    assert reader.load_calls == ["/cloud/source.taco"]
    assert reader.top.read_calls == [7]
    assert (tmp_path / "sample" / "lr.tif").read_bytes() == lr_bytes
    assert (tmp_path / "sample" / "hr.tif").read_bytes() == hr_bytes
    assert lr_asset.relative_path == "lr.tif"
    assert lr_asset.size_bytes == len(lr_bytes)
    assert lr_asset.sha256 == hashlib.sha256(lr_bytes).hexdigest()
    assert lr_asset.shape == (4, 130, 130)
    assert lr_asset.dtype == "uint16"
    assert lr_asset.crs == "EPSG:32618"
    assert lr_asset.transform == (10.0, 0.0, 500000.0, 0.0, -10.0, 400000.0)
    assert lr_asset.nodata is None
    assert lr_asset.minimum == 100.0
    assert lr_asset.maximum == 100.0
    assert lr_asset.time_start == _LR_TIME
    assert hr_asset.relative_path == "hr.tif"
    assert hr_asset.size_bytes == len(hr_bytes)
    assert hr_asset.sha256 == hashlib.sha256(hr_bytes).hexdigest()
    assert hr_asset.shape == (4, 520, 520)
    assert hr_asset.dtype == "uint16"
    assert hr_asset.crs == "EPSG:32618"
    assert hr_asset.transform == (2.5, 0.0, 500000.0, 0.0, -2.5, 400000.0)
    assert hr_asset.nodata is None
    assert hr_asset.minimum == 120.0
    assert hr_asset.maximum == 120.0
    assert hr_asset.time_start == _HR_TIME


@pytest.mark.parametrize(
    ("assets", "rows", "message"),
    [
        (
            [
                _three_band_geotiff_bytes(),
                _geotiff_bytes(
                    width=520,
                    height=520,
                    value=120,
                    transform=Affine(2.5, 0.0, 500000.0, 0.0, -2.5, 400000.0),
                ),
            ],
            None,
            "four bands",
        ),
        (
            [
                _geotiff_bytes(width=129, height=130, value=100),
                _geotiff_bytes(
                    width=520,
                    height=520,
                    value=120,
                    transform=Affine(2.5, 0.0, 500000.0, 0.0, -2.5, 400000.0),
                ),
            ],
            None,
            "dimensions",
        ),
        (
            [
                _geotiff_bytes(width=130, height=130, value=100),
                _geotiff_bytes(
                    width=520,
                    height=520,
                    value=120,
                    crs="EPSG:32619",
                    transform=Affine(2.5, 0.0, 500000.0, 0.0, -2.5, 400000.0),
                ),
            ],
            None,
            "CRS",
        ),
        (
            [
                _geotiff_bytes(width=130, height=130, value=100),
                _geotiff_bytes(
                    width=520,
                    height=520,
                    value=120,
                    transform=Affine(2.5, 0.0, 500001.0, 0.0, -2.5, 400000.0),
                ),
            ],
            None,
            "bounds",
        ),
        (
            [
                _geotiff_bytes(width=130, height=130, value=float("nan"), dtype="float32"),
                _geotiff_bytes(
                    width=520,
                    height=520,
                    value=120,
                    transform=Affine(2.5, 0.0, 500000.0, 0.0, -2.5, 400000.0),
                ),
            ],
            None,
            "finite",
        ),
        (
            [
                _geotiff_bytes(width=130, height=130, value=100),
                _geotiff_bytes(width=130, height=130, value=120),
            ],
            None,
            "one LR.*one HR",
        ),
        (
            [
                _geotiff_bytes(width=130, height=130, value=100),
                _geotiff_bytes(
                    width=520,
                    height=520,
                    value=120,
                    transform=Affine(2.5, 0.0, 500000.0, 0.0, -2.5, 400000.0),
                ),
                _geotiff_bytes(
                    width=520,
                    height=520,
                    value=130,
                    transform=Affine(2.5, 0.0, 500000.0, 0.0, -2.5, 400000.0),
                ),
            ],
            None,
            "exactly two nested assets",
        ),
    ],
)
def test_extract_pair_rejects_invalid_raster_pairs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    assets: list[bytes],
    rows: list[dict[str, object]] | None,
    message: str,
) -> None:
    _install_reader(monkeypatch, metadata={"taco_version": "0.4.0"}, assets=assets, rows=rows)

    with pytest.raises(ValueError, match=message):
        taco_v1_adapter.extract_pair(Path("/cloud/source.taco"), 0, tmp_path, _BANDS)

    assert list(tmp_path.iterdir()) == []


def test_extract_pair_rejects_wrong_source_band_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_reader(monkeypatch, metadata={"taco_version": "0.4.0"}, assets=[])

    with pytest.raises(ValueError, match="B04"):
        taco_v1_adapter.extract_pair(
            Path("/cloud/source.taco"), 0, tmp_path, ("B02", "B03", "B04", "B08")
        )


def test_extract_pair_rejects_contradictory_present_band_descriptions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lr = _geotiff_bytes(
        width=130,
        height=130,
        value=100,
        descriptions=("B04", "B03", "B02", "wrong"),
    )
    hr = _geotiff_bytes(
        width=520,
        height=520,
        value=120,
        transform=Affine(2.5, 0.0, 500000.0, 0.0, -2.5, 400000.0),
    )
    _install_reader(monkeypatch, metadata={"taco_version": "0.4.0"}, assets=[lr, hr])

    with pytest.raises(ValueError, match="band descriptions"):
        taco_v1_adapter.extract_pair(Path("/cloud/source.taco"), 0, tmp_path, _BANDS)


def test_extract_pair_rejects_a_non_geotiff_raster_before_writing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hr = _geotiff_bytes(
        width=520,
        height=520,
        value=120,
        transform=Affine(2.5, 0.0, 500000.0, 0.0, -2.5, 400000.0),
    )
    _install_reader(monkeypatch, metadata={"taco_version": "0.4.0"}, assets=[_vrt_bytes(), hr])

    with pytest.raises(ValueError, match="GTiff"):
        taco_v1_adapter.extract_pair(Path("/cloud/source.taco"), 0, tmp_path, _BANDS)

    assert list(tmp_path.iterdir()) == []


def test_extract_pair_reuses_identical_cached_assets_without_writing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lr_bytes = _geotiff_bytes(width=130, height=130, value=100)
    hr_bytes = _geotiff_bytes(
        width=520,
        height=520,
        value=120,
        transform=Affine(2.5, 0.0, 500000.0, 0.0, -2.5, 400000.0),
    )
    _install_reader(monkeypatch, metadata={"taco_version": "0.4.0"}, assets=[lr_bytes, hr_bytes])
    output_root = tmp_path / "sample"
    output_root.mkdir()
    atomic_write_bytes(output_root / "lr.tif", lr_bytes)
    atomic_write_bytes(output_root / "hr.tif", hr_bytes)

    monkeypatch.setattr(
        taco_v1_adapter,
        "atomic_write_bytes",
        lambda *_: pytest.fail("identical cached bytes must not be rewritten"),
    )

    lr_asset, hr_asset = taco_v1_adapter.extract_pair(
        Path("/cloud/source.taco"), 0, output_root, _BANDS
    )

    assert (lr_asset.sha256, hr_asset.sha256) == (
        hashlib.sha256(lr_bytes).hexdigest(),
        hashlib.sha256(hr_bytes).hexdigest(),
    )


def test_extract_pair_never_replaces_different_cached_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lr_bytes = _geotiff_bytes(width=130, height=130, value=100)
    hr_bytes = _geotiff_bytes(
        width=520,
        height=520,
        value=120,
        transform=Affine(2.5, 0.0, 500000.0, 0.0, -2.5, 400000.0),
    )
    _install_reader(monkeypatch, metadata={"taco_version": "0.4.0"}, assets=[lr_bytes, hr_bytes])
    output_root = tmp_path / "sample"
    output_root.mkdir()
    (output_root / "lr.tif").write_bytes(b"different")

    with pytest.raises(ValueError, match="different bytes"):
        taco_v1_adapter.extract_pair(Path("/cloud/source.taco"), 0, output_root, _BANDS)

    assert (output_root / "lr.tif").read_bytes() == b"different"
    assert not (output_root / "hr.tif").exists()
