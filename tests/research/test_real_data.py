"""Synthetic-only selected-container boundary tests; never use source rasters."""

import hashlib
import io
import json

import numpy as np
import pytest
import torch


def test_provider_module_exists():
    from research.trustmask import real_data

    assert callable(real_data.LocalTacoProvider)


def parquet(rows):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    out = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), out)
    return out.getvalue()


def raster(size, value, nodata=65535, invalid_mask=False):
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin

    with MemoryFile() as mem:
        with mem.open(
            driver="GTiff",
            count=4,
            width=size,
            height=size,
            dtype="uint16",
            nodata=nodata,
            crs="EPSG:32611",
            transform=from_origin(500000, 4000000, 1300 / size, 1300 / size),
        ) as ds:
            ds.write(np.full((4, size, size), value, dtype=np.uint16))
            if invalid_mask:
                mask = np.full((size, size), 255, dtype=np.uint8)
                mask[0, 0] = 0
                ds.write_mask(mask)
            ds.descriptions = ("B04", "B03", "B02", "B08")
        return mem.read()


def fixture(
    tmp_path, monkeypatch, *, bad_offset=False, nodata=65535, invalid_mask=False, geometry=None
):
    from research.trustmask import real_data as mod

    lr, hr = raster(130, 20000, nodata, invalid_mask), raster(520, 5000)
    nested = parquet(
        [
            {
                "tortilla:id": "lr",
                "stac:crs": geometry.get("crs", "EPSG:32611") if geometry else "EPSG:32611",
                "stac:raster_shape": geometry.get("shape", [130, 130]) if geometry else [130, 130],
                "stac:geotransform": geometry.get("transform", [500000, 10, 0, 4000000, 0, -10])
                if geometry
                else [500000, 10, 0, 4000000, 0, -10],
                "tortilla:file_format": "GTiff",
                "tortilla:offset": 18,
                "tortilla:length": len(lr),
            },
            {
                "tortilla:id": "hr",
                "stac:crs": "EPSG:32611",
                "stac:raster_shape": [520, 520],
                "stac:geotransform": [500000, 2.5, 0, 4000000, 0, -2.5],
                "tortilla:file_format": "GTiff",
                "tortilla:offset": 18 + len(lr) if not bad_offset else 0,
                "tortilla:length": len(hr),
            },
        ]
    )
    pos = 18 + len(lr) + len(hr)
    parent = (
        b"#y" + pos.to_bytes(8, "little") + len(nested).to_bytes(8, "little") + lr + hr + nested
    )
    top = parquet(
        [{"tortilla:id": "selected", "tortilla:offset": 42, "tortilla:length": len(parent)}]
    )
    col = json.dumps({"version": "0.4.0"}).encode()
    start = 42 + len(parent)
    header = (
        b"WX"
        + start.to_bytes(8, "little")
        + len(top).to_bytes(8, "little")
        + bytes(8)
        + (start + len(top)).to_bytes(8, "little")
        + len(col).to_bytes(8, "little")
    )
    raw = header + parent + top + col
    path = tmp_path / "synthetic.taco"
    path.write_bytes(raw)
    monkeypatch.setattr(mod, "SOURCE_SIZE", len(raw))
    monkeypatch.setattr(
        mod,
        "METADATA_SHA256",
        {
            k: hashlib.sha256(v).hexdigest()
            for k, v in [("header", header), ("collection", col), ("directory", top)]
        },
    )
    return mod, path, (42 + 18, len(lr))


def test_preflight_reads_no_pixels_and_wrong_identity_reads_nothing(tmp_path, monkeypatch):
    mod, path, pixel = fixture(tmp_path, monkeypatch)
    reads = []
    orig = mod.os.pread
    monkeypatch.setattr(mod.os, "pread", lambda fd, n, o: (reads.append((o, n)), orig(fd, n, o))[1])
    with mod.LocalTacoProvider(path, {"selected": "group"}, tmp_path) as provider:
        provider.preflight()
        assert len(reads) == 3
        with pytest.raises(ValueError, match="assignment"):
            provider("selected", "other")
        with pytest.raises(ValueError, match="assignment"):
            provider("unknown", "group")
        assert len(reads) == 3
        assert provider.binding_metadata()["full_source_sha256_verified"] is False


def test_selected_pair_keeps_full_support_and_v2_radiometry(tmp_path, monkeypatch):
    mod, path, pixel = fixture(tmp_path, monkeypatch)

    def predictions(self, lr):
        assert tuple(lr.shape) == (4, 128, 128)
        assert torch.all(lr == 1)
        return torch.full((5, 4, 512, 512), 0.25)

    monkeypatch.setattr(mod.LocalTacoProvider, "_predict", predictions)
    with mod.LocalTacoProvider(path, {"selected": "group"}, tmp_path) as provider:
        obs = provider("selected", "group")
        assert obs.risk.shape == (512, 512)
        np.testing.assert_allclose(obs.risk, 0.25)
        assert obs.sample_id == "selected"
        assert len(provider.last_provenance["assets"]) == 2


def test_invalid_nested_range_rejected_before_any_pixel_read(tmp_path, monkeypatch):
    mod, path, pixel = fixture(tmp_path, monkeypatch, bad_offset=True)
    reads = []
    orig = mod.os.pread
    monkeypatch.setattr(mod.os, "pread", lambda fd, n, o: (reads.append((o, n)), orig(fd, n, o))[1])
    with mod.LocalTacoProvider(path, {"selected": "group"}, tmp_path) as provider:
        with pytest.raises(ValueError, match="asset"):
            provider("selected", "group")
    assert pixel not in reads


def test_bad_top_digest_rejected_before_nested_reads(tmp_path, monkeypatch):
    mod, path, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setitem(mod.METADATA_SHA256, "directory", "0" * 64)
    with pytest.raises(ValueError, match="digest"):
        mod.LocalTacoProvider(path, {"selected": "group"}, tmp_path)


def test_lazy_model_shared_across_five_seeds(tmp_path, monkeypatch):
    mod, path, _ = fixture(tmp_path, monkeypatch)
    builds, seeds = [], []

    class FakeModel:
        def for_seed(self, seed):
            seeds.append(seed)
            return self

        def predict(self, lr):
            return torch.full((4, 512, 512), 0.25)

        def provenance(self):
            return {"synthetic": True}

    def build(model_dir, *, device):
        builds.append(device)
        return FakeModel()

    monkeypatch.setattr(mod.LDSRS2X4, "from_pretrained", build)
    with mod.LocalTacoProvider(path, {"selected": "group"}, tmp_path) as provider:
        assert builds == []
        provider("selected", "group")
        assert builds == ["cuda:0"]
        assert seeds == [3407, 3408, 3409, 3410, 3411]
        assert provider.preflight()["pixel_payload_read"] is True


@pytest.mark.parametrize(
    "geometry",
    [
        {"crs": "EPSG:32610"},
        {"shape": [128, 128]},
        {"transform": [500010, 10, 0, 4000000, 0, -10]},
    ],
)
def test_nested_geometry_mismatch_rejected(tmp_path, monkeypatch, geometry):
    mod, path, _ = fixture(tmp_path, monkeypatch, geometry=geometry)
    with mod.LocalTacoProvider(path, {"selected": "group"}, tmp_path) as provider:
        with pytest.raises(ValueError, match="geometry"):
            provider._load_pair("selected")


@pytest.mark.parametrize("options", [{"nodata": None}, {"invalid_mask": True}])
def test_invalid_nodata_policy_rejected(tmp_path, monkeypatch, options):
    mod, path, _ = fixture(tmp_path, monkeypatch, **options)
    with mod.LocalTacoProvider(path, {"selected": "group"}, tmp_path) as provider:
        with pytest.raises(ValueError, match="nodata"):
            provider._load_pair("selected")


def test_crop_clipping_receipts(tmp_path, monkeypatch):
    mod, path, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        mod.LocalTacoProvider, "_predict", lambda self, lr: torch.full((5, 4, 512, 512), 0.25)
    )
    with mod.LocalTacoProvider(path, {"selected": "group"}, tmp_path) as provider:
        provider("selected", "group")
        receipts = {r["kind"]: r for r in provider.last_provenance["assets"]}
        assert receipts["lr"]["clipped_high_count"] == 65536
        assert receipts["lr"]["crop_value_count"] == 65536
        assert receipts["hr"]["clipped_high_count"] == 0
        assert receipts["hr"]["crop_value_count"] == 1048576


def test_source_must_be_regular_file(tmp_path, monkeypatch):
    mod, path, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="regular"):
        mod.LocalTacoProvider(tmp_path, {"selected": "group"}, tmp_path)


def test_corrupt_header_cannot_redirect_metadata_reads_into_pixels(tmp_path, monkeypatch):
    mod, path, _ = fixture(tmp_path, monkeypatch)
    with path.open("r+b") as stream:
        stream.seek(2)
        stream.write((100).to_bytes(8, "little"))
    reads = []
    orig = mod.os.pread
    monkeypatch.setattr(mod.os, "pread", lambda fd, n, o: (reads.append((o, n)), orig(fd, n, o))[1])
    with pytest.raises(ValueError, match="digest"):
        mod.LocalTacoProvider(path, {"selected": "group"}, tmp_path)
    assert reads == [(0, 42)]
