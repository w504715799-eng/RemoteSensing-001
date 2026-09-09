"""Selected-only local access to the pinned public SEN2NAIPv2 TACO release.

Only authenticated top metadata and assigned nested metadata are read at preflight.
Pixels and predictions are transient; the full source digest is declared, not checked.
"""

from __future__ import annotations

import hashlib
import io
import os
import stat
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch

from research.trustmask.catalog import fetch_metadata, nested_directory_interval
from research.trustmask.pipeline import Observation, extract_features
from trustsr.models.ldsr_assets import CHECKPOINT_SHA256, CONFIG_SHA256
from trustsr.models.ldsr_s2 import LDSRS2X4
from trustsr.risk.local import local_l1_risk

SOURCE_SIZE = 9717583850
SOURCE_REVISION = "c370504201072fdb1dd388013ab8c0fc7d00a57e"
SOURCE_DECLARED_SHA256 = "c6f29d8e80dc5e856e2b4510c0e6830043d4b15c9228a9ca249a4f618e7475a5"
METADATA_SHA256 = {
    "header": "0ae2b7363e73464ffd0c0c64e890277d9c0718caa3786bf7166aa8fc12937c93",
    "collection": "ef1505137d5431bfcc1098a91482f1c52e6a99edb8ab58c5e1f45a73ae38af03",
    "directory": "9eb6bc8f5e52b3ca9f0aba0a733c98b95e579e6b5c02db515f44cc6d36921ea1",
}
SEEDS = tuple(range(3407, 3412))


def binding_metadata():
    """Static scientific binding; this does not load pixels, models, or the source."""
    return {
        "provider": "local-selected-taco-v1",
        "source_revision": SOURCE_REVISION,
        "source_object": "sen2naipv2-crosssensor.taco",
        "source_size": SOURCE_SIZE,
        "source_declared_sha256": SOURCE_DECLARED_SHA256,
        "full_source_sha256_verified": False,
        "expected_metadata_sha256": dict(METADATA_SHA256),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "config_sha256": CONFIG_SHA256,
        "seeds": list(SEEDS),
        "center_seed": 3407,
        "sampling_steps": 100,
        "sampling_eta": 0.95,
        "sampling_temperature": 1.0,
        "histogram_matching": True,
        "nodata_policy": "uint16_65535_sentinel_all_raw_masks_valid",
        "bands": ["B04", "B03", "B02", "B08"],
        "normalization_policy": "uint16_saturate_10000_divide_10000_v2",
        "crop_policy": "fixed_lr_[1:129,1:129]_hr_[4:516,4:516]",
        "raw_lr_shape": [4, 130, 130],
        "raw_hr_shape": [4, 520, 520],
        "lr_shape": [4, 128, 128],
        "hr_shape": [4, 512, 512],
        "risk_window": 9,
        "support": "entire_fixed_512x512_grid",
    }


def _rows(raw, columns):
    import pyarrow.parquet as pq

    return pq.read_table(io.BytesIO(raw), columns=columns).to_pylist()


def _interval(row, lower, upper):
    offset, length = row.get("tortilla:offset"), row.get("tortilla:length")
    if (
        type(offset) is not int
        or type(length) is not int
        or offset < lower
        or not 1 <= length <= 8 * 1024 * 1024
        or offset + length > upper
    ):
        raise ValueError("asset or parent interval outside authorized bounds")
    return offset, length


class LocalTacoProvider:
    """Callable provider restricted to an immutable sample_id -> group_id assignment."""

    binding_metadata = staticmethod(binding_metadata)

    def __init__(self, taco_path, assignments, model_dir, device="cuda:0"):
        assignments = dict(assignments)
        if not assignments or any(
            not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
            for k, v in assignments.items()
        ):
            raise ValueError("nonempty sample/group assignments required")
        self.assignments = MappingProxyType(assignments)
        self.model_dir, self.device = Path(model_dir), device
        self._model = None
        self.last_provenance = None
        self._pixel_payload_read = False
        self._fd = os.open(Path(taco_path), os.O_RDONLY)
        try:
            self._stat = os.fstat(self._fd)
            if not stat.S_ISREG(self._stat.st_mode):
                raise ValueError("source must be a regular file")
            if self._stat.st_size != SOURCE_SIZE:
                raise ValueError("source size differs from fixed public source")
            header = self._read(0, 42)
            if hashlib.sha256(header).hexdigest() != METADATA_SHA256["header"]:
                raise ValueError("header metadata digest mismatch")
            metadata = fetch_metadata(
                lambda offset, length: (
                    header if (offset, length) == (0, 42) else self._read(offset, length)
                ),
                SOURCE_SIZE,
            )
            for key, digest in METADATA_SHA256.items():
                if hashlib.sha256(metadata[key]).hexdigest() != digest:
                    raise ValueError(f"{key} metadata digest mismatch")
            self._directory_start = metadata["intervals"]["directory"][0]
            self._parents = {}
            seen, intervals = set(), []
            for row in _rows(
                metadata["directory"], ["tortilla:id", "tortilla:offset", "tortilla:length"]
            ):
                identity = row["tortilla:id"]
                if not isinstance(identity, str) or not identity or identity in seen:
                    raise ValueError("top directory identity invalid or duplicate")
                seen.add(identity)
                offset, length = _interval(row, 42, self._directory_start)
                intervals.append((offset, offset + length))
                if identity in self.assignments:
                    self._parents[identity] = (offset, length)
            intervals.sort()
            if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:], strict=False)):
                raise ValueError("top directory parents overlap")
            if set(self._parents) != set(self.assignments):
                raise ValueError("assignment missing from authenticated top directory")
        except BaseException:
            self.close()
            raise

    def _read(self, offset, length):
        if (
            type(offset) is not int
            or type(length) is not int
            or offset < 0
            or not 1 <= length <= 8 * 1024 * 1024
            or offset + length > SOURCE_SIZE
        ):
            raise ValueError("invalid bounded local range")
        now = os.fstat(self._fd)
        if (now.st_size, now.st_mtime_ns, now.st_ctime_ns) != (
            self._stat.st_size,
            self._stat.st_mtime_ns,
            self._stat.st_ctime_ns,
        ):
            raise ValueError("source changed during selected access")
        raw = os.pread(self._fd, length, offset)
        if len(raw) != length:
            raise ValueError("short local range read")
        return raw

    def preflight(self):
        return {
            "metadata_verified": True,
            "source_file_fingerprint": {
                "size": self._stat.st_size,
                "mtime_ns": self._stat.st_mtime_ns,
                "ctime_ns": self._stat.st_ctime_ns,
                "device": self._stat.st_dev,
                "inode": self._stat.st_ino,
            },
            "assigned_members": len(self.assignments),
            "pixel_payload_read": self._pixel_payload_read,
            "model_loaded": self._model is not None,
        }

    def close(self):
        if getattr(self, "_fd", None) is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _load_pair(self, sample_id):
        from rasterio.io import MemoryFile

        from trustsr.data.taco_v1_adapter import (
            _inspect_raster,
            _pair_by_dimensions,
            _validate_pair_geometry,
        )

        parent, size = self._parents[sample_id]
        header = self._read(parent, 18)
        start, length = nested_directory_interval(header, parent, size, self._directory_start)
        raw = self._read(start, length)
        rows = _rows(
            raw,
            [
                "tortilla:id",
                "tortilla:file_format",
                "tortilla:offset",
                "tortilla:length",
                "stac:crs",
                "stac:geotransform",
                "stac:raster_shape",
            ],
        )
        if len(rows) != 2 or {r["tortilla:id"] for r in rows} != {"lr", "hr"}:
            raise ValueError("selected nested directory must contain LR and HR assets")
        ranges = []
        for row in rows:
            if row["tortilla:file_format"] != "GTiff":
                raise ValueError("selected asset must be GTiff")
            offset, count = _interval(row, 18, start - parent)
            ranges.append((parent + offset, count))
        (a, n), (b, m) = ranges
        if max(a, b) < min(a + n, b + m):
            raise ValueError("selected asset ranges overlap")
        rasters, receipts = [], []
        for row, (offset, count) in zip(rows, ranges, strict=True):
            self._pixel_payload_read = True
            payload = self._read(offset, count)
            inspected = _inspect_raster(payload, "selected")
            expected = (4, 130, 130) if row["tortilla:id"] == "lr" else (4, 520, 520)
            if (
                inspected.shape != expected
                or inspected.dtype != "uint16"
                or inspected.maximum > 32767
            ):
                raise ValueError("selected asset identity, dtype, or radiometry invalid")
            if inspected.nodata != 65535:
                raise ValueError("asset nodata must equal frozen sentinel 65535")
            with MemoryFile(payload) as mem, mem.open() as dataset:
                if np.any(dataset.read_masks() == 0):
                    raise ValueError("asset contains invalid nodata pixels")
            t = inspected.transform
            declared_transform = row["stac:geotransform"]
            if (
                row["stac:crs"] != inspected.crs
                or row["stac:raster_shape"] != list(inspected.shape[1:])
                or not isinstance(declared_transform, list)
                or len(declared_transform) != 6
                or not np.allclose(
                    declared_transform, [t[2], t[0], t[1], t[5], t[3], t[4]], rtol=0, atol=1e-6
                )
            ):
                raise ValueError("nested declared geometry differs from selected raster")
            expected_res = 10 if row["tortilla:id"] == "lr" else 2.5
            if not np.allclose(
                [t[0], t[1], t[3], t[4]], [expected_res, 0, 0, -expected_res], rtol=0, atol=1e-6
            ):
                raise ValueError("selected asset transform invalid")
            rasters.append(inspected)
            receipts.append(
                {
                    "kind": row["tortilla:id"],
                    "offset": offset,
                    "length": count,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        lr, hr = _pair_by_dimensions(tuple(rasters))
        _validate_pair_geometry(lr, hr)
        tensors = []
        for kind, raster, cut in (("lr", lr, slice(1, 129)), ("hr", hr, slice(4, 516))):
            with MemoryFile(raster.payload) as mem, mem.open() as ds:
                values = ds.read()[:, cut, cut]
            receipt = next(item for item in receipts if item["kind"] == kind)
            receipt["crop_value_count"] = int(values.size)
            receipt["clipped_high_count"] = int(np.count_nonzero(values > 10000))
            receipt["clipped_high_by_band"] = [int(np.count_nonzero(b > 10000)) for b in values]
            tensors.append(
                torch.from_numpy(np.minimum(values, 10000).astype(np.float32)).div_(10000)
            )
        return *tensors, receipts, hashlib.sha256(raw).hexdigest()

    def _predict(self, lr):
        if self._model is None:
            self._model = LDSRS2X4.from_pretrained(self.model_dir, device=self.device)
        return torch.stack([self._model.for_seed(seed).predict(lr) for seed in SEEDS])

    def __call__(self, sample_id, group_id):
        if sample_id not in self.assignments or self.assignments[sample_id] != group_id:
            raise ValueError("sample/group outside authorized assignment")
        self.last_provenance = None
        lr, hr, assets, nested_digest = self._load_pair(sample_id)
        samples = self._predict(lr)
        if (
            samples.dtype != torch.float32
            or tuple(samples.shape) != (5, 4, 512, 512)
            or not torch.isfinite(samples).all()
            or (samples < 0).any()
            or (samples > 1).any()
        ):
            raise ValueError("prediction support or values invalid")
        features = extract_features(lr, samples)
        risk = local_l1_risk(samples[0], hr, window=9).numpy()
        observation = Observation(sample_id, group_id, features, risk)
        self.last_provenance = {
            "sample_id": sample_id,
            "group_id": group_id,
            "assets": assets,
            "nested_directory_sha256": nested_digest,
            "model": self._model.provenance() if self._model is not None else None,
        }
        return observation
