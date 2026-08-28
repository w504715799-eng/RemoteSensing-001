# Phase 2B1A Crosssensor Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a cloud-only, integrity-checked SEN2NAIPv2 crosssensor pipeline that creates a deterministic 5 km leak-safe manifest and extracts exactly 36 stratified LR/HR pilot pairs.

**Architecture:** Keep metadata normalization, spatial grouping, sampling, and canonical serialization independent of `tacoreader`. Put the legacy TACO v1 import behind a cloud adapter, and expose download, manifest, pilot, and audit as restartable CLI stages. Real pixels and the full manifest stay in user-supplied cloud storage; Git receives only code, synthetic tests, and a small canonical audit summary.

**Tech Stack:** Python 3.12, NumPy, pandas, SciPy `cKDTree`, Rasterio, pytest, Ruff, Bash, curl, `tacoreader==0.4.5` on the cloud base environment only.

**Spec:** `docs/superpowers/specs/2026-08-28-phase2b1a-crosssensor-pilot.md`

## Global Constraints

- The only real object is `sen2naipv2-crosssensor.taco`, size `9_717_583_850`, SHA-256 `c6f29d8e80dc5e856e2b4510c0e6830043d4b15c9228a9ca249a4f618e7475a5`, from revision `c370504201072fdb1dd388013ab8c0fc7d00a57e`.
- Never download a real TACO object or real GeoTIFF into the WSL workspace; local tests generate synthetic bytes only.
- Real execution requires an explicit cloud storage root and an explicit HTTPS transport URL; neither has a project default.
- `tacoreader==0.4.5` is cloud-extraction-only and must not enter `pyproject.toml` or `uv.lock`.
- The cloud command uses `/opt/conda/bin/python` in the base environment and creates no Conda environment.
- Do not modify PyTorch, torchvision, triton, CUDA packages, or packages whose normalized names start with `nvidia-` while installing the legacy reader.
- Do not hard-code an SSH host, port, username, password, cloud root, GPU model, or CUDA version.
- The spatial threshold is inclusive `<= 5 km`, the Earth radius is `6_371.0088 km`, and an entire connected component receives one split.
- Split buckets are `[0,.5)` development, `[.5,.75)` calibration, and `[.75,1)` internal test, using the first 16 hex characters of `spatial_group_id` divided by `2**64`.
- Pilot strata are `days_between` `-1/0/1` crossed with the four fixed correlation bins at `0.8842208864`, `0.9041984739`, and `0.9265462586`.
- The final pilot contains exactly 12 distinct spatial groups per split, 36 sample pairs, and 72 GeoTIFF files.
- This phase does not use a GPU, run a super-resolution model, compute paper metrics, or expand to 360 pairs.
- Every task follows red-green-refactor TDD and ends with a focused commit.

## File Structure

- `src/trustsr/jsonio.py`: canonical JSON bytes and durable atomic file writes shared by metadata artifacts.
- `src/trustsr/data/crosssensor_schema.py`: strict top-level TACO metadata normalization and immutable sample records.
- `src/trustsr/data/spatial_split.py`: haversine distances, 5 km connected components, deterministic split assignment, and cross-split distance audit.
- `src/trustsr/data/pilot_sampling.py`: fixed correlation bins and deterministic 36-sample selection.
- `src/trustsr/data/crosssensor_manifest.py`: JSONL manifest records, digests, and canonical audit payloads.
- `src/trustsr/data/crosssensor_source.py`: frozen object lookup, resumable curl transport, size/SHA verification, and quarantine.
- `src/trustsr/data/taco_v1_adapter.py`: optional `tacoreader==0.4.5` import, nested asset reads, GeoTIFF validation, and atomic extraction.
- `src/trustsr/cli/phase2b1a.py`: restartable `download`, `manifest`, `pilot`, and `audit` commands.
- `requirements/cloud-taco-v1.txt`: exact cloud-only reader and geospatial dependency snapshot.
- `scripts/phase2b1a/bootstrap_reader.sh`: base-environment install with a protected PyTorch/CUDA dry-run gate.
- `scripts/phase2b1a/run_cloud.sh`: validated cloud storage wrapper around the CLI.
- `tests/data/test_crosssensor_schema.py`: strict schema normalization tests.
- `tests/data/test_spatial_split.py`: boundary, transitive component, determinism, and no-leakage tests.
- `tests/data/test_pilot_sampling.py`: 12 strata per split and distinct-group selection tests.
- `tests/data/test_crosssensor_manifest.py`: canonical JSONL, digest, and audit tests.
- `tests/data/test_crosssensor_source.py`: offline acquisition and integrity tests with a fake curl runner.
- `tests/data/test_taco_v1_adapter.py`: fake reader plus runtime-generated GeoTIFF extraction tests.
- `tests/cli/test_phase2b1a.py`: CLI stage and fail-closed behavior tests.
- `tests/scripts/test_phase2b1a_scripts.py`: base-only bootstrap and safe-path shell contract tests.
- `artifacts/datasets/sen2naipv2-phase2b1a-audit-v1.json`: added only after the real cloud gate passes; contains no paths or pixels.

---

### Task 1: Shared Canonical JSON and Atomic Writes

**Files:**
- Create: `src/trustsr/jsonio.py`
- Modify: `src/trustsr/artifacts/predictions.py`
- Test: `tests/test_jsonio.py`
- Test: `tests/artifacts/test_predictions.py`

**Interfaces:**
- Consumes: JSON-native values and a destination `Path`.
- Produces: `canonical_json(value: Any) -> bytes` and `atomic_write_bytes(path: Path, payload: bytes) -> None`.

- [ ] **Step 1: Write failing canonicalization and atomic-write tests**

```python
import json
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
```

- [ ] **Step 2: Run the new tests and confirm the missing module failure**

Run: `uv run pytest tests/test_jsonio.py -v`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'trustsr.jsonio'`.

- [ ] **Step 3: Add the shared implementation and preserve the old import surface**

```python
# src/trustsr/jsonio.py
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not canonical JSON") from exc


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    if not isinstance(path, Path) or not isinstance(payload, bytes):
        raise TypeError("path must be Path and payload must be bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
```

In `predictions.py`, delete its local `canonical_json` function and import the same name from
`trustsr.jsonio`. Keep the symbol importable from `trustsr.artifacts` so existing callers do not change.

- [ ] **Step 4: Run focused and regression tests**

Run: `uv run pytest tests/test_jsonio.py tests/artifacts/test_predictions.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the focused refactor**

```bash
git add src/trustsr/jsonio.py src/trustsr/artifacts/predictions.py tests/test_jsonio.py tests/artifacts/test_predictions.py
git commit -m "refactor: share canonical json writes"
```

### Task 2: Strict Crosssensor Metadata Schema

**Files:**
- Create: `src/trustsr/data/crosssensor_schema.py`
- Create: `tests/data/test_crosssensor_schema.py`

**Interfaces:**
- Consumes: `normalize_top_level(records: Sequence[Mapping[str, object]]) -> tuple[CrosssensorSample, ...]`.
- Produces: immutable `CrosssensorSample` records with `source_index`, `sample_id`, centroid, CRS, transform, shape, timestamp, administrative labels, `days_between`, `correlation`, and `scale_factor`.

- [ ] **Step 1: Write strict normalization tests**

Create a `_row(sample_id="sample-1", longitude=-76.5, latitude=3.5)` helper containing all required
fields. Assert the normalized record equals:

```python
CrosssensorSample(
    source_index=0,
    sample_id="sample-1",
    longitude=-76.5,
    latitude=3.5,
    crs="EPSG:32618",
    geotransform=(10.0, 0.0, 500000.0, 0.0, -10.0, 400000.0),
    raster_shape=(130, 130),
    time_start="2020-01-02T10:00:00Z",
    admin0="Colombia",
    admin1=None,
    admin2="Cali",
    days_between=0,
    correlation=0.91,
    scale_factor=4,
)
```

Also parameterize failures for duplicate `tortilla:id`, non-POINT WKT, longitude outside `[-180,180]`,
latitude outside `[-90,90]`, non-finite correlation, `days_between` outside `{-1,0,1}`, scale other than 4,
shape other than `(130,130)`, missing required fields, and an input length other than 8,000 when
`expected_count=8_000`.

- [ ] **Step 2: Run the schema tests and confirm the missing module failure**

Run: `uv run pytest tests/data/test_crosssensor_schema.py -v`

Expected: FAIL during collection because `crosssensor_schema` does not exist.

- [ ] **Step 3: Implement immutable types and fail-closed validators**

Use this public surface:

```python
REQUIRED_COLUMNS = frozenset({
    "tortilla:id",
    "stac:crs",
    "stac:geotransform",
    "stac:raster_shape",
    "stac:time_start",
    "stac:centroid",
    "rai:admin0",
    "rai:admin1",
    "rai:admin2",
    "days_between",
    "correlation",
    "scale_factor",
})


@dataclass(frozen=True)
class CrosssensorSample:
    source_index: int
    sample_id: str
    longitude: float
    latitude: float
    crs: str
    geotransform: tuple[float, float, float, float, float, float]
    raster_shape: tuple[int, int]
    time_start: str
    admin0: str | None
    admin1: str | None
    admin2: str | None
    days_between: int
    correlation: float
    scale_factor: int


def normalize_top_level(
    records: Sequence[Mapping[str, object]], *, expected_count: int = 8_000
) -> tuple[CrosssensorSample, ...]:
    normalized = tuple(_normalize_row(index, row) for index, row in enumerate(records))
    if len(normalized) != expected_count:
        raise ValueError(f"expected {expected_count} crosssensor rows, observed {len(normalized)}")
    identifiers = [sample.sample_id for sample in normalized]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("tortilla:id values must be unique")
    return normalized
```

Parse `stac:centroid` only from a complete `POINT (longitude latitude)` WKT match. Convert NumPy scalars
with `.item()` before exact type checks, reject booleans as integers, reject extra coordinate values, and
require every float to pass `math.isfinite`.

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/data/test_crosssensor_schema.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the schema**

```bash
git add src/trustsr/data/crosssensor_schema.py tests/data/test_crosssensor_schema.py
git commit -m "feat: normalize crosssensor metadata"
```

### Task 3: Five-Kilometre Connected Components and Splits

**Files:**
- Create: `src/trustsr/data/spatial_split.py`
- Create: `tests/data/test_spatial_split.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: `Sequence[CrosssensorSample]`.
- Produces: `assign_spatial_splits(samples, threshold_km=5.0) -> tuple[AssignedSample, ...]` and `minimum_cross_split_distances(assignments) -> dict[str, float]`.

- [ ] **Step 1: Write spatial boundary and component tests**

Use synthetic points at the equator so `math.degrees(5 / EARTH_RADIUS_KM)` is exactly the angular 5 km
boundary. Assert that points at the inclusive boundary share a group, a point 5.01 km away does not, and
the chain A—B—C becomes one component even when A—C exceeds 5 km. Add dateline points at `179.99` and
`-179.99` degrees to prove the unit-sphere search does not split neighbors. Assert every group has exactly
one split, outputs are sorted by `sample_id`, repeated calls are equal, and all reported cross-split minima
are greater than 5 km.

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run: `uv run pytest tests/data/test_spatial_split.py -v`

Expected: FAIL during collection because `spatial_split` does not exist.

- [ ] **Step 3: Declare direct SciPy ownership and implement the algorithm**

Add `"scipy>=1.14,<2"` to project dependencies and run `uv lock`. Implement:

```python
EARTH_RADIUS_KM = 6_371.0088


@dataclass(frozen=True)
class AssignedSample:
    sample: CrosssensorSample
    spatial_group_id: str
    split: Literal["development", "calibration", "internal_test"]


def component_split(spatial_group_id: str) -> str:
    value = int(spatial_group_id[:16], 16) / 2**64
    if value < 0.50:
        return "development"
    if value < 0.75:
        return "calibration"
    return "internal_test"
```

Convert centroids to `(cos(lat)cos(lon), cos(lat)sin(lon), sin(lat))`, query `cKDTree.query_pairs` with
chord `2*sin((5/EARTH_RADIUS_KM)/2)`, confirm every candidate with the haversine formula, and union only
distances `<= threshold_km`. Hash the newline-joined sorted sample IDs in each component. For minimum
cross-split distances, query each split pair's unit-vector tree for nearest chords and convert the smallest
chord to arc distance.

- [ ] **Step 4: Run spatial tests and the dependency lock check**

Run: `uv sync --dev && uv run pytest tests/data/test_spatial_split.py -v && uv lock --check`

Expected: PASS and lock file unchanged by `uv lock --check`.

- [ ] **Step 5: Commit the spatial core**

```bash
git add pyproject.toml uv.lock src/trustsr/data/spatial_split.py tests/data/test_spatial_split.py
git commit -m "feat: add leak-safe spatial splits"
```

### Task 4: Deterministic Pilot Sampling and Manifest

**Files:**
- Create: `src/trustsr/data/pilot_sampling.py`
- Create: `src/trustsr/data/crosssensor_manifest.py`
- Create: `tests/data/test_pilot_sampling.py`
- Create: `tests/data/test_crosssensor_manifest.py`

**Interfaces:**
- Consumes: assigned samples and optional extracted LR/HR asset records.
- Produces: `select_pilot(assignments: Sequence[AssignedSample]) -> tuple[PilotChoice, ...]`, canonical manifest read/write functions, and a strict audit builder.

- [ ] **Step 1: Write failing sampling tests**

Build 36 strata per split with at least two candidate groups per stratum. Assert:

```python
choices = select_pilot(assignments)
assert len(choices) == 36
assert {choice.split for choice in choices} == {
    "development", "calibration", "internal_test"
}
assert len({(choice.split, choice.days_between, choice.correlation_bin) for choice in choices}) == 36
for split in ("development", "calibration", "internal_test"):
    selected = [choice for choice in choices if choice.split == split]
    assert len({choice.spatial_group_id for choice in selected}) == 12
```

Assert `0.8842208864`, `0.9041984739`, and `0.9265462586` enter bins 1, 2, and 3 respectively. Add a
fixture where the only candidate for a later stratum reuses a selected group and assert a fail-closed
`ValueError` instead of a duplicate group.

- [ ] **Step 2: Run sampling tests and confirm the missing module failure**

Run: `uv run pytest tests/data/test_pilot_sampling.py -v`

Expected: FAIL during collection because `pilot_sampling` does not exist.

- [ ] **Step 3: Implement the fixed-bin greedy selector**

```python
CORRELATION_CUTS = (0.8842208864, 0.9041984739, 0.9265462586)


@dataclass(frozen=True)
class PilotChoice:
    sample_id: str
    split: str
    days_between: int
    correlation_bin: int
    spatial_group_id: str
    selection_sha256: str


def correlation_bin(value: float) -> int:
    return bisect.bisect_right(CORRELATION_CUTS, value)
```

For each split, iterate days `(-1,0,1)` and bins `(0,1,2,3)`. Sort candidates by
`sha256(b"trustsr-pilot-v1\n" + sample_id.encode()).hexdigest()`, then choose the first candidate whose
group has not been used in that split. Return choices sorted by `(split, days_between, correlation_bin)`.

- [ ] **Step 4: Write failing canonical manifest and audit tests**

Assert two writes from the same assigned samples and choices are byte-identical JSONL, sorted by
`sample_id`, end with one newline per record, and return the same SHA-256. Assert the audit contains:

```python
{
    "schema": "trustsr.phase2b1a-audit.v1",
    "source_revision": "c370504201072fdb1dd388013ab8c0fc7d00a57e",
    "source_object_sha256": "c6f29d8e80dc5e856e2b4510c0e6830043d4b15c9228a9ca249a4f618e7475a5",
    "sample_count": 8000,
    "component_count": 6695,
    "pilot_pair_count": 36,
    "pilot_geotiff_count": 0,
    "real_pixels_local": False,
    "gpu_used": False,
}
```

The test payload may use smaller synthetic counts by passing an explicit `ExpectedCounts`; production
defaults must use `(8000, 6695, 3967, 2070, 1963, 3317, 1719, 1659)`.

- [ ] **Step 5: Implement canonical JSONL and strict audit builders**

Define and use these exact public records and signatures:

```python
@dataclass(frozen=True)
class ExpectedCounts:
    samples: int
    components: int
    development_samples: int
    calibration_samples: int
    internal_test_samples: int
    development_components: int
    calibration_components: int
    internal_test_components: int


@dataclass(frozen=True)
class ExtractedAsset:
    relative_path: str
    size_bytes: int
    sha256: str
    shape: tuple[int, int, int]
    dtype: str
    crs: str
    transform: tuple[float, float, float, float, float, float]
    nodata: float | int | None
    minimum: float
    maximum: float
    time_start: str


@dataclass(frozen=True)
class ManifestArtifact:
    path: Path
    size_bytes: int
    sha256: str


```

The exact function signatures are:

- `write_manifest(path: Path, assignments: Sequence[AssignedSample], choices: Sequence[PilotChoice], assets: Mapping[str, tuple[ExtractedAsset, ExtractedAsset]]) -> ManifestArtifact`;
- `load_manifest(path: Path, *, expected_sha256: str) -> tuple[dict[str, object], ...]`;
- `build_audit(records: Sequence[Mapping[str, object]], *, manifest_sha256: str, minimum_distances: Mapping[str, float], expected: ExpectedCounts) -> dict[str, object]`.

`ExtractedAsset` stores relative POSIX path,
byte size, SHA-256, `(bands,height,width)`, dtype, CRS, six-value transform, nodata, minimum, maximum, and
acquisition time. Non-extracted records serialize `lr_asset` and `hr_asset` as JSON `null`. Write each record
with `canonical_json(record) + b"\n"`, hash the complete byte stream, and commit via `atomic_write_bytes`.
Reject duplicate sample IDs, absolute/parent paths, unknown schemas, non-finite numbers, mismatched expected
counts, shared groups, and any minimum cross-split distance `<= 5`.

- [ ] **Step 6: Run the sampling and manifest tests**

Run: `uv run pytest tests/data/test_pilot_sampling.py tests/data/test_crosssensor_manifest.py -v`

Expected: PASS.

- [ ] **Step 7: Commit the selection and manifest layer**

```bash
git add src/trustsr/data/pilot_sampling.py src/trustsr/data/crosssensor_manifest.py tests/data/test_pilot_sampling.py tests/data/test_crosssensor_manifest.py
git commit -m "feat: build crosssensor pilot manifest"
```

### Task 5: Resumable Verified Cloud Acquisition

**Files:**
- Create: `src/trustsr/data/crosssensor_source.py`
- Create: `tests/data/test_crosssensor_source.py`

**Interfaces:**
- Consumes: validated Phase 2B0 `DatasetSource`, explicit storage root, explicit transport URL, and confirmation flag.
- Produces: `acquire_crosssensor(source: DatasetSource, storage_root: Path, transport_url: str, *, confirmed_cloud_storage: bool, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> VerifiedSourceObject` and `verify_crosssensor(path: Path, object_spec: LfsObject) -> VerifiedSourceObject`.

- [ ] **Step 1: Write offline downloader tests with a fake runner**

Use a fake runner that records its argument list and writes a small payload to the `--output` path. Test
that the command contains `curl --fail --location --retry 5 --continue-at -`, that the URL and paths are
separate arguments, and that success atomically moves `.part` to the digest-qualified source directory.
Test these failures before any runner call: non-HTTPS URL, URL user info, missing confirmation, less than
15 GiB free space, missing crosssensor inventory entry, and an existing invalid final object. Test that an
interrupted runner leaves `.part` for resume and that a completed size/hash mismatch moves the part into
`quarantine/` without creating a final object.

- [ ] **Step 2: Run acquisition tests and confirm the missing module failure**

Run: `uv run pytest tests/data/test_crosssensor_source.py -v`

Expected: FAIL during collection because `crosssensor_source` does not exist.

- [ ] **Step 3: Implement source lookup, transport, verification, and quarantine**

Use this public signature:

```python
@dataclass(frozen=True)
class VerifiedSourceObject:
    path: Path
    size_bytes: int
    sha256: str


def acquire_crosssensor(
    source: DatasetSource,
    storage_root: Path,
    transport_url: str,
    *,
    confirmed_cloud_storage: bool,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> VerifiedSourceObject:
    object_spec = require_crosssensor_object(source)
    require_cloud_confirmation(storage_root, confirmed_cloud_storage)
    require_free_space(storage_root, minimum_bytes=15 * 1024**3)
    paths = source_paths(storage_root, object_spec)
    if paths.final.exists():
        return verify_crosssensor(paths.final, object_spec)
    runner(curl_arguments(transport_url, paths.partial), check=True, text=True)
    try:
        verified = verify_crosssensor(paths.partial, object_spec)
    except SourceIntegrityError:
        quarantine_completed_partial(paths.partial, paths.quarantine)
        raise
    os.replace(paths.partial, paths.final)
    return VerifiedSourceObject(paths.final, verified.size_bytes, verified.sha256)
```

Resolve and confine every derived path below `storage_root`, reject a storage root that is `/`, the home
directory, a symlink, or not an existing directory, and never delete an invalid real object.

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/data/test_crosssensor_source.py -v`

Expected: PASS without network access.

- [ ] **Step 5: Commit cloud acquisition**

```bash
git add src/trustsr/data/crosssensor_source.py tests/data/test_crosssensor_source.py
git commit -m "feat: acquire verified crosssensor source"
```

### Task 6: Isolated TACO v1 Adapter and GeoTIFF Extraction

**Files:**
- Create: `src/trustsr/data/taco_v1_adapter.py`
- Create: `tests/data/test_taco_v1_adapter.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: verified local-cloud TACO path, selected `source_index`, output root, and source bands.
- Produces: `load_top_level_records(taco_path: Path) -> tuple[Mapping[str, object], ...]` and `extract_pair(taco_path: Path, source_index: int, output_root: Path, bands: tuple[str, ...]) -> tuple[ExtractedAsset, ExtractedAsset]`.

- [ ] **Step 1: Write the optional-import and version-gate tests**

Inject a fake module and fake `importlib.metadata.version`. Assert `require_tacoreader_v1()` accepts only
`0.4.5`, rejects missing installation with a cloud-bootstrap instruction, and rejects `2.4.21` even if that
module exposes a v1 namespace. Assert importing `trustsr.data.taco_v1_adapter` itself does not import
`tacoreader`. Add a collection test that accepts exactly one legacy version field resolving to `0.4.0` and
rejects a missing, ambiguous, or different TACO format version.

- [ ] **Step 2: Write extraction tests using generated GeoTIFF bytes**

Generate four-band `uint16` LR and HR GeoTIFF bytes with Rasterio `MemoryFile`: LR `130×130` at 10 m,
HR `520×520` at 2.5 m, identical bounds and CRS, values 100 and 120. A fake top table's `read(index)`
returns a fake two-row nested table whose `read(asset_index)` returns these bytes. Assert extraction writes
`lr.tif` and `hr.tif` atomically, preserves the exact bytes, records hashes and acquisition times, and calls
the requested source index. Parameterize rejection of three bands, wrong shapes, mismatched CRS, mismatched
bounds, float NaN pixels, duplicate dimensions, and three nested assets.

- [ ] **Step 3: Run adapter tests and confirm the missing module failure**

Run: `uv run pytest tests/data/test_taco_v1_adapter.py -v`

Expected: FAIL during collection because `taco_v1_adapter` does not exist.

- [ ] **Step 4: Declare Rasterio ownership and implement the isolated adapter**

Add `"rasterio>=1.4,<2"` to project dependencies and run `uv lock`. The module must import `tacoreader`
inside `require_tacoreader_v1`, verify `importlib.metadata.version("tacoreader") == "0.4.5"`, and always
call `tacoreader.load(str(taco_path))` because the old `load` implementation calls `.startswith()` before
checking for `Path`.

For a selected sample, call `top.read(source_index)`, require exactly two nested assets, call `nested.read`
for each, inspect bytes using `rasterio.io.MemoryFile`, and identify LR/HR by exact dimensions rather than
row order. Verify four bands, source-declared `B04/B03/B02/B08`, CRS equality, exact expected resolutions,
and equal bounds within absolute tolerance `1e-6`. Use `atomic_write_bytes` for raw bytes and return
`ExtractedAsset` values. Call `tacoreader.load_metadata(str(taco_path))` before the table read and accept
`0.4.0` only when exactly one of the legacy collection keys `taco_version` or `version` supplies that value;
two present keys are accepted only when their values are identical.

- [ ] **Step 5: Run adapter and lock tests**

Run: `uv sync --dev && uv run pytest tests/data/test_taco_v1_adapter.py -v && uv lock --check`

Expected: PASS, with `tacoreader` still absent from `uv.lock`.

- [ ] **Step 6: Commit the adapter**

```bash
git add pyproject.toml uv.lock src/trustsr/data/taco_v1_adapter.py tests/data/test_taco_v1_adapter.py
git commit -m "feat: isolate taco v1 pilot extraction"
```

### Task 7: Restartable Phase 2B1A CLI

**Files:**
- Create: `src/trustsr/cli/phase2b1a.py`
- Create: `tests/cli/test_phase2b1a.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: explicit `--source`, `--storage-root`, and stage-specific inputs.
- Produces: console script `trustsr-phase2b1a` with `download`, `manifest`, `pilot`, and `audit` subcommands.

- [ ] **Step 1: Write parser and stage-isolation tests**

Assert the parser has no transport URL or storage-root default. Assert:

```text
trustsr-phase2b1a download --source SOURCE --storage-root ROOT --transport-url URL --confirm-cloud-storage
trustsr-phase2b1a manifest --source SOURCE --storage-root ROOT --confirm-cloud-storage
trustsr-phase2b1a pilot --source SOURCE --storage-root ROOT --manifest MANIFEST --confirm-cloud-storage
trustsr-phase2b1a audit --source SOURCE --storage-root ROOT --manifest MANIFEST --confirm-cloud-storage
```

Each test monkeypatches the called service and asserts other stages are untouched. Add fail-closed tests for
missing confirmation, source object mismatch, manifest digest mismatch, wrong production counts, pilot not
equal to 36, and audit before all 72 files exist. Assert emitted stdout is one canonical JSON line without
absolute paths, hostnames, timestamps, or credentials.

- [ ] **Step 2: Run CLI tests and confirm the missing module failure**

Run: `uv run pytest tests/cli/test_phase2b1a.py -v`

Expected: FAIL during collection because `phase2b1a` does not exist.

- [ ] **Step 3: Implement four idempotent service stages**

`download` invokes `acquire_crosssensor`. `manifest` verifies the source object, reads and normalizes exactly
8,000 rows and exactly 26 top-level columns, assigns spatial groups/splits, checks production counts and minimum distances, selects 36 pilot
choices, then writes a digest-addressed JSONL manifest. `pilot` reloads and verifies that manifest, extracts
only its 36 selected rows, and writes a new manifest containing 72 asset records. `audit` rehashes every
selected GeoTIFF, validates every record, and writes a canonical audit under `audits/<manifest-sha256>/`.
Every stage reuses a valid result and rejects an existing invalid result; no stage silently repairs it.

Add this entry point:

```toml
trustsr-phase2b1a = "trustsr.cli.phase2b1a:main"
```

- [ ] **Step 4: Run CLI, policy, and lock tests**

Run: `uv sync --dev && uv run pytest tests/cli/test_phase2b1a.py tests/data/test_local_data_policy.py -v && uv lock --check`

Expected: PASS, and the local policy still rejects real TACO/large dataset artifacts.

- [ ] **Step 5: Commit the CLI**

```bash
git add pyproject.toml uv.lock src/trustsr/cli/phase2b1a.py tests/cli/test_phase2b1a.py
git commit -m "feat: add phase2b1a data stages"
```

### Task 8: Cloud Base Bootstrap and Safe Runner

**Files:**
- Create: `requirements/cloud-taco-v1.txt`
- Create: `scripts/phase2b1a/bootstrap_reader.sh`
- Create: `scripts/phase2b1a/run_cloud.sh`
- Create: `tests/scripts/test_phase2b1a_scripts.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `/opt/conda/bin/python`, a checked-out repository, an explicit persistent storage root, and a stage.
- Produces: a validated base reader install and a safe shell entry point without any SSH or GPU assumptions.

- [ ] **Step 1: Write shell contract tests**

Follow the existing fake-command pattern in `tests/scripts/test_phase1b_scripts.py`. Assert both scripts use
`set -euo pipefail`, reject `/`, `/root`, home, symlinks, wildcards, newlines, relative roots, missing mounts,
and less than 15 GiB. Assert neither script contains `conda create`, a GPU model, an SSH endpoint, a password,
`StrictHostKeyChecking=no`, shutdown commands, recursive deletion, or a default transport URL.

For bootstrap, record pip calls and assert: Python is `/opt/conda/bin/python`; dry-run happens before install;
the dry-run report rejects torch, torchvision, triton, and `nvidia-*`; installation uses the requirements
file with `--upgrade-strategy only-if-needed`; and post-install metadata matches every exact version. For the
runner, assert the stage and arguments reach `python -m trustsr.cli.phase2b1a` as distinct argv elements.

- [ ] **Step 2: Run script tests and confirm missing scripts fail**

Run: `uv run pytest tests/scripts/test_phase2b1a_scripts.py -v`

Expected: FAIL because the Phase 2B1A scripts do not exist.

- [ ] **Step 3: Add exact cloud-only requirements**

```text
tacoreader==0.4.5
geopandas==1.1.4
pyarrow==25.0.1
shapely==2.1.2
pyproj==3.7.2
pyogrio==0.13.0
```

The bootstrap script performs `pip install --dry-run --report`, parses normalized package names, aborts on
protected CUDA packages, installs into the base interpreter, runs `pip check`, verifies all six versions,
and confirms `tacoreader.load` plus `tacoreader.load_metadata` are callable. It creates no environment.

- [ ] **Step 4: Add the validated runner and operator documentation**

`run_cloud.sh` takes exactly `STORAGE_ROOT REPO_DIR STAGE` followed by stage arguments, canonicalizes and
confines the storage root, requires the user confirmation flag in forwarded arguments, and executes the base
Python module. Document the four stage commands in README with shell variables
`PHASE2B1A_STORAGE_ROOT` and `PHASE2B1A_TRANSPORT_URL`; do not put a real host or credential in the file.

- [ ] **Step 5: Run script tests**

Run: `uv run pytest tests/scripts/test_phase2b1a_scripts.py -v`

Expected: PASS.

- [ ] **Step 6: Commit cloud operations**

```bash
git add requirements/cloud-taco-v1.txt scripts/phase2b1a tests/scripts/test_phase2b1a_scripts.py README.md
git commit -m "ops: add phase2b1a cloud runner"
```

### Task 9: Local End-to-End Gate and Cloud Handoff Checkpoint

**Files:**
- Modify: `tests/cli/test_phase2b1a.py`
- Modify after cloud success: `artifacts/datasets/sen2naipv2-phase2b1a-audit-v1.json`

**Interfaces:**
- Consumes: the complete local implementation and then, only after user starts a cloud instance, the frozen real object.
- Produces: a clean local verification commit first; later a separate evidence-only cloud commit.

- [ ] **Step 1: Add a synthetic stage-sequence integration test**

Run all four service stages against a temporary storage root, a tiny fake source object, 36 synthetic strata,
a fake TACO reader, and generated GeoTIFFs. Assert restart does not call the downloader or extractor again,
tampering causes the next stage to fail, the final audit is byte-identical across runs, and no output escapes
the temporary storage root.

- [ ] **Step 2: Run the complete local verification gate**

Run:

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv lock --check
git diff --check
git ls-files -z | xargs -0 -r du -b | awk '$1 > 1048576 {print; failed=1} END {exit failed}'
```

Expected: all tests pass, Ruff passes, the lock is current, no whitespace errors exist, and no tracked file
exceeds 1 MiB.

- [ ] **Step 3: Commit the local integration gate**

```bash
git add tests/cli/test_phase2b1a.py
git commit -m "test: verify phase2b1a stage sequence"
```

- [ ] **Step 4: Stop and request the cloud connection**

Report that local development is complete and explicitly tell the user the next gate needs a running cloud
instance with `/root/rivermind-fs` mounted. State that GPU compute is not needed; the rental instance is used
only for network, CPU, and persistent storage. Do not attempt a stale SSH endpoint.

- [ ] **Step 5: Execute the real gate only after receiving the current SSH target**

Use the user-provided SSH target, verify its host key through the configured SSH trust, verify the persistent
mount and at least 15 GiB free space, synchronize the committed branch, run bootstrap, then execute
`download`, `manifest`, `pilot`, and `audit` in order. Do not copy the `.taco`, full JSONL manifest, or any
GeoTIFF back to WSL.

- [ ] **Step 6: Verify and commit only the small real audit summary**

Before copying, verify the cloud audit states 8,000 samples, 6,695 components, split sample/component counts
`3967/3317`, `2070/1719`, `1963/1659`, 36 pairs, 72 GeoTIFFs, correct source size/SHA, no shared group, and
all cross-split minima above 5 km. Copy only the canonical audit JSON, reject it if over 1 MiB or if it
contains absolute paths, hosts, timestamps, or credentials, rerun the complete local gate, then commit:

```bash
git add artifacts/datasets/sen2naipv2-phase2b1a-audit-v1.json
git commit -m "data: record phase2b1a pilot audit"
```

- [ ] **Step 7: Push the branch and update the Phase 2B pull request**

Run `git push -u origin feature/phase2b1-crosssensor-manifest`, confirm the remote commit equals local HEAD,
and create or update a pull request whose base is `feature/phase2b0-data-provenance`. Include the exact test
count and cloud audit digest in the PR body, but no cloud path or access detail.
