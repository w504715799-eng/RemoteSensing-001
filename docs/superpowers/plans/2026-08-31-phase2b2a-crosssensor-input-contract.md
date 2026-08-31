# Phase 2B2-A Crosssensor Input Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a strict CPU-only loader and reproducibility audit that turns 12 deterministic Phase 2B1B GeoTIFF pairs into spatially aligned RGBN reflectance tensors without copying real pixels into Git or local WSL storage.

**Architecture:** Keep deterministic smoke selection, digest-addressed sidecar loading, raster validation/cropping, and audit construction in focused data modules. Expose one restartable `audit-inputs` CLI and a cloud wrapper; the CLI loads the same 12 pairs twice and commits only a canonical host-free audit. Model inference remains outside this phase.

**Tech Stack:** Python 3.12, dataclasses, hashlib, pathlib, NumPy, Rasterio, PyTorch, pytest, Ruff, Bash, existing canonical JSON and Phase 2B1B schema utilities.

**Spec:** `docs/superpowers/specs/2026-08-31-phase2b2a-crosssensor-input-contract.md`

## Global Constraints

- Accept only Phase 2B1B post-manifest SHA-256 `c7f8ffa8415575d85daafe284a0796ec3f111442f0ac662f1d01311c4a851d4a`.
- Preserve the frozen Phase 2B1B audit SHA-256 `d8964033958594a23ac7056519894d508977bfd2cc13da50a5833024274f3e90` and base-manifest SHA-256 `7487b0af2ebef86910e918d5d6b2fb927a6f5e46bac7c2e30be7ffb2ce994482` in the audit.
- Read only the Phase 2B1B `subset-v1` pixel tree; never overwrite, relocate, or repair upstream files.
- Require raw four-band `uint16`, `nodata=None`, finite values in `[0,10000]`, and exact sidecar metadata.
- Normalize only by `torch.float32(raw) / 10000.0`; never clamp an invalid input.
- Crop LR with `[:,1:129,1:129]` and HR with `[:,4:516,4:516]`; never resize or use random crops.
- Verify cropped LR/HR geographic bounds within `1e-3 m` absolute tolerance.
- The smoke set is exactly 12 pairs: round 1, `days_between=-1`, one sample per split/correlation-bin key.
- Do not run or download bicubic, SEN2SRLite, LDSR-S2, model weights, prediction caches, metrics, or conformal calibration.
- Do not create a Conda environment or modify PyTorch, torchvision, triton, CUDA, or any `nvidia-*` package.
- Do not hard-code SSH endpoints, passwords, cloud vendor names, GPU models, CUDA versions, or storage mountpoints.
- Real TIFFs, full sidecars, model assets, and generated tensors remain outside Git and the local WSL workspace.
- Every production-code task uses red-green-refactor TDD and ends in a focused commit.

## File Structure

- `src/trustsr/data/crosssensor_pairs.py`: frozen identities, deterministic 12-record selection, sidecar path verification, strict GeoTIFF loading, crop and normalization.
- `src/trustsr/data/input_audit.py`: immutable digest records and canonical Phase 2B2-A audit construction.
- `src/trustsr/cli/phase2b2a.py`: fail-closed `audit-inputs` command and digest-scoped audit commit.
- `scripts/phase2b2a/run_cloud.sh`: base-Python, mount-checked CPU runner.
- `tests/data/test_crosssensor_pairs.py`: selection, path, raster, crop, reflectance and failure tests.
- `tests/data/test_input_audit.py`: exact audit schema, counts, repeatability and host-free payload tests.
- `tests/cli/test_phase2b2a.py`: parser, preflight, double-load, atomic commit and reuse tests.
- `tests/scripts/test_phase2b2a_scripts.py`: executable shell contract using harmless command fakes.
- `artifacts/datasets/sen2naipv2-phase2b2a-input-audit-v1.json`: created only after the real cloud gate passes.

---

### Task 1: Deterministic 12-Record Smoke Selection

**Files:**
- Create: `src/trustsr/data/crosssensor_pairs.py`
- Create: `tests/data/test_crosssensor_pairs.py`

**Interfaces:**
- Consumes: a sequence of already schema-validated Phase 2B1B record mappings.
- Produces: `select_input_smoke_records(records: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]` and frozen constants used by later tasks.

- [ ] **Step 1: Write the failing selection tests**

Create literal mappings for all 12 required keys and additional non-eligible records. The key behavioral tests are:

```python
from copy import deepcopy

import pytest

from trustsr.data.crosssensor_pairs import select_input_smoke_records


SPLITS = ("development", "calibration", "internal_test")


def _eligible_records() -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"sample-{split}-{bin_index}",
            "split": split,
            "spatial_group_id": f"group-{split}-{bin_index}",
            "days_between": -1,
            "correlation_bin": bin_index,
            "selection_round": 1,
        }
        for split in SPLITS
        for bin_index in range(4)
    ]


def test_smoke_selection_has_four_bins_per_split_in_canonical_order() -> None:
    records = _eligible_records()
    records.extend(
        {
            **deepcopy(records[0]),
            "sample_id": "not-eligible",
            "spatial_group_id": "not-eligible-group",
            "selection_round": 2,
        }
        for _ in range(3)
    )

    selected = select_input_smoke_records(tuple(reversed(records)))

    assert [(record["split"], record["correlation_bin"]) for record in selected] == [
        (split, bin_index) for split in sorted(SPLITS) for bin_index in range(4)
    ]
    assert len({record["sample_id"] for record in selected}) == 12
    assert len({record["spatial_group_id"] for record in selected}) == 12


def test_smoke_selection_rejects_a_missing_or_duplicate_required_cell() -> None:
    records = _eligible_records()
    with pytest.raises(ValueError, match="exactly one record"):
        select_input_smoke_records(records[:-1])
    duplicate = records + [{**records[0], "sample_id": "duplicate"}]
    with pytest.raises(ValueError, match="exactly one record"):
        select_input_smoke_records(duplicate)
```

The production mutation these tests catch is selecting a fallback row, changing the smoke filters, or relying on input order.

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run: `uv run pytest tests/data/test_crosssensor_pairs.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'trustsr.data.crosssensor_pairs'`.

- [ ] **Step 3: Implement the minimal deterministic selector and constants**

Create these exact constants:

```python
POST_MANIFEST_SHA256 = "c7f8ffa8415575d85daafe284a0796ec3f111442f0ac662f1d01311c4a851d4a"
PHASE2B1B_AUDIT_SHA256 = "d8964033958594a23ac7056519894d508977bfd2cc13da50a5833024274f3e90"
REFLECTANCE_SCALE = 10_000.0
RAW_DTYPE = "uint16"
CROP_POLICY = "center_crop_lr_1_hr_4_v1"
NORMALIZATION_POLICY = "uint16_divide_10000_no_clip_v1"
SMOKE_SPLITS = ("calibration", "development", "internal_test")
SMOKE_BINS = (0, 1, 2, 3)
```

Implement selection by building a dictionary keyed by `(split, correlation_bin)`. Include a record only when `selection_round == 1` and `days_between == -1`. Require exactly one record for every Cartesian key, 12 unique non-empty string sample IDs, and 12 unique non-empty string spatial group IDs. Return records in `SMOKE_SPLITS` then `SMOKE_BINS` order.

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/data/test_crosssensor_pairs.py -v`

Expected: all selection tests pass.

- [ ] **Step 5: Commit the selector**

```bash
git add src/trustsr/data/crosssensor_pairs.py tests/data/test_crosssensor_pairs.py
git commit -m "feat: select phase2b2a input smoke records"
```

---

### Task 2: Frozen Sidecar Loading Contract

**Files:**
- Modify: `src/trustsr/data/crosssensor_pairs.py`
- Modify: `tests/data/test_crosssensor_pairs.py`

**Interfaces:**
- Consumes: explicit storage root, manifest path and expected SHA-256.
- Produces: `load_crosssensor_records(storage_root: Path, manifest_path: Path, *, expected_sha256: str) -> tuple[dict[str, object], ...]`.

- [ ] **Step 1: Add failing digest-path and post-manifest tests**

Generate a valid 360-row synthetic Phase 2B1B sidecar using `write_subset_manifest()` and its existing test fixture pattern, then place its bytes under an explicitly constructed digest directory. Because the production contract accepts only the real frozen digest, monkeypatch the module constant for this isolated path test:

```python
def test_load_records_requires_digest_addressed_all_assets_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest, digest = _synthetic_post_manifest(tmp_path)
    monkeypatch.setattr(crosssensor_pairs, "POST_MANIFEST_SHA256", digest)

    records = load_crosssensor_records(
        tmp_path,
        manifest,
        expected_sha256=digest,
    )

    assert len(records) == 360
    assert all(record["lr_asset"] is not None for record in records)
    assert all(record["hr_asset"] is not None for record in records)


def test_load_records_rejects_wrong_digest_or_layout_before_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest, digest = _synthetic_post_manifest(tmp_path)
    monkeypatch.setattr(crosssensor_pairs, "POST_MANIFEST_SHA256", digest)
    misplaced = tmp_path / "misplaced.jsonl"
    misplaced.write_bytes(manifest.read_bytes())

    with pytest.raises(ValueError, match="frozen post-manifest"):
        load_crosssensor_records(tmp_path, misplaced, expected_sha256=digest)
    with pytest.raises(ValueError, match="frozen post-manifest SHA-256"):
        load_crosssensor_records(tmp_path, manifest, expected_sha256="0" * 64)
```

Also cover symlink manifest, symlink storage root, a manifest with any null asset, and a resolved path escaping the storage root.

- [ ] **Step 2: Run the new tests and verify the missing-function failure**

Run: `uv run pytest tests/data/test_crosssensor_pairs.py -k load_records -v`

Expected: failures report that `load_crosssensor_records` is unavailable.

- [ ] **Step 3: Implement strict sidecar loading**

The function must:

1. require `Path` arguments and exact `expected_sha256 == POST_MANIFEST_SHA256`;
2. require an existing, non-symlink storage directory and manifest regular file;
3. resolve both paths and require the storage root itself not to change through symlink resolution;
4. require the manifest path to equal:

```python
storage_root / "trustsr" / "phase2b1b" / "selections" / expected_sha256 / "samples.jsonl"
```

5. call `load_subset_manifest(manifest_path, expected_sha256=expected_sha256)`;
6. require every LR and HR asset field to be present;
7. return the validated 360-record tuple without copying or weakening schema values.

- [ ] **Step 4: Run focused and Phase 2B1B regression tests**

Run: `uv run pytest tests/data/test_crosssensor_pairs.py tests/data/test_subset_manifest.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit the sidecar loader**

```bash
git add src/trustsr/data/crosssensor_pairs.py tests/data/test_crosssensor_pairs.py
git commit -m "feat: load frozen phase2b1b post manifest"
```

---

### Task 3: Strict Raster Crop and Reflectance Loader

**Files:**
- Modify: `src/trustsr/data/crosssensor_pairs.py`
- Modify: `tests/data/test_crosssensor_pairs.py`

**Interfaces:**
- Consumes: one validated all-assets record and the Phase 2B1B pixel tree.
- Produces: immutable `CrosssensorPairMetadata`, `LoadedCrosssensorPair`, and `load_crosssensor_pair(storage_root: Path, record: Mapping[str, object], *, manifest_sha256: str) -> LoadedCrosssensorPair`.

- [ ] **Step 1: Add failing real-GeoTIFF crop and normalization test**

Create real temporary LR and HR GeoTIFFs. Use four `uint16` bands, `EPSG:32618`, LR transform `Affine(10,0,500000,0,-10,400000)`, HR transform `Affine(2.5,0,500000,0,-2.5,400000)`, and no nodata. Fill arrays with `5000`, then place hand-checked values inside and outside the crop:

```python
def test_load_pair_center_crops_aligns_and_normalizes_without_clipping(tmp_path: Path) -> None:
    record = _real_pair_fixture(tmp_path)
    loaded = load_crosssensor_pair(
        tmp_path,
        record,
        manifest_sha256=POST_MANIFEST_SHA256,
    )

    assert loaded.pair.lr.shape == (4, 128, 128)
    assert loaded.pair.hr.shape == (4, 512, 512)
    assert loaded.pair.lr.dtype == torch.float32
    assert loaded.pair.hr.dtype == torch.float32
    assert loaded.pair.lr.device.type == "cpu"
    assert loaded.pair.lr.is_contiguous()
    assert loaded.pair.lr[0, 0, 0].item() == pytest.approx(0.1234)
    assert loaded.pair.hr[0, 0, 0].item() == pytest.approx(0.1234)
    assert loaded.pair.lr.max().item() <= 1.0
    assert loaded.pair.hr.max().item() <= 1.0
    loaded.pair.validate()
    assert loaded.metadata.crop_policy == CROP_POLICY
    assert loaded.metadata.normalization_policy == NORMALIZATION_POLICY
```

The fixture sets raw LR `[0,0,0]=9999` and LR `[0,1,1]=1234`, while HR `[0,0,0]=9999` and HR `[0,4,4]=1234`. The assertion proves border values are removed and the aligned interior value becomes `0.1234`.

- [ ] **Step 2: Add failing integrity and scientific-policy tests**

Parameterize independent mutations and expected messages:

```python
@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("hash", "asset bytes"),
        ("symlink", "regular GeoTIFF"),
        ("dtype", "uint16"),
        ("nodata", "nodata"),
        ("maximum", r"\[0, 10000\]"),
        ("transform", "metadata"),
        ("bounds", "cropped LR/HR bounds"),
    ],
)
def test_load_pair_rejects_integrity_or_reflectance_contract_violation(
    tmp_path: Path, damage: str, message: str
) -> None:
    record = _damaged_pair_fixture(tmp_path, damage)
    with pytest.raises(ValueError, match=message):
        load_crosssensor_pair(
            tmp_path,
            record,
            manifest_sha256=POST_MANIFEST_SHA256,
        )
```

Also test wrong manifest digest, path traversal in `relative_path`, band count other than four, non-matching band descriptions when present, and sidecar min/max that disagrees with pixels.

- [ ] **Step 3: Run the raster tests and verify the expected failures**

Run: `uv run pytest tests/data/test_crosssensor_pairs.py -k 'load_pair' -v`

Expected: tests fail because the dataclasses and loader are absent.

- [ ] **Step 4: Implement immutable metadata and confined raster loading**

Use these exact public records:

```python
@dataclass(frozen=True)
class CrosssensorPairMetadata:
    manifest_sha256: str
    sample_id: str
    split: str
    spatial_group_id: str
    days_between: int
    correlation_bin: int
    selection_round: int
    lr_asset_sha256: str
    hr_asset_sha256: str
    lr_crop_transform: tuple[float, float, float, float, float, float]
    hr_crop_transform: tuple[float, float, float, float, float, float]
    crop_bounds: tuple[float, float, float, float]
    crop_policy: str
    normalization_policy: str


@dataclass(frozen=True)
class LoadedCrosssensorPair:
    pair: SRPair
    metadata: CrosssensorPairMetadata
```

For each asset:

1. require the exact canonical sidecar path `subset-v1/<split>/<sample_id>/<kind>.tif`;
2. resolve under `<storage-root>/trustsr/phase2b1b` and reject a symlink or escape;
3. stream SHA-256 and compare both size and digest before Rasterio opens it;
4. require GTiff, four bands, uniform `uint16`, optional descriptions equal `B04/B03/B02/B08`, declared CRS, `nodata=None`, and finite integer pixels;
5. compare observed shape, dtype, CRS, six transform values, nodata, minimum and maximum with the sidecar;
6. reject any raw value outside `[0,10000]`.

Use `rasterio.windows.Window(1,1,128,128)` for LR and `Window(4,4,512,512)` for HR. Compute crop transforms with `rasterio.windows.transform()` and bounds with `rasterio.windows.bounds()`. Compare the four LR/HR bounds with `math.isclose(..., rel_tol=0.0, abs_tol=1e-3)`.

Convert the cropped arrays by copied NumPy storage, `torch.float32`, and division by `REFLECTANCE_SCALE`. Do not call `clamp`. Construct `SRPair` with source `f"sen2naipv2-crosssensor/{manifest_sha256}"`, call `validate()`, and return the frozen wrapper.

- [ ] **Step 5: Run focused data tests and model-contract regressions**

Run:

```bash
uv run pytest tests/data/test_crosssensor_pairs.py tests/test_contracts.py \
  tests/models/test_sen2srlite.py tests/models/test_ldsr_s2.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit the raster loader**

```bash
git add src/trustsr/data/crosssensor_pairs.py tests/data/test_crosssensor_pairs.py
git commit -m "feat: load aligned crosssensor reflectance pairs"
```

---

### Task 4: Canonical Phase 2B2-A Audit Builder

**Files:**
- Create: `src/trustsr/data/input_audit.py`
- Create: `tests/data/test_input_audit.py`

**Interfaces:**
- Consumes: two independently loaded 12-pair sequences.
- Produces: `build_input_audit(first: Sequence[LoadedCrosssensorPair], second: Sequence[LoadedCrosssensorPair]) -> dict[str, object]`.

- [ ] **Step 1: Write the failing exact audit test**

Construct 12 `LoadedCrosssensorPair` values with literal tensors and metadata. Assert stable source identity, counts and per-pair hashes using the existing `tensor_sha256()` helper:

```python
def test_input_audit_records_exact_counts_and_repeatable_tensor_digests() -> None:
    first = _loaded_pairs()
    second = _loaded_pairs()

    audit = build_input_audit(first, second)

    assert audit["schema"] == "trustsr.phase2b2a-input-audit.v1"
    assert audit["post_manifest_sha256"] == POST_MANIFEST_SHA256
    assert audit["phase2b1b_audit_sha256"] == PHASE2B1B_AUDIT_SHA256
    assert audit["smoke_pair_count"] == 12
    assert audit["smoke_geotiff_count"] == 24
    assert audit["split_counts"] == {
        "calibration": 4,
        "development": 4,
        "internal_test": 4,
    }
    assert audit["correlation_bin_counts"] == {"0": 3, "1": 3, "2": 3, "3": 3}
    assert len(audit["pairs"]) == 12
    assert audit["repeated_load_equal"] is True
    assert audit["model_inference_run"] is False
    assert audit["gpu_used"] is False
    assert audit["real_pixels_local"] is False
```

Add tests that reject reordered pairs, one changed tensor, duplicate sample IDs, wrong split/bin counts, wrong crop/normalization policy, or a tensor that fails `SRPair.validate()`.

- [ ] **Step 2: Run tests and verify the missing-module failure**

Run: `uv run pytest tests/data/test_input_audit.py -v`

Expected: collection fails because `trustsr.data.input_audit` does not exist.

- [ ] **Step 3: Implement host-free canonical audit construction**

For each pair produce only:

```python
{
    "sample_id": metadata.sample_id,
    "split": metadata.split,
    "correlation_bin": metadata.correlation_bin,
    "lr_asset_sha256": metadata.lr_asset_sha256,
    "hr_asset_sha256": metadata.hr_asset_sha256,
    "lr_crop_transform": list(metadata.lr_crop_transform),
    "hr_crop_transform": list(metadata.hr_crop_transform),
    "crop_bounds": list(metadata.crop_bounds),
    "lr_tensor_sha256": tensor_sha256(pair.lr),
    "hr_tensor_sha256": tensor_sha256(pair.hr),
}
```

Require both sequences to have identical ordered metadata and tensor digest records. Return exactly the spec fields, including `BASE_MANIFEST_SHA256`, `SOURCE_OBJECT_SHA256`, raw shapes `[4,130,130]` and `[4,520,520]`, cropped shapes `[4,128,128]` and `[4,512,512]`, crop policy, raw dtype, reflectance scale, normalization policy and the three false execution flags. Call `canonical_json(result)` before return to reject non-JSON or non-finite values.

- [ ] **Step 4: Run audit and tensor-cache regression tests**

Run: `uv run pytest tests/data/test_input_audit.py tests/artifacts/test_predictions.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit the audit builder**

```bash
git add src/trustsr/data/input_audit.py tests/data/test_input_audit.py
git commit -m "feat: build phase2b2a input audit"
```

---

### Task 5: Restartable CPU Audit CLI

**Files:**
- Create: `src/trustsr/cli/phase2b2a.py`
- Create: `tests/cli/test_phase2b2a.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: explicit mounted storage root and frozen Phase 2B1B post-manifest.
- Produces: `run_audit_inputs(storage_root: Path, selection_manifest_path: Path, selection_manifest_sha256: str, *, confirmed_cloud_storage: bool) -> dict[str, object]`, canonical audit bytes and console command `trustsr-phase2b2a`.

- [ ] **Step 1: Write failing parser and fail-before-write tests**

```python
def test_parser_requires_explicit_frozen_manifest_and_confirmation() -> None:
    args = phase2b2a.build_parser().parse_args([
        "audit-inputs",
        "--storage-root", "/persistent",
        "--selection-manifest", "/persistent/samples.jsonl",
        "--selection-manifest-sha256", POST_MANIFEST_SHA256,
        "--confirm-cloud-storage",
    ])
    assert args.stage == "audit-inputs"
    assert args.selection_manifest_sha256 == POST_MANIFEST_SHA256
    assert args.confirm_cloud_storage is True


def test_invalid_manifest_stops_before_audit_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(phase2b2a, "require_cloud_confirmation", lambda root, _: root)
    with pytest.raises(ValueError, match="frozen post-manifest"):
        phase2b2a.run_audit_inputs(
            tmp_path,
            tmp_path / "missing.jsonl",
            "0" * 64,
            confirmed_cloud_storage=True,
        )
    assert not (tmp_path / "trustsr" / "phase2b2a").exists()
```

- [ ] **Step 2: Write failing double-load and reuse test**

Patch only the real disk loader boundary; keep selection, audit building and canonical commit real:

```python
def test_run_audit_inputs_loads_12_pairs_twice_and_reuses_identical_audit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = _patch_valid_services(monkeypatch, tmp_path)

    first = phase2b2a.run_audit_inputs(
        tmp_path,
        state.manifest,
        POST_MANIFEST_SHA256,
        confirmed_cloud_storage=True,
    )
    second = phase2b2a.run_audit_inputs(
        tmp_path,
        state.manifest,
        POST_MANIFEST_SHA256,
        confirmed_cloud_storage=True,
    )

    assert state.loaded_sample_ids == state.expected_ids * 4
    assert first["counts"] == {"smoke_pairs": 12, "smoke_geotiffs": 24}
    assert first["digests"]["audit_sha256"] == second["digests"]["audit_sha256"]
    assert first["reused"] is False
    assert second["reused"] is True
```

Also test that a pre-existing audit directory with extra files, a symlink, different bytes or missing audit file fails closed.

- [ ] **Step 3: Run CLI tests and verify missing behavior**

Run: `uv run pytest tests/cli/test_phase2b2a.py -v`

Expected: collection or assertions fail because the CLI does not exist.

- [ ] **Step 4: Implement parser, run function and immutable audit commit**

The parser has exactly one stage and these arguments:

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    command = subparsers.add_parser("audit-inputs")
    command.add_argument("--storage-root", type=Path, required=True)
    command.add_argument("--selection-manifest", type=Path, required=True)
    command.add_argument("--selection-manifest-sha256", required=True)
    command.add_argument("--confirm-cloud-storage", action="store_true")
    return parser
```

`run_audit_inputs()` must call `require_cloud_confirmation()` first, then load the frozen records, select 12 records, load them once into a tuple, load them independently a second time, and build the audit. Serialize with `canonical_json()`.

Commit to:

```text
<storage-root>/trustsr/phase2b2a/input-audits/<manifest-sha256>/phase2b2a-input-audit.json
```

Use a temporary sibling directory and atomic rename. Existing content is reusable only when the digest directory contains exactly one regular non-symlink audit file whose bytes equal the candidate. Return stage, audit/manifest digests, 12/24 counts and reuse status. `main()` prints exactly one canonical JSON object.

- [ ] **Step 5: Register the console script and run focused tests**

Add:

```toml
trustsr-phase2b2a = "trustsr.cli.phase2b2a:main"
```

Run: `uv run pytest tests/cli/test_phase2b2a.py tests/cli/test_phase2b1b.py -v`

Expected: all tests pass and Phase 2B1B behavior is unchanged.

- [ ] **Step 6: Commit the CLI**

```bash
git add pyproject.toml src/trustsr/cli/phase2b2a.py tests/cli/test_phase2b2a.py
git commit -m "feat: audit phase2b2a model inputs"
```

---

### Task 6: CPU Cloud Runner and Operator Runbook

**Files:**
- Create: `scripts/phase2b2a/run_cloud.sh`
- Create: `tests/scripts/test_phase2b2a_scripts.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: explicit mounted storage root, repository and Phase 2B2-A CLI arguments.
- Produces: safe `/opt/conda/bin/python -m trustsr.cli.phase2b2a audit-inputs` execution and a canonical JSONL log.

- [ ] **Step 1: Write failing shell contract tests**

Adapt the harmless fake-command harness from `tests/scripts/test_phase2b1b_scripts.py`. Assert:

```python
def test_runner_uses_base_python_without_conda_or_gpu_commands(tmp_path: Path) -> None:
    completed, calls, conda_called = _invoke(
        tmp_path,
        stage_arguments=[
            "--selection-manifest", "/persistent/samples.jsonl",
            "--selection-manifest-sha256", POST_MANIFEST_SHA256,
            "--confirm-cloud-storage",
        ],
    )
    assert completed.returncode == 0, completed.stderr
    assert calls[0][:3] == ["-m", "trustsr.cli.phase2b2a", "audit-inputs"]
    assert not conda_called.exists()


@pytest.mark.parametrize("bad_root", ["/", "/root", "relative", "/tmp/*"])
def test_runner_rejects_unsafe_storage_before_python(tmp_path: Path, bad_root: str) -> None:
    completed, calls, _ = _invoke(tmp_path, storage_root=bad_root)
    assert completed.returncode == 2
    assert calls == []
```

Also require an actual mountpoint, more than 1 GiB free, at least 1024 free inodes, explicit cloud confirmation, a repository without colon/symlink components, and rejection of CLI arguments beginning `--st` that could override the storage root.

- [ ] **Step 2: Run the script tests and verify missing-script failure**

Run: `uv run pytest tests/scripts/test_phase2b2a_scripts.py -v`

Expected: fails because `scripts/phase2b2a/run_cloud.sh` is absent.

- [ ] **Step 3: Implement the runner**

Reuse the validated path, mountpoint and repository functions from the Phase 2B1B script by copying the shell functions into the standalone runner. The runner must:

1. receive `BASE_PYTHON STORAGE_ROOT REPO_DIR STAGE_ARGS` in its testable `run_main()`;
2. validate `/opt/conda/bin/python` when executed directly;
3. require the mount with `mountpoint -q`, disk with `df -Pk`, and inodes with `df -Pi`;
4. require `--confirm-cloud-storage` in forwarded arguments;
5. write only canonical CLI stdout to `<storage-root>/trustsr/phase2b2a/logs/audit-inputs.jsonl`;
6. invoke `PYTHONPATH=<repo>/src /opt/conda/bin/python -m trustsr.cli.phase2b2a audit-inputs --storage-root <root> ...`;
7. never call `conda`, `pip`, `nvidia-smi`, `cuda`, `wget`, or `curl`.

Set executable mode with `chmod +x scripts/phase2b2a/run_cloud.sh`.

- [ ] **Step 4: Add exact README commands and shutdown boundary**

Document this operator sequence without host or secrets:

```bash
: "${PHASE2B2A_STORAGE_ROOT:?set this to the persistent filesystem mountpoint}"
PHASE2B2A_POST_MANIFEST_SHA256=c7f8ffa8415575d85daafe284a0796ec3f111442f0ac662f1d01311c4a851d4a
PHASE2B2A_POST_MANIFEST="${PHASE2B2A_STORAGE_ROOT%/}/trustsr/phase2b1b/selections/${PHASE2B2A_POST_MANIFEST_SHA256}/samples.jsonl"

scripts/phase2b2a/run_cloud.sh \
  "$PHASE2B2A_STORAGE_ROOT" "$PWD" \
  --selection-manifest "$PHASE2B2A_POST_MANIFEST" \
  --selection-manifest-sha256 "$PHASE2B2A_POST_MANIFEST_SHA256" \
  --confirm-cloud-storage
```

State that the instance can be paused after the digest-scoped audit is copied and verified, and that no GPU is used.

- [ ] **Step 5: Run script regressions and repository gates**

Run:

```bash
uv run pytest tests/scripts/test_phase2b2a_scripts.py tests/scripts/test_phase2b1b_scripts.py -v
uv run ruff check .
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 6: Commit the runner and runbook**

```bash
git add scripts/phase2b2a/run_cloud.sh tests/scripts/test_phase2b2a_scripts.py README.md
git commit -m "docs: add phase2b2a CPU input runbook"
```

---

### Task 7: Local Full Gate, Remote 12-Pair Gate and Git-Safe Audit

**Files:**
- Create after real gate: `artifacts/datasets/sen2naipv2-phase2b2a-input-audit-v1.json`
- Modify only when evidence proves a contract defect: Phase 2B2-A code and tests.

**Interfaces:**
- Consumes: exact reviewed feature commit and a user-provided current SSH endpoint with mounted Phase 2B1B storage.
- Produces: verified real 12-pair CPU audit and one small canonical Git artifact.

- [ ] **Step 1: Run the complete local gate**

```bash
uv run pytest -q
uv run ruff check .
uv lock --check
git diff --check main...HEAD
git status --short
```

Expected: all tests and quality commands pass, and the working tree is clean.

- [ ] **Step 2: Verify repository data policy**

Run a tracked-file check that does not traverse `.venv`:

```bash
test -z "$(git ls-files '*.taco' '*.tif' '*.tiff')"
test -z "$(git ls-files -z | xargs -0 -r stat --printf='%s %n\n' | awk '$1 > 1048576 {print}')"
```

Expected: both commands exit 0 with no output.

- [ ] **Step 3: Push the exact locally verified feature tip**

```bash
git push -u origin feature/phase2b2a-input-contract
```

Record the commit SHA. Do not start a cloud instance before this gate.

- [ ] **Step 4: Request only a current SSH endpoint when remote access is needed**

Tell the user that Phase 2B2-A uses CPU but requires the cloud instance only to access persistent Phase 2B1B pixels. Do not request a password when key authentication is configured. Confirm the user-supplied long-term storage path is mounted before reading.

- [ ] **Step 5: Inspect remote state read-only**

Over the current SSH endpoint, run:

```bash
mountpoint -q /root/rivermind-fs
df -h /root/rivermind-fs
df -i /root/rivermind-fs
/opt/conda/bin/python --version
git -C /root/rivermind-fs/RemoteSensing001 status --short
test -f /root/rivermind-fs/trustsr/phase2b1b/selections/c7f8ffa8415575d85daafe284a0796ec3f111442f0ac662f1d01311c4a851d4a/samples.jsonl
find /root/rivermind-fs/trustsr/phase2b1b/subset-v1 -type f -name '*.tif' | wc -l
```

Expected: mount is present, free bytes and inodes satisfy the runner, repository state is understood, frozen sidecar exists, and the pixel count is 720. Preserve unrelated remote changes.

- [ ] **Step 6: Update non-destructively and run the CPU audit twice**

Fetch and fast-forward/switch to the exact feature commit without `git reset --hard`. Run the README command twice. Require both outputs to report 12 pairs and 24 GeoTIFFs, the same audit SHA-256, and `reused=true` on the second run.

- [ ] **Step 7: Copy only canonical audit bytes into Git**

Transfer only:

```text
<storage-root>/trustsr/phase2b2a/input-audits/
  c7f8ffa8415575d85daafe284a0796ec3f111442f0ac662f1d01311c4a851d4a/
  phase2b2a-input-audit.json
```

Save it as `artifacts/datasets/sen2naipv2-phase2b2a-input-audit-v1.json`. Do not transfer TIFFs, the 360-row sidecar, tensors, logs or model files.

- [ ] **Step 8: Validate and commit real evidence**

```bash
uv run python -m json.tool artifacts/datasets/sen2naipv2-phase2b2a-input-audit-v1.json >/dev/null
test "$(stat -c %s artifacts/datasets/sen2naipv2-phase2b2a-input-audit-v1.json)" -lt 1048576
uv run pytest -q
uv run ruff check .
uv lock --check
git diff --check
git add artifacts/datasets/sen2naipv2-phase2b2a-input-audit-v1.json
git commit -m "data: record phase2b2a input audit"
git push
```

After checking that no remote process remains, explicitly tell the user the cloud server can be paused.

---

### Task 8: Review and Integration Gate

**Files:**
- No new implementation files; review the complete branch against the approved spec.

**Interfaces:**
- Consumes: all Phase 2B2-A commits and real audit evidence.
- Produces: a reviewed pull request with explicit scientific and engineering evidence.

- [ ] **Step 1: Run final fresh verification on the exact branch tip**

```bash
uv run pytest -q
uv run ruff check .
uv lock --check
git diff --check main...HEAD
git status --short
```

Expected: all commands pass and the working tree is clean.

- [ ] **Step 2: Inspect scope and history**

```bash
git log --oneline main..HEAD
git diff --stat main...HEAD
git diff --name-status main...HEAD
```

Require only the approved spec/plan, input loader/tests, audit builder/tests, CLI/tests, shell runner/tests, README, console entry point and Git-safe real audit. Reject credentials, TIFFs, full sidecars, tensors, weights, model outputs or unrelated refactors.

- [ ] **Step 3: Create the PR**

Create a PR from `feature/phase2b2a-input-contract` to `main`. Include frozen manifest/audit hashes, exact crop and normalization policies, 12/24 counts, repeat-load evidence, test/Ruff/lock results, and explicit statements that no model/GPU ran and no real pixels entered Git.

- [ ] **Step 4: Complete evidence-backed code review before merge**

Use `superpowers:requesting-code-review`; apply `superpowers:receiving-code-review` to findings; rerun affected focused tests and the full gate. Merge only after review is clean and the user-approved integration choice is confirmed by the current conversation.
