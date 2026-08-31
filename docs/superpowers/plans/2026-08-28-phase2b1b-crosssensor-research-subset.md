# Phase 2B1B Crosssensor Research Subset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, cloud-only Phase 2B1B pipeline that expands the audited 36-pair pilot into an independently extracted and verified 360-pair research subset.

**Architecture:** Preserve the Phase 2B1A manifest and pilot interfaces unchanged. Add a ten-round selector and a digest-addressed 360-row sidecar schema, then expose `select`, `extract`, and `audit` as restartable Phase 2B1B CLI stages. Reuse the frozen source verifier and TACO v1 pixel adapter; keep all real manifests and pixels in explicit cloud storage and commit only a small audit summary.

**Tech Stack:** Python 3.12, standard-library dataclasses/hashlib/json/pathlib, Rasterio through the existing adapter, pytest, Ruff, Bash, `tacoreader==0.4.5` in the cloud base environment only.

**Spec:** `docs/superpowers/specs/2026-08-28-phase2b1b-crosssensor-research-subset.md`

## Global Constraints

- The only accepted Phase 2B1A post-extraction manifest SHA-256 is `7487b0af2ebef86910e918d5d6b2fb927a6f5e46bac7c2e30be7ffb2ce994482`.
- The only source object is `sen2naipv2-crosssensor.taco`, size `9_717_583_850`, SHA-256 `c6f29d8e80dc5e856e2b4510c0e6830043d4b15c9228a9ca249a4f618e7475a5`, revision `c370504201072fdb1dd388013ab8c0fc7d00a57e`.
- Never copy the real TACO object, the 360-row sidecar, or real GeoTIFF pixels into the WSL workspace or Git.
- The final selection is exactly 10 rounds × 12 strata × 3 splits = 360 pairs and 720 GeoTIFF files.
- Each split has exactly 120 different `spatial_group_id` values; no fallback may duplicate a group or reduce a stratum.
- Round 1 must equal the existing `select_pilot()` result exactly as a set of 36 sample IDs.
- Correlation cuts remain `0.8842208864`, `0.9041984739`, and `0.9265462586`, with boundaries assigned to the higher bin.
- Candidate ordering remains `sha256(b"trustsr-pilot-v1\n" + sample_id.encode("utf-8"))`.
- Phase 2B1A code, schemas, paths, audit bytes, and behavior remain compatible.
- `tacoreader==0.4.5` remains cloud-extraction-only and must not enter `pyproject.toml` or `uv.lock`.
- Cloud execution uses `/opt/conda/bin/python` in the base environment and creates no Conda environment.
- Do not modify PyTorch, torchvision, triton, CUDA, or any normalized package name beginning with `nvidia-`.
- Do not hard-code an SSH host, port, username, password, storage root, GPU model, CUDA version, or transport URL.
- This phase runs no super-resolution model, computes no paper metric, and requires no GPU.
- Every implementation task uses red-green-refactor TDD and ends in a focused commit.

## File Structure

- `src/trustsr/data/research_subset.py`: immutable selection records and the ten-round, 360-pair selector.
- `src/trustsr/data/subset_manifest.py`: strict sidecar schema, canonical JSONL, base-manifest cross-check, and Phase 2B1B audit payload.
- `src/trustsr/cli/phase2b1b.py`: restartable `select`, `extract`, and `audit` stages plus confined, digest-addressed artifact handling.
- `src/trustsr/data/pilot_sampling.py`: unchanged; imported to prove the first round remains identical.
- `src/trustsr/data/taco_v1_adapter.py`: unchanged public `extract_pair()` reused for raw pair extraction and pixel validation.
- `pyproject.toml`: add only the `trustsr-phase2b1b` console entry point.
- `scripts/phase2b1b/run_cloud.sh`: base-Python cloud runner with mount, path, confirmation, and 5 GiB free-space gates.
- `README.md`: exact Phase 2B1B operator sequence and shutdown boundary.
- `tests/data/test_research_subset.py`: ten-round selection, determinism, boundary, prefix, and exhaustion tests.
- `tests/data/test_subset_manifest.py`: sidecar schema, canonical serialization, base cross-check, asset-state, and audit tests.
- `tests/cli/test_phase2b1b.py`: parser, stage preflight, recovery, extraction, audit, and fail-closed CLI tests.
- `tests/scripts/test_phase2b1b_scripts.py`: executable shell contract tests using harmless command fakes.
- `artifacts/datasets/sen2naipv2-phase2b1b-audit-v1.json`: small real-data audit added only after the cloud gate passes.

---

### Task 1: Ten-Round Research Subset Selector

**Files:**
- Create: `src/trustsr/data/research_subset.py`
- Create: `tests/data/test_research_subset.py`

**Interfaces:**
- Consumes: `Sequence[AssignedSample]` from the verified 8,000-row Phase 2B1A manifest.
- Produces: `SubsetChoice` and `select_research_subset(assignments: Sequence[AssignedSample]) -> tuple[SubsetChoice, ...]`.

- [ ] **Step 1: Write the failing selection tests**

Create a synthetic fixture with ten unique candidates for every split/day/bin, followed by the selection assertions:

```python
import hashlib
from collections import Counter

import pytest

from trustsr.data.crosssensor_schema import CrosssensorSample
from trustsr.data.pilot_sampling import select_pilot
from trustsr.data.research_subset import select_research_subset
from trustsr.data.spatial_split import AssignedSample


def _assignment(
    sample_id: str,
    split: str,
    days_between: int,
    correlation: float,
    group_id: str,
) -> AssignedSample:
    lr_time = {
        -1: "2020-01-03T10:00:00Z",
        0: "2020-01-02T10:00:00Z",
        1: "2020-01-01T10:00:00Z",
    }[days_between]
    return AssignedSample(
        sample=CrosssensorSample(
            source_index=int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16),
            sample_id=sample_id,
            longitude=-76.5,
            latitude=3.5,
            crs="EPSG:32618",
            geotransform=(10.0, 0.0, 500000.0, 0.0, -10.0, 400000.0),
            raster_shape=(130, 130),
            time_start="2020-01-02T10:00:00Z",
            lr_time_start=lr_time,
            hr_time_start="2020-01-02T10:00:00Z",
            admin0="Colombia",
            admin1=None,
            admin2="Cali",
            days_between=days_between,
            correlation=correlation,
            scale_factor=4,
        ),
        spatial_group_id=group_id,
        split=split,
    )


def _complete_assignments(candidates_per_stratum: int) -> tuple[AssignedSample, ...]:
    correlations = (0.8, 0.89, 0.91, 0.94)
    return tuple(
        _assignment(
            sample_id=f"{split}-{day}-{bin_index}-{rank}",
            split=split,
            days_between=day,
            correlation=correlation,
            group_id=f"group-{split}-{day}-{bin_index}-{rank}",
        )
        for split in ("development", "calibration", "internal_test")
        for day in (-1, 0, 1)
        for bin_index, correlation in enumerate(correlations)
        for rank in range(candidates_per_stratum)
    )


def test_research_subset_has_ten_rounds_per_stratum_and_unique_groups() -> None:
    assignments = _complete_assignments(candidates_per_stratum=10)
    choices = select_research_subset(assignments)

    assert len(choices) == 360
    assert Counter(choice.split for choice in choices) == {
        "development": 120,
        "calibration": 120,
        "internal_test": 120,
    }
    assert set(Counter(
        (choice.split, choice.days_between, choice.correlation_bin)
        for choice in choices
    ).values()) == {10}
    assert set(choice.selection_round for choice in choices) == set(range(1, 11))
    for split in ("development", "calibration", "internal_test"):
        selected = [choice for choice in choices if choice.split == split]
        assert len({choice.spatial_group_id for choice in selected}) == 120


def test_round_one_exactly_preserves_phase2b1a_pilot() -> None:
    assignments = _complete_assignments(candidates_per_stratum=10)
    expected = {choice.sample_id for choice in select_pilot(assignments)}
    observed = {
        choice.sample_id
        for choice in select_research_subset(assignments)
        if choice.selection_round == 1
    }
    assert observed == expected


def test_selection_is_independent_of_input_order() -> None:
    assignments = _complete_assignments(candidates_per_stratum=10)
    assert select_research_subset(assignments) == select_research_subset(
        tuple(reversed(assignments))
    )


def test_selection_rejects_a_tenth_round_that_can_only_reuse_a_group() -> None:
    assignments = list(_complete_assignments(candidates_per_stratum=10))
    target = [
        item for item in assignments
        if item.split == "development"
        and item.sample.days_between == -1
        and item.sample.correlation == 0.8
    ]
    assignments = [item for item in assignments if item not in target[-1:]]
    with pytest.raises(ValueError, match="selection_round=10"):
        select_research_subset(assignments)
```

- [ ] **Step 2: Run the new tests and verify the missing-module failure**

Run: `uv run pytest tests/data/test_research_subset.py -v`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'trustsr.data.research_subset'`.

- [ ] **Step 3: Implement the immutable record and round-major selector**

Use the existing public correlation binning function and the exact selection prefix:

```python
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from trustsr.data.pilot_sampling import correlation_bin, select_pilot
from trustsr.data.spatial_split import AssignedSample

SPLITS = ("development", "calibration", "internal_test")
DAYS_BETWEEN = (-1, 0, 1)
CORRELATION_BINS = (0, 1, 2, 3)
SELECTION_ROUNDS = 10
SELECTION_PREFIX = b"trustsr-pilot-v1\n"


@dataclass(frozen=True)
class SubsetChoice:
    sample_id: str
    split: str
    days_between: int
    correlation_bin: int
    selection_round: int
    spatial_group_id: str
    selection_sha256: str


def selection_sha256(sample_id: str) -> str:
    if type(sample_id) is not str or not sample_id:
        raise ValueError("sample_id must be a non-empty string")
    return hashlib.sha256(SELECTION_PREFIX + sample_id.encode("utf-8")).hexdigest()


def select_research_subset(
    assignments: Sequence[AssignedSample],
) -> tuple[SubsetChoice, ...]:
    choices: list[SubsetChoice] = []
    for split in SPLITS:
        used_groups: set[str] = set()
        strata = {
            (day, bin_index): sorted(
                (
                    item for item in assignments
                    if item.split == split
                    and item.sample.days_between == day
                    and correlation_bin(item.sample.correlation) == bin_index
                ),
                key=lambda item: selection_sha256(item.sample.sample_id),
            )
            for day in DAYS_BETWEEN
            for bin_index in CORRELATION_BINS
        }
        for selection_round in range(1, SELECTION_ROUNDS + 1):
            for day in DAYS_BETWEEN:
                for bin_index in CORRELATION_BINS:
                    choice = next(
                        (
                            item for item in strata[(day, bin_index)]
                            if item.spatial_group_id not in used_groups
                        ),
                        None,
                    )
                    if choice is None:
                        raise ValueError(
                            "research subset stratum cannot select a distinct spatial group: "
                            f"split={split}, selection_round={selection_round}, "
                            f"days_between={day}, correlation_bin={bin_index}"
                        )
                    used_groups.add(choice.spatial_group_id)
                    choices.append(SubsetChoice(
                        sample_id=choice.sample.sample_id,
                        split=split,
                        days_between=day,
                        correlation_bin=bin_index,
                        selection_round=selection_round,
                        spatial_group_id=choice.spatial_group_id,
                        selection_sha256=selection_sha256(choice.sample.sample_id),
                    ))
    result = tuple(sorted(
        choices,
        key=lambda item: (
            item.split,
            item.selection_round,
            item.days_between,
            item.correlation_bin,
        ),
    ))
    if {item.sample_id for item in result if item.selection_round == 1} != {
        item.sample_id for item in select_pilot(assignments)
    }:
        raise ValueError("research subset round one must equal the Phase 2B1A pilot")
    return result
```

- [ ] **Step 4: Run focused selector tests and the Phase 2B1A regression**

Run: `uv run pytest tests/data/test_research_subset.py tests/data/test_pilot_sampling.py -v`

Expected: PASS, with 360 choices and unchanged Phase 2B1A tests.

- [ ] **Step 5: Commit the selector**

```bash
git add src/trustsr/data/research_subset.py tests/data/test_research_subset.py
git commit -m "feat: select phase2b1b research subset"
```

---

### Task 2: Canonical 360-Row Sidecar Contract

**Files:**
- Create: `src/trustsr/data/subset_manifest.py`
- Create: `tests/data/test_subset_manifest.py`

**Interfaces:**
- Consumes: validated Phase 2B1A records, `SubsetChoice` records, and optional `ExtractedAsset` pairs.
- Produces: `write_subset_manifest(path: Path, base_records: Sequence[Mapping[str, object]], choices: Sequence[SubsetChoice], assets: Mapping[str, tuple[ExtractedAsset, ExtractedAsset]]) -> ManifestArtifact`, `load_subset_manifest(path: Path, *, expected_sha256: str) -> tuple[dict[str, object], ...]`, `validate_subset_against_base(records: Sequence[Mapping[str, object]], base_records: Sequence[Mapping[str, object]]) -> tuple[SubsetChoice, ...]`, and constants `SUBSET_SCHEMA`, `BASE_MANIFEST_SHA256`.

- [ ] **Step 1: Write failing canonical sidecar tests**

Build 360 synthetic base records using the existing `write_manifest()` fixture style, then assert exact state and provenance:

```python
def test_subset_manifest_round_trip_is_canonical_and_digest_addressed(tmp_path: Path) -> None:
    assignments = _complete_assignments()
    base_records = _base_records(assignments)
    choices = select_research_subset(assignments)
    path = tmp_path / "samples.jsonl"

    artifact = write_subset_manifest(path, base_records, choices, {})
    records = load_subset_manifest(path, expected_sha256=artifact.sha256)

    assert len(records) == 360
    assert artifact.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert all(record["schema"] == "trustsr.phase2b1b-selection.v1" for record in records)
    assert all(record["base_manifest_sha256"] == BASE_MANIFEST_SHA256 for record in records)
    assert all(record["lr_asset"] is None and record["hr_asset"] is None for record in records)
    assert path.read_bytes().endswith(b"\n")


def test_manifest_requires_all_or_none_assets(tmp_path: Path) -> None:
    assignments = _complete_assignments()
    choices = select_research_subset(assignments)
    assets = {_one_choice_id(choices): _asset_pair()}
    with pytest.raises(ValueError, match="exactly every selected sample"):
        write_subset_manifest(tmp_path / "samples.jsonl", _base_records(assignments), choices, assets)


def test_subset_must_equal_deterministic_selection_from_base() -> None:
    assignments = _complete_assignments()
    choices = list(select_research_subset(assignments))
    choices[0] = replace(choices[0], selection_round=2)
    with pytest.raises(ValueError, match="deterministic research subset"):
        write_subset_manifest(Path("unused"), _base_records(assignments), choices, {})
```

- [ ] **Step 2: Run the new tests and verify failure before implementation**

Run: `uv run pytest tests/data/test_subset_manifest.py -v`

Expected: FAIL during collection because `trustsr.data.subset_manifest` does not exist.

- [ ] **Step 3: Define the exact sidecar record and validators**

Create constants and reuse the established asset/artifact dataclasses:

```python
SUBSET_SCHEMA = "trustsr.phase2b1b-selection.v1"
AUDIT_SCHEMA = "trustsr.phase2b1b-audit.v1"
BASE_MANIFEST_SHA256 = "7487b0af2ebef86910e918d5d6b2fb927a6f5e46bac7c2e30be7ffb2ce994482"
BANDS = ("B04", "B03", "B02", "B08")
SUBSET_FIELDS = frozenset({
    "schema", "base_manifest_sha256", "source", "source_index", "sample_id",
    "centroid", "crs", "geotransform", "raster_shape", "time_start",
    "lr_time_start", "hr_time_start", "days_between", "correlation",
    "correlation_bin", "scale_factor", "bands", "spatial_group_id", "split",
    "selection_round", "selection_sha256", "lr_asset", "hr_asset",
})
```

Implement `_subset_record()` by copying the named values from the matching base record, assigning the five selection fields from `SubsetChoice`, setting `bands=list(BANDS)`, and serializing assets with the existing `ExtractedAsset` field names. Implement `_validate_record()` with exact type checks, lowercase 64-character digest checks, fixed source identity, fixed ×4 scale, exact bands, allowed split/day/bin/round values, selection hash recomputation, safe relative POSIX asset paths, and LR/HR all-or-none validation.

- [ ] **Step 4: Implement base cross-check and canonical JSONL I/O**

Use these public signatures and fail on duplicate/missing IDs:

```python
def validate_subset_against_base(
    records: Sequence[Mapping[str, object]],
    base_records: Sequence[Mapping[str, object]],
) -> tuple[SubsetChoice, ...]:
    assignments = tuple(_assignment_from_base_record(record) for record in base_records)
    expected = select_research_subset(assignments)
    expected_by_id = {choice.sample_id: choice for choice in expected}
    if len(records) != 360 or {record["sample_id"] for record in records} != set(expected_by_id):
        raise ValueError("sidecar does not equal the deterministic research subset")
    for record in records:
        _require_record_matches_base(record, _base_by_id(base_records)[record["sample_id"]])
        _require_record_matches_choice(record, expected_by_id[record["sample_id"]])
    return expected


def write_subset_manifest(
    path: Path,
    base_records: Sequence[Mapping[str, object]],
    choices: Sequence[SubsetChoice],
    assets: Mapping[str, tuple[ExtractedAsset, ExtractedAsset]],
) -> ManifestArtifact:
    base_by_id = _base_by_id(base_records)
    expected = select_research_subset(tuple(
        _assignment_from_base_record(record) for record in base_records
    ))
    if tuple(choices) != expected:
        raise ValueError("choices do not equal the deterministic research subset")
    selected_ids = {choice.sample_id for choice in expected}
    if set(assets) not in (set(), selected_ids):
        raise ValueError("assets must contain exactly every selected sample or be empty")
    choice_by_id = {choice.sample_id: choice for choice in expected}
    records = tuple(
        _validate_record(_subset_record(
            base_by_id[sample_id],
            choice_by_id[sample_id],
            assets.get(sample_id),
        ))
        for sample_id in sorted(selected_ids)
    )
    _validate_collection(records)
    payload = b"".join(canonical_json(record) + b"\n" for record in records)
    digest = hashlib.sha256(payload).hexdigest()
    atomic_write_bytes(path, payload)
    return ManifestArtifact(path=path, size_bytes=len(payload), sha256=digest)


def load_subset_manifest(
    path: Path, *, expected_sha256: str
) -> tuple[dict[str, object], ...]:
    _require_sha256(expected_sha256, "expected selection manifest SHA-256")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("selection manifest SHA-256 does not match")
    if not payload.endswith(b"\n"):
        raise ValueError("selection manifest must end with one newline")
    records: list[dict[str, object]] = []
    for line in payload.splitlines(keepends=True):
        decoded = json.loads(line)
        record = _validate_record(decoded)
        if canonical_json(record) + b"\n" != line:
            raise ValueError("selection manifest record is not canonical JSON")
        records.append(record)
    if [record["sample_id"] for record in records] != sorted(
        record["sample_id"] for record in records
    ):
        raise ValueError("selection manifest must be sorted by sample_id")
    _validate_collection(records)
    return tuple(records)
```

`_validate_collection()` must require 360 unique IDs, exactly 120 rows and 120 unique groups per
split, ten rows for each `(split, day, bin)` key, 36 rows for every round from 1 through 10, and a
uniform asset state in which either all 360 pairs are null or all 360 pairs are present.

- [ ] **Step 5: Run sidecar and upstream manifest tests**

Run: `uv run pytest tests/data/test_subset_manifest.py tests/data/test_crosssensor_manifest.py tests/data/test_research_subset.py -v`

Expected: PASS.

- [ ] **Step 6: Commit the sidecar contract**

```bash
git add src/trustsr/data/subset_manifest.py tests/data/test_subset_manifest.py
git commit -m "feat: define phase2b1b sidecar manifest"
```

---

### Task 3: Fail-Closed `select` CLI Stage

**Files:**
- Create: `src/trustsr/cli/phase2b1b.py`
- Create: `tests/cli/test_phase2b1b.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: source provenance JSON, explicit cloud root, and the exact Phase 2B1A post-extraction manifest path.
- Produces: `build_parser() -> argparse.ArgumentParser`, `run_select(source_path: Path, storage_root: Path, base_manifest_path: Path, *, confirmed_cloud_storage: bool) -> dict[str, object]`, and console command `trustsr-phase2b1b`.

- [ ] **Step 1: Write failing parser and select-stage tests**

```python
def test_parser_requires_cloud_confirmation_and_stage_specific_manifest() -> None:
    parser = phase2b1b.build_parser()
    args = parser.parse_args([
        "select", "--source", "source.json", "--storage-root", "/persistent",
        "--base-manifest", "samples.jsonl", "--confirm-cloud-storage",
    ])
    assert args.stage == "select"
    assert args.base_manifest == Path("samples.jsonl")
    assert args.confirm_cloud_storage is True


def test_select_writes_digest_addressed_pre_extraction_sidecar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = _patch_select_services(monkeypatch, tmp_path)
    result = phase2b1b.run_select(
        state.source_path,
        tmp_path,
        state.base_manifest,
        confirmed_cloud_storage=True,
    )
    assert result["stage"] == "select"
    assert result["counts"] == {"subset_pairs": 360, "subset_geotiffs": 0}
    digest = result["digests"]["selection_manifest_sha256"]
    assert (tmp_path / "trustsr" / "phase2b1b" / "selections" / digest / "samples.jsonl").is_file()


def test_select_rejects_wrong_base_digest_before_creating_phase_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = _patch_select_services(monkeypatch, tmp_path, base_digest="0" * 64)
    with pytest.raises(ValueError, match="frozen Phase 2B1A manifest"):
        phase2b1b.run_select(
            state.source_path, tmp_path, state.base_manifest,
            confirmed_cloud_storage=True,
        )
    assert not (tmp_path / "trustsr" / "phase2b1b").exists()
```

- [ ] **Step 2: Verify the CLI tests fail before implementation**

Run: `uv run pytest tests/cli/test_phase2b1b.py -v`

Expected: FAIL during collection because `trustsr.cli.phase2b1b` does not exist.

- [ ] **Step 3: Implement parser, frozen-source preflight, and base-manifest loading**

The parser has exactly three stages and no download option:

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for stage in ("select", "extract", "audit"):
        command = subparsers.add_parser(stage)
        command.add_argument("--source", type=Path, required=True)
        command.add_argument("--storage-root", type=Path, required=True)
        command.add_argument("--confirm-cloud-storage", action="store_true")
        if stage == "select":
            command.add_argument("--base-manifest", type=Path, required=True)
        else:
            command.add_argument("--selection-manifest", type=Path, required=True)
    return parser
```

Implement `_load_frozen_source()` with the same repository, revision, license, object name, size, and SHA checks as Phase 2B1A. Implement `_load_base_manifest(root, path)` so the resolved regular file must be exactly:

```text
<root>/trustsr/phase2b1a/manifests/7487b0af2ebef86910e918d5d6b2fb927a6f5e46bac7c2e30be7ffb2ce994482/samples.jsonl
```

and load it with `load_manifest(path, expected_sha256=BASE_MANIFEST_SHA256)`.

- [ ] **Step 4: Implement `run_select` with no writes before complete validation**

```python
def run_select(
    source_path: Path,
    storage_root: Path,
    base_manifest_path: Path,
    *,
    confirmed_cloud_storage: bool,
) -> dict[str, object]:
    root = require_cloud_confirmation(storage_root, confirmed_cloud_storage)
    source, object_spec = _load_frozen_source(source_path)
    verified = verify_crosssensor(source_paths(root, object_spec).final, object_spec)
    base_records = _load_base_manifest(root, base_manifest_path)
    assignments = _assignments_from_base_records(base_records)
    choices = select_research_subset(assignments)
    if len(choices) != 360:
        raise ValueError("select stage must choose exactly 360 pairs")

    selection_root = _phase_root(root) / "selections"
    selection_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".candidate-", dir=selection_root) as temporary:
        candidate = Path(temporary) / "samples.jsonl"
        artifact = write_subset_manifest(candidate, base_records, choices, {})
        records = load_subset_manifest(candidate, expected_sha256=artifact.sha256)
        validate_subset_against_base(records, base_records)
        _, reused = _commit_selection(selection_root, artifact)
    return {
        "stage": "select",
        "digests": {
            "base_manifest_sha256": BASE_MANIFEST_SHA256,
            "selection_manifest_sha256": artifact.sha256,
            "source_sha256": verified.sha256,
        },
        "counts": {"subset_pairs": 360, "subset_geotiffs": 0},
        "reused": reused,
    }
```

`_commit_selection()` must use a digest directory containing only `samples.jsonl`; existing directories are reusable only when the file is regular, canonical, and byte-identical.

- [ ] **Step 5: Register the console script and run focused tests**

Add to `[project.scripts]`:

```toml
trustsr-phase2b1b = "trustsr.cli.phase2b1b:main"
```

Run: `uv run pytest tests/cli/test_phase2b1b.py tests/cli/test_phase2b1a.py -v`

Expected: PASS, including unchanged Phase 2B1A CLI behavior.

- [ ] **Step 6: Commit the select stage**

```bash
git add pyproject.toml src/trustsr/cli/phase2b1b.py tests/cli/test_phase2b1b.py
git commit -m "feat: add phase2b1b selection stage"
```

---

### Task 4: Restartable Independent Extraction Stage

**Files:**
- Modify: `src/trustsr/cli/phase2b1b.py`
- Modify: `tests/cli/test_phase2b1b.py`

**Interfaces:**
- Consumes: the digest-addressed all-null sidecar, the derived frozen base manifest, and cached source TACO.
- Produces: `run_extract(source_path: Path, storage_root: Path, selection_manifest_path: Path, *, confirmed_cloud_storage: bool) -> dict[str, object]`, 360 independent pair directories, and an all-assets sidecar.

- [ ] **Step 1: Add failing extraction, resume, and preflight tests**

```python
def test_extract_rebases_all_720_assets_and_commits_post_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = _patch_extract_services(monkeypatch, tmp_path)
    result = phase2b1b.run_extract(
        state.source_path, tmp_path, state.pre_manifest,
        confirmed_cloud_storage=True,
    )
    assert result["counts"] == {"subset_pairs": 360, "subset_geotiffs": 720}
    assert state.extracted_indices == list(range(360))
    for record in state.post_records:
        prefix = f"subset-v1/{record['split']}/{record['sample_id']}"
        assert record["lr_asset"]["relative_path"] == f"{prefix}/lr.tif"
        assert record["hr_asset"]["relative_path"] == f"{prefix}/hr.tif"


def test_extract_reuses_complete_pairs_but_rejects_partial_pair_before_reader_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = _patch_extract_services(monkeypatch, tmp_path)
    complete = state.output_for(state.choices[0])
    complete.mkdir(parents=True)
    (complete / "lr.tif").write_bytes(state.lr_bytes)
    (complete / "hr.tif").write_bytes(state.hr_bytes)
    partial = state.output_for(state.choices[1])
    partial.mkdir(parents=True)
    (partial / "lr.tif").write_bytes(state.lr_bytes)
    with pytest.raises(ValueError, match="partial or invalid"):
        phase2b1b.run_extract(
            state.source_path, tmp_path, state.pre_manifest,
            confirmed_cloud_storage=True,
        )
    assert state.extracted_indices == []


def test_extract_rejects_mixed_asset_manifest_before_any_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = _patch_extract_services(monkeypatch, tmp_path, manifest_state="mixed")
    with pytest.raises(ValueError, match="all-null pre-extraction"):
        phase2b1b.run_extract(
            state.source_path, tmp_path, state.pre_manifest,
            confirmed_cloud_storage=True,
        )
    assert not (tmp_path / "trustsr" / "phase2b1b" / "subset-v1").exists()
```

- [ ] **Step 2: Run the extraction tests and verify the missing-stage failure**

Run: `uv run pytest tests/cli/test_phase2b1b.py -k extract -v`

Expected: FAIL because `run_extract` is absent or does not implement extraction.

- [ ] **Step 3: Implement input validation and complete-pair preflight**

Before any extraction, require:

```python
input_digest, records = _load_digest_selection(root, selection_manifest_path)
if any(record["lr_asset"] is not None or record["hr_asset"] is not None for record in records):
    raise ValueError("extract requires an all-null pre-extraction sidecar")
base_records = _load_frozen_base_from_storage(root)
choices = validate_subset_against_base(records, base_records)
outputs = {
    choice.sample_id: _phase_root(root) / "subset-v1" / choice.split / choice.sample_id
    for choice in choices
}
for output in outputs.values():
    _require_confined(root, output)
    _require_absent_or_complete_pair(output)
```

`_require_absent_or_complete_pair()` accepts an absent directory or exactly two regular, non-symlink files named `lr.tif` and `hr.tif`. It rejects every other state before calling `extract_pair()` for any sample.

- [ ] **Step 4: Implement extraction, time checks, path rebasing, and final commit**

For every canonical choice, call:

```python
lr_asset, hr_asset = extract_pair(
    verified.path,
    int(record["source_index"]),
    outputs[choice.sample_id],
    source.bands,
)
if lr_asset.time_start != record["lr_time_start"]:
    raise ValueError("extracted LR time_start must equal the sidecar")
if hr_asset.time_start != record["hr_time_start"]:
    raise ValueError("extracted HR time_start must equal the sidecar")
prefix = PurePosixPath("subset-v1", choice.split, choice.sample_id)
assets[choice.sample_id] = (
    replace(lr_asset, relative_path=(prefix / "lr.tif").as_posix()),
    replace(hr_asset, relative_path=(prefix / "hr.tif").as_posix()),
)
```

After all 360 calls return validated asset records, write an all-assets sidecar in a temporary directory, reload it, cross-check it against the frozen base, verify all 720 files, and then commit it to the digest-addressed `selections` tree. `_find_reusable_post_selection()` may return one matching post sidecar; zero means extract/resume, more than one is ambiguous and fails.

- [ ] **Step 5: Run extraction and adapter regressions**

Run: `uv run pytest tests/cli/test_phase2b1b.py -k extract -v && uv run pytest tests/data/test_taco_v1_adapter.py tests/cli/test_phase2b1a.py -v`

Expected: PASS.

- [ ] **Step 6: Commit the extraction stage**

```bash
git add src/trustsr/cli/phase2b1b.py tests/cli/test_phase2b1b.py
git commit -m "feat: extract phase2b1b research pairs"
```

---

### Task 5: Strict Phase 2B1B Audit

**Files:**
- Modify: `src/trustsr/data/subset_manifest.py`
- Modify: `src/trustsr/cli/phase2b1b.py`
- Modify: `tests/data/test_subset_manifest.py`
- Modify: `tests/cli/test_phase2b1b.py`

**Interfaces:**
- Consumes: an all-assets sidecar, its frozen base manifest, the source object, and all 720 confined files.
- Produces: `build_subset_audit(records: Sequence[Mapping[str, object]], *, manifest_sha256: str, base_records: Sequence[Mapping[str, object]], minimum_distances: Mapping[str, float]) -> dict[str, object]` and `run_audit(source_path: Path, storage_root: Path, selection_manifest_path: Path, *, confirmed_cloud_storage: bool) -> dict[str, object]`.

- [ ] **Step 1: Write failing audit payload tests**

```python
def test_audit_records_exact_counts_strata_round_one_and_frozen_distances() -> None:
    records, base_records = _complete_post_extraction_records()
    audit = build_subset_audit(
        records,
        manifest_sha256="a" * 64,
        base_records=base_records,
        minimum_distances={
            "calibration:development": 5.001488639974653,
            "calibration:internal_test": 5.012057656530008,
            "development:internal_test": 5.001592281637432,
        },
    )
    assert audit["schema"] == "trustsr.phase2b1b-audit.v1"
    assert audit["subset_pair_count"] == 360
    assert audit["subset_geotiff_count"] == 720
    assert audit["split_sample_counts"] == {
        "development": 120, "calibration": 120, "internal_test": 120,
    }
    assert audit["split_spatial_group_counts"] == audit["split_sample_counts"]
    assert set(audit["stratum_counts"].values()) == {10}
    assert audit["round_one_matches_phase2b1a"] is True
    assert audit["real_pixels_local"] is False
    assert audit["gpu_used"] is False


def test_audit_rejects_one_reused_group() -> None:
    records, base_records = _complete_post_extraction_records()
    records[1]["spatial_group_id"] = records[0]["spatial_group_id"]
    with pytest.raises(ValueError, match="120 distinct spatial groups"):
        build_subset_audit(
            records, manifest_sha256="a" * 64,
            base_records=base_records, minimum_distances=_distances(),
        )
```

- [ ] **Step 2: Run audit tests and verify failure before implementation**

Run: `uv run pytest tests/data/test_subset_manifest.py -k audit -v`

Expected: FAIL because `build_subset_audit` is absent.

- [ ] **Step 3: Implement the canonical audit builder**

The builder first calls `validate_subset_against_base()`, requires 360 asset pairs, and returns only host-free JSON-native values:

```python
return {
    "schema": AUDIT_SCHEMA,
    **audit_source_identity(),
    "base_manifest_sha256": BASE_MANIFEST_SHA256,
    "manifest_sha256": manifest_sha256,
    "subset_pair_count": 360,
    "subset_geotiff_count": 720,
    "split_sample_counts": split_counts,
    "split_spatial_group_counts": group_counts,
    "stratum_counts": stratum_counts,
    "selection_round_counts": round_counts,
    "round_one_matches_phase2b1a": round_one_ids == pilot_ids,
    "minimum_cross_split_distances": validated_distances,
    "real_pixels_local": False,
    "gpu_used": False,
}
```

Use stable string keys `"<split>:day=<day>:bin=<bin>"` for the 36 stratum counts and `"1"` through `"10"` for round counts. Require all stratum values to equal 10, all round values to equal 36, every distance key/value to equal the Phase 2B1A frozen audit evidence exactly, and the round-one boolean to be true.

- [ ] **Step 4: Write failing CLI file-integrity tests**

```python
def test_audit_rehashes_all_720_files_and_writes_digest_scoped_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = _patch_audit_services(monkeypatch, tmp_path)
    result = phase2b1b.run_audit(
        state.source_path, tmp_path, state.post_manifest,
        confirmed_cloud_storage=True,
    )
    assert state.hash_calls == 720
    assert result["counts"] == {"subset_pairs": 360, "subset_geotiffs": 720}
    assert result["digests"]["audit_sha256"] == hashlib.sha256(
        canonical_json(state.audit)
    ).hexdigest()


@pytest.mark.parametrize("damage", ["symlink", "changed-bytes"])
def test_audit_rejects_a_symlink_or_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, damage: str
) -> None:
    state = _patch_audit_services(monkeypatch, tmp_path)
    target = state.first_lr_path
    if damage == "symlink":
        original = target.with_name("original-lr.tif")
        target.rename(original)
        target.symlink_to(original)
        match = "regular GeoTIFF"
    else:
        target.write_bytes(b"changed")
        match = "asset bytes"
    with pytest.raises(ValueError, match=match):
        phase2b1b.run_audit(
            state.source_path, tmp_path, state.post_manifest,
            confirmed_cloud_storage=True,
        )
    assert not (tmp_path / "trustsr" / "phase2b1b" / "audits").exists()
```

- [ ] **Step 5: Implement `run_audit` and immutable audit commit**

Load and verify the post sidecar by digest, derive and load the base manifest, and require exact deterministic equality. For each record, require exact paths:

```python
expected = PurePosixPath("subset-v1", split, sample_id, f"{kind}.tif").as_posix()
if asset["relative_path"] != expected:
    raise ValueError("asset relative_path must use the exact Phase 2B1B layout")
path = _phase_root(root) / expected
_require_confined(root, path)
if path.is_symlink() or not path.is_file():
    raise ValueError("audit requires all 720 confined regular GeoTIFF files")
size_bytes, sha256 = _hash_file(path)
if (size_bytes, sha256) != (asset["size_bytes"], asset["sha256"]):
    raise ValueError("asset bytes do not match the selection manifest")
```

Write `canonical_json(audit)` to `audits/<manifest-sha256>/phase2b1b-audit.json`. An existing audit is reusable only if it is the sole regular file in that digest directory and its bytes are identical.

- [ ] **Step 6: Run focused and cross-phase audit regressions**

Run: `uv run pytest tests/data/test_subset_manifest.py tests/cli/test_phase2b1b.py tests/data/test_crosssensor_manifest.py tests/cli/test_phase2b1a.py -v`

Expected: PASS.

- [ ] **Step 7: Commit the audit stage**

```bash
git add src/trustsr/data/subset_manifest.py src/trustsr/cli/phase2b1b.py tests/data/test_subset_manifest.py tests/cli/test_phase2b1b.py
git commit -m "feat: audit phase2b1b research subset"
```

---

### Task 6: Cloud Runner and Operator Runbook

**Files:**
- Create: `scripts/phase2b1b/run_cloud.sh`
- Create: `tests/scripts/test_phase2b1b_scripts.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: explicit mounted storage root, checked-out repository, stage name, and stage arguments.
- Produces: safe `/opt/conda/bin/python -m trustsr.cli.phase2b1b <stage> --storage-root <root> <stage-arguments>` execution and exact operator commands.

- [ ] **Step 1: Write failing executable shell contract tests**

Reuse harmless `mountpoint`, `df`, and base-Python fakes from the Phase 2B1A test style. Assert:

```python
@pytest.mark.parametrize("stage", ["select", "extract", "audit"])
def test_runner_uses_base_python_and_forwards_only_phase2b1b_stages(
    tmp_path: Path, stage: str
) -> None:
    completed, calls = _run_fixture(tmp_path, stage, ["--confirm-cloud-storage"])
    assert completed.returncode == 0
    assert calls[-1][:3] == ["-m", "trustsr.cli.phase2b1b"]
    assert calls[-1][3] == stage


@pytest.mark.parametrize("bad_root", ["/", "/root", "relative", "/tmp/*"])
def test_runner_rejects_unsafe_or_nonmounted_storage_before_python(
    tmp_path: Path, bad_root: str
) -> None:
    completed, calls = _invoke_runner(tmp_path, storage_root=bad_root)
    assert completed.returncode == 2
    assert calls == []


def test_runner_requires_more_than_five_gib_and_explicit_confirmation(tmp_path: Path) -> None:
    low_space, calls = _run_fixture(tmp_path, "select", ["--confirm-cloud-storage"], available_kib=5 * 1024 * 1024)
    assert low_space.returncode == 2
    assert calls == []
    unconfirmed, calls = _run_fixture(tmp_path, "select", [])
    assert unconfirmed.returncode == 2
    assert calls == []
```

Also test rejection of symlink path components, storage-root override abbreviations, unknown stages, repository paths containing colons, and any call to `conda`.

- [ ] **Step 2: Run script tests and verify the missing-script failure**

Run: `uv run pytest tests/scripts/test_phase2b1b_scripts.py -v`

Expected: FAIL because `scripts/phase2b1b/run_cloud.sh` does not exist.

- [ ] **Step 3: Implement the Phase 2B1B cloud runner**

Create `scripts/phase2b1b/run_cloud.sh` with this complete content:

```bash
#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'invalid cloud run input: %s\n' "$1" >&2
  exit 2
}

validate_raw_path() {
  local value="$1"
  [[ -n "$value" && "$value" == /* && "$value" != / && "$value" != /root ]] || return 1
  [[ "$value" != ~* && "$value" != *$'\n'* && "$value" != *$'\r'* ]] || return 1
  [[ "$value" != *[\*\?\[]* ]] || return 1
}

reject_symlink_components() {
  local value="$1"
  local current=/
  local component
  local -a components
  IFS=/ read -r -a components <<< "${value#/}"
  for component in "${components[@]}"; do
    [[ -n "$component" ]] || continue
    current="${current%/}/${component}"
    [[ ! -L "$current" ]] || return 1
  done
}

require_storage_root() {
  local value="$1"
  local resolved
  local current_home
  validate_raw_path "$value" || die 'storage root'
  reject_symlink_components "$value" || die 'storage root must not contain a symlink'
  [[ -d "$value" && ! -L "$value" ]] || die 'storage root must be an existing directory'
  resolved="$(realpath -e -- "$value")" || die 'storage root'
  current_home="$(realpath -e -- "${HOME:?HOME must be set}")" || die 'current home cannot be resolved'
  [[ "$resolved" != / && "$resolved" != /root && "$resolved" != "$current_home" ]] ||
    die 'prohibited storage root'
  mountpoint -q -- "$resolved" || die "persistent mountpoint is unavailable: $resolved"
  printf '%s\n' "$resolved"
}

require_repository() {
  local value="$1"
  local resolved
  validate_raw_path "$value" || die 'repository directory'
  [[ "$value" != *:* ]] || die 'repository directory must not contain a colon'
  reject_symlink_components "$value" || die 'repository directory must not contain a symlink'
  [[ -d "$value" && ! -L "$value" && -f "$value/pyproject.toml" && -d "$value/src/trustsr" ]] ||
    die 'repository directory must be a checked-out project'
  resolved="$(realpath -e -- "$value")" || die 'repository directory'
  [[ "$resolved" != *:* ]] || die 'repository directory must not contain a colon'
  printf '%s\n' "$resolved"
}

run_main() {
  local base_python="$1"
  local storage_root
  local repo_dir
  local stage
  local argument
  local confirmed=false
  local available_kib
  shift
  (( $# >= 4 )) ||
    die 'argument count; usage: run_cloud.sh STORAGE_ROOT REPO_DIR STAGE STAGE_ARGS'
  [[ -x "$base_python" ]] ||
    die "required cloud-image interpreter is unavailable: $base_python"

  storage_root="$(require_storage_root "$1")"
  repo_dir="$(require_repository "$2")"
  stage="$3"
  shift 3
  case "$stage" in
    select|extract|audit) ;;
    *) die 'stage must be select, extract, or audit' ;;
  esac
  for argument in "$@"; do
    case "$argument" in
      --confirm-cloud-storage) confirmed=true ;;
      --st*) die 'stage arguments must not override storage root' ;;
    esac
  done
  [[ "$confirmed" == true ]] ||
    die 'stage arguments must include --confirm-cloud-storage'

  available_kib="$(df -Pk -- "$storage_root" | awk 'NR == 2 {print $4}')"
  [[ "$available_kib" =~ ^[0-9]+$ ]] || die 'could not determine free disk space'
  (( available_kib > 5 * 1024 * 1024 )) ||
    die 'more than 5 GiB of free disk space is required'

  cd -- "$repo_dir"
  PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}" \
    exec "$base_python" -m trustsr.cli.phase2b1b "$stage" \
      --storage-root "$storage_root" "$@"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  run_main /opt/conda/bin/python "$@"
fi
```

Run `chmod +x scripts/phase2b1b/run_cloud.sh`. Do not create a Phase 2B1B bootstrap script:
Phase 2B1A has already installed and verified the pinned reader/runtime in the base environment.

- [ ] **Step 4: Add the exact README operator sequence**

Document variables and commands without SSH secrets:

```bash
: "${PHASE2B1B_STORAGE_ROOT:?set this to the persistent filesystem mountpoint}"
PHASE2B1B_SOURCE="$PWD/artifacts/datasets/sen2naipv2-source-v1.json"
PHASE2B1B_BASE_MANIFEST_SHA256=7487b0af2ebef86910e918d5d6b2fb927a6f5e46bac7c2e30be7ffb2ce994482
PHASE2B1B_BASE_MANIFEST="${PHASE2B1B_STORAGE_ROOT%/}/trustsr/phase2b1a/manifests/${PHASE2B1B_BASE_MANIFEST_SHA256}/samples.jsonl"

phase2b1b_json_digest() {
  /opt/conda/bin/python -c 'import json, string, sys
value = json.load(sys.stdin)["digests"][sys.argv[1]]
if not isinstance(value, str) or len(value) != 64 or any(c not in string.hexdigits.lower() for c in value):
    raise SystemExit("stage output did not contain a lowercase SHA-256 digest")
print(value)' "$1"
}

PHASE2B1B_SELECT_JSON="$(scripts/phase2b1b/run_cloud.sh \
  "$PHASE2B1B_STORAGE_ROOT" "$PWD" select --confirm-cloud-storage \
  --source "$PHASE2B1B_SOURCE" --base-manifest "$PHASE2B1B_BASE_MANIFEST")"
PHASE2B1B_PRE_MANIFEST_SHA256="$(
  printf '%s\n' "$PHASE2B1B_SELECT_JSON" | phase2b1b_json_digest selection_manifest_sha256
)"
PHASE2B1B_PRE_MANIFEST="${PHASE2B1B_STORAGE_ROOT%/}/trustsr/phase2b1b/selections/${PHASE2B1B_PRE_MANIFEST_SHA256}/samples.jsonl"

PHASE2B1B_EXTRACT_JSON="$(scripts/phase2b1b/run_cloud.sh \
  "$PHASE2B1B_STORAGE_ROOT" "$PWD" extract --confirm-cloud-storage \
  --source "$PHASE2B1B_SOURCE" --selection-manifest "$PHASE2B1B_PRE_MANIFEST")"
PHASE2B1B_POST_MANIFEST_SHA256="$(
  printf '%s\n' "$PHASE2B1B_EXTRACT_JSON" | phase2b1b_json_digest selection_manifest_sha256
)"
PHASE2B1B_POST_MANIFEST="${PHASE2B1B_STORAGE_ROOT%/}/trustsr/phase2b1b/selections/${PHASE2B1B_POST_MANIFEST_SHA256}/samples.jsonl"

PHASE2B1B_AUDIT_JSON="$(scripts/phase2b1b/run_cloud.sh \
  "$PHASE2B1B_STORAGE_ROOT" "$PWD" audit --confirm-cloud-storage \
  --source "$PHASE2B1B_SOURCE" --selection-manifest "$PHASE2B1B_POST_MANIFEST")"
PHASE2B1B_AUDIT="${PHASE2B1B_STORAGE_ROOT%/}/trustsr/phase2b1b/audits/${PHASE2B1B_POST_MANIFEST_SHA256}/phase2b1b-audit.json"
printf 'Pre-extraction sidecar: %s\nPost-extraction sidecar: %s\nAudit: %s\n' \
  "$PHASE2B1B_PRE_MANIFEST" "$PHASE2B1B_POST_MANIFEST" "$PHASE2B1B_AUDIT"
```

State directly below the block that the instance can be stopped only after the audit file and the
Git-safe audit copy have both been verified.

- [ ] **Step 5: Run script, README, and full local quality gates**

Run:

```bash
uv run pytest tests/scripts/test_phase2b1b_scripts.py tests/scripts/test_phase2b1a_scripts.py -v
uv run pytest -q
uv run ruff check .
uv lock --check
git diff --check
```

Expected: all tests pass; Ruff reports `All checks passed!`; lock and diff checks exit 0.

- [ ] **Step 6: Verify repository data policy before commit**

Run:

```bash
test -z "$(find . -path ./.git -prune -o -type f \( -name '*.taco' -o -name '*.tif' -o -name '*.tiff' \) -print)"
test -z "$(git ls-files -z | xargs -0 -r stat --printf='%s %n\n' | awk '$1 > 1048576 {print}')"
```

Expected: both commands exit 0 with no output.

- [ ] **Step 7: Commit the runner and runbook**

```bash
git add scripts/phase2b1b/run_cloud.sh tests/scripts/test_phase2b1b_scripts.py README.md
git commit -m "docs: add phase2b1b cloud runbook"
```

---

### Task 7: Remote 360-Pair Gate and Git-Safe Audit

**Files:**
- Create after real gate: `artifacts/datasets/sen2naipv2-phase2b1b-audit-v1.json`
- Modify only if evidence requires correction: Phase 2B1B code/tests from Tasks 1–6.

**Interfaces:**
- Consumes: user-provided SSH endpoint at execution time, mounted persistent root, the exact feature-branch commit, existing Phase 2B1A source/manifest, and the Phase 2B1A base environment.
- Produces: verified cloud-only 360-pair data, a canonical real audit, a Git-safe audit copy, and a focused audit commit.

- [ ] **Step 1: Stop locally and request the cloud endpoint**

Tell the user that local implementation gates passed and that a running cloud instance is now required. Request only the current SSH endpoint if it is not already current. Do not request or store a password when key authentication is configured. Confirm `/root/rivermind-fs` (or the user-provided replacement) is an actual mountpoint before writing.

- [ ] **Step 2: Inspect remote state read-only before changing it**

Run over SSH with the current endpoint:

```bash
mountpoint -q /root/rivermind-fs
df -h /root/rivermind-fs
/opt/conda/bin/python --version
git -C /root/rivermind-fs/RemoteSensing001 status --short
git -C /root/rivermind-fs/RemoteSensing001 rev-parse HEAD
test -f /root/rivermind-fs/trustsr/phase2b1a/manifests/7487b0af2ebef86910e918d5d6b2fb927a6f5e46bac7c2e30be7ffb2ce994482/samples.jsonl
```

Expected: mount succeeds, more than 5 GiB is free, Python is 3.12, repository state is understood before update, and the frozen base manifest exists. Preserve unrelated remote changes.

- [ ] **Step 3: Update the remote checkout to the exact reviewed feature commit**

Fetch the feature branch and switch/fast-forward non-destructively. Record the resulting commit SHA. Do not use `git reset --hard`, do not delete remote data, and do not reinstall dependencies unless `/opt/conda/bin/python -c 'import tacoreader; print(tacoreader.__version__)'` fails to report `0.4.5`.

- [ ] **Step 4: Run `select` and verify the pre-extraction evidence**

Execute the README command with the mounted root. Verify canonical output reports:

```json
{"counts":{"subset_geotiffs":0,"subset_pairs":360},"stage":"select"}
```

Record the full selection manifest SHA-256 from `digests.selection_manifest_sha256`, verify its digest-addressed file, and run the same `select` command a second time. The second result must report the same digest and `reused=true`.

- [ ] **Step 5: Run restartable extraction and report progress without GPU claims**

Run `extract` using the pre-selection path. If the SSH session disconnects, reconnect and rerun the same command; complete pairs are reused and partial/invalid pairs stop for diagnosis. Do not delete or overwrite a failed pair without first identifying the exact cause and obtaining evidence. The successful output must report exactly 360 pairs and 720 GeoTIFFs.

- [ ] **Step 6: Run audit twice and compare canonical bytes**

Run `audit` against the post-extraction sidecar, then repeat it. Require the same audit SHA-256 and `reused=true` on the repeat. Independently verify:

```bash
find "$PHASE2B1B_STORAGE_ROOT/trustsr/phase2b1b/subset-v1" -type f -name '*.tif' | wc -l
sha256sum "$PHASE2B1B_STORAGE_ROOT/trustsr/phase2b1b/audits/$PHASE2B1B_POST_MANIFEST_SHA256/phase2b1b-audit.json"
```

Expected: `720` files and the SHA-256 printed by the CLI.

- [ ] **Step 7: Copy only the canonical audit payload into Git and validate policy**

Use a secure byte transfer or stdout capture to obtain only `phase2b1b-audit.json`; do not transfer the TACO, sidecar, or TIFF tree. Save the bytes as `artifacts/datasets/sen2naipv2-phase2b1b-audit-v1.json`, then run:

```bash
uv run python -m json.tool artifacts/datasets/sen2naipv2-phase2b1b-audit-v1.json >/dev/null
test "$(stat -c %s artifacts/datasets/sen2naipv2-phase2b1b-audit-v1.json)" -lt 1048576
git diff --check
uv run pytest -q
uv run ruff check .
uv lock --check
```

Expected: valid sub-1-MiB JSON, all tests pass, and all quality gates exit 0.

- [ ] **Step 8: Commit and push the real audit evidence**

```bash
git add artifacts/datasets/sen2naipv2-phase2b1b-audit-v1.json
git commit -m "data: record phase2b1b subset audit"
git push
```

After verifying the push and confirming no remote process is still extracting or hashing, tell the user the server is no longer needed and can be stopped through the provider console.

---

### Task 8: Review and Integration Gate

**Files:**
- No new implementation files; review all commits on `feature/phase2b1b-crosssensor-subset` against the approved spec.

**Interfaces:**
- Consumes: Tasks 1–7 and their fresh verification evidence.
- Produces: a reviewable GitHub pull request with explicit scientific and engineering evidence.

- [ ] **Step 1: Run final fresh verification on the exact branch tip**

```bash
uv run pytest -q
uv run ruff check .
uv lock --check
git diff --check main...HEAD
git status --short
```

Expected: tests and Ruff pass, lock/diff checks exit 0, and the working tree is clean.

- [ ] **Step 2: Inspect the complete change set and commit sequence**

```bash
git log --oneline main..HEAD
git diff --stat main...HEAD
git diff --name-status main...HEAD
```

Require only the approved specification, plan, selector, sidecar, Phase 2B1B CLI/tests/scripts/docs, console entry point, and Git-safe audit. Reject any credential, real pixel, full sidecar, model output, or unrelated refactor.

- [ ] **Step 3: Create the pull request without merging it implicitly**

Create a PR from `feature/phase2b1b-crosssensor-subset` to `main`. The description must include:

- exact source and base-manifest digests;
- 360/720 counts and 120-per-split/10-per-stratum gates;
- round-one preservation and 120 unique groups per split;
- local test/Ruff/lock results;
- cloud audit and post-manifest SHA-256 values;
- explicit statements that no GPU/model/metric ran and no real pixels entered Git.

- [ ] **Step 4: Request code review and address only evidence-backed findings**

Use `superpowers:requesting-code-review` before merge. Apply `superpowers:receiving-code-review` to any review feedback, rerun affected focused tests and the full gate, then update the PR. Merge only after the review and all checks pass.
