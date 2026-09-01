"""Verify pulled Phase 2B3-A JSON bundles without pixels, models, or CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from trustsr.jsonio import atomic_write_bytes, canonical_json

POST_MANIFEST_SHA256 = "c7f8ffa8415575d85daafe284a0796ec3f111442f0ac662f1d01311c4a851d4a"
INPUT_AUDIT_SHA256 = "fceb2ec04680ddf46bf4d0ed5a4a93edd33d58a09fc176d936bdef783114b44b"
_MAX_FILE_BYTES = 5 * 1024**2
_MANIFEST = "phase2b3a-bundle-manifest.json"
_PHASE_FILES = {
    "a1": (
        "phase2b3a-a1-result.json",
        "phase2b3a-a1-cache-audit.json",
        "phase2b3a-a1-runtime.json",
        "phase2b3a-a1-replay.json",
    ),
    "a2": (
        "phase2b3a-a2-result.json",
        "phase2b3a-a2-cache-audit.json",
        "phase2b3a-a2-runtime.json",
        "phase2b3a-a2-replay.json",
    ),
}
_OUTPUT_NAMES = {
    "a1": "sen2naipv2-development-smoke-acceptance-v1.json",
    "a2": "sen2naipv2-development-score-acceptance-v1.json",
}
_SECRET_KEY = re.compile(
    r"(?:^|_)(?:password|passwd|secret|token|api_key|private_key|ssh|hostname|host|username|user|env)(?:$|_)",
    re.IGNORECASE,
)
_HEX64 = re.compile(r"[0-9a-f]{64}")


def resolve_project_root() -> Path:
    """Resolve the reviewed checkout without importing GPU/model support."""

    for candidate in Path(__file__).resolve().parents:
        if (
            candidate != Path(candidate.anchor)
            and (candidate / "uv.lock").is_file()
            and not (candidate / "uv.lock").is_symlink()
            and (candidate / "pyproject.toml").is_file()
            and not (candidate / "pyproject.toml").is_symlink()
            and (candidate / "src" / "trustsr").is_dir()
            and not (candidate / "src" / "trustsr").is_symlink()
        ):
            return candidate
    raise ValueError("project root must identify the reviewed TrustSR checkout")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    for phase, handler in (("a1", verify_a1_bundle), ("a2", verify_a2_bundle)):
        child = subparsers.add_parser(phase)
        child.add_argument("--bundle", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        child.set_defaults(handler=handler)
    return parser


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_canonical(path: Path) -> tuple[bytes, dict[str, object]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"bundle file must be a regular non-symlink file: {path.name}")
    size = path.stat().st_size
    if size > _MAX_FILE_BYTES:
        raise ValueError(f"bundle file exceeds the 5 MiB limit: {path.name}")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"bundle file is not valid JSON: {path.name}") from exc
    if not isinstance(value, dict) or canonical_json(value) != raw:
        raise ValueError(f"bundle file is not canonical JSON: {path.name}")
    return raw, value


def _verify_allowlisted_bundle(
    bundle: Path, *, expected_phase: str
) -> dict[str, tuple[bytes, dict[str, object]]]:
    if expected_phase not in _PHASE_FILES:
        raise ValueError("unknown Phase 2B3-A bundle phase")
    if not isinstance(bundle, Path) or bundle.is_symlink() or not bundle.is_dir():
        raise ValueError("bundle must be an existing non-symlink directory")
    resolved = bundle.resolve(strict=True)
    if resolved != bundle.absolute():
        raise ValueError("bundle path must not contain symlink components")
    expected_names = {_MANIFEST, *_PHASE_FILES[expected_phase]}
    observed = {entry.name for entry in bundle.iterdir()}
    if observed != expected_names:
        raise ValueError("bundle must contain the exact allowlisted files; missing or extra file")
    manifest_raw, manifest = _read_canonical(bundle / _MANIFEST)
    if (
        manifest.get("schema") != "trustsr.phase2b3a-bundle-manifest.v1"
        or manifest.get("phase") != expected_phase
    ):
        raise ValueError("bundle manifest schema or phase is invalid")
    entries = manifest.get("files")
    if not isinstance(entries, list) or len(entries) != 4:
        raise ValueError("bundle manifest must contain exactly four file entries")
    entry_names: list[str] = []
    files: dict[str, tuple[bytes, dict[str, object]]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "basename",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("bundle manifest file entry schema is invalid")
        name = entry["basename"]
        size = entry["size_bytes"]
        digest = entry["sha256"]
        if (
            type(name) is not str
            or PurePosixPath(name).name != name
            or PureWindowsPath(name).name != name
            or name not in _PHASE_FILES[expected_phase]
            or type(size) is not int
            or size < 0
            or size > _MAX_FILE_BYTES
            or type(digest) is not str
            or _HEX64.fullmatch(digest) is None
        ):
            raise ValueError("bundle manifest contains an invalid basename, size, or digest")
        raw, value = _read_canonical(bundle / name)
        if len(raw) != size or _sha256(raw) != digest:
            raise ValueError("bundle file size or SHA-256 digest mismatch")
        entry_names.append(name)
        files[name] = (raw, value)
    if entry_names != sorted(_PHASE_FILES[expected_phase]) or len(set(entry_names)) != 4:
        raise ValueError("bundle manifest file allowlist/order is invalid")
    files[_MANIFEST] = (manifest_raw, manifest)
    for name, (_, value) in files.items():
        _reject_secrets_paths_and_leakage(value, label=name)
    return files


def _reject_secrets_paths_and_leakage(value: Any, *, label: str) -> None:
    def visit(item: Any, key: str | None = None) -> None:
        if isinstance(item, Mapping):
            for child_key, child in item.items():
                if type(child_key) is not str:
                    raise ValueError(f"{label} contains a non-string JSON key")
                if _SECRET_KEY.search(child_key):
                    raise ValueError(f"{label} contains a secret-like key")
                visit(child, child_key)
            return
        if isinstance(item, list):
            for child in item:
                visit(child, key)
            return
        if isinstance(item, str):
            lowered = item.lower()
            if "calibration" in lowered or "internal_test" in lowered:
                raise ValueError(f"{label} contains prohibited non-development leakage evidence")
            if (
                item.startswith(("/", "~", "file://", "ssh://"))
                or "\\" in item
                or ".." in PurePosixPath(item).parts
                or "-----begin" in lowered
                or ("@" in item and (":" in item or "." in item))
            ):
                raise ValueError(f"{label} contains a path-like or secret-like value")

    visit(value)


def _require_scientific_payload_is_runtime_free(result: Mapping[str, object]) -> None:
    prohibited = ("runtime", "duration", "timestamp", "gpu", "cuda", "path", "host")

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                lowered = str(key).lower()
                if key != "runtime_manifest_sha256" and any(
                    token in lowered for token in prohibited
                ):
                    raise ValueError("scientific result contains prohibited runtime fields")
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(result)


def _file(
    files: Mapping[str, tuple[bytes, dict[str, object]]], name: str
) -> tuple[bytes, dict[str, object]]:
    try:
        return files[name]
    except KeyError as exc:
        raise ValueError(f"bundle is missing required file: {name}") from exc


def _validate_common_references(
    result: Mapping[str, object], audit: Mapping[str, object]
) -> None:
    expected = {
        "post_manifest_sha256": POST_MANIFEST_SHA256,
        "input_audit_sha256": INPUT_AUDIT_SHA256,
    }
    if result.get("upstream") != expected or any(
        audit.get(key) != digest for key, digest in expected.items()
    ):
        raise ValueError("bundle upstream digest references are invalid")


def _validate_replay(
    replay: Mapping[str, object],
    *,
    phase: str,
    result_bytes: bytes,
    audit_bytes: bytes,
    runtime_bytes: bytes,
) -> None:
    if replay != {
        "schema": f"trustsr.phase2b3a-{phase}-replay.v1",
        "byte_identical": True,
        "result_sha256": _sha256(result_bytes),
        "cache_audit_sha256": _sha256(audit_bytes),
        "runtime_manifest_sha256": _sha256(runtime_bytes),
    }:
        raise ValueError("bundle replay receipt is invalid or not byte-identical")


def _bundle_digests(
    files: Mapping[str, tuple[bytes, dict[str, object]]]
) -> dict[str, str]:
    return {name: _sha256(raw) for name, (raw, _) in sorted(files.items())}


def _validate_file_evidence(value: object, cache_key: str) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("cache entry file evidence is incomplete")
    suffixes: set[str] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "filename",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("cache entry file evidence schema is invalid")
        filename = item["filename"]
        if type(filename) is not str or not filename.startswith(f"{cache_key}."):
            raise ValueError("cache file internal key reference is invalid")
        suffixes.add(filename.removeprefix(cache_key))
        if (
            type(item["size_bytes"]) is not int
            or not 0 <= item["size_bytes"] <= _MAX_FILE_BYTES
            or type(item["sha256"]) is not str
            or _HEX64.fullmatch(item["sha256"]) is None
        ):
            raise ValueError("cache file size or SHA-256 evidence is invalid")
    if suffixes != {".json", ".safetensors"}:
        raise ValueError("cache file evidence has invalid basenames")


def _validate_prediction_entry(
    entry: object,
    *,
    sample_id: object,
    model_name: str,
    seed: int | None,
    require_files: bool,
) -> dict[str, object]:
    if not isinstance(entry, dict) or not isinstance(entry.get("identity"), dict):
        raise ValueError("prediction cache entry schema is invalid")
    identity = entry["identity"]
    cache_key = _sha256(canonical_json(identity))
    digest = entry.get("prediction_sha256")
    if (
        entry.get("sample_id") != sample_id
        or entry.get("model_name") != model_name
        or entry.get("seed") != seed
        or entry.get("cache_key") != cache_key
        or type(digest) is not str
        or _HEX64.fullmatch(digest) is None
    ):
        raise ValueError("prediction cache internal identity/key/SHA reference is invalid")
    if require_files:
        _validate_file_evidence(entry.get("files"), cache_key)
    elif "files" in entry:
        raise ValueError("A1 prediction cache entry has unexpected file evidence")
    return entry


def _validate_score_entry(
    entry: object, *, sample_id: object, name: str
) -> dict[str, object]:
    if not isinstance(entry, dict) or not isinstance(entry.get("identity"), dict):
        raise ValueError("score cache entry schema is invalid")
    cache_key = _sha256(canonical_json(entry["identity"]))
    digest = entry.get("score_sha256")
    if (
        entry.get("sample_id") != sample_id
        or entry.get("name") != name
        or entry.get("cache_key") != cache_key
        or type(digest) is not str
        or _HEX64.fullmatch(digest) is None
    ):
        raise ValueError("score cache internal identity/key/SHA reference is invalid")
    _validate_file_evidence(entry.get("files"), cache_key)
    return entry


def _validate_sample_score_reference(
    score: object, audit_entry: Mapping[str, object], name: str
) -> None:
    if (
        not isinstance(score, dict)
        or score.get("name") != name
        or score.get("cache_key") != audit_entry.get("cache_key")
        or score.get("score_sha256") != audit_entry.get("score_sha256")
    ):
        raise ValueError("scientific score/cache audit internal references differ")


def _verify_a1_result_audit_runtime_replay(
    files: Mapping[str, tuple[bytes, dict[str, object]]],
) -> dict[str, object]:
    result_bytes, result = _file(files, _PHASE_FILES["a1"][0])
    audit_bytes, audit = _file(files, _PHASE_FILES["a1"][1])
    runtime_bytes, runtime = _file(files, _PHASE_FILES["a1"][2])
    _, replay = _file(files, _PHASE_FILES["a1"][3])
    if result.get("schema") != "trustsr.phase2b3a-development-smoke.v1":
        raise ValueError("A1 result schema is invalid")
    if audit.get("schema") != "trustsr.phase2b3a-development-smoke-cache-audit.v1":
        raise ValueError("A1 cache-audit schema is invalid")
    if audit.get("experiment_schema") != result.get("schema"):
        raise ValueError("A1 cache-audit experiment schema reference is invalid")
    if runtime.get("schema") != "trustsr.phase2b3a-a1-runtime.v1":
        raise ValueError("A1 runtime schema is invalid")
    _validate_common_references(result, audit)
    _require_scientific_payload_is_runtime_free(result)
    samples = result.get("samples")
    if (
        result.get("sample_count") != 4
        or result.get("prediction_count") != 108
        or result.get("score_count") != 20
        or audit.get("sample_count") != 4
        or audit.get("prediction_count") != 108
        or audit.get("score_count") != 20
        or not isinstance(samples, list)
        or len(samples) != 4
    ):
        raise ValueError("A1 result/audit ROI or stage counts are invalid")
    observed = []
    sample_ids: set[object] = set()
    groups: set[object] = set()
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("A1 sample schema is invalid")
        if sample.get("split", "development") != "development":
            raise ValueError("A1 contains non-development leakage evidence")
        observed.append(
            (
                sample.get("days_between"),
                sample.get("correlation_bin"),
                sample.get("selection_round"),
            )
        )
        sample_ids.add(sample.get("sample_id"))
        groups.add(sample.get("spatial_group_id"))
    if (
        observed != [(-1, index, 1) for index in range(4)]
        or len(sample_ids) != 4
        or len(groups) != 4
    ):
        raise ValueError("A1 canonical ROI bins or identities are invalid")
    prediction_entries = audit.get("prediction_entries")
    score_entries = audit.get("score_entries")
    if not isinstance(prediction_entries, list) or len(prediction_entries) != 108:
        raise ValueError("A1 cache audit prediction count is invalid")
    if not isinstance(score_entries, list) or len(score_entries) != 20:
        raise ValueError("A1 cache audit score count is invalid")
    prediction_slots = (
        ("bicubic-x4", None),
        ("sen2srlite-x4", None),
        *(("ldsr-s2-x4", seed) for seed in range(3407, 3432)),
    )
    score_names = (
        "ldsr_variance_k5a",
        "ldsr_variance_k5b",
        "ldsr_variance_k25",
        "lr_reprojection_l1",
        "three_model_disagreement",
    )
    for index, sample in enumerate(samples):
        start = index * len(prediction_slots)
        verified_predictions = [
            _validate_prediction_entry(
                entry,
                sample_id=sample["sample_id"],
                model_name=model_name,
                seed=seed,
                require_files=False,
            )
            for entry, (model_name, seed) in zip(
                prediction_entries[start : start + len(prediction_slots)],
                prediction_slots,
                strict=True,
            )
        ]
        if sample.get("central_prediction_sha256") != verified_predictions[2].get(
            "prediction_sha256"
        ):
            raise ValueError("A1 central prediction SHA reference is invalid")
        sample_scores = sample.get("scores")
        if not isinstance(sample_scores, list) or len(sample_scores) != 5:
            raise ValueError("A1 scientific score records are incomplete")
        score_start = index * len(score_names)
        for score, entry, name in zip(
            sample_scores,
            score_entries[score_start : score_start + len(score_names)],
            score_names,
            strict=True,
        ):
            verified_score = _validate_score_entry(
                entry, sample_id=sample["sample_id"], name=name
            )
            _validate_sample_score_reference(score, verified_score, name)
    runtime_sha256 = _sha256(runtime_bytes)
    if result.get("runtime_manifest_sha256") != runtime_sha256:
        raise ValueError("A1 runtime-manifest SHA reference is invalid")
    projection = runtime.get("projected_a2_uncached_seconds")
    missing = runtime.get("missing_a2_seed_predictions")
    median = runtime.get("a1_median_uncached_ldsr_prediction_seconds")
    if (
        runtime.get("single_repeatability_pass") is not True
        or type(missing) is not int
        or missing < 0
        or not isinstance(median, int | float)
        or not isinstance(projection, int | float)
        or float(projection) != float(median) * missing
    ):
        raise ValueError("A1 runtime repeatability or missing-seed projection is invalid")
    peak = runtime.get("single_peak_memory_bytes")
    total = runtime.get("gpu_total_memory_bytes")
    free = runtime.get("persistent_free_bytes")
    if any(type(value) is not int for value in (peak, total, free)):
        raise ValueError("A1 runtime resource measurements are invalid")
    expected_gate = bool(
        peak <= int(0.80 * total)
        and free >= 10 * 1024**3
        and 1.5 * float(projection) <= 2 * 60 * 60
    )
    if runtime.get("resource_gate_pass") is not expected_gate:
        raise ValueError("A1 runtime resource gate is inconsistent")
    _validate_replay(
        replay,
        phase="a1",
        result_bytes=result_bytes,
        audit_bytes=audit_bytes,
        runtime_bytes=runtime_bytes,
    )
    include = result.get("include_ldsr_variance_k5")
    stable = result.get("k5_statistically_stable")
    if type(include) is not bool or type(stable) is not bool or include is not stable:
        raise ValueError("A1 K=5 acceptance/stability decision is invalid")
    return {
        "schema": "trustsr.phase2b3a-development-smoke-acceptance.v1",
        "digests": _bundle_digests(files),
        "bundle_integrity_pass": True,
        "replay_pass": True,
        "repeatability_pass": True,
        "resource_gate_pass": expected_gate,
        "include_ldsr_variance_k5": include,
    }


def _validate_a2_samples(
    result: Mapping[str, object],
    audit: Mapping[str, object],
    *,
    include_ldsr_variance_k5: bool,
) -> None:
    samples = result.get("samples")
    groups = audit.get("groups")
    if not isinstance(samples, list) or len(samples) != 120:
        raise ValueError("A2 result must contain exactly 120 ROI")
    if not isinstance(groups, list) or len(groups) != 120:
        raise ValueError("A2 cache audit must contain exactly 120 ROI groups")
    sample_ids: set[object] = set()
    spatial_groups: set[object] = set()
    cells: Counter[tuple[object, object]] = Counter()
    rounds: dict[tuple[object, object], list[object]] = {}
    keys = (
        "sample_id",
        "spatial_group_id",
        "days_between",
        "correlation_bin",
        "selection_round",
    )
    for sample, group in zip(samples, groups, strict=True):
        if not isinstance(sample, dict) or not isinstance(group, dict):
            raise ValueError("A2 ROI or audit group schema is invalid")
        if sample.get("split") != "development":
            raise ValueError("A2 contains non-development leakage evidence")
        if tuple(sample.get(key) for key in keys) != tuple(group.get(key) for key in keys):
            raise ValueError("A2 result/audit ROI strata references do not match")
        sample_ids.add(sample.get("sample_id"))
        spatial_groups.add(sample.get("spatial_group_id"))
        cell = (sample.get("days_between"), sample.get("correlation_bin"))
        cells[cell] += 1
        rounds.setdefault(cell, []).append(sample.get("selection_round"))
        prediction_slots = (
            ("bicubic-x4", None),
            ("sen2srlite-x4", None),
            *(("ldsr-s2-x4", seed) for seed in (
                range(3407, 3412) if include_ldsr_variance_k5 else (3407,)
            )),
        )
        score_names = (
            "lr_reprojection_l1",
            "three_model_disagreement",
            *(("ldsr_variance_k5",) if include_ldsr_variance_k5 else ()),
        )
        prediction_entries = group.get("prediction_entries")
        score_entries = group.get("score_entries")
        sample_scores = sample.get("scores")
        if (
            not isinstance(prediction_entries, list)
            or len(prediction_entries) != len(prediction_slots)
            or not isinstance(score_entries, list)
            or len(score_entries) != len(score_names)
            or not isinstance(sample_scores, list)
            or len(sample_scores) != len(score_names)
        ):
            raise ValueError("A2 cache group prediction/score counts are incomplete")
        verified_predictions = [
            _validate_prediction_entry(
                entry,
                sample_id=sample["sample_id"],
                model_name=model_name,
                seed=seed,
                require_files=True,
            )
            for entry, (model_name, seed) in zip(
                prediction_entries, prediction_slots, strict=True
            )
        ]
        if sample.get("central_prediction_sha256") != verified_predictions[2].get(
            "prediction_sha256"
        ):
            raise ValueError("A2 central prediction SHA reference is invalid")
        for score, entry, name in zip(
            sample_scores, score_entries, score_names, strict=True
        ):
            verified_score = _validate_score_entry(
                entry, sample_id=sample["sample_id"], name=name
            )
            _validate_sample_score_reference(score, verified_score, name)
    expected_cells = {(day, bin_index) for day in (-1, 0, 1) for bin_index in range(4)}
    if (
        len(sample_ids) != 120
        or len(spatial_groups) != 120
        or set(cells) != expected_cells
        or any(cells[cell] != 10 for cell in expected_cells)
        or any(sorted(rounds[cell]) != list(range(1, 11)) for cell in expected_cells)
    ):
        raise ValueError("A2 ROI identities or 12x10 strata are invalid")


def _verify_a2_result_audit_runtime_replay(
    files: Mapping[str, tuple[bytes, dict[str, object]]],
) -> dict[str, object]:
    result_bytes, result = _file(files, _PHASE_FILES["a2"][0])
    audit_bytes, audit = _file(files, _PHASE_FILES["a2"][1])
    runtime_bytes, runtime = _file(files, _PHASE_FILES["a2"][2])
    _, replay = _file(files, _PHASE_FILES["a2"][3])
    if result.get("schema") != "trustsr.phase2b3a-development-score-audit.v1":
        raise ValueError("A2 result schema is invalid")
    if audit.get("schema") != "trustsr.phase2b3a-development-score-cache-audit.v1":
        raise ValueError("A2 cache-audit schema is invalid")
    if audit.get("experiment_schema") != result.get("schema"):
        raise ValueError("A2 cache-audit experiment schema reference is invalid")
    if runtime.get("schema") != "trustsr.phase2b3a-a2-runtime.v1":
        raise ValueError("A2 runtime schema is invalid")
    _validate_common_references(result, audit)
    _require_scientific_payload_is_runtime_free(result)
    if result.get("sample_count") != 120 or audit.get("sample_count") != 120:
        raise ValueError("A2 result/audit must report exactly 120 ROI")
    include = result.get("include_ldsr_variance_k5")
    if type(include) is not bool:
        raise ValueError("A2 K=5 inclusion decision is invalid")
    expected_predictions = 120 * (7 if include else 3)
    expected_scores = 120 * (3 if include else 2)
    if (
        result.get("prediction_count") != expected_predictions
        or audit.get("prediction_count") != expected_predictions
        or result.get("score_count") != expected_scores
        or audit.get("score_count") != expected_scores
    ):
        raise ValueError("A2 result/audit stage counts differ")
    if result.get("code_revision") != audit.get("code_revision") or result.get(
        "code_revision"
    ) != runtime.get("git_commit"):
        raise ValueError("A2 code-revision references differ")
    bootstrap = result.get("bootstrap")
    if bootstrap != {
        "algorithm": "numpy.PCG64",
        "seed": 23031,
        "resamples": 10_000,
        "ci_percentiles": [2.5, 97.5],
    }:
        raise ValueError("A2 bootstrap schema is invalid")
    _validate_a2_samples(result, audit, include_ldsr_variance_k5=include)
    runtime_sha256 = _sha256(runtime_bytes)
    if result.get("runtime_manifest_sha256") != runtime_sha256:
        raise ValueError("A2 runtime-manifest SHA reference is invalid")
    if runtime.get("a1_acceptance_pass") is not True or runtime.get("sample_count") != 120:
        raise ValueError("A2 runtime acceptance evidence is invalid")
    _validate_replay(
        replay,
        phase="a2",
        result_bytes=result_bytes,
        audit_bytes=audit_bytes,
        runtime_bytes=runtime_bytes,
    )
    frozen = result.get("frozen_score")
    decision = result.get("phase_decision")
    acceptance: dict[str, object] = {
        "schema": "trustsr.phase2b3a-development-score-acceptance.v1",
        "digests": _bundle_digests(files),
        "bundle_integrity_pass": True,
        "replay_pass": True,
        "development_only_pass": True,
    }
    if decision == "freeze_score" and isinstance(frozen, dict):
        acceptance["frozen_score"] = frozen
        acceptance["no_eligible_score"] = False
    elif decision == "stop_no_eligible_score" and frozen is None:
        acceptance["frozen_score"] = None
        acceptance["no_eligible_score"] = True
    else:
        raise ValueError("A2 frozen-score/no-eligible decision is invalid")
    return acceptance


def verify_a1_bundle(bundle: Path) -> dict[str, object]:
    files = _verify_allowlisted_bundle(bundle, expected_phase="a1")
    return _verify_a1_result_audit_runtime_replay(files)


def verify_a2_bundle(bundle: Path) -> dict[str, object]:
    files = _verify_allowlisted_bundle(bundle, expected_phase="a2")
    return _verify_a2_result_audit_runtime_replay(files)


def _validate_output_path(output: Path, *, phase: str) -> Path:
    repository = resolve_project_root()
    expected = repository / "artifacts" / "phase2b3a" / _OUTPUT_NAMES[phase]
    if output.absolute() != expected.absolute():
        raise ValueError("acceptance output must use the declared artifacts/phase2b3a basename")
    current = repository
    for component in expected.relative_to(repository).parts[:-1]:
        current = current / component
        if current.is_symlink():
            raise ValueError("acceptance output path must not contain symlink components")
        if current.exists() and not current.is_dir():
            raise ValueError("acceptance output parent must be a directory")
    if expected.is_symlink() or (expected.exists() and not expected.is_file()):
        raise ValueError("acceptance output must be a regular file")
    return expected


def _commit_acceptance(path: Path, acceptance: Mapping[str, object]) -> None:
    payload = canonical_json(dict(acceptance))
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("existing acceptance output has different bytes")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, payload)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = _validate_output_path(args.output, phase=args.phase)
    acceptance = args.handler(args.bundle)
    _commit_acceptance(output, acceptance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
