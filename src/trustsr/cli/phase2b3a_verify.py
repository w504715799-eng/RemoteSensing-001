"""Verify pulled Phase 2B3-A JSON bundles without pixels, models, or CUDA."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import stat
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
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
    r"(?:^|_)(?:password|passwd|secret|token|api_key|access_key|credential|credentials|authorization|auth|private_key|ssh|hostname|host|username|user|env|environment)(?:$|_)",
    re.IGNORECASE,
)
_HEX64 = re.compile(r"[0-9a-f]{64}")
_SOURCE = f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}"
_PREDICTION_COMMON = {
    "name",
    "scale",
    "experiment_schema",
    "post_manifest_sha256",
    "input_audit_sha256",
    "implementation_schema_version",
    "output_policy",
    "torch_version",
}
_PROVENANCE_KEYS = {
    "bicubic-x4": _PREDICTION_COMMON | {"implementation", "mode", "align_corners", "antialias"},
    "sen2srlite-x4": _PREDICTION_COMMON
    | {
        "model_id",
        "manifest_url",
        "mlstac_version",
        "sen2sr_version",
        "device",
        "asset_sha256:example_data.safetensor",
        "asset_sha256:hard_constraint.safetensor",
        "asset_sha256:load.py",
        "asset_sha256:mlm.json",
        "asset_sha256:model.safetensor",
    },
    "ldsr-s2-x4": _PREDICTION_COMMON
    | {
        "opensr_model_version",
        "cuda_runtime",
        "checkpoint_name",
        "checkpoint_url",
        "checkpoint_size",
        "checkpoint_sha256",
        "config_sha256",
        "device",
        "seed",
        "sampling_steps",
        "sampling_eta",
        "sampling_temperature",
        "histogram_matching",
    },
}
_DIAGNOSTIC_KEYS = {
    "rho",
    "constant_score",
    "coverages",
    "selective_mean_risks",
    "aurc",
    "random_aurc",
    "aurc_gain",
    "high_risk_miss_rate_at_80",
}
_SEN2SRLITE_ASSETS = {
    "asset_sha256:example_data.safetensor": (
        "c895c7da8a8d48882b73a2a1955e4260714b97540eea290229a284d73f129985"
    ),
    "asset_sha256:hard_constraint.safetensor": (
        "fbad981519066387c413ead1d6af7ef3e0d2947c34147ba90163fc79ae539239"
    ),
    "asset_sha256:load.py": "4b6c836b1f73078c62c84d4374b2d8daee5345f6239f64e0b6be29432383bac6",
    "asset_sha256:mlm.json": "59caa5c6af96a6fbebdbd771d93c91cc2d3a770302cd2f262b5409e77a40e3f7",
    "asset_sha256:model.safetensor": (
        "479aa796d5068d0b1206118ccbca27bd3223df0214db1a9b31a1e18349ed1c7e"
    ),
}


def _require_exact_keys(value: object, keys: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} schema is invalid")
    return value


def _is_sha(value: object) -> bool:
    return type(value) is str and _HEX64.fullmatch(value) is not None


def _score_configuration(name: str) -> dict[str, object]:
    if name in {"ldsr_variance_k5a", "ldsr_variance_k5"}:
        first, last = 3407, 3411
    elif name == "ldsr_variance_k5b":
        first, last = 3412, 3416
    elif name == "ldsr_variance_k25":
        first, last = 3407, 3431
    elif name == "lr_reprojection_l1":
        return {
            "algorithm": "lr_reprojection_l1_score",
            "downsample_mode": "area",
            "scale": 4,
            "upsample_mode": "repeat_interleave",
        }
    elif name == "three_model_disagreement":
        return {
            "algorithm": "three_model_disagreement_score",
            "band_reduction": "mean",
            "correction": 0,
            "model_order": "bicubic-x4,sen2srlite-x4,ldsr-s2-x4",
        }
    else:
        raise ValueError("unknown score name")
    return {
        "algorithm": "ensemble_variance_score",
        "band_reduction": "mean",
        "correction": 0,
        "seed_first": first,
        "seed_last": last,
        "seed_count": last - first + 1,
    }


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
            authorized_slash = (key == "source" and item == _SOURCE) or (
                key in {"manifest_url", "checkpoint_url"}
                and item.startswith("https://")
                and ".." not in PurePosixPath(item).parts
            )
            if "calibration" in lowered or "internal_test" in lowered:
                raise ValueError(f"{label} contains prohibited non-development leakage evidence")
            if (
                item.startswith(("/", "~", "file://", "ssh://"))
                or "\\" in item
                or ("/" in item and not authorized_slash)
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


def _validate_common_references(result: Mapping[str, object], audit: Mapping[str, object]) -> None:
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


def _bundle_digests(files: Mapping[str, tuple[bytes, dict[str, object]]]) -> dict[str, str]:
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
    lr_sha256: object,
    correlation_bin: object,
) -> dict[str, object]:
    entry_keys = {"sample_id", "model_name", "seed", "cache_key", "identity", "prediction_sha256"}
    if model_name == "bicubic-x4" or model_name == "sen2srlite-x4" or model_name == "ldsr-s2-x4":
        entry_keys.add("correlation_bin") if not require_files else entry_keys.add("files")
    entry = _require_exact_keys(entry, entry_keys, "prediction cache entry")
    identity = entry["identity"]
    identity = _require_exact_keys(
        identity,
        {"model_provenance", "source", "sample_id", "lr"},
        "prediction identity",
    )
    lr = _require_exact_keys(identity["lr"], {"shape", "dtype", "sha256"}, "prediction LR identity")
    provenance = _require_exact_keys(
        identity["model_provenance"], _PROVENANCE_KEYS[model_name], "prediction provenance"
    )
    cache_key = _sha256(canonical_json(identity))
    digest = entry.get("prediction_sha256")
    if (
        entry.get("sample_id") != sample_id
        or entry.get("model_name") != model_name
        or entry.get("seed") != seed
        or entry.get("cache_key") != cache_key
        or (not require_files and entry.get("correlation_bin") != correlation_bin)
        or type(digest) is not str
        or _HEX64.fullmatch(digest) is None
        or identity["source"] != _SOURCE
        or identity["sample_id"] != sample_id
        or lr != {"shape": [4, 128, 128], "dtype": "torch.float32", "sha256": lr_sha256}
        or provenance.get("name") != model_name
        or provenance.get("scale") != 4
        or provenance.get("experiment_schema") != "trustsr.phase2b3a-predictions.v1"
        or provenance.get("post_manifest_sha256") != POST_MANIFEST_SHA256
        or provenance.get("input_audit_sha256") != INPUT_AUDIT_SHA256
        or provenance.get("implementation_schema_version") != 1
        or provenance.get("output_policy") != "clip_to_[0,1]"
        or type(provenance.get("torch_version")) is not str
    ):
        raise ValueError("prediction cache internal identity/key/SHA reference is invalid")
    if model_name == "bicubic-x4" and any(
        provenance.get(key) != value
        for key, value in {
            "implementation": "torch.nn.functional.interpolate",
            "mode": "bicubic",
            "align_corners": False,
            "antialias": True,
        }.items()
    ):
        raise ValueError("prediction provenance is not the frozen bicubic adapter")
    if model_name == "sen2srlite-x4":
        if (
            provenance.get("device") != "cpu"
            or provenance.get("model_id") != "SEN2SRLite_NonReference_RGBN_x4"
            or provenance.get("manifest_url")
            != (
                "https://huggingface.co/tacofoundation/sen2sr/resolve/main/"
                "SEN2SRLite/NonReference_RGBN_x4/mlm.json"
            )
            or any(
                type(provenance.get(key)) is not str or not provenance.get(key)
                for key in ("mlstac_version", "sen2sr_version")
            )
            or any(provenance.get(key) != digest for key, digest in _SEN2SRLITE_ASSETS.items())
        ):
            raise ValueError("prediction provenance is not the frozen SEN2SRLite adapter")
    if model_name == "ldsr-s2-x4" and (
        provenance.get("seed") != seed
        or provenance.get("device") != "cuda"
        or provenance.get("opensr_model_version") != "1.1.1"
        or type(provenance.get("cuda_runtime")) is not str
        or not provenance.get("cuda_runtime")
        or provenance.get("checkpoint_name") != "opensr-ldsrs2_v1_0_0.ckpt"
        or provenance.get("checkpoint_url")
        != "https://huggingface.co/simon-donike/RS-SR-LTDF/resolve/main/opensr-ldsrs2_v1_0_0.ckpt"
        or provenance.get("checkpoint_size") != 1_130_715_795
        or provenance.get("checkpoint_sha256")
        != "e2621e3912eb7c14867c3d20c9029607ba941be8e166dc09621860fcac27dc3a"
        or provenance.get("config_sha256")
        != "ac76685d354bfec32e3e0641aef574bedd7d650402c97dbd0ade86304e69ca6f"
        or provenance.get("sampling_steps") != 100
        or provenance.get("sampling_eta") != 0.95
        or provenance.get("sampling_temperature") != 1.0
        or provenance.get("histogram_matching") is not True
    ):
        raise ValueError("prediction provenance is not the frozen LDSR adapter")
    if require_files:
        _validate_file_evidence(entry.get("files"), cache_key)
    elif "files" in entry:
        raise ValueError("A1 prediction cache entry has unexpected file evidence")
    return entry


def _validate_score_entry(
    entry: object,
    *,
    sample_id: object,
    name: str,
    lr_sha256: object,
    correlation_bin: object,
    input_sha256s: list[object],
) -> dict[str, object]:
    entry = _require_exact_keys(
        entry,
        {"sample_id", "correlation_bin", "name", "cache_key", "identity", "score_sha256", "files"},
        "score cache entry",
    )
    identity = _require_exact_keys(
        entry["identity"],
        {"score_name", "score_schema_version", "sample_id", "input_sha256s", "operator_parameters"},
        "score identity",
    )
    cache_key = _sha256(canonical_json(identity))
    digest = entry.get("score_sha256")
    expected_parameters = {**_score_configuration(name), "lr_sha256": lr_sha256}
    if (
        entry.get("sample_id") != sample_id
        or entry.get("name") != name
        or entry.get("correlation_bin") != correlation_bin
        or entry.get("cache_key") != cache_key
        or type(digest) is not str
        or _HEX64.fullmatch(digest) is None
        or identity.get("score_name") != name
        or identity.get("score_schema_version") != 1
        or identity.get("sample_id") != sample_id
        or identity.get("input_sha256s") != input_sha256s
        or identity.get("operator_parameters") != expected_parameters
    ):
        raise ValueError("score cache internal identity/key/SHA reference is invalid")
    _validate_file_evidence(entry.get("files"), cache_key)
    return entry


def _validate_sample_score_reference(
    score: object, audit_entry: Mapping[str, object], name: str
) -> None:
    score = _require_exact_keys(
        score,
        {"name", "cache_key", "score_sha256", "primary_window_9", "sensitivity_window_1"},
        "scientific score record",
    )
    for key in ("primary_window_9", "sensitivity_window_1"):
        diagnostic = _require_exact_keys(score[key], _DIAGNOSTIC_KEYS, "score diagnostic")
        coverages = diagnostic["coverages"]
        risks = diagnostic["selective_mean_risks"]
        if (
            type(diagnostic["constant_score"]) is not bool
            or any(
                not isinstance(diagnostic[field], int | float)
                or not math.isfinite(float(diagnostic[field]))
                for field in (
                    "rho",
                    "aurc",
                    "random_aurc",
                    "aurc_gain",
                    "high_risk_miss_rate_at_80",
                )
            )
            or not isinstance(coverages, list)
            or not isinstance(risks, list)
            or coverages != [index / 10 for index in range(1, 11)]
            or len(risks) != 10
            or any(
                not isinstance(value, int | float) or not math.isfinite(float(value))
                for value in [*coverages, *risks]
            )
        ):
            raise ValueError("score diagnostic values are invalid")
    if (
        score.get("name") != name
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
    _require_exact_keys(
        result,
        {
            "schema",
            "dataset_role",
            "upstream",
            "bands",
            "scale",
            "sample_count",
            "prediction_count",
            "score_count",
            "seed_sets",
            "stability_thresholds",
            "k5_statistically_stable",
            "include_ldsr_variance_k5",
            "samples",
            "runtime_manifest_sha256",
        },
        "A1 result",
    )
    _require_exact_keys(
        audit,
        {
            "schema",
            "experiment_schema",
            "post_manifest_sha256",
            "input_audit_sha256",
            "sample_count",
            "prediction_count",
            "score_count",
            "prediction_entries",
            "score_entries",
        },
        "A1 audit",
    )
    _require_exact_keys(
        runtime,
        {
            "schema",
            "git_commit",
            "single_repeatability_pass",
            "single_peak_memory_bytes",
            "gpu_total_memory_bytes",
            "persistent_free_bytes",
            "a1_uncached_ldsr_prediction_seconds",
            "a1_median_uncached_ldsr_prediction_seconds",
            "missing_a2_seed_predictions",
            "projected_a2_uncached_seconds",
            "resource_gate_pass",
        },
        "A1 runtime",
    )
    if result.get("schema") != "trustsr.phase2b3a-development-smoke.v1":
        raise ValueError("A1 result schema is invalid")
    if audit.get("schema") != "trustsr.phase2b3a-development-smoke-cache-audit.v1":
        raise ValueError("A1 cache-audit schema is invalid")
    if audit.get("experiment_schema") != result.get("schema"):
        raise ValueError("A1 cache-audit experiment schema reference is invalid")
    if runtime.get("schema") != "trustsr.phase2b3a-a1-runtime.v1":
        raise ValueError("A1 runtime schema is invalid")
    if (
        result.get("dataset_role") != "development_engineering_smoke_only"
        or result.get("bands") != ["B04", "B03", "B02", "B08"]
        or result.get("scale") != 4
        or result.get("seed_sets")
        != {
            "k5a": list(range(3407, 3412)),
            "k5b": list(range(3412, 3417)),
            "k25": list(range(3407, 3432)),
        }
        or result.get("stability_thresholds")
        != {
            "k5a_k5b_median_minimum": 0.6,
            "k5a_k5b_worst_minimum": 0.4,
            "k5a_k25_median_minimum": 0.8,
            "k5a_k25_worst_minimum": 0.6,
            "k5a_k25_top10_jaccard_median_minimum": 0.5,
        }
    ):
        raise ValueError("A1 frozen experiment configuration is invalid")
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
        sample = _require_exact_keys(
            sample,
            {
                "sample_id",
                "spatial_group_id",
                "correlation_bin",
                "days_between",
                "selection_round",
                "lr_tensor_sha256",
                "hr_tensor_sha256",
                "central_prediction_sha256",
                "risks",
                "scores",
                "stability",
            },
            "A1 sample",
        )
        risks = _require_exact_keys(sample["risks"], {"primary", "sensitivity"}, "A1 risks")
        if risks != {
            "primary": {
                "name": "local_l1_risk",
                "window": 9,
                "risk_sha256": risks["primary"].get("risk_sha256")
                if isinstance(risks["primary"], dict)
                else None,
            },
            "sensitivity": {
                "name": "local_l1_risk",
                "window": 1,
                "risk_sha256": risks["sensitivity"].get("risk_sha256")
                if isinstance(risks["sensitivity"], dict)
                else None,
            },
        } or not all(_is_sha(risks[key]["risk_sha256"]) for key in risks):
            raise ValueError("A1 risk schema is invalid")
        stability = _require_exact_keys(
            sample["stability"],
            {
                "k5a_k5b_spearman",
                "k5a_k25_spearman",
                "k5a_k25_top10_jaccard",
                "k5a_constant_score",
                "k5b_constant_score",
                "k25_constant_score",
            },
            "A1 stability",
        )
        if any(
            not isinstance(stability[key], int | float)
            or isinstance(stability[key], bool)
            or not math.isfinite(float(stability[key]))
            for key in (
                "k5a_k5b_spearman",
                "k5a_k25_spearman",
                "k5a_k25_top10_jaccard",
            )
        ) or any(
            type(stability[key]) is not bool
            for key in ("k5a_constant_score", "k5b_constant_score", "k25_constant_score")
        ):
            raise ValueError("A1 stability evidence is invalid")
        if not all(
            _is_sha(sample[key])
            for key in ("lr_tensor_sha256", "hr_tensor_sha256", "central_prediction_sha256")
        ):
            raise ValueError("A1 sample tensor SHA evidence is invalid")
        if (
            type(sample["sample_id"]) is not str
            or not sample["sample_id"]
            or type(sample["spatial_group_id"]) is not str
            or not sample["spatial_group_id"]
            or type(sample["days_between"]) is not int
            or type(sample["correlation_bin"]) is not int
            or type(sample["selection_round"]) is not int
        ):
            raise ValueError("A1 sample membership types are invalid")
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
                lr_sha256=sample.get("lr_tensor_sha256"),
                correlation_bin=sample.get("correlation_bin"),
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
        prediction_hashes = [item["prediction_sha256"] for item in verified_predictions]
        score_inputs = (
            prediction_hashes[2:7],
            prediction_hashes[7:12],
            prediction_hashes[2:27],
            [prediction_hashes[2]],
            [prediction_hashes[0], prediction_hashes[1], prediction_hashes[2]],
        )
        for score, entry, name, inputs in zip(
            sample_scores,
            score_entries[score_start : score_start + len(score_names)],
            score_names,
            score_inputs,
            strict=True,
        ):
            verified_score = _validate_score_entry(
                entry,
                sample_id=sample["sample_id"],
                name=name,
                lr_sha256=sample.get("lr_tensor_sha256"),
                input_sha256s=inputs,
                correlation_bin=sample.get("correlation_bin"),
            )
            _validate_sample_score_reference(score, verified_score, name)
    runtime_sha256 = _sha256(runtime_bytes)
    if result.get("runtime_manifest_sha256") != runtime_sha256:
        raise ValueError("A1 runtime-manifest SHA reference is invalid")
    projection = runtime.get("projected_a2_uncached_seconds")
    missing = runtime.get("missing_a2_seed_predictions")
    median = runtime.get("a1_median_uncached_ldsr_prediction_seconds")
    durations = runtime.get("a1_uncached_ldsr_prediction_seconds")
    if (
        runtime.get("single_repeatability_pass") is not True
        or type(missing) is not int
        or missing < 0
        or not isinstance(median, int | float)
        or not isinstance(projection, int | float)
        or not math.isfinite(float(median))
        or not math.isfinite(float(projection))
        or float(median) < 0.0
        or float(projection) < 0.0
        or float(projection) != float(median) * missing
        or not isinstance(durations, list)
        or not durations
        or any(
            not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in durations
        )
        or float(median) != float(statistics.median(float(value) for value in durations))
    ):
        raise ValueError("A1 runtime repeatability or missing-seed projection is invalid")
    peak = runtime.get("single_peak_memory_bytes")
    total = runtime.get("gpu_total_memory_bytes")
    free = runtime.get("persistent_free_bytes")
    if (
        type(runtime.get("git_commit")) is not str
        or re.fullmatch(r"[0-9a-f]{40}", runtime["git_commit"]) is None
    ):
        raise ValueError("A1 runtime Git commit is invalid")
    if any(type(value) is not int or value < 0 for value in (peak, total, free)) or total == 0:
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
        sample = _require_exact_keys(
            sample,
            {
                "sample_id",
                "spatial_group_id",
                "split",
                "days_between",
                "correlation_bin",
                "selection_round",
                "lr_tensor_sha256",
                "hr_tensor_sha256",
                "central_prediction_sha256",
                "risks",
                "scores",
            },
            "A2 ROI",
        )
        group = _require_exact_keys(
            group,
            {
                "sample_id",
                "spatial_group_id",
                "days_between",
                "correlation_bin",
                "selection_round",
                "prediction_entries",
                "score_entries",
            },
            "A2 audit group",
        )
        risks = _require_exact_keys(
            sample["risks"], {"primary_window_9", "sensitivity_window_1"}, "A2 risks"
        )
        for risk_name, window in (("primary_window_9", 9), ("sensitivity_window_1", 1)):
            risk = _require_exact_keys(
                risks[risk_name], {"name", "window", "risk_sha256"}, "A2 risk"
            )
            if risk != {
                "name": "local_l1_risk",
                "window": window,
                "risk_sha256": risk["risk_sha256"],
            } or not _is_sha(risk["risk_sha256"]):
                raise ValueError("A2 risk evidence is invalid")
        if not all(
            _is_sha(sample[key])
            for key in ("lr_tensor_sha256", "hr_tensor_sha256", "central_prediction_sha256")
        ):
            raise ValueError("A2 sample tensor SHA evidence is invalid")
        if (
            type(sample["sample_id"]) is not str
            or not sample["sample_id"]
            or type(sample["spatial_group_id"]) is not str
            or not sample["spatial_group_id"]
            or type(sample["days_between"]) is not int
            or type(sample["correlation_bin"]) is not int
            or type(sample["selection_round"]) is not int
        ):
            raise ValueError("A2 sample membership types are invalid")
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
            *(
                ("ldsr-s2-x4", seed)
                for seed in (range(3407, 3412) if include_ldsr_variance_k5 else (3407,))
            ),
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
                lr_sha256=sample.get("lr_tensor_sha256"),
                correlation_bin=sample.get("correlation_bin"),
            )
            for entry, (model_name, seed) in zip(prediction_entries, prediction_slots, strict=True)
        ]
        if sample.get("central_prediction_sha256") != verified_predictions[2].get(
            "prediction_sha256"
        ):
            raise ValueError("A2 central prediction SHA reference is invalid")
        hashes = [item["prediction_sha256"] for item in verified_predictions]
        inputs_by_name = {
            "lr_reprojection_l1": [hashes[2]],
            "three_model_disagreement": [hashes[0], hashes[1], hashes[2]],
            "ldsr_variance_k5": hashes[2:7],
        }
        for score, entry, name in zip(sample_scores, score_entries, score_names, strict=True):
            verified_score = _validate_score_entry(
                entry,
                sample_id=sample["sample_id"],
                name=name,
                lr_sha256=sample.get("lr_tensor_sha256"),
                correlation_bin=sample.get("correlation_bin"),
                input_sha256s=inputs_by_name[name],
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
    _require_exact_keys(
        result,
        {
            "schema",
            "dataset_role",
            "upstream",
            "code_revision",
            "bands",
            "scale",
            "sample_count",
            "statistical_unit",
            "prediction_count",
            "score_count",
            "include_ldsr_variance_k5",
            "candidate_names",
            "score_configuration",
            "risk_configuration",
            "bootstrap",
            "selection_risk",
            "candidate_summaries",
            "frozen_score",
            "phase_decision",
            "samples",
            "runtime_manifest_sha256",
        },
        "A2 result",
    )
    _require_exact_keys(
        audit,
        {
            "schema",
            "experiment_schema",
            "post_manifest_sha256",
            "input_audit_sha256",
            "code_revision",
            "sample_count",
            "prediction_count",
            "score_count",
            "groups",
        },
        "A2 audit",
    )
    _require_exact_keys(
        runtime,
        {
            "schema",
            "git_commit",
            "a1_acceptance_pass",
            "a1_replay_sha256",
            "sample_count",
        },
        "A2 runtime",
    )
    if result.get("schema") != "trustsr.phase2b3a-development-score-audit.v1":
        raise ValueError("A2 result schema is invalid")
    if audit.get("schema") != "trustsr.phase2b3a-development-score-cache-audit.v1":
        raise ValueError("A2 cache-audit schema is invalid")
    if audit.get("experiment_schema") != result.get("schema"):
        raise ValueError("A2 cache-audit experiment schema reference is invalid")
    if runtime.get("schema") != "trustsr.phase2b3a-a2-runtime.v1":
        raise ValueError("A2 runtime schema is invalid")
    if (
        result.get("dataset_role") != "development_score_selection_only"
        or result.get("bands") != ["B04", "B03", "B02", "B08"]
        or result.get("scale") != 4
        or result.get("statistical_unit") != "roi"
        or result.get("risk_configuration")
        != {
            "name": "local_l1_risk",
            "primary_window": 9,
            "sensitivity_window": 1,
        }
        or result.get("selection_risk") != "primary_window_9"
    ):
        raise ValueError("A2 frozen experiment configuration is invalid")
    _validate_common_references(result, audit)
    _require_scientific_payload_is_runtime_free(result)
    if result.get("sample_count") != 120 or audit.get("sample_count") != 120:
        raise ValueError("A2 result/audit must report exactly 120 ROI")
    include = result.get("include_ldsr_variance_k5")
    if type(include) is not bool:
        raise ValueError("A2 K=5 inclusion decision is invalid")
    expected_predictions = 120 * (7 if include else 3)
    expected_scores = 120 * (3 if include else 2)
    names = ["lr_reprojection_l1", "three_model_disagreement"] + (
        ["ldsr_variance_k5"] if include else []
    )
    if result.get("candidate_names") != names or result.get("score_configuration") != {
        name: _score_configuration(name) for name in names
    }:
        raise ValueError("A2 candidate names or score configuration is invalid")
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
    if (
        type(result.get("code_revision")) is not str
        or re.fullmatch(r"[0-9a-f]{40}", result["code_revision"]) is None
    ):
        raise ValueError("A2 code revision is invalid")
    bootstrap = result.get("bootstrap")
    if bootstrap != {
        "algorithm": "numpy.PCG64",
        "seed": 23031,
        "resamples": 10_000,
        "ci_percentiles": [2.5, 97.5],
    }:
        raise ValueError("A2 bootstrap schema is invalid")
    summaries = result.get("candidate_summaries")
    if not isinstance(summaries, list) or len(summaries) != len(names):
        raise ValueError("A2 candidate summaries are incomplete")
    primary_keys = {
        "name",
        "eligible",
        "failure_reasons",
        "nonconstant_count",
        "mean_rho",
        "mean_rho_ci95",
        "mean_aurc_gain",
        "mean_aurc_gain_ci95",
        "positive_strata",
        "minimum_stratum_mean_rho",
        "median_rho",
        "rho_quartiles",
        "median_aurc_gain",
        "aurc_gain_quartiles",
        "stratum_mean_rho",
    }
    sensitivity_keys = {
        "selection_use",
        "nonconstant_count",
        "mean_rho",
        "mean_rho_ci95",
        "mean_aurc_gain",
        "mean_aurc_gain_ci95",
        "median_rho",
        "rho_quartiles",
        "median_aurc_gain",
        "aurc_gain_quartiles",
        "stratum_mean_rho",
    }
    for summary, name in zip(summaries, names, strict=True):
        summary = _require_exact_keys(
            summary,
            {"name", "operator_parameters", "seeds", "primary_window_9", "sensitivity_window_1"},
            "A2 candidate summary",
        )
        primary = _require_exact_keys(
            summary["primary_window_9"], primary_keys, "A2 primary candidate evidence"
        )
        sensitivity = _require_exact_keys(
            summary["sensitivity_window_1"], sensitivity_keys, "A2 sensitivity evidence"
        )
        expected_seeds = list(range(3407, 3412)) if name == "ldsr_variance_k5" else [3407]
        numeric_fields = ("mean_rho", "mean_aurc_gain", "median_rho", "median_aurc_gain")
        strata = primary["stratum_mean_rho"]
        sensitivity_strata = sensitivity["stratum_mean_rho"]
        if (
            summary["name"] != name
            or summary["operator_parameters"] != _score_configuration(name)
            or summary["seeds"] != expected_seeds
            or primary["name"] != name
            or type(primary["eligible"]) is not bool
            or not isinstance(primary["failure_reasons"], list)
            or any(
                not isinstance(primary[field], int | float)
                or not math.isfinite(float(primary[field]))
                for field in numeric_fields
            )
            or any(
                not isinstance(sensitivity[field], int | float)
                or not math.isfinite(float(sensitivity[field]))
                for field in numeric_fields
            )
            or not isinstance(strata, list)
            or not isinstance(sensitivity_strata, list)
            or len(strata) != 12
            or len(sensitivity_strata) != 12
            or any(
                not isinstance(item, dict)
                or set(item) != {"days_between", "correlation_bin", "mean_rho", "mean_aurc_gain"}
                for item in [*strata, *sensitivity_strata]
            )
            or {
                (item.get("days_between"), item.get("correlation_bin"))
                for item in strata
                if isinstance(item, dict)
            }
            != {(day, bin_index) for day in (-1, 0, 1) for bin_index in range(4)}
            or {
                (item.get("days_between"), item.get("correlation_bin"))
                for item in sensitivity_strata
                if isinstance(item, dict)
            }
            != {(day, bin_index) for day in (-1, 0, 1) for bin_index in range(4)}
        ):
            raise ValueError("A2 candidate summary/freeze evidence is invalid")
    _validate_a2_samples(result, audit, include_ldsr_variance_k5=include)
    runtime_sha256 = _sha256(runtime_bytes)
    if result.get("runtime_manifest_sha256") != runtime_sha256:
        raise ValueError("A2 runtime-manifest SHA reference is invalid")
    if (
        runtime.get("a1_acceptance_pass") is not True
        or runtime.get("sample_count") != 120
        or not _is_sha(runtime.get("a1_replay_sha256"))
    ):
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
        frozen = _require_exact_keys(
            frozen,
            {
                "name",
                "operator_parameters",
                "seeds",
                "post_manifest_sha256",
                "code_revision",
                "cost_rank",
                "statistical_leader",
                "indistinguishable_candidates",
                "selected_candidate_evidence",
                "candidate_eligibility_evidence",
            },
            "A2 frozen score",
        )
        selected_name = frozen["name"]
        expected_seeds = list(range(3407, 3412)) if selected_name == "ldsr_variance_k5" else [3407]
        primary_evidence = [summary["primary_window_9"] for summary in summaries]
        if (
            selected_name not in names
            or frozen["operator_parameters"] != _score_configuration(selected_name)
            or frozen["seeds"] != expected_seeds
            or frozen["post_manifest_sha256"] != POST_MANIFEST_SHA256
            or frozen["code_revision"] != result["code_revision"]
            or frozen["selected_candidate_evidence"] != primary_evidence[names.index(selected_name)]
            or frozen["candidate_eligibility_evidence"] != primary_evidence
            or frozen["cost_rank"] != names.index(selected_name)
            or frozen["statistical_leader"] not in names
            or not isinstance(frozen["indistinguishable_candidates"], list)
            or frozen["indistinguishable_candidates"]
            != [name for name in names if name in frozen["indistinguishable_candidates"]]
            or selected_name not in frozen["indistinguishable_candidates"]
        ):
            raise ValueError("A2 frozen score provenance/selection evidence is invalid")
        acceptance["frozen_score"] = frozen
        acceptance["no_eligible_score"] = False
    elif decision == "stop_no_eligible_score" and frozen is None:
        if any(summary["primary_window_9"]["eligible"] is not False for summary in summaries):
            raise ValueError("A2 no-eligible stop lacks complete ineligibility evidence")
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with _acceptance_lock(path):
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file():
                raise ValueError("existing acceptance output must be a regular file")
            if path.read_bytes() != payload:
                raise ValueError("existing acceptance output has different bytes")
            return
        atomic_write_bytes(path, payload)


@contextmanager
def _acceptance_lock(path: Path) -> Iterable[None]:
    lock_path = path.with_name(f".{path.name}.lock")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("acceptance lock must be a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = _validate_output_path(args.output, phase=args.phase)
    acceptance = args.handler(args.bundle)
    _commit_acceptance(output, acceptance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
