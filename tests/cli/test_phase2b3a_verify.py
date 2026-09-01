"""Local-only verification of small Phase 2B3-A evidence bundles."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from trustsr.cli import phase2b3a_verify
from trustsr.data.crosssensor_pairs import POST_MANIFEST_SHA256
from trustsr.evaluation.crosssensor_smoke import INPUT_AUDIT_SHA256
from trustsr.jsonio import canonical_json


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _identity(sample_id: str, slot: str) -> tuple[dict[str, object], str]:
    identity = {"sample_id": sample_id, "slot": slot}
    return identity, _digest(canonical_json(identity))


def _files(cache_key: str) -> list[dict[str, object]]:
    return [
        {
            "filename": f"{cache_key}{suffix}",
            "size_bytes": 10,
            "sha256": _digest(f"{cache_key}{suffix}".encode()),
        }
        for suffix in (".json", ".safetensors")
    ]


def _write_bundle(root: Path, phase: str) -> dict[str, dict[str, object]]:
    root.mkdir()
    if phase == "a1":
        result = {
            "schema": "trustsr.phase2b3a-development-smoke.v1",
            "dataset_role": "development_engineering_smoke_only",
            "upstream": {
                "post_manifest_sha256": POST_MANIFEST_SHA256,
                "input_audit_sha256": INPUT_AUDIT_SHA256,
            },
            "sample_count": 4,
            "prediction_count": 108,
            "score_count": 20,
            "k5_statistically_stable": True,
            "include_ldsr_variance_k5": True,
            "samples": [
                {
                    "sample_id": f"development-{index}",
                    "spatial_group_id": f"group-{index}",
                    "correlation_bin": index,
                    "days_between": -1,
                    "selection_round": 1,
                }
                for index in range(4)
            ],
        }
        prediction_entries = []
        score_entries = []
        score_names = (
            "ldsr_variance_k5a",
            "ldsr_variance_k5b",
            "ldsr_variance_k25",
            "lr_reprojection_l1",
            "three_model_disagreement",
        )
        for sample in result["samples"]:
            predictions = []
            for model_name, seed in (
                ("bicubic-x4", None),
                ("sen2srlite-x4", None),
                *(("ldsr-s2-x4", seed) for seed in range(3407, 3432)),
            ):
                identity, cache_key = _identity(
                    sample["sample_id"], f"{model_name}:{seed}"
                )
                predictions.append(
                    {
                        "sample_id": sample["sample_id"],
                        "correlation_bin": sample["correlation_bin"],
                        "model_name": model_name,
                        "seed": seed,
                        "cache_key": cache_key,
                        "identity": identity,
                        "prediction_sha256": _digest(cache_key.encode()),
                    }
                )
            prediction_entries.extend(predictions)
            sample["central_prediction_sha256"] = predictions[2]["prediction_sha256"]
            sample["scores"] = []
            for name in score_names:
                identity, cache_key = _identity(sample["sample_id"], name)
                digest = _digest(cache_key.encode())
                score = {
                    "sample_id": sample["sample_id"],
                    "correlation_bin": sample["correlation_bin"],
                    "name": name,
                    "cache_key": cache_key,
                    "identity": identity,
                    "score_sha256": digest,
                    "files": _files(cache_key),
                }
                score_entries.append(score)
                sample["scores"].append(
                    {"name": name, "cache_key": cache_key, "score_sha256": digest}
                )
        audit = {
            "schema": "trustsr.phase2b3a-development-smoke-cache-audit.v1",
            "experiment_schema": result["schema"],
            "post_manifest_sha256": POST_MANIFEST_SHA256,
            "input_audit_sha256": INPUT_AUDIT_SHA256,
            "sample_count": 4,
            "prediction_count": 108,
            "score_count": 20,
            "prediction_entries": prediction_entries,
            "score_entries": score_entries,
        }
        runtime = {
            "schema": "trustsr.phase2b3a-a1-runtime.v1",
            "git_commit": "a" * 40,
            "single_repeatability_pass": True,
            "single_peak_memory_bytes": 80,
            "gpu_total_memory_bytes": 100,
            "persistent_free_bytes": 10 * 1024**3,
            "a1_median_uncached_ldsr_prediction_seconds": 2.0,
            "missing_a2_seed_predictions": 120,
            "projected_a2_uncached_seconds": 240.0,
            "resource_gate_pass": True,
        }
        names = {
            "result": "phase2b3a-a1-result.json",
            "audit": "phase2b3a-a1-cache-audit.json",
            "runtime": "phase2b3a-a1-runtime.json",
            "replay": "phase2b3a-a1-replay.json",
        }
    else:
        samples = [
            {
                "sample_id": f"development-{day}-{bin_index}-{selection_round}",
                "spatial_group_id": f"group-{day}-{bin_index}-{selection_round}",
                "split": "development",
                "days_between": day,
                "correlation_bin": bin_index,
                "selection_round": selection_round,
            }
            for day in (-1, 0, 1)
            for bin_index in range(4)
            for selection_round in range(1, 11)
        ]
        result = {
            "schema": "trustsr.phase2b3a-development-score-audit.v1",
            "dataset_role": "development_score_selection_only",
            "upstream": {
                "post_manifest_sha256": POST_MANIFEST_SHA256,
                "input_audit_sha256": INPUT_AUDIT_SHA256,
            },
            "code_revision": "a" * 40,
            "sample_count": 120,
            "prediction_count": 360,
            "score_count": 240,
            "include_ldsr_variance_k5": False,
            "phase_decision": "freeze_score",
            "frozen_score": {"name": "lr_reprojection_l1"},
            "bootstrap": {
                "algorithm": "numpy.PCG64",
                "seed": 23031,
                "resamples": 10_000,
                "ci_percentiles": [2.5, 97.5],
            },
            "samples": samples,
        }
        groups = []
        for sample in samples:
            predictions = []
            for model_name, seed in (
                ("bicubic-x4", None),
                ("sen2srlite-x4", None),
                ("ldsr-s2-x4", 3407),
            ):
                identity, cache_key = _identity(
                    sample["sample_id"], f"{model_name}:{seed}"
                )
                predictions.append(
                    {
                        "sample_id": sample["sample_id"],
                        "model_name": model_name,
                        "seed": seed,
                        "cache_key": cache_key,
                        "identity": identity,
                        "prediction_sha256": _digest(cache_key.encode()),
                        "files": _files(cache_key),
                    }
                )
            sample["central_prediction_sha256"] = predictions[2]["prediction_sha256"]
            sample["scores"] = []
            scores = []
            for name in ("lr_reprojection_l1", "three_model_disagreement"):
                identity, cache_key = _identity(sample["sample_id"], name)
                digest = _digest(cache_key.encode())
                score = {
                    "sample_id": sample["sample_id"],
                    "name": name,
                    "cache_key": cache_key,
                    "identity": identity,
                    "score_sha256": digest,
                    "files": _files(cache_key),
                }
                scores.append(score)
                sample["scores"].append(
                    {"name": name, "cache_key": cache_key, "score_sha256": digest}
                )
            groups.append(
                {
                    **{
                        key: sample[key]
                        for key in (
                            "sample_id",
                            "spatial_group_id",
                            "days_between",
                            "correlation_bin",
                            "selection_round",
                        )
                    },
                    "prediction_entries": predictions,
                    "score_entries": scores,
                }
            )
        audit = {
            "schema": "trustsr.phase2b3a-development-score-cache-audit.v1",
            "experiment_schema": result["schema"],
            "post_manifest_sha256": POST_MANIFEST_SHA256,
            "input_audit_sha256": INPUT_AUDIT_SHA256,
            "code_revision": "a" * 40,
            "sample_count": 120,
            "prediction_count": 360,
            "score_count": 240,
            "groups": groups,
        }
        runtime = {
            "schema": "trustsr.phase2b3a-a2-runtime.v1",
            "git_commit": "a" * 40,
            "a1_acceptance_pass": True,
            "sample_count": 120,
        }
        names = {
            "result": "phase2b3a-a2-result.json",
            "audit": "phase2b3a-a2-cache-audit.json",
            "runtime": "phase2b3a-a2-runtime.json",
            "replay": "phase2b3a-a2-replay.json",
        }

    runtime_bytes = canonical_json(runtime)
    result["runtime_manifest_sha256"] = _digest(runtime_bytes)
    payloads = {
        names["result"]: canonical_json(result),
        names["audit"]: canonical_json(audit),
        names["runtime"]: runtime_bytes,
    }
    replay = {
        "schema": f"trustsr.phase2b3a-{phase}-replay.v1",
        "byte_identical": True,
        "result_sha256": _digest(payloads[names["result"]]),
        "cache_audit_sha256": _digest(payloads[names["audit"]]),
        "runtime_manifest_sha256": _digest(runtime_bytes),
    }
    payloads[names["replay"]] = canonical_json(replay)
    for name, payload in payloads.items():
        (root / name).write_bytes(payload)
    manifest = {
        "schema": "trustsr.phase2b3a-bundle-manifest.v1",
        "phase": phase,
        "files": [
            {"basename": name, "size_bytes": len(payload), "sha256": _digest(payload)}
            for name, payload in sorted(payloads.items())
        ],
    }
    (root / "phase2b3a-bundle-manifest.json").write_bytes(canonical_json(manifest))
    return {"result": result, "audit": audit, "runtime": runtime, "replay": replay}


@pytest.mark.parametrize("stage", ["a1", "a2"])
def test_parser_requires_bundle_and_output(stage: str) -> None:
    parser = phase2b3a_verify.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([stage])
    with pytest.raises(SystemExit):
        parser.parse_args([stage, "--bundle", "bundle"])


def test_verifier_imports_without_model_or_geotiff_modules() -> None:
    script = """
import importlib.abc
import sys

class BlockHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "rasterio" or fullname.startswith("trustsr.models"):
            raise RuntimeError(f"verifier imported prohibited module: {fullname}")
        return None

sys.meta_path.insert(0, BlockHeavyImports())
from trustsr.cli import phase2b3a_verify
assert not any(name == "rasterio" or name.startswith("trustsr.models") for name in sys.modules)
assert phase2b3a_verify.build_parser().parse_args(["a1", "--help"]) is None
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("phase", "acceptance_name"),
    [
        ("a1", "sen2naipv2-development-smoke-acceptance-v1.json"),
        ("a2", "sen2naipv2-development-score-acceptance-v1.json"),
    ],
)
def test_valid_bundle_writes_exact_canonical_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    acceptance_name: str,
) -> None:
    repository = tmp_path / "repository"
    (repository / "artifacts" / "phase2b3a").mkdir(parents=True)
    (repository / "uv.lock").write_text("fixture", encoding="utf-8")
    (repository / "pyproject.toml").write_text("fixture", encoding="utf-8")
    (repository / "src" / "trustsr").mkdir(parents=True)
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, phase)
    monkeypatch.setattr(phase2b3a_verify, "resolve_project_root", lambda: repository)
    output = repository / "artifacts" / "phase2b3a" / acceptance_name

    acceptance = getattr(phase2b3a_verify, f"verify_{phase}_bundle")(bundle)
    assert phase2b3a_verify.main(
        [phase, "--bundle", str(bundle), "--output", str(output)]
    ) == 0
    assert output.read_bytes() == canonical_json(acceptance)


def test_output_must_be_declared_direct_child_and_not_traverse_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(phase2b3a_verify, "resolve_project_root", lambda: repository)
    outside = tmp_path / "outside"
    outside.mkdir()
    (repository / "artifacts").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="output|symlink"):
        phase2b3a_verify._validate_output_path(
            repository / "artifacts/phase2b3a/sen2naipv2-development-smoke-acceptance-v1.json",
            phase="a1",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "exact|missing"),
        ("extra", "exact|extra"),
        ("digest", "digest|SHA"),
        ("noncanonical", "canonical"),
        ("secret", "secret"),
        ("schema", "schema"),
        ("acceptance", "K=5|acceptance|stability"),
        ("internal_reference", "internal|cache"),
        ("replay", "replay|identical"),
        ("leakage", "development|leakage"),
    ],
)
def test_a1_bundle_mutations_fail_closed(
    tmp_path: Path, mutation: str, message: str
) -> None:
    bundle = tmp_path / "bundle"
    payloads = _write_bundle(bundle, "a1")
    result_path = bundle / "phase2b3a-a1-result.json"
    manifest_path = bundle / "phase2b3a-bundle-manifest.json"
    if mutation == "missing":
        (bundle / "phase2b3a-a1-runtime.json").unlink()
    elif mutation == "extra":
        (bundle / "extra.json").write_text("{}", encoding="utf-8")
    elif mutation == "digest":
        payloads["result"]["digest_mutation"] = True
        result_path.write_bytes(canonical_json(payloads["result"]))
    elif mutation == "noncanonical":
        result_path.write_text(json.dumps(payloads["result"], indent=2), encoding="utf-8")
        _refresh_manifest(manifest_path, result_path)
    elif mutation == "secret":
        payloads["result"]["api_token"] = "not-a-real-secret"
        result_path.write_bytes(canonical_json(payloads["result"]))
        _refresh_manifest(manifest_path, result_path)
    elif mutation == "schema":
        payloads["result"]["schema"] = "wrong"
        result_path.write_bytes(canonical_json(payloads["result"]))
        _refresh_manifest(manifest_path, result_path)
    elif mutation == "acceptance":
        replay_path = bundle / "phase2b3a-a1-replay.json"
        payloads["result"]["include_ldsr_variance_k5"] = False
        result_path.write_bytes(canonical_json(payloads["result"]))
        payloads["replay"]["result_sha256"] = _digest(result_path.read_bytes())
        replay_path.write_bytes(canonical_json(payloads["replay"]))
        _refresh_manifest(manifest_path, result_path)
        _refresh_manifest(manifest_path, replay_path)
    elif mutation == "internal_reference":
        audit_path = bundle / "phase2b3a-a1-cache-audit.json"
        payloads["audit"]["prediction_entries"][0]["cache_key"] = "0" * 64
        audit_path.write_bytes(canonical_json(payloads["audit"]))
        _refresh_manifest(manifest_path, audit_path)
    elif mutation == "replay":
        replay_path = bundle / "phase2b3a-a1-replay.json"
        payloads["replay"]["byte_identical"] = False
        replay_path.write_bytes(canonical_json(payloads["replay"]))
        _refresh_manifest(manifest_path, replay_path)
    else:
        payloads["result"]["samples"][0]["split"] = "calibration"
        result_path.write_bytes(canonical_json(payloads["result"]))
        _refresh_manifest(manifest_path, result_path)

    with pytest.raises((ValueError, RuntimeError), match=message):
        phase2b3a_verify.verify_a1_bundle(bundle)


def _refresh_manifest(manifest_path: Path, changed: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = changed.read_bytes()
    for item in manifest["files"]:
        if item["basename"] == changed.name:
            item["size_bytes"] = len(payload)
            item["sha256"] = _digest(payload)
    manifest_path.write_bytes(canonical_json(manifest))


@pytest.mark.parametrize("mutation", ["roi_count", "strata", "leakage"])
def test_a2_rejects_wrong_roi_strata_or_non_development_evidence(
    tmp_path: Path, mutation: str
) -> None:
    bundle = tmp_path / "bundle"
    payloads = _write_bundle(bundle, "a2")
    path = bundle / "phase2b3a-a2-result.json"
    if mutation == "roi_count":
        payloads["result"]["sample_count"] = 119
    elif mutation == "strata":
        payloads["result"]["samples"][0]["selection_round"] = 2
    else:
        payloads["result"]["samples"][0]["split"] = "internal_test"
    path.write_bytes(canonical_json(payloads["result"]))
    _refresh_manifest(bundle / "phase2b3a-bundle-manifest.json", path)

    with pytest.raises(ValueError, match="120|strat|development|leakage"):
        phase2b3a_verify.verify_a2_bundle(bundle)
