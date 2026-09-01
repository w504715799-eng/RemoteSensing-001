"""Local-only verification of small Phase 2B3-A evidence bundles."""

from __future__ import annotations

import hashlib
import json
import math
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


_SOURCE = f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}"


def _prediction_identity(
    sample_id: str, model_name: str, seed: int | None, lr_sha256: str
) -> tuple[dict[str, object], str]:
    common: dict[str, object] = {
        "name": model_name,
        "scale": 4,
        "experiment_schema": "trustsr.phase2b3a-predictions.v1",
        "post_manifest_sha256": POST_MANIFEST_SHA256,
        "input_audit_sha256": INPUT_AUDIT_SHA256,
        "implementation_schema_version": 1,
        "output_policy": "clip_to_[0,1]",
        "torch_version": "2.8.0",
    }
    if model_name == "bicubic-x4":
        common.update(
            implementation="torch.nn.functional.interpolate",
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
    elif model_name == "sen2srlite-x4":
        common.update(
            model_id="SEN2SRLite_NonReference_RGBN_x4",
            manifest_url=(
                "https://huggingface.co/tacofoundation/sen2sr/resolve/main/"
                "SEN2SRLite/NonReference_RGBN_x4/mlm.json"
            ),
            mlstac_version="0.9",
            sen2sr_version="1.0",
            device="cpu",
        )
        common.update(
            {
                "asset_sha256:example_data.safetensor": (
                    "c895c7da8a8d48882b73a2a1955e4260714b97540eea290229a284d73f129985"
                ),
                "asset_sha256:hard_constraint.safetensor": (
                    "fbad981519066387c413ead1d6af7ef3e0d2947c34147ba90163fc79ae539239"
                ),
                "asset_sha256:load.py": (
                    "4b6c836b1f73078c62c84d4374b2d8daee5345f6239f64e0b6be29432383bac6"
                ),
                "asset_sha256:mlm.json": (
                    "59caa5c6af96a6fbebdbd771d93c91cc2d3a770302cd2f262b5409e77a40e3f7"
                ),
                "asset_sha256:model.safetensor": (
                    "479aa796d5068d0b1206118ccbca27bd3223df0214db1a9b31a1e18349ed1c7e"
                ),
            }
        )
    else:
        common.update(
            opensr_model_version="1.1.1",
            cuda_runtime="12.8",
            checkpoint_name="opensr-ldsrs2_v1_0_0.ckpt",
            checkpoint_url="https://huggingface.co/simon-donike/RS-SR-LTDF/resolve/main/opensr-ldsrs2_v1_0_0.ckpt",
            checkpoint_size=1_130_715_795,
            checkpoint_sha256="e2621e3912eb7c14867c3d20c9029607ba941be8e166dc09621860fcac27dc3a",
            config_sha256="ac76685d354bfec32e3e0641aef574bedd7d650402c97dbd0ade86304e69ca6f",
            device="cuda",
            seed=seed,
            sampling_steps=100,
            sampling_eta=0.95,
            sampling_temperature=1.0,
            histogram_matching=True,
        )
    identity = {
        "model_provenance": common,
        "source": _SOURCE,
        "sample_id": sample_id,
        "lr": {"shape": [4, 128, 128], "dtype": "torch.float32", "sha256": lr_sha256},
    }
    return identity, _digest(canonical_json(identity))


def _score_configuration(name: str) -> dict[str, object]:
    if name.startswith("ldsr_variance"):
        first, last = (3412, 3416) if name == "ldsr_variance_k5b" else (3407, 3411)
        if name == "ldsr_variance_k25":
            last = 3431
        return {
            "algorithm": "ensemble_variance_score",
            "band_reduction": "mean",
            "correction": 0,
            "seed_first": first,
            "seed_last": last,
            "seed_count": last - first + 1,
        }
    if name == "lr_reprojection_l1":
        return {
            "algorithm": "lr_reprojection_l1_score",
            "downsample_mode": "area",
            "scale": 4,
            "upsample_mode": "repeat_interleave",
        }
    return {
        "algorithm": "three_model_disagreement_score",
        "band_reduction": "mean",
        "correction": 0,
        "model_order": "bicubic-x4,sen2srlite-x4,ldsr-s2-x4",
    }


def _score_identity(
    sample_id: str, name: str, lr_sha256: str, input_sha256s: list[str]
) -> tuple[dict[str, object], str]:
    identity = {
        "score_name": name,
        "score_schema_version": 1,
        "sample_id": sample_id,
        "input_sha256s": input_sha256s,
        "operator_parameters": {**_score_configuration(name), "lr_sha256": lr_sha256},
    }
    return identity, _digest(canonical_json(identity))


def _diagnostic(name: str) -> dict[str, object]:
    return {
        "rho": 0.75,
        "constant_score": False,
        "coverages": [index / 10 for index in range(1, 11)],
        "selective_mean_risks": [0.1 + index / 100 for index in range(10)],
        "aurc": 0.15,
        "random_aurc": 0.2,
        "aurc_gain": 0.05,
        "high_risk_miss_rate_at_80": 0.1,
        **({"name": name} if False else {}),
    }


def _summary(name: str, *, eligible: bool = True) -> dict[str, object]:
    return {
        "name": name,
        "eligible": eligible,
        "failure_reasons": [],
        "nonconstant_count": 120,
        "mean_rho": 0.75,
        "mean_rho_ci95": [0.7, 0.8],
        "mean_aurc_gain": 0.05,
        "mean_aurc_gain_ci95": [0.01, 0.09],
        "positive_strata": 12,
        "minimum_stratum_mean_rho": 0.7,
        "median_rho": 0.75,
        "rho_quartiles": [0.7, 0.8],
        "median_aurc_gain": 0.05,
        "aurc_gain_quartiles": [0.01, 0.09],
        "stratum_mean_rho": [
            {
                "days_between": day,
                "correlation_bin": bin_index,
                "mean_rho": 0.75,
                "mean_aurc_gain": 0.05,
            }
            for day in (-1, 0, 1)
            for bin_index in range(4)
        ],
    }


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
            "bands": ["B04", "B03", "B02", "B08"],
            "scale": 4,
            "sample_count": 4,
            "prediction_count": 108,
            "score_count": 20,
            "k5_statistically_stable": True,
            "include_ldsr_variance_k5": True,
            "seed_sets": {
                "k5a": list(range(3407, 3412)),
                "k5b": list(range(3412, 3417)),
                "k25": list(range(3407, 3432)),
            },
            "stability_thresholds": {
                "k5a_k5b_median_minimum": 0.6,
                "k5a_k5b_worst_minimum": 0.4,
                "k5a_k25_median_minimum": 0.8,
                "k5a_k25_worst_minimum": 0.6,
                "k5a_k25_top10_jaccard_median_minimum": 0.5,
            },
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
            lr_sha256 = _digest(f"lr:{sample['sample_id']}".encode())
            hr_sha256 = _digest(f"hr:{sample['sample_id']}".encode())
            predictions = []
            for model_name, seed in (
                ("bicubic-x4", None),
                ("sen2srlite-x4", None),
                *(("ldsr-s2-x4", seed) for seed in range(3407, 3432)),
            ):
                identity, cache_key = _prediction_identity(
                    sample["sample_id"], model_name, seed, lr_sha256
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
            sample["lr_tensor_sha256"] = lr_sha256
            sample["hr_tensor_sha256"] = hr_sha256
            sample["central_prediction_sha256"] = predictions[2]["prediction_sha256"]
            sample["risks"] = {
                "primary": {"name": "local_l1_risk", "window": 9, "risk_sha256": _digest(b"risk9")},
                "sensitivity": {
                    "name": "local_l1_risk",
                    "window": 1,
                    "risk_sha256": _digest(b"risk1"),
                },
            }
            sample["stability"] = {
                "k5a_k5b_spearman": 0.9,
                "k5a_k25_spearman": 0.9,
                "k5a_k25_top10_jaccard": 0.8,
                "k5a_constant_score": False,
                "k5b_constant_score": False,
                "k25_constant_score": False,
            }
            sample["scores"] = []
            for name in score_names:
                hashes = [item["prediction_sha256"] for item in predictions]
                inputs = {
                    "ldsr_variance_k5a": hashes[2:7],
                    "ldsr_variance_k5b": hashes[7:12],
                    "ldsr_variance_k25": hashes[2:27],
                    "lr_reprojection_l1": [hashes[2]],
                    "three_model_disagreement": [hashes[0], hashes[1], hashes[2]],
                }[name]
                identity, cache_key = _score_identity(sample["sample_id"], name, lr_sha256, inputs)
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
                    {
                        "name": name,
                        "cache_key": cache_key,
                        "score_sha256": digest,
                        "primary_window_9": _diagnostic(name),
                        "sensitivity_window_1": _diagnostic(name),
                    }
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
            "a1_uncached_ldsr_prediction_seconds": [2.0],
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
            "bands": ["B04", "B03", "B02", "B08"],
            "scale": 4,
            "sample_count": 120,
            "statistical_unit": "roi",
            "prediction_count": 360,
            "score_count": 240,
            "include_ldsr_variance_k5": False,
            "candidate_names": ["lr_reprojection_l1", "three_model_disagreement"],
            "score_configuration": {
                name: _score_configuration(name)
                for name in ("lr_reprojection_l1", "three_model_disagreement")
            },
            "risk_configuration": {
                "name": "local_l1_risk",
                "primary_window": 9,
                "sensitivity_window": 1,
            },
            "selection_risk": "primary_window_9",
            "phase_decision": "freeze_score",
            "bootstrap": {
                "algorithm": "numpy.PCG64",
                "seed": 23031,
                "resamples": 10_000,
                "ci_percentiles": [2.5, 97.5],
            },
            "samples": samples,
        }
        primary_summaries = [_summary(name) for name in result["candidate_names"]]
        result["candidate_summaries"] = [
            {
                "name": name,
                "operator_parameters": _score_configuration(name),
                "seeds": [3407],
                "primary_window_9": summary,
                "sensitivity_window_1": {
                    "selection_use": "descriptive_only",
                    "nonconstant_count": 120,
                    "mean_rho": 0.75,
                    "mean_rho_ci95": [0.7, 0.8],
                    "mean_aurc_gain": 0.05,
                    "mean_aurc_gain_ci95": [0.01, 0.09],
                    "median_rho": 0.75,
                    "rho_quartiles": [0.7, 0.8],
                    "median_aurc_gain": 0.05,
                    "aurc_gain_quartiles": [0.01, 0.09],
                    "stratum_mean_rho": [
                        {
                            "days_between": day,
                            "correlation_bin": bin_index,
                            "mean_rho": 0.75,
                            "mean_aurc_gain": 0.05,
                        }
                        for day in (-1, 0, 1)
                        for bin_index in range(4)
                    ],
                },
            }
            for name, summary in zip(result["candidate_names"], primary_summaries, strict=True)
        ]
        result["frozen_score"] = {
            "name": "lr_reprojection_l1",
            "operator_parameters": _score_configuration("lr_reprojection_l1"),
            "seeds": [3407],
            "post_manifest_sha256": POST_MANIFEST_SHA256,
            "code_revision": "a" * 40,
            "cost_rank": 0,
            "statistical_leader": "lr_reprojection_l1",
            "indistinguishable_candidates": ["lr_reprojection_l1"],
            "selected_candidate_evidence": primary_summaries[0],
            "candidate_eligibility_evidence": primary_summaries,
        }
        groups = []
        for sample in samples:
            lr_sha256 = _digest(f"lr:{sample['sample_id']}".encode())
            hr_sha256 = _digest(f"hr:{sample['sample_id']}".encode())
            predictions = []
            for model_name, seed in (
                ("bicubic-x4", None),
                ("sen2srlite-x4", None),
                ("ldsr-s2-x4", 3407),
            ):
                identity, cache_key = _prediction_identity(
                    sample["sample_id"], model_name, seed, lr_sha256
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
            sample["lr_tensor_sha256"] = lr_sha256
            sample["hr_tensor_sha256"] = hr_sha256
            sample["central_prediction_sha256"] = predictions[2]["prediction_sha256"]
            sample["risks"] = {
                "primary_window_9": {
                    "name": "local_l1_risk",
                    "window": 9,
                    "risk_sha256": _digest(b"risk9"),
                },
                "sensitivity_window_1": {
                    "name": "local_l1_risk",
                    "window": 1,
                    "risk_sha256": _digest(b"risk1"),
                },
            }
            sample["scores"] = []
            scores = []
            for name in ("lr_reprojection_l1", "three_model_disagreement"):
                inputs = (
                    [predictions[2]["prediction_sha256"]]
                    if name == "lr_reprojection_l1"
                    else [item["prediction_sha256"] for item in predictions]
                )
                identity, cache_key = _score_identity(sample["sample_id"], name, lr_sha256, inputs)
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
                scores.append(score)
                sample["scores"].append(
                    {
                        "name": name,
                        "cache_key": cache_key,
                        "score_sha256": digest,
                        "primary_window_9": _diagnostic(name),
                        "sensitivity_window_1": _diagnostic(name),
                    }
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
            "a1_replay_sha256": _digest(b"a1-replay"),
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
    assert phase2b3a_verify.main([phase, "--bundle", str(bundle), "--output", str(output)]) == 0
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
def test_a1_bundle_mutations_fail_closed(tmp_path: Path, mutation: str, message: str) -> None:
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


@pytest.mark.parametrize("phase", ["a1", "a2"])
@pytest.mark.parametrize(
    "mutation", ["prediction_shape", "provenance", "score_version", "score_inputs", "extra_key"]
)
def test_verifier_rejects_fabricated_or_mutated_cache_identities(
    tmp_path: Path, phase: str, mutation: str
) -> None:
    bundle = tmp_path / "bundle"
    payloads = _write_bundle(bundle, phase)
    audit_name = f"phase2b3a-{phase}-cache-audit.json"
    audit_path = bundle / audit_name
    predictions = (
        payloads["audit"]["prediction_entries"]
        if phase == "a1"
        else payloads["audit"]["groups"][0]["prediction_entries"]
    )
    scores = (
        payloads["audit"]["score_entries"]
        if phase == "a1"
        else payloads["audit"]["groups"][0]["score_entries"]
    )
    if mutation == "prediction_shape":
        predictions[0]["identity"]["lr"]["shape"] = [4, 1, 1]
        predictions[0]["cache_key"] = _digest(canonical_json(predictions[0]["identity"]))
    elif mutation == "provenance":
        predictions[0]["identity"]["model_provenance"]["mode"] = "nearest"
        predictions[0]["cache_key"] = _digest(canonical_json(predictions[0]["identity"]))
    elif mutation == "score_version":
        scores[0]["identity"]["score_schema_version"] = 2
        scores[0]["cache_key"] = _digest(canonical_json(scores[0]["identity"]))
    elif mutation == "score_inputs":
        scores[0]["identity"]["input_sha256s"] = [_digest(b"fabricated")]
        scores[0]["cache_key"] = _digest(canonical_json(scores[0]["identity"]))
    else:
        scores[0]["identity"]["extra"] = True
        scores[0]["cache_key"] = _digest(canonical_json(scores[0]["identity"]))
    target = predictions[0] if mutation in {"prediction_shape", "provenance"} else scores[0]
    if "files" in target:
        target["files"] = _files(target["cache_key"])
    audit_path.write_bytes(canonical_json(payloads["audit"]))
    _refresh_manifest(bundle / "phase2b3a-bundle-manifest.json", audit_path)
    with pytest.raises(ValueError, match="identity|provenance|score|prediction|schema|reference"):
        getattr(phase2b3a_verify, f"verify_{phase}_bundle")(bundle)


@pytest.mark.parametrize(
    "field",
    [
        "peak_memory_bytes",
        "elapsed_seconds",
        "driver_version",
        "device_uuid",
        "environment",
        "unexpected",
    ],
)
def test_scientific_result_rejects_every_unrecognized_runtime_field(
    tmp_path: Path, field: str
) -> None:
    bundle = tmp_path / "bundle"
    payloads = _write_bundle(bundle, "a1")
    result_path = bundle / "phase2b3a-a1-result.json"
    replay_path = bundle / "phase2b3a-a1-replay.json"
    payloads["result"][field] = "value"
    result_path.write_bytes(canonical_json(payloads["result"]))
    payloads["replay"]["result_sha256"] = _digest(result_path.read_bytes())
    replay_path.write_bytes(canonical_json(payloads["replay"]))
    _refresh_manifest(bundle / "phase2b3a-bundle-manifest.json", result_path)
    _refresh_manifest(bundle / "phase2b3a-bundle-manifest.json", replay_path)
    with pytest.raises(ValueError, match="runtime|schema|field|secret"):
        phase2b3a_verify.verify_a1_bundle(bundle)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_local_resource_projection_rejects_negative_or_nonfinite(
    tmp_path: Path, value: float
) -> None:
    bundle = tmp_path / "bundle"
    payloads = _write_bundle(bundle, "a1")
    runtime_path = bundle / "phase2b3a-a1-runtime.json"
    result_path = bundle / "phase2b3a-a1-result.json"
    replay_path = bundle / "phase2b3a-a1-replay.json"
    payloads["runtime"]["a1_median_uncached_ldsr_prediction_seconds"] = value
    payloads["runtime"]["projected_a2_uncached_seconds"] = value * 120
    runtime_path.write_bytes(
        canonical_json(payloads["runtime"])
        if math.isfinite(value)
        else json.dumps(payloads["runtime"], sort_keys=True, separators=(",", ":")).encode()
    )
    payloads["result"]["runtime_manifest_sha256"] = _digest(runtime_path.read_bytes())
    result_path.write_bytes(canonical_json(payloads["result"]))
    payloads["replay"]["runtime_manifest_sha256"] = _digest(runtime_path.read_bytes())
    payloads["replay"]["result_sha256"] = _digest(result_path.read_bytes())
    replay_path.write_bytes(canonical_json(payloads["replay"]))
    for path in (runtime_path, result_path, replay_path):
        _refresh_manifest(bundle / "phase2b3a-bundle-manifest.json", path)
    with pytest.raises(ValueError, match="resource|projection|canonical"):
        phase2b3a_verify.verify_a1_bundle(bundle)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("single_peak_memory_bytes", -1),
        ("gpu_total_memory_bytes", 0),
        ("persistent_free_bytes", -1),
    ],
)
def test_local_resource_measurements_reject_negative_or_zero_total(
    tmp_path: Path, key: str, value: int
) -> None:
    bundle = tmp_path / "bundle"
    payloads = _write_bundle(bundle, "a1")
    runtime_path = bundle / "phase2b3a-a1-runtime.json"
    result_path = bundle / "phase2b3a-a1-result.json"
    replay_path = bundle / "phase2b3a-a1-replay.json"
    payloads["runtime"][key] = value
    runtime_path.write_bytes(canonical_json(payloads["runtime"]))
    payloads["result"]["runtime_manifest_sha256"] = _digest(runtime_path.read_bytes())
    result_path.write_bytes(canonical_json(payloads["result"]))
    payloads["replay"]["runtime_manifest_sha256"] = _digest(runtime_path.read_bytes())
    payloads["replay"]["result_sha256"] = _digest(result_path.read_bytes())
    replay_path.write_bytes(canonical_json(payloads["replay"]))
    for path in (runtime_path, result_path, replay_path):
        _refresh_manifest(bundle / "phase2b3a-bundle-manifest.json", path)
    with pytest.raises(ValueError, match="resource"):
        phase2b3a_verify.verify_a1_bundle(bundle)


@pytest.mark.parametrize("key", ["credential", "authorization", "access_key", "client_secret"])
def test_secret_key_variants_fail_closed(tmp_path: Path, key: str) -> None:
    bundle = tmp_path / "bundle"
    payloads = _write_bundle(bundle, "a1")
    path = bundle / "phase2b3a-a1-cache-audit.json"
    payloads["audit"][key] = "redacted"
    path.write_bytes(canonical_json(payloads["audit"]))
    _refresh_manifest(bundle / "phase2b3a-bundle-manifest.json", path)
    with pytest.raises(ValueError, match="secret"):
        phase2b3a_verify.verify_a1_bundle(bundle)


def test_relative_path_value_and_bundle_file_symlink_fail_closed(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    payloads = _write_bundle(bundle, "a1")
    path = bundle / "phase2b3a-a1-cache-audit.json"
    payloads["audit"]["note"] = "relative/file.txt"
    path.write_bytes(canonical_json(payloads["audit"]))
    _refresh_manifest(bundle / "phase2b3a-bundle-manifest.json", path)
    with pytest.raises(ValueError, match="path|schema"):
        phase2b3a_verify.verify_a1_bundle(bundle)

    path.unlink()
    outside = tmp_path / "outside.json"
    outside.write_bytes(canonical_json(payloads["audit"]))
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        phase2b3a_verify.verify_a1_bundle(bundle)


def test_oversized_bundle_file_is_rejected_before_json_read(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, "a1")
    path = bundle / "phase2b3a-a1-cache-audit.json"
    path.write_bytes(b" " * (5 * 1024**2 + 1))
    with pytest.raises(ValueError, match="5 MiB"):
        phase2b3a_verify.verify_a1_bundle(bundle)


def test_bundle_directory_symlink_is_rejected(tmp_path: Path) -> None:
    real_bundle = tmp_path / "real"
    _write_bundle(real_bundle, "a1")
    bundle = tmp_path / "bundle"
    bundle.symlink_to(real_bundle, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        phase2b3a_verify.verify_a1_bundle(bundle)


def test_acceptance_output_collision_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "acceptance.json"
    path.write_bytes(b"different")
    with pytest.raises(ValueError, match="different bytes"):
        phase2b3a_verify._commit_acceptance(path, {"accepted": True})


def test_acceptance_atomic_write_failure_leaves_no_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "acceptance.json"
    monkeypatch.setattr(
        phase2b3a_verify,
        "atomic_write_bytes",
        lambda path, payload: (_ for _ in ()).throw(OSError("injected write failure")),
    )
    with pytest.raises(OSError, match="injected"):
        phase2b3a_verify._commit_acceptance(path, {"accepted": True})
    assert not path.exists()


def test_concurrent_acceptance_writers_commit_identical_bytes(tmp_path: Path) -> None:
    path = tmp_path / "acceptance.json"
    script = (
        "import sys; from pathlib import Path; "
        "from trustsr.cli.phase2b3a_verify import _commit_acceptance; "
        "_commit_acceptance(Path(sys.argv[1]), {'accepted': True})"
    )
    processes = [subprocess.Popen([sys.executable, "-c", script, str(path)]) for _ in range(4)]
    assert [process.wait() for process in processes] == [0, 0, 0, 0]
    assert path.read_bytes() == canonical_json({"accepted": True})


@pytest.mark.parametrize(
    "mutation", ["candidate_names", "configuration", "summaries", "frozen", "extra"]
)
def test_a2_requires_complete_exact_candidate_and_freeze_evidence(
    tmp_path: Path, mutation: str
) -> None:
    bundle = tmp_path / "bundle"
    payloads = _write_bundle(bundle, "a2")
    result = payloads["result"]
    if mutation == "candidate_names":
        result.pop("candidate_names")
    elif mutation == "configuration":
        result["score_configuration"]["lr_reprojection_l1"]["scale"] = 2
    elif mutation == "summaries":
        result["candidate_summaries"][0].pop("sensitivity_window_1")
    elif mutation == "frozen":
        result["frozen_score"].pop("candidate_eligibility_evidence")
    else:
        result["runtime_seconds"] = 1
    result_path = bundle / "phase2b3a-a2-result.json"
    replay_path = bundle / "phase2b3a-a2-replay.json"
    result_path.write_bytes(canonical_json(result))
    payloads["replay"]["result_sha256"] = _digest(result_path.read_bytes())
    replay_path.write_bytes(canonical_json(payloads["replay"]))
    for path in (result_path, replay_path):
        _refresh_manifest(bundle / "phase2b3a-bundle-manifest.json", path)
    with pytest.raises(ValueError, match="A2|candidate|score|frozen|schema|runtime"):
        phase2b3a_verify.verify_a2_bundle(bundle)
