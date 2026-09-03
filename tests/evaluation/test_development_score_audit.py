"""Four-ROI A1 score construction, stability, evidence, and replay."""

from __future__ import annotations

import builtins
import gc
import os
import weakref
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from trustsr.artifacts.predictions import PredictionCache, build_identity, tensor_sha256
from trustsr.artifacts.scores import ScoreCache, ScoreIdentity, score_entry_evidence
from trustsr.contracts import SRPair
from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    PHASE2B3A_NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
    CrosssensorPairMetadata,
    LoadedCrosssensorPair,
    RadiometricSaturation,
)
from trustsr.evaluation import development_predictions, development_score_audit
from trustsr.evaluation.development_predictions import (
    A1_SEEDS,
    K5A_SEEDS,
    K5B_SEEDS,
    CachedDevelopmentPrediction,
    DevelopmentPredictionBundle,
    build_cache_provenance,
)
from trustsr.evaluation.development_score_audit import (
    A1_CACHE_AUDIT_SCHEMA,
    A1_RESULT_SCHEMA,
    A2_CACHE_AUDIT_SCHEMA,
    A2_RESULT_SCHEMA,
    A2_SCORE_NAMES,
    PRIMARY_RISK_WINDOW,
    SENSITIVITY_RISK_WINDOW,
    _is_k5_statistically_stable,
    build_a1_score_maps,
    evaluate_a1_smoke,
    evaluate_a2_development,
    replay_a1_smoke,
    replay_a2_development,
)
from trustsr.evaluation.score_diagnostics import (
    RoiScoreDiagnostics,
    score_map_spearman,
    top_fraction_jaccard,
)
from trustsr.evaluation.score_selection import COST_ORDER
from trustsr.jsonio import canonical_json
from trustsr.models import bicubic, ldsr_s2, sen2srlite


def _small_loaded_pair(correlation_bin: int) -> LoadedCrosssensorPair:
    sample_id = f"development-{correlation_bin}"
    row = torch.linspace(0.0, 1.0, 12, dtype=torch.float32)
    hr = row.view(1, 1, 12).expand(4, 12, 12).contiguous() * 0.2 + 0.35
    return LoadedCrosssensorPair(
        pair=SRPair(
            source=f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}",
            sample_id=sample_id,
            lr=torch.full((4, 3, 3), 0.4, dtype=torch.float32),
            hr=hr,
            scale=4,
        ),
        metadata=CrosssensorPairMetadata(
            manifest_sha256=POST_MANIFEST_SHA256,
            sample_id=sample_id,
            split="development",
            spatial_group_id=f"group-{correlation_bin}",
            days_between=-1,
            correlation_bin=correlation_bin,
            selection_round=1,
            lr_asset_sha256=f"{correlation_bin + 1:x}" * 64,
            hr_asset_sha256=f"{correlation_bin + 5:x}" * 64,
            lr_crop_transform=(10.0, 0.0, 10.0, 0.0, -10.0, -10.0),
            hr_crop_transform=(2.5, 0.0, 10.0, 0.0, -2.5, -10.0),
            crop_bounds=(10.0, -30.0, 40.0, -10.0),
            crop_policy=CROP_POLICY,
            normalization_policy=PHASE2B3A_NORMALIZATION_POLICY,
            lr_saturation=RadiometricSaturation(4000, 4000, 0, (0, 0, 0, 0)),
            hr_saturation=RadiometricSaturation(3500, 5500, 0, (0, 0, 0, 0)),
        ),
    )


def _prediction(
    pair: LoadedCrosssensorPair,
    *,
    model_name: str,
    value: torch.Tensor,
    seed: int | None,
    cache: PredictionCache | None,
) -> CachedDevelopmentPrediction:
    provenance: dict[str, object] = {"name": model_name, "scale": 4, "synthetic": True}
    if seed is not None:
        provenance["seed"] = seed
    identity = build_identity(
        build_cache_provenance(provenance),
        pair.pair.source,
        pair.pair.sample_id,
        pair.pair.lr,
    )
    tensor = value.to(torch.float32).contiguous()
    if cache is not None:
        cache.put(identity, tensor)
        loaded = cache.get(identity)
        assert loaded is not None
        tensor = loaded
    return CachedDevelopmentPrediction(
        model_name=model_name,
        seed=seed,
        identity=identity,
        prediction_sha256=tensor_sha256(tensor),
        tensor=tensor,
    )


def _synthetic_prediction_bundle(
    pair: LoadedCrosssensorPair,
    *,
    cache: PredictionCache | None = None,
    constant_maps: bool = False,
    seeds: tuple[int, ...] = A1_SEEDS,
) -> DevelopmentPredictionBundle:
    height, width = pair.pair.hr.shape[1:]
    ramp = torch.linspace(0.0, 1.0, height * width, dtype=torch.float32).reshape(height, width)
    spatial = torch.ones_like(ramp) if constant_maps else ramp
    channels = spatial.unsqueeze(0).expand(4, -1, -1)
    bicubic = _prediction(
        pair,
        model_name="bicubic-x4",
        value=torch.full_like(channels, 0.3),
        seed=None,
        cache=cache,
    )
    sen2srlite = _prediction(
        pair,
        model_name="sen2srlite-x4",
        value=0.35 + 0.01 * channels,
        seed=None,
        cache=cache,
    )
    ldsr = tuple(
        _prediction(
            pair,
            model_name="ldsr-s2-x4",
            value=(
                0.5
                + (((seed - 3407) % 5) - 2)
                * (0.01 if seed in K5A_SEEDS else 0.015 if seed in K5B_SEEDS else 0.02)
                * channels
            ),
            seed=seed,
            cache=cache,
        )
        for seed in seeds
    )
    return DevelopmentPredictionBundle(
        sample_id=pair.pair.sample_id,
        bicubic=bicubic,
        sen2srlite=sen2srlite,
        ldsr=ldsr,
    )


def _a1_inputs(
    prediction_cache: PredictionCache | None = None,
    *,
    constant_maps: bool = False,
) -> tuple[tuple[LoadedCrosssensorPair, ...], tuple[DevelopmentPredictionBundle, ...]]:
    pairs = tuple(_small_loaded_pair(bin_index) for bin_index in range(4))
    bundles = tuple(
        _synthetic_prediction_bundle(pair, cache=prediction_cache, constant_maps=constant_maps)
        for pair in pairs
    )
    return pairs, bundles


def test_build_score_maps_uses_frozen_central_prediction_and_seed_sets(
    tmp_path: Path,
) -> None:
    pair = _small_loaded_pair(correlation_bin=0)
    bundle = _synthetic_prediction_bundle(pair)

    scores = build_a1_score_maps(pair, bundle, ScoreCache(tmp_path))

    assert tuple(item.name for item in scores) == (
        "ldsr_variance_k5a",
        "ldsr_variance_k5b",
        "ldsr_variance_k25",
        "lr_reprojection_l1",
        "three_model_disagreement",
    )
    assert all(item.tensor.dtype == torch.float64 for item in scores)
    assert all(item.tensor.device.type == "cpu" for item in scores)
    assert scores[0].identity.input_sha256s == tuple(
        bundle.ldsr_for_seed(seed).prediction_sha256 for seed in K5A_SEEDS
    )
    assert scores[1].identity.input_sha256s == tuple(
        bundle.ldsr_for_seed(seed).prediction_sha256 for seed in K5B_SEEDS
    )
    assert scores[2].identity.input_sha256s == tuple(
        bundle.ldsr_for_seed(seed).prediction_sha256 for seed in A1_SEEDS
    )
    assert scores[3].identity.input_sha256s == (bundle.ldsr_for_seed(3407).prediction_sha256,)
    assert scores[4].identity.input_sha256s == (
        bundle.bicubic.prediction_sha256,
        bundle.sen2srlite.prediction_sha256,
        bundle.ldsr_for_seed(3407).prediction_sha256,
    )
    assert all(
        item.identity.operator_parameters["lr_sha256"] == tensor_sha256(pair.pair.lr)
        for item in scores
    )
    assert all(item.score_sha256 == tensor_sha256(item.tensor) for item in scores)
    ramp_squared = torch.linspace(0.0, 1.0, 144, dtype=torch.float64).reshape(12, 12).square()
    torch.testing.assert_close(scores[0].tensor, 0.0002 * ramp_squared)
    torch.testing.assert_close(scores[1].tensor, 0.00045 * ramp_squared)
    torch.testing.assert_close(scores[2].tensor, 0.00061 * ramp_squared)
    assert scores[3].tensor[0, 0] == pytest.approx(0.09727272)
    assert scores[3].tensor[-1, -1] == pytest.approx(0.08272727)
    assert scores[4].tensor[0, 0] == pytest.approx(0.00722222)
    assert scores[4].tensor[-1, -1] == pytest.approx(0.0056)


def test_score_map_warm_cache_reuses_exact_bytes_without_recomputing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pair = _small_loaded_pair(correlation_bin=0)
    bundle = _synthetic_prediction_bundle(pair)
    cache = ScoreCache(tmp_path)
    first = build_a1_score_maps(pair, bundle, cache)
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in tmp_path.iterdir()
    }

    def prohibited(*_args, **_kwargs):
        raise AssertionError("warm score cache recomputed a score")

    monkeypatch.setattr(
        "trustsr.evaluation.development_score_audit.ensemble_variance_score", prohibited
    )
    monkeypatch.setattr(
        "trustsr.evaluation.development_score_audit.lr_reprojection_l1_score", prohibited
    )
    monkeypatch.setattr(
        "trustsr.evaluation.development_score_audit.three_model_disagreement_score",
        prohibited,
    )

    second = build_a1_score_maps(pair, bundle, cache)

    assert second == first
    assert {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in tmp_path.iterdir()
    } == before


def test_score_map_rejects_cache_reload_that_differs_from_computed_tensor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pair = _small_loaded_pair(correlation_bin=0)
    bundle = _synthetic_prediction_bundle(pair)
    original_get = ScoreCache.get
    calls = 0

    def changed_reload(cache: ScoreCache, identity):
        nonlocal calls
        calls += 1
        loaded = original_get(cache, identity)
        if calls == 2 and loaded is not None:
            return (loaded + 0.001).contiguous()
        return loaded

    monkeypatch.setattr(ScoreCache, "get", changed_reload)

    with pytest.raises(RuntimeError, match="differs after cache commit"):
        build_a1_score_maps(pair, bundle, ScoreCache(tmp_path))


def test_a1_result_has_hand_controlled_stability_and_distinct_r9_r1(
    tmp_path: Path,
) -> None:
    pairs, bundles = _a1_inputs()

    result, audit = evaluate_a1_smoke(pairs, bundles, ScoreCache(tmp_path))

    assert result["schema"] == A1_RESULT_SCHEMA
    assert audit["schema"] == A1_CACHE_AUDIT_SCHEMA
    assert A1_RESULT_SCHEMA == "trustsr.phase2b3a-development-smoke.v2"
    assert A1_CACHE_AUDIT_SCHEMA == "trustsr.phase2b3a-development-smoke-cache-audit.v2"
    assert result["normalization_policy"] == PHASE2B3A_NORMALIZATION_POLICY
    assert audit["normalization_policy"] == PHASE2B3A_NORMALIZATION_POLICY
    assert result["sample_count"] == audit["sample_count"] == 4
    assert result["prediction_count"] == audit["prediction_count"] == 108
    assert result["score_count"] == audit["score_count"] == 20
    assert result["k5_statistically_stable"] is True
    assert result["include_ldsr_variance_k5"] is True
    assert [record["correlation_bin"] for record in result["samples"]] == [0, 1, 2, 3]
    assert [(entry["model_name"], entry["seed"]) for entry in audit["prediction_entries"][:27]] == [
        ("bicubic-x4", None),
        ("sen2srlite-x4", None),
        *(("ldsr-s2-x4", seed) for seed in A1_SEEDS),
    ]
    assert [entry["name"] for entry in audit["score_entries"][:5]] == [
        "ldsr_variance_k5a",
        "ldsr_variance_k5b",
        "ldsr_variance_k25",
        "lr_reprojection_l1",
        "three_model_disagreement",
    ]
    assert all(len(entry["files"]) == 2 for entry in audit["score_entries"])
    for record in result["samples"]:
        assert record["stability"] == {
            "k5a_k5b_spearman": 1.0,
            "k5a_k25_spearman": 1.0,
            "k5a_k25_top10_jaccard": 1.0,
            "k5a_constant_score": False,
            "k5b_constant_score": False,
            "k25_constant_score": False,
        }
        assert record["risks"]["primary"]["window"] == PRIMARY_RISK_WINDOW
        assert record["risks"]["sensitivity"]["window"] == SENSITIVITY_RISK_WINDOW
        assert (
            record["risks"]["primary"]["risk_sha256"]
            != record["risks"]["sensitivity"]["risk_sha256"]
        )
    assert canonical_json(result) == canonical_json(deepcopy(result))
    assert canonical_json(audit) == canonical_json(deepcopy(audit))
    prohibited_keys = {"host", "hostname", "path", "runtime", "timestamp", "gpu", "cuda"}
    assert not prohibited_keys.intersection(str(key).lower() for key in _all_keys(result))
    assert not prohibited_keys.intersection(str(key).lower() for key in _all_keys(audit))


def test_a1_serializes_known_saturation_and_builds_literal_aggregate(
    tmp_path: Path,
) -> None:
    pairs, bundles = _a1_inputs()
    object.__setattr__(
        pairs[0].metadata,
        "lr_saturation",
        RadiometricSaturation(208, 11968, 8, (4, 0, 0, 4)),
    )
    object.__setattr__(
        pairs[0].metadata,
        "hr_saturation",
        RadiometricSaturation(208, 11968, 117, (56, 0, 0, 61)),
    )

    result, audit = evaluate_a1_smoke(pairs, bundles, ScoreCache(tmp_path))

    assert result["samples"][0]["radiometric_saturation"] == {
        "lr": {
            "raw_crop_minimum": 208,
            "raw_crop_maximum": 11968,
            "clipped_high_count": 8,
            "clipped_high_by_band": [4, 0, 0, 4],
        },
        "hr": {
            "raw_crop_minimum": 208,
            "raw_crop_maximum": 11968,
            "clipped_high_count": 117,
            "clipped_high_by_band": [56, 0, 0, 61],
        },
    }
    assert result["radiometric_policy"] == {
        "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
        "raw_radiometric_max": 32767,
        "saturation_threshold": 10000,
        "bands": ["B04", "B03", "B02", "B08"],
        "sample_count": 4,
        "affected_sample_count": 1,
        "affected_asset_count": 2,
        "lr_clipped_high_count": 8,
        "hr_clipped_high_count": 117,
        "raw_crop_maximum": 11968,
    }
    assert "radiometric_saturation" not in audit


def _all_keys(value: object):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


def test_a1_constant_maps_record_zero_correlations_and_constant_flags(
    tmp_path: Path,
) -> None:
    pairs, bundles = _a1_inputs(constant_maps=True)

    result, _ = evaluate_a1_smoke(pairs, bundles, ScoreCache(tmp_path))

    for record in result["samples"]:
        assert record["stability"]["k5a_k5b_spearman"] == 0.0
        assert record["stability"]["k5a_k25_spearman"] == 0.0
        assert record["stability"]["k5a_constant_score"] is True
        assert record["stability"]["k5b_constant_score"] is True
        assert record["stability"]["k25_constant_score"] is True
    assert result["k5_statistically_stable"] is False


def test_a1_independent_score_caches_produce_byte_identical_payloads(
    tmp_path: Path,
) -> None:
    pairs, bundles = _a1_inputs()

    first = evaluate_a1_smoke(pairs, bundles, ScoreCache(tmp_path / "first"))
    second = evaluate_a1_smoke(pairs, bundles, ScoreCache(tmp_path / "second"))

    assert canonical_json(first[0]) == canonical_json(second[0])
    assert canonical_json(first[1]) == canonical_json(second[1])


def _rho(permutation: tuple[float, ...]) -> float:
    increasing = torch.tensor((1.0, 2.0, 3.0, 4.0), dtype=torch.float64)
    # These four-rank permutations have exact rational Spearman values at one
    # decimal place; normalize floating linear-algebra noise to that hand result.
    return round(
        score_map_spearman(increasing, torch.tensor(permutation, dtype=torch.float64)),
        1,
    )


def _jaccard_with_shared_top_pixels(shared: int) -> float:
    first = torch.arange(30, dtype=torch.float64)
    top = {27, 28, 29}
    replacement = tuple(sorted(top)[:shared]) + tuple(range(24, 27 - shared))
    second = torch.zeros(30, dtype=torch.float64)
    for rank, index in enumerate(replacement, start=1):
        second[index] = 30.0 + rank
    return top_fraction_jaccard(first, second, fraction=0.10)


def test_a1_stability_threshold_boundaries_use_hand_controlled_maps() -> None:
    rho_1 = _rho((1.0, 2.0, 3.0, 4.0))
    rho_08 = _rho((1.0, 2.0, 4.0, 3.0))
    rho_06 = _rho((2.0, 1.0, 4.0, 3.0))
    rho_04 = _rho((1.0, 3.0, 4.0, 2.0))
    rho_02 = _rho((1.0, 4.0, 3.0, 2.0))
    jaccard_05 = _jaccard_with_shared_top_pixels(2)
    jaccard_below = _jaccard_with_shared_top_pixels(1)
    assert (rho_1, rho_08, rho_06, rho_04, rho_02) == pytest.approx((1.0, 0.8, 0.6, 0.4, 0.2))
    assert (jaccard_05, jaccard_below) == pytest.approx((0.5, 0.2))

    # Every comparison is inclusive exactly at its frozen boundary.
    assert _is_k5_statistically_stable(
        (rho_04, rho_06, rho_06, rho_1),
        (rho_06, rho_08, rho_08, rho_1),
        (jaccard_05,) * 4,
    )
    # Each individual boundary rejects a hand-controlled value below it.
    assert not _is_k5_statistically_stable(
        (rho_02, rho_08, rho_08, rho_1),
        (rho_06, rho_08, rho_08, rho_1),
        (jaccard_05,) * 4,
    )
    assert not _is_k5_statistically_stable(
        (rho_04, rho_04, rho_06, rho_1),
        (rho_06, rho_08, rho_08, rho_1),
        (jaccard_05,) * 4,
    )
    assert not _is_k5_statistically_stable(
        (rho_04, rho_06, rho_06, rho_1),
        (rho_04, rho_08, rho_08, rho_1),
        (jaccard_05,) * 4,
    )
    assert not _is_k5_statistically_stable(
        (rho_04, rho_06, rho_06, rho_1),
        (rho_06, rho_06, rho_08, rho_1),
        (jaccard_05,) * 4,
    )
    assert not _is_k5_statistically_stable(
        (rho_04, rho_06, rho_06, rho_1),
        (rho_06, rho_08, rho_08, rho_1),
        (jaccard_below, jaccard_below, jaccard_05, jaccard_05),
    )


@pytest.mark.parametrize(
    "damage",
    [
        "wrong-order",
        "missing-bin",
        "wrong-split",
        "wrong-day",
        "wrong-round",
        "duplicate-sample",
        "duplicate-group",
        "missing-saturation",
    ],
)
def test_a1_rejects_any_noncanonical_four_roi_input(tmp_path: Path, damage: str) -> None:
    pairs, bundles = _a1_inputs()
    pairs = list(pairs)
    bundles = list(bundles)
    if damage == "wrong-order":
        pairs[0], pairs[1] = pairs[1], pairs[0]
        bundles[0], bundles[1] = bundles[1], bundles[0]
    elif damage == "missing-bin":
        pairs.pop()
        bundles.pop()
    else:
        field, value = {
            "wrong-split": ("split", "calibration"),
            "wrong-day": ("days_between", 0),
            "wrong-round": ("selection_round", 2),
            "duplicate-sample": ("sample_id", pairs[1].metadata.sample_id),
            "duplicate-group": ("spatial_group_id", pairs[1].metadata.spatial_group_id),
            "missing-saturation": ("lr_saturation", None),
        }[damage]
        object.__setattr__(pairs[0].metadata, field, value)

    with pytest.raises(ValueError, match="four|canonical|development|identit|spatial|saturation"):
        evaluate_a1_smoke(pairs, bundles, ScoreCache(tmp_path))


def _committed_a1(tmp_path: Path):
    prediction_cache = PredictionCache(tmp_path / "predictions")
    score_cache = ScoreCache(tmp_path / "scores")
    pairs, bundles = _a1_inputs(prediction_cache)
    result, audit = evaluate_a1_smoke(pairs, bundles, score_cache)
    return pairs, result, audit, prediction_cache, score_cache


def test_a1_replay_is_inference_free_and_byte_mtime_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs, result, audit, prediction_cache, score_cache = _committed_a1(tmp_path)
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for root in (prediction_cache.root, score_cache.root)
        for path in root.iterdir()
    }

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("trustsr.models.ldsr"):
            raise AssertionError("A1 replay imported an LDSR implementation")
        return original_import(name, *args, **kwargs)

    def prohibited(*_args, **_kwargs):
        raise AssertionError("A1 replay touched a model, generation, or compute seam")

    monkeypatch.setattr(bicubic, "BicubicX4", prohibited)
    monkeypatch.setattr(sen2srlite, "SEN2SRLiteX4", prohibited)
    monkeypatch.setattr(ldsr_s2, "LDSRS2X4", prohibited)
    monkeypatch.setattr(development_predictions, "load_or_generate_prediction_bundle", prohibited)
    monkeypatch.setattr(development_score_audit, "evaluate_a1_smoke", prohibited)
    monkeypatch.setattr(development_score_audit, "build_a1_score_maps", prohibited)
    monkeypatch.setattr(development_score_audit, "ensemble_variance_score", prohibited)
    monkeypatch.setattr(development_score_audit, "lr_reprojection_l1_score", prohibited)
    monkeypatch.setattr(development_score_audit, "three_model_disagreement_score", prohibited)
    monkeypatch.setattr(PredictionCache, "put", prohibited)
    monkeypatch.setattr(ScoreCache, "put", prohibited)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    rebuilt_result, rebuilt_audit = replay_a1_smoke(
        pairs, result, audit, prediction_cache, score_cache
    )

    assert canonical_json(rebuilt_result) == canonical_json(result)
    assert canonical_json(rebuilt_audit) == canonical_json(audit)
    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for root in (prediction_cache.root, score_cache.root)
        for path in root.iterdir()
    } == before


@pytest.mark.parametrize(
    "target",
    [
        "result-schema",
        "audit-schema",
        "sample-order",
        "sample-bin",
        "prediction-seed",
        "prediction-key",
        "prediction-sha",
        "score-key",
        "score-sha",
        "score-file-sha",
        "prediction-identity-sha",
        "prediction-count",
        "score-count",
        "prediction-entry-count",
        "score-entry-count",
    ],
)
def test_a1_replay_rejects_mutated_schema_order_bin_seed_key_hash_or_count(
    tmp_path: Path, target: str
) -> None:
    pairs, result, audit, prediction_cache, score_cache = _committed_a1(tmp_path)
    result = deepcopy(result)
    audit = deepcopy(audit)
    if target == "result-schema":
        result["schema"] = "changed"
    elif target == "audit-schema":
        audit["schema"] = "changed"
    elif target == "sample-order":
        result["samples"].reverse()
    elif target == "sample-bin":
        result["samples"][0]["correlation_bin"] = 3
    elif target == "prediction-seed":
        audit["prediction_entries"][2]["seed"] = 9999
    elif target == "prediction-key":
        audit["prediction_entries"][0]["cache_key"] = "0" * 64
    elif target == "prediction-sha":
        audit["prediction_entries"][0]["prediction_sha256"] = "0" * 64
    elif target == "score-key":
        audit["score_entries"][0]["cache_key"] = "0" * 64
    elif target == "score-sha":
        audit["score_entries"][0]["score_sha256"] = "0" * 64
    elif target == "score-file-sha":
        audit["score_entries"][0]["files"][0]["sha256"] = "0" * 64
    elif target == "prediction-identity-sha":
        audit["prediction_entries"][0]["identity"]["lr"]["sha256"] = "0" * 64
    elif target == "prediction-count":
        audit["prediction_count"] = 107
    elif target == "score-count":
        result["score_count"] = 19
    elif target == "prediction-entry-count":
        audit["prediction_entries"].pop()
    else:
        audit["score_entries"].pop()

    with pytest.raises((ValueError, RuntimeError), match="A1|cache|committed|audit|result"):
        replay_a1_smoke(pairs, result, audit, prediction_cache, score_cache)


@pytest.mark.parametrize(
    "target",
    [
        "negative-count",
        "wrong-band-length",
        "wrong-band-total",
        "boolean-integer",
        "minimum-above-maximum",
        "maximum-out-of-domain",
        "inconsistent-aggregate",
        "boolean-aggregate",
        "wrong-result-policy",
        "wrong-audit-policy",
    ],
)
def test_a1_replay_rejects_malformed_or_inconsistent_radiometric_evidence(
    tmp_path: Path, target: str
) -> None:
    pairs, result, audit, prediction_cache, score_cache = _committed_a1(tmp_path)
    result, audit = deepcopy(result), deepcopy(audit)
    lr = result["samples"][0]["radiometric_saturation"]["lr"]
    if target == "negative-count":
        lr["clipped_high_count"] = -1
    elif target == "wrong-band-length":
        lr["clipped_high_by_band"] = [0, 0, 0]
    elif target == "wrong-band-total":
        lr["clipped_high_count"] = 1
    elif target == "boolean-integer":
        lr["raw_crop_minimum"] = True
    elif target == "minimum-above-maximum":
        lr["raw_crop_minimum"] = 4001
    elif target == "maximum-out-of-domain":
        lr["raw_crop_maximum"] = 32768
    elif target == "inconsistent-aggregate":
        result["radiometric_policy"]["affected_sample_count"] = 1
    elif target == "boolean-aggregate":
        result["radiometric_policy"]["affected_sample_count"] = False
    elif target == "wrong-result-policy":
        result["normalization_policy"] = "uint16_divide_10000_no_clip_v1"
    else:
        audit["normalization_policy"] = "uint16_divide_10000_no_clip_v1"

    with pytest.raises((TypeError, ValueError), match="radiometric|normalization|policy"):
        replay_a1_smoke(pairs, result, audit, prediction_cache, score_cache)


@pytest.mark.parametrize(
    ("maximum", "clipped"),
    [(10000, 1), (10001, 0)],
)
def test_online_radiometric_policy_rejects_threshold_mismatches_from_evidence(
    maximum: int, clipped: int
) -> None:
    """Fails if online aggregate accepts evidence the offline verifier rejects."""

    sample = {
        "radiometric_saturation": {
            "lr": {
                "raw_crop_minimum": 5000,
                "raw_crop_maximum": maximum,
                "clipped_high_count": clipped,
                "clipped_high_by_band": [clipped, 0, 0, 0],
            },
            "hr": {
                "raw_crop_minimum": 5000,
                "raw_crop_maximum": 10000,
                "clipped_high_count": 0,
                "clipped_high_by_band": [0, 0, 0, 0],
            },
        }
    }

    with pytest.raises(ValueError, match="maximum and clipped count are inconsistent"):
        development_score_audit._build_radiometric_policy(
            [sample], expected_sample_count=1
        )


def test_a1_replay_detects_cache_mtime_change_during_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs, result, audit, prediction_cache, score_cache = _committed_a1(tmp_path)
    first_path = next(prediction_cache.root.glob("*.json"))
    original_get = ScoreCache.get
    changed = False

    def changing_get(cache: ScoreCache, identity):
        nonlocal changed
        value = original_get(cache, identity)
        if not changed:
            stat = first_path.stat()
            os.utime(first_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            changed = True
        return value

    monkeypatch.setattr(ScoreCache, "get", changing_get)

    with pytest.raises(RuntimeError, match="changed during A1 replay"):
        replay_a1_smoke(pairs, result, audit, prediction_cache, score_cache)


@pytest.mark.parametrize("kind", ["prediction", "score"])
def test_a1_replay_preserves_cache_integrity_failures_without_recompute(
    tmp_path: Path, kind: str
) -> None:
    pairs, result, audit, prediction_cache, score_cache = _committed_a1(tmp_path)
    if kind == "prediction":
        first_key = audit["prediction_entries"][0]["cache_key"]
        (prediction_cache.root / f"{first_key}.json").unlink()
    else:
        first_key = audit["score_entries"][0]["cache_key"]
        (score_cache.root / f"{first_key}.json").unlink()

    with pytest.raises((ValueError, RuntimeError), match="cache"):
        replay_a1_smoke(pairs, result, audit, prediction_cache, score_cache)


def test_a1_replay_rejects_self_consistent_alternate_score_identity_before_mutation(
    tmp_path: Path,
) -> None:
    pairs, result, audit, prediction_cache, score_cache = _committed_a1(tmp_path)
    result = deepcopy(result)
    audit = deepcopy(audit)
    entry = audit["score_entries"][0]
    canonical_key = entry["cache_key"]
    canonical_paths = tuple(
        score_cache.root / f"{canonical_key}{suffix}"
        for suffix in (".json", ".safetensors", ".lock")
    )
    canonical_identity = ScoreIdentity(
        score_name=entry["identity"]["score_name"],
        score_schema_version=entry["identity"]["score_schema_version"],
        sample_id=entry["identity"]["sample_id"],
        input_sha256s=tuple(entry["identity"]["input_sha256s"]),
        operator_parameters=entry["identity"]["operator_parameters"],
    )
    canonical_score = score_cache.get(canonical_identity)
    assert canonical_score is not None
    alternate_identity = ScoreIdentity(
        score_name=canonical_identity.score_name,
        score_schema_version=canonical_identity.score_schema_version,
        sample_id=canonical_identity.sample_id,
        input_sha256s=canonical_identity.input_sha256s,
        operator_parameters={
            **canonical_identity.operator_parameters,
            "seed_count": 999,
        },
    )
    alternate_score = (canonical_score + 0.25).contiguous()
    alternate_sha256 = tensor_sha256(alternate_score)
    score_cache.put(alternate_identity, alternate_score)
    alternate_evidence = score_entry_evidence(score_cache.root, alternate_identity)
    entry["identity"] = alternate_identity.as_dict()
    entry["cache_key"] = alternate_identity.key
    entry["score_sha256"] = alternate_sha256
    entry["files"] = [
        {
            "filename": alternate_evidence[kind]["filename"],
            "size_bytes": (score_cache.root / alternate_evidence[kind]["filename"]).stat().st_size,
            "sha256": alternate_evidence[kind]["sha256"],
        }
        for kind in ("json", "safetensors")
    ]
    result["samples"][0]["scores"][0]["cache_key"] = alternate_identity.key
    result["samples"][0]["scores"][0]["score_sha256"] = alternate_sha256
    for path in canonical_paths:
        path.unlink()
    assert all(not path.exists() for path in canonical_paths)

    with pytest.raises(ValueError):
        replay_a1_smoke(pairs, result, audit, prediction_cache, score_cache)

    assert all(not path.exists() for path in canonical_paths)


def _small_a2_pair(day: int, bin_index: int, round_index: int) -> LoadedCrosssensorPair:
    pair = _small_loaded_pair(bin_index)
    sample_id = f"development-{day}-{bin_index}-{round_index}"
    slope = 0.12 + 0.0005 * ((day + 1) * 40 + bin_index * 10 + round_index)
    row = torch.linspace(0.0, 1.0, 12, dtype=torch.float32)
    hr = (0.35 + slope * row).view(1, 1, 12).expand(4, 12, 12).contiguous()
    if (day, bin_index, round_index) == (-1, 0, 1):
        lr_saturation = RadiometricSaturation(208, 11968, 8, (4, 0, 0, 4))
        hr_saturation = RadiometricSaturation(208, 11968, 117, (56, 0, 0, 61))
    else:
        lr_saturation = RadiometricSaturation(4000, 4000, 0, (0, 0, 0, 0))
        hr_saturation = RadiometricSaturation(3500, 5500, 0, (0, 0, 0, 0))
    return LoadedCrosssensorPair(
        pair=SRPair(
            source=pair.pair.source,
            sample_id=sample_id,
            lr=pair.pair.lr.clone(),
            hr=hr,
            scale=4,
        ),
        metadata=CrosssensorPairMetadata(
            manifest_sha256=POST_MANIFEST_SHA256,
            sample_id=sample_id,
            split="development",
            spatial_group_id=f"group-{day}-{bin_index}-{round_index}",
            days_between=day,
            correlation_bin=bin_index,
            selection_round=round_index,
            lr_asset_sha256="a" * 64,
            hr_asset_sha256="b" * 64,
            lr_crop_transform=pair.metadata.lr_crop_transform,
            hr_crop_transform=pair.metadata.hr_crop_transform,
            crop_bounds=pair.metadata.crop_bounds,
            crop_policy=CROP_POLICY,
            normalization_policy=PHASE2B3A_NORMALIZATION_POLICY,
            lr_saturation=lr_saturation,
            hr_saturation=hr_saturation,
        ),
    )


def _a2_inputs(
    prediction_cache: PredictionCache | None = None,
    *,
    include_ldsr_variance_k5: bool = True,
    constant_maps: bool = False,
) -> tuple[tuple[LoadedCrosssensorPair, ...], tuple[DevelopmentPredictionBundle, ...]]:
    pairs = tuple(
        _small_a2_pair(day, bin_index, round_index)
        for day in (-1, 0, 1)
        for bin_index in range(4)
        for round_index in range(1, 11)
    )
    seeds = K5A_SEEDS if include_ldsr_variance_k5 else (3407,)
    bundles = tuple(
        _synthetic_prediction_bundle(
            pair,
            cache=prediction_cache,
            constant_maps=constant_maps,
            seeds=seeds,
        )
        for pair in pairs
    )
    return pairs, bundles


def _ordered_a2_sample_ids(
    pairs: tuple[LoadedCrosssensorPair, ...] | list[LoadedCrosssensorPair],
) -> tuple[str, ...]:
    return tuple(pair.pair.sample_id for pair in pairs)


def _committed_a2(
    tmp_path: Path,
    *,
    include_ldsr_variance_k5: bool = True,
    constant_maps: bool = False,
):
    prediction_cache = PredictionCache(tmp_path / "predictions")
    score_cache = ScoreCache(tmp_path / "scores")
    pairs, bundles = _a2_inputs(
        prediction_cache,
        include_ldsr_variance_k5=include_ldsr_variance_k5,
        constant_maps=constant_maps,
    )
    result, audit = evaluate_a2_development(
        pairs,
        iter(bundles),
        prediction_cache=prediction_cache,
        score_cache=score_cache,
        include_ldsr_variance_k5=include_ldsr_variance_k5,
        code_revision="d" * 40,
        ordered_development_sample_ids=_ordered_a2_sample_ids(pairs),
    )
    return pairs, result, audit, prediction_cache, score_cache


@pytest.fixture(scope="module")
def committed_a2(tmp_path_factory: pytest.TempPathFactory):
    return _committed_a2(tmp_path_factory.mktemp("committed-a2"))


def test_a2_freezes_only_from_exact_complete_development_set(committed_a2) -> None:
    pairs, result, audit, _, _ = committed_a2

    assert result["schema"] == A2_RESULT_SCHEMA
    assert A2_RESULT_SCHEMA == "trustsr.phase2b3a-development-score-audit.v1"
    assert A2_CACHE_AUDIT_SCHEMA == "trustsr.phase2b3a-development-score-cache-audit.v1"
    assert result["normalization_policy"] == PHASE2B3A_NORMALIZATION_POLICY
    assert audit["normalization_policy"] == PHASE2B3A_NORMALIZATION_POLICY
    assert result["samples"][0]["radiometric_saturation"] == {
        "lr": {
            "raw_crop_minimum": 208,
            "raw_crop_maximum": 11968,
            "clipped_high_count": 8,
            "clipped_high_by_band": [4, 0, 0, 4],
        },
        "hr": {
            "raw_crop_minimum": 208,
            "raw_crop_maximum": 11968,
            "clipped_high_count": 117,
            "clipped_high_by_band": [56, 0, 0, 61],
        },
    }
    assert result["radiometric_policy"] == {
        "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
        "raw_radiometric_max": 32767,
        "saturation_threshold": 10000,
        "bands": ["B04", "B03", "B02", "B08"],
        "sample_count": 120,
        "affected_sample_count": 1,
        "affected_asset_count": 2,
        "lr_clipped_high_count": 8,
        "hr_clipped_high_count": 117,
        "raw_crop_maximum": 11968,
    }
    assert result["sample_count"] == 120
    assert result["statistical_unit"] == "roi"
    assert result["bootstrap"] == {
        "algorithm": "numpy.PCG64",
        "seed": 23031,
        "resamples": 10_000,
        "ci_percentiles": [2.5, 97.5],
    }
    assert result["phase_decision"] == "freeze_score"
    assert result["frozen_score"]["name"] in COST_ORDER
    assert result["frozen_score"]["post_manifest_sha256"] == POST_MANIFEST_SHA256
    assert result["frozen_score"]["code_revision"] == "d" * 40
    assert result["frozen_score"]["operator_parameters"]
    assert result["frozen_score"]["seeds"]
    assert [item["name"] for item in result["candidate_summaries"]] == list(A2_SCORE_NAMES)
    assert all("sensitivity_window_1" in item for item in result["candidate_summaries"])
    assert all(record["split"] == "development" for record in result["samples"])
    assert len({record["spatial_group_id"] for record in result["samples"]}) == 120
    assert audit["schema"] == A2_CACHE_AUDIT_SCHEMA
    assert audit["sample_count"] == len(pairs) == 120
    assert len(audit["groups"]) == 120
    assert all("radiometric_saturation" in sample for sample in result["samples"])
    assert all("radiometric_saturation" not in group for group in audit["groups"])


def test_a2_rejects_synchronized_pair_bundle_reversal_against_authoritative_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_pairs, original_bundles = _a2_inputs()
    reversed_pairs = tuple(reversed(original_pairs))
    reversed_bundles = tuple(reversed(original_bundles))

    def prohibited(*_args, **_kwargs):
        raise AssertionError("non-authoritative A2 order touched a cache")

    monkeypatch.setattr(PredictionCache, "get", prohibited)
    monkeypatch.setattr(ScoreCache, "get", prohibited)
    with pytest.raises(ValueError, match="authoritative|ordered.*sample"):
        evaluate_a2_development(
            reversed_pairs,
            reversed_bundles,
            prediction_cache=PredictionCache(tmp_path / "predictions"),
            score_cache=ScoreCache(tmp_path / "scores"),
            include_ldsr_variance_k5=True,
            code_revision="d" * 40,
            ordered_development_sample_ids=_ordered_a2_sample_ids(original_pairs),
        )


@pytest.mark.parametrize("damage", ["count", "empty", "duplicate", "non-string"])
def test_a2_rejects_invalid_authoritative_sample_ids_before_cache_access(
    tmp_path: Path, damage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs, bundles = _a2_inputs()
    ordered_ids: list[object] = list(_ordered_a2_sample_ids(pairs))
    if damage == "count":
        ordered_ids.pop()
    elif damage == "empty":
        ordered_ids[0] = ""
    elif damage == "duplicate":
        ordered_ids[-1] = ordered_ids[0]
    else:
        ordered_ids[0] = 1

    def prohibited(*_args, **_kwargs):
        raise AssertionError("invalid authoritative IDs touched a cache")

    monkeypatch.setattr(PredictionCache, "get", prohibited)
    monkeypatch.setattr(ScoreCache, "get", prohibited)
    with pytest.raises((TypeError, ValueError), match="ordered.*sample|120|unique|string"):
        evaluate_a2_development(
            pairs,
            bundles,
            prediction_cache=PredictionCache(tmp_path / "predictions"),
            score_cache=ScoreCache(tmp_path / "scores"),
            include_ldsr_variance_k5=True,
            code_revision="d" * 40,
            ordered_development_sample_ids=ordered_ids,  # type: ignore[arg-type]
        )


def test_a2_consumes_one_bundle_at_a_time_and_releases_prediction_tensors(
    tmp_path: Path,
) -> None:
    pairs, _ = _a2_inputs(include_ldsr_variance_k5=False)
    prediction_cache = PredictionCache(tmp_path / "predictions")
    released_checks = 0

    def one_shot_bundles():
        nonlocal released_checks
        previous_tensors: tuple[weakref.ReferenceType[torch.Tensor], ...] = ()
        for pair in pairs:
            gc.collect()
            assert all(reference() is None for reference in previous_tensors)
            if previous_tensors:
                released_checks += 1
            bundle = _synthetic_prediction_bundle(
                pair,
                cache=prediction_cache,
                seeds=(3407,),
            )
            previous_tensors = tuple(
                weakref.ref(item.tensor)
                for item in (bundle.bicubic, bundle.sen2srlite, *bundle.ldsr)
            )
            yield bundle
            del bundle
        gc.collect()
        assert all(reference() is None for reference in previous_tensors)
        released_checks += 1

    result, _ = evaluate_a2_development(
        pairs,
        one_shot_bundles(),
        prediction_cache=prediction_cache,
        score_cache=ScoreCache(tmp_path / "scores"),
        include_ldsr_variance_k5=False,
        code_revision="d" * 40,
        ordered_development_sample_ids=_ordered_a2_sample_ids(pairs),
    )

    assert result["sample_count"] == 120
    assert released_checks == 120


def test_a2_rejects_a_short_bundle_iterable_explicitly(tmp_path: Path) -> None:
    pairs, _ = _a2_inputs(include_ldsr_variance_k5=False)

    with pytest.raises(ValueError, match="bundle iterable ended before 120"):
        evaluate_a2_development(
            pairs,
            iter(()),
            prediction_cache=PredictionCache(tmp_path / "predictions"),
            score_cache=ScoreCache(tmp_path / "scores"),
            include_ldsr_variance_k5=False,
            code_revision="d" * 40,
            ordered_development_sample_ids=_ordered_a2_sample_ids(pairs),
        )


def test_a2_rejects_an_extra_bundle_after_processing_exactly_120(tmp_path: Path) -> None:
    prediction_cache = PredictionCache(tmp_path / "predictions")
    pairs, bundles = _a2_inputs(
        prediction_cache,
        include_ldsr_variance_k5=False,
    )

    with pytest.raises(ValueError, match="more than 120"):
        evaluate_a2_development(
            pairs,
            iter((*bundles, bundles[0])),
            prediction_cache=prediction_cache,
            score_cache=ScoreCache(tmp_path / "scores"),
            include_ldsr_variance_k5=False,
            code_revision="d" * 40,
            ordered_development_sample_ids=_ordered_a2_sample_ids(pairs),
        )


def _controlled_a2_score_maps(
    pair: LoadedCrosssensorPair,
    bundle: DevelopmentPredictionBundle,
    score_cache: ScoreCache,
    *,
    include_ldsr_variance_k5: bool,
):
    assert include_ldsr_variance_k5 is True
    records = []
    for name, base_marker in zip(A2_SCORE_NAMES, (1.0, 2.0, 3.0), strict=True):
        marker = base_marker
        if (
            name == "ldsr_variance_k5"
            and pair.metadata.days_between == -1
            and pair.metadata.correlation_bin == 0
        ):
            marker = 4.0
        identity = development_score_audit._a2_score_identity(
            pair,
            name=name,
            input_sha256s=development_score_audit._a2_score_input_hashes(bundle, name),
        )
        records.append(
            development_score_audit._load_or_compute_score(
                name,
                identity,
                score_cache,
                lambda marker=marker: torch.full(
                    pair.pair.hr.shape[1:], marker, dtype=torch.float64
                ),
            )
        )
    return tuple(records)


def _controlled_a2_diagnostic(
    score: torch.Tensor,
    _risk: torch.Tensor,
    *,
    sensitivity_direction: int = 1,
) -> RoiScoreDiagnostics:
    marker = int(score[0, 0].item())
    if marker == 1:
        rho, gain, constant = 0.4 * sensitivity_direction, 0.02 * sensitivity_direction, False
    elif marker == 2:
        rho, gain, constant = 0.0, 0.0, True
    elif marker == 3:
        rho, gain, constant = 0.4, 0.02, False
    else:
        rho, gain, constant = -0.11, 0.02, False
    random_aurc = 0.1
    aurc = random_aurc - gain
    return RoiScoreDiagnostics(
        rho=rho,
        constant_score=constant,
        coverages=tuple(index / 10 for index in range(1, 11)),
        selective_mean_risks=(aurc,) * 10,
        aurc=aurc,
        random_aurc=random_aurc,
        aurc_gain=gain,
        high_risk_miss_rate_at_80=0.2,
    )


def _controlled_a2_bundles(
    pairs: tuple[LoadedCrosssensorPair, ...], prediction_cache: PredictionCache
):
    for pair in pairs:
        yield _synthetic_prediction_bundle(
            pair,
            cache=prediction_cache,
            seeds=K5A_SEEDS,
        )


def test_a2_orchestration_retains_positive_random_and_one_failed_stratum_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs, _ = _a2_inputs()
    prediction_cache = PredictionCache(tmp_path / "predictions")
    monkeypatch.setattr(
        development_score_audit, "_build_a2_score_maps", _controlled_a2_score_maps
    )
    monkeypatch.setattr(
        development_score_audit,
        "evaluate_roi_score",
        lambda score, risk: _controlled_a2_diagnostic(score, risk),
    )

    result, _ = evaluate_a2_development(
        pairs,
        _controlled_a2_bundles(pairs, prediction_cache),
        prediction_cache=prediction_cache,
        score_cache=ScoreCache(tmp_path / "scores"),
        include_ldsr_variance_k5=True,
        code_revision="d" * 40,
        ordered_development_sample_ids=_ordered_a2_sample_ids(pairs),
    )

    summaries = {item["name"]: item["primary_window_9"] for item in result["candidate_summaries"]}
    positive = summaries["lr_reprojection_l1"]
    random_equivalent = summaries["three_model_disagreement"]
    failed_stratum = summaries["ldsr_variance_k5"]
    assert positive["eligible"] is True
    assert positive["mean_rho"] == pytest.approx(0.4)
    assert random_equivalent["mean_rho"] == 0.0
    assert random_equivalent["mean_aurc_gain"] == 0.0
    assert random_equivalent["eligible"] is False
    assert failed_stratum["positive_strata"] == 11
    assert failed_stratum["minimum_stratum_mean_rho"] == pytest.approx(-0.11)
    assert failed_stratum["failure_reasons"] == ["a stratum mean rho is below -0.10"]
    assert result["frozen_score"]["name"] == "lr_reprojection_l1"
    assert [
        item["name"] for item in result["frozen_score"]["candidate_eligibility_evidence"]
    ] == list(A2_SCORE_NAMES)
    for candidate in result["candidate_summaries"]:
        for window in ("primary_window_9", "sensitivity_window_1"):
            strata = candidate[window]["stratum_mean_rho"]
            assert len(strata) == 12
            assert all(
                set(entry)
                == {
                    "days_between",
                    "correlation_bin",
                    "mean_rho",
                    "mean_aurc_gain",
                }
                for entry in strata
            )
            expected_gain = 0.0 if candidate["name"] == "three_model_disagreement" else 0.02
            assert [entry["mean_aurc_gain"] for entry in strata] == pytest.approx(
                [expected_gain] * 12
            )


def test_a2_r1_only_perturbation_changes_sensitivity_but_not_frozen_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs, _ = _a2_inputs()
    prediction_cache = PredictionCache(tmp_path / "predictions")
    score_cache = ScoreCache(tmp_path / "scores")
    sensitivity_direction = 1
    monkeypatch.setattr(
        development_score_audit, "_build_a2_score_maps", _controlled_a2_score_maps
    )
    monkeypatch.setattr(
        development_score_audit,
        "local_l1_risk",
        lambda sr, _hr, *, window: torch.full(
            sr.shape[1:], 0.9 if window == 9 else 0.1, dtype=torch.float64
        ),
    )

    def diagnostics(score: torch.Tensor, risk: torch.Tensor) -> RoiScoreDiagnostics:
        direction = 1 if float(risk[0, 0]) == 0.9 else sensitivity_direction
        return _controlled_a2_diagnostic(
            score, risk, sensitivity_direction=direction
        )

    monkeypatch.setattr(development_score_audit, "evaluate_roi_score", diagnostics)

    def evaluate_once():
        return evaluate_a2_development(
            pairs,
            _controlled_a2_bundles(pairs, prediction_cache),
            prediction_cache=prediction_cache,
            score_cache=score_cache,
            include_ldsr_variance_k5=True,
            code_revision="d" * 40,
            ordered_development_sample_ids=_ordered_a2_sample_ids(pairs),
        )[0]

    first = evaluate_once()
    sensitivity_direction = -1
    second = evaluate_once()

    assert first["frozen_score"] == second["frozen_score"]
    assert [item["primary_window_9"] for item in first["candidate_summaries"]] == [
        item["primary_window_9"] for item in second["candidate_summaries"]
    ]
    assert [item["sensitivity_window_1"] for item in first["candidate_summaries"]] != [
        item["sensitivity_window_1"] for item in second["candidate_summaries"]
    ]


@pytest.mark.parametrize(
    "damage",
    [
        "119-roi",
        "121-roi",
        "duplicate-sample",
        "duplicate-group",
        "reordered-bundle",
        "calibration",
        "broken-stratum",
        "missing-saturation",
    ],
)
def test_a2_rejects_incomplete_leaky_or_mismatched_membership_before_cache_access(
    tmp_path: Path, damage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs, bundles = _a2_inputs()
    pairs, bundles = list(pairs), list(bundles)
    if damage == "119-roi":
        pairs.pop()
        bundles.pop()
    elif damage == "121-roi":
        pairs.append(pairs[0])
        bundles.append(bundles[0])
    elif damage == "duplicate-sample":
        object.__setattr__(pairs[-1].metadata, "sample_id", pairs[0].metadata.sample_id)
    elif damage == "duplicate-group":
        object.__setattr__(
            pairs[-1].metadata, "spatial_group_id", pairs[0].metadata.spatial_group_id
        )
    elif damage == "reordered-bundle":
        bundles[0], bundles[1] = bundles[1], bundles[0]
    elif damage == "calibration":
        object.__setattr__(pairs[0].metadata, "split", "calibration")
    elif damage == "missing-saturation":
        object.__setattr__(pairs[0].metadata, "hr_saturation", None)
    else:
        object.__setattr__(pairs[-1].metadata, "selection_round", 9)

    def prohibited(*_args, **_kwargs):
        raise AssertionError("invalid A2 membership touched a cache")

    monkeypatch.setattr(PredictionCache, "get", prohibited)
    monkeypatch.setattr(ScoreCache, "get", prohibited)
    with pytest.raises(
        ValueError, match="120|unique|bundle|development|strat|identit|saturation"
    ):
        evaluate_a2_development(
            pairs,
            bundles,
            prediction_cache=PredictionCache(tmp_path / "predictions"),
            score_cache=ScoreCache(tmp_path / "scores"),
            include_ldsr_variance_k5=True,
            code_revision="d" * 40,
            ordered_development_sample_ids=_ordered_a2_sample_ids(pairs),
        )


@pytest.mark.parametrize("value", [None, 1, "true"])
def test_a2_rejects_non_boolean_a1_variance_acceptance(
    tmp_path: Path, value: object
) -> None:
    pairs, bundles = _a2_inputs()

    with pytest.raises((TypeError, ValueError), match="include_ldsr_variance_k5|boolean"):
        evaluate_a2_development(
            pairs,
            bundles,
            prediction_cache=PredictionCache(tmp_path / "predictions"),
            score_cache=ScoreCache(tmp_path / "scores"),
            include_ldsr_variance_k5=value,  # type: ignore[arg-type]
            code_revision="d" * 40,
            ordered_development_sample_ids=_ordered_a2_sample_ids(pairs),
        )


@pytest.mark.parametrize("revision", ["d" * 39, "D" * 40, "g" * 40, 1])
def test_a2_rejects_any_non_commit_code_revision(tmp_path: Path, revision: object) -> None:
    pairs, bundles = _a2_inputs()

    with pytest.raises((TypeError, ValueError), match="code_revision"):
        evaluate_a2_development(
            pairs,
            bundles,
            prediction_cache=PredictionCache(tmp_path / "predictions"),
            score_cache=ScoreCache(tmp_path / "scores"),
            include_ldsr_variance_k5=True,
            code_revision=revision,  # type: ignore[arg-type]
            ordered_development_sample_ids=_ordered_a2_sample_ids(pairs),
        )


def test_a2_rejects_variance_seed_bundle_after_a1_removed_variance(tmp_path: Path) -> None:
    pairs, bundles = _a2_inputs(include_ldsr_variance_k5=True)

    with pytest.raises(ValueError, match="A1|seed|variance"):
        evaluate_a2_development(
            pairs,
            bundles,
            prediction_cache=PredictionCache(tmp_path / "predictions"),
            score_cache=ScoreCache(tmp_path / "scores"),
            include_ldsr_variance_k5=False,
            code_revision="d" * 40,
            ordered_development_sample_ids=_ordered_a2_sample_ids(pairs),
        )


def test_a2_no_eligible_score_is_an_explicit_stop_with_all_evidence(tmp_path: Path) -> None:
    _, result, _, _, _ = _committed_a2(
        tmp_path,
        include_ldsr_variance_k5=False,
        constant_maps=True,
    )

    assert result["frozen_score"] is None
    assert result["phase_decision"] == "stop_no_eligible_score"
    assert [item["name"] for item in result["candidate_summaries"]] == [
        "lr_reprojection_l1",
        "three_model_disagreement",
    ]
    assert all(
        item["primary_window_9"]["eligible"] is False
        and item["primary_window_9"]["failure_reasons"]
        for item in result["candidate_summaries"]
    )


def test_a2_replay_is_cache_only_and_byte_mtime_stable(
    committed_a2, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs, result, audit, prediction_cache, score_cache = committed_a2
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for root in (prediction_cache.root, score_cache.root)
        for path in root.iterdir()
    }

    def prohibited(*_args, **_kwargs):
        raise AssertionError("A2 replay touched a model, generation, score compute, or write seam")

    monkeypatch.setattr(bicubic, "BicubicX4", prohibited)
    monkeypatch.setattr(sen2srlite, "SEN2SRLiteX4", prohibited)
    monkeypatch.setattr(ldsr_s2, "LDSRS2X4", prohibited)
    monkeypatch.setattr(development_predictions, "load_or_generate_prediction_bundle", prohibited)
    monkeypatch.setattr(development_score_audit, "evaluate_a2_development", prohibited)
    monkeypatch.setattr(development_score_audit, "ensemble_variance_score", prohibited)
    monkeypatch.setattr(development_score_audit, "lr_reprojection_l1_score", prohibited)
    monkeypatch.setattr(development_score_audit, "three_model_disagreement_score", prohibited)
    monkeypatch.setattr(PredictionCache, "put", prohibited)
    monkeypatch.setattr(ScoreCache, "put", prohibited)

    rebuilt_result, rebuilt_audit = replay_a2_development(
        pairs,
        result,
        audit,
        prediction_cache,
        score_cache,
        ordered_development_sample_ids=_ordered_a2_sample_ids(pairs),
    )

    assert canonical_json(rebuilt_result) == canonical_json(result)
    assert canonical_json(rebuilt_audit) == canonical_json(audit)
    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for root in (prediction_cache.root, score_cache.root)
        for path in root.iterdir()
    } == before


def test_a2_replay_recomputes_bootstrap_selection(
    committed_a2, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs, result, audit, prediction_cache, score_cache = committed_a2

    monkeypatch.setattr(
        "trustsr.evaluation.score_selection.build_bootstrap_indices",
        lambda: torch.zeros((10_000, 120), dtype=torch.int64).numpy(),
    )

    with pytest.raises(ValueError, match="rebuilt A2 result"):
        replay_a2_development(
            pairs,
            result,
            audit,
            prediction_cache,
            score_cache,
            ordered_development_sample_ids=_ordered_a2_sample_ids(pairs),
        )


@pytest.mark.parametrize(
    "target",
    [
        "result-schema",
        "audit-schema",
        "sample-count",
        "group-count",
        "sample-order",
        "prediction-key",
        "prediction-sha",
        "score-key",
        "score-sha",
        "score-file-sha",
        "score-config",
        "code-revision",
        "normalization-policy",
        "radiometric-aggregate",
    ],
)
def test_a2_replay_rejects_mutated_commitment_or_cache_identity(
    committed_a2, target: str
) -> None:
    pairs, result, audit, prediction_cache, score_cache = committed_a2
    result, audit = deepcopy(result), deepcopy(audit)
    if target == "result-schema":
        result["schema"] = "changed"
    elif target == "audit-schema":
        audit["schema"] = "changed"
    elif target == "sample-count":
        result["sample_count"] = 119
    elif target == "group-count":
        audit["groups"].pop()
    elif target == "sample-order":
        result["samples"].reverse()
    elif target == "prediction-key":
        audit["groups"][0]["prediction_entries"][0]["cache_key"] = "0" * 64
    elif target == "prediction-sha":
        audit["groups"][0]["prediction_entries"][0]["prediction_sha256"] = "0" * 64
    elif target == "score-key":
        audit["groups"][0]["score_entries"][0]["cache_key"] = "0" * 64
    elif target == "score-sha":
        audit["groups"][0]["score_entries"][0]["score_sha256"] = "0" * 64
    elif target == "score-file-sha":
        audit["groups"][0]["score_entries"][0]["files"][0]["sha256"] = "0" * 64
    elif target == "score-config":
        result["score_configuration"]["lr_reprojection_l1"]["scale"] = 2
    elif target == "normalization-policy":
        audit["normalization_policy"] = "uint16_divide_10000_no_clip_v1"
    elif target == "radiometric-aggregate":
        result["radiometric_policy"]["affected_asset_count"] = 1
    else:
        result["code_revision"] = "e" * 40

    with pytest.raises(
        (ValueError, RuntimeError), match="A2|cache|committed|rebuilt|radiometric"
    ):
        replay_a2_development(
            pairs,
            result,
            audit,
            prediction_cache,
            score_cache,
            ordered_development_sample_ids=_ordered_a2_sample_ids(pairs),
        )


@pytest.mark.parametrize("kind", ["prediction", "score"])
def test_a2_replay_requires_every_existing_cache_entry(committed_a2, kind: str) -> None:
    pairs, result, audit, prediction_cache, score_cache = committed_a2
    entry = audit["groups"][0][f"{kind}_entries"][0]
    cache = prediction_cache if kind == "prediction" else score_cache
    path = cache.root / f"{entry['cache_key']}.json"
    original = path.read_bytes()
    path.unlink()
    try:
        with pytest.raises((ValueError, RuntimeError), match="cache"):
            replay_a2_development(
                pairs,
                result,
                audit,
                prediction_cache,
                score_cache,
                ordered_development_sample_ids=_ordered_a2_sample_ids(pairs),
            )
    finally:
        path.write_bytes(original)


@pytest.mark.parametrize("kind", ["prediction", "score"])
def test_a2_replay_rebuilds_canonical_identity_before_any_cache_access(
    committed_a2, kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs, result, audit, prediction_cache, score_cache = committed_a2
    result, audit = deepcopy(result), deepcopy(audit)
    entry = audit["groups"][0][f"{kind}_entries"][0]
    if kind == "prediction":
        entry["identity"]["lr"]["sha256"] = "0" * 64
        altered = development_score_audit._prediction_identity_from_dict(entry["identity"])
        entry["cache_key"] = altered.key
    else:
        entry["identity"]["operator_parameters"]["scale"] = 2
        altered = ScoreIdentity(
            score_name=entry["identity"]["score_name"],
            score_schema_version=entry["identity"]["score_schema_version"],
            sample_id=entry["identity"]["sample_id"],
            input_sha256s=tuple(entry["identity"]["input_sha256s"]),
            operator_parameters=entry["identity"]["operator_parameters"],
        )
        entry["cache_key"] = altered.key

    def prohibited(*_args, **_kwargs):
        raise AssertionError("noncanonical A2 identity reached cache access")

    monkeypatch.setattr(ScoreCache, "get", prohibited)
    if kind == "prediction":
        monkeypatch.setattr(PredictionCache, "get", prohibited)
    else:
        monkeypatch.setattr(
            development_score_audit, "_snapshot_score_files", prohibited
        )
    with pytest.raises(ValueError, match="committed A2.*identity|score identity"):
        replay_a2_development(
            pairs,
            result,
            audit,
            prediction_cache,
            score_cache,
            ordered_development_sample_ids=_ordered_a2_sample_ids(pairs),
        )


def test_a2_replay_verifies_prediction_digest_before_touching_score_namespace(
    committed_a2, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs, result, audit, prediction_cache, score_cache = committed_a2
    audit = deepcopy(audit)
    group = audit["groups"][0]
    prediction_entry = group["prediction_entries"][2]
    original_digest = prediction_entry["prediction_sha256"]
    altered_digest = "e" * 64
    assert original_digest != altered_digest
    prediction_entry["prediction_sha256"] = altered_digest
    for score_entry in group["score_entries"]:
        identity = score_entry["identity"]
        identity["input_sha256s"] = [
            altered_digest if digest == original_digest else digest
            for digest in identity["input_sha256s"]
        ]
        altered_identity = ScoreIdentity(
            score_name=identity["score_name"],
            score_schema_version=identity["score_schema_version"],
            sample_id=identity["sample_id"],
            input_sha256s=tuple(identity["input_sha256s"]),
            operator_parameters=identity["operator_parameters"],
        )
        score_entry["cache_key"] = altered_identity.key

    def prohibited_score_access(*_args, **_kwargs):
        raise AssertionError("A2 replay touched score namespace before prediction verification")

    monkeypatch.setattr(development_score_audit, "_snapshot_score_files", prohibited_score_access)
    monkeypatch.setattr(ScoreCache, "get", prohibited_score_access)
    with pytest.raises(ValueError, match="prediction logical tensor SHA"):
        replay_a2_development(
            pairs,
            result,
            audit,
            prediction_cache,
            score_cache,
            ordered_development_sample_ids=_ordered_a2_sample_ids(pairs),
        )


def test_a2_replay_rejects_incomplete_diagnostics_before_cache_access(
    committed_a2, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs, result, audit, prediction_cache, score_cache = committed_a2
    result = deepcopy(result)
    del result["samples"][0]["scores"][0]["sensitivity_window_1"]["aurc_gain"]

    def prohibited(*_args, **_kwargs):
        raise AssertionError("incomplete A2 diagnostics reached cache access")

    monkeypatch.setattr(PredictionCache, "get", prohibited)
    monkeypatch.setattr(ScoreCache, "get", prohibited)
    with pytest.raises(ValueError, match="diagnostics.*incomplete"):
        replay_a2_development(
            pairs,
            result,
            audit,
            prediction_cache,
            score_cache,
            ordered_development_sample_ids=_ordered_a2_sample_ids(pairs),
        )


def test_a2_replay_detects_cache_mtime_change_during_rebuild(
    committed_a2, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs, result, audit, prediction_cache, score_cache = committed_a2
    path = next(prediction_cache.root.glob("*.json"))
    original_stat = path.stat()
    original_get = ScoreCache.get
    changed = False

    def changing_get(cache: ScoreCache, identity):
        nonlocal changed
        value = original_get(cache, identity)
        if not changed:
            os.utime(
                path,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1_000_000),
            )
            changed = True
        return value

    monkeypatch.setattr(ScoreCache, "get", changing_get)
    try:
        with pytest.raises(RuntimeError, match="changed during A2 replay"):
            replay_a2_development(
                pairs,
                result,
                audit,
                prediction_cache,
                score_cache,
                ordered_development_sample_ids=_ordered_a2_sample_ids(pairs),
            )
    finally:
        os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
