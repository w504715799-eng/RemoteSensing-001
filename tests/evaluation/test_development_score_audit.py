"""Four-ROI A1 score construction, stability, evidence, and replay."""

from __future__ import annotations

import builtins
import os
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from trustsr.artifacts.predictions import PredictionCache, build_identity, tensor_sha256
from trustsr.artifacts.scores import ScoreCache
from trustsr.contracts import SRPair
from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
    CrosssensorPairMetadata,
    LoadedCrosssensorPair,
)
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
    PRIMARY_RISK_WINDOW,
    SENSITIVITY_RISK_WINDOW,
    _is_k5_statistically_stable,
    build_a1_score_maps,
    evaluate_a1_smoke,
    replay_a1_smoke,
)
from trustsr.evaluation.score_diagnostics import (
    score_map_spearman,
    top_fraction_jaccard,
)
from trustsr.jsonio import canonical_json


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
            normalization_policy=NORMALIZATION_POLICY,
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
        for seed in A1_SEEDS
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
        }[damage]
        object.__setattr__(pairs[0].metadata, field, value)

    with pytest.raises(ValueError, match="four|canonical|development|identit|spatial"):
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
