"""Pure cache identity audit for fixed Phase 2B3-C internal-test maps."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from trustsr.evaluation.internal_test_maps import InternalTestMaps
from trustsr.evaluation.internal_test_predictions import (
    SEEDS,
    CachedInternalTestPrediction,
    InternalTestPredictionBundle,
)
from trustsr.jsonio import canonical_json


def _ordered_sample_ids_sha256(sample_ids: tuple[str, ...]) -> str:
    if any(type(sample_id) is not str or not sample_id for sample_id in sample_ids):
        raise ValueError("internal_test sample identities must be non-empty strings")
    return hashlib.sha256(canonical_json(sample_ids)).hexdigest()


def _prediction_entry(item: CachedInternalTestPrediction) -> dict[str, object]:
    return {
        "model_name": item.model_name,
        "seed": item.seed,
        "cache_key": item.identity.key,
        "identity": item.identity.as_dict(),
        "prediction_sha256": item.prediction_sha256,
    }


def _validate_inputs(
    bundles: Sequence[InternalTestPredictionBundle],
    maps: Sequence[InternalTestMaps],
) -> tuple[tuple[InternalTestPredictionBundle, InternalTestMaps], ...]:
    if not isinstance(bundles, Sequence) or not isinstance(maps, Sequence):
        raise TypeError("internal_test cache audit inputs must be sequences")
    bundle_values = tuple(bundles)
    map_values = tuple(maps)
    if len(bundle_values) != 120 or len(map_values) != 120:
        raise ValueError("internal_test cache audit requires exactly 120 bundles and maps")
    validated: list[tuple[InternalTestPredictionBundle, InternalTestMaps]] = []
    global_model_identity: dict[str, object] | None = None
    for bundle, sample_maps in zip(bundle_values, map_values, strict=True):
        if not isinstance(bundle, InternalTestPredictionBundle):
            raise TypeError("internal_test cache audit requires prediction bundles")
        if not isinstance(sample_maps, InternalTestMaps):
            raise TypeError("internal_test cache audit requires internal_test maps")
        if type(bundle.items) is not tuple or len(bundle.items) != len(SEEDS):
            raise ValueError("internal_test cache audit requires the fixed K5 bundle")
        for item in bundle.items:
            if not isinstance(item, CachedInternalTestPrediction):
                raise TypeError("internal_test cache audit bundle item is invalid")
            item.__post_init__()
        bundle.__post_init__()
        sample_maps.__post_init__()
        model_identity = {
            key: value
            for key, value in bundle.items[0].identity.model_provenance.items()
            if key != "seed"
        }
        if global_model_identity is None:
            global_model_identity = model_identity
        elif model_identity != global_model_identity:
            raise ValueError(
                "internal_test cache audit has mismatched model scientific identities"
            )
        if bundle.sample_id != sample_maps.sample_id:
            raise ValueError("internal_test bundle and maps sample order differs")
        prediction_sha256s = tuple(item.prediction_sha256 for item in bundle.items)
        if sample_maps.score_prediction_sha256s != prediction_sha256s:
            raise ValueError("internal_test score input digests differ from the prediction bundle")
        first_identity = bundle.items[0].identity
        parameters = sample_maps.score.identity.operator_parameters
        if (
            parameters.get("lr_sha256") != first_identity.lr_sha256
            or parameters.get("source") != first_identity.source
        ):
            raise ValueError(
                "internal_test score LR input identity differs from the prediction bundle"
            )
        validated.append((bundle, sample_maps))
    sample_ids = tuple(bundle.sample_id for bundle, _ in validated)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("internal_test cache audit requires unique sample identities")
    return tuple(validated)


def _sample_entry(
    bundle: InternalTestPredictionBundle, maps: InternalTestMaps
) -> dict[str, object]:
    score = maps.score
    return {
        "sample_id": bundle.sample_id,
        "predictions": tuple(_prediction_entry(item) for item in bundle.items),
        "score": {
            "name": score.name,
            "cache_key": score.identity.key,
            "identity": score.identity.as_dict(),
            "score_sha256": score.score_sha256,
        },
        "risk": {
            "name": maps.risk_name,
            "window": maps.risk_window,
            "risk_sha256": maps.risk_sha256,
        },
    }


def build_internal_test_cache_audit(
    bundles: Sequence[InternalTestPredictionBundle],
    maps: Sequence[InternalTestMaps],
) -> dict[str, object]:
    """Enumerate the fixed cache identities without reading cache files or pixels."""

    validated = _validate_inputs(bundles, maps)
    samples = tuple(_sample_entry(bundle, sample_maps) for bundle, sample_maps in validated)
    return {
        "schema": "trustsr.phase2b3c-evaluation-cache-audit.v1",
        "split": "internal_test",
        "ordered_sample_ids_sha256": _ordered_sample_ids_sha256(
            tuple(bundle.sample_id for bundle, _ in validated)
        ),
        "sample_count": len(samples),
        "prediction_count": sum(len(bundle.items) for bundle, _ in validated),
        "score_count": len(validated),
        "samples": samples,
    }
