"""Pure cache identity audit for fixed Phase 2B3-B calibration maps."""

from __future__ import annotations

from collections.abc import Sequence

from trustsr.evaluation.calibration_maps import CalibrationMaps
from trustsr.evaluation.calibration_predictions import (
    SEEDS,
    CachedCalibrationPrediction,
    CalibrationPredictionBundle,
)


def _prediction_entry(item: CachedCalibrationPrediction) -> dict[str, object]:
    return {
        "model_name": item.model_name,
        "seed": item.seed,
        "cache_key": item.identity.key,
        "identity": item.identity.as_dict(),
        "prediction_sha256": item.prediction_sha256,
    }


def _validate_inputs(
    bundles: Sequence[CalibrationPredictionBundle],
    maps: Sequence[CalibrationMaps],
) -> tuple[tuple[CalibrationPredictionBundle, CalibrationMaps], ...]:
    if not isinstance(bundles, Sequence) or not isinstance(maps, Sequence):
        raise TypeError("calibration cache audit inputs must be sequences")
    bundle_values = tuple(bundles)
    map_values = tuple(maps)
    if len(bundle_values) != 120 or len(map_values) != 120:
        raise ValueError("calibration cache audit requires exactly 120 bundles and maps")
    validated: list[tuple[CalibrationPredictionBundle, CalibrationMaps]] = []
    for bundle, sample_maps in zip(bundle_values, map_values, strict=True):
        if not isinstance(bundle, CalibrationPredictionBundle):
            raise TypeError("calibration cache audit requires prediction bundles")
        if not isinstance(sample_maps, CalibrationMaps):
            raise TypeError("calibration cache audit requires calibration maps")
        if type(bundle.items) is not tuple or len(bundle.items) != len(SEEDS):
            raise ValueError("calibration cache audit requires the fixed K5 bundle")
        for item in bundle.items:
            if not isinstance(item, CachedCalibrationPrediction):
                raise TypeError("calibration cache audit bundle item is invalid")
            item.__post_init__()
        bundle.__post_init__()
        sample_maps.__post_init__()
        if bundle.sample_id != sample_maps.sample_id:
            raise ValueError("calibration bundle and maps sample order differs")
        prediction_sha256s = tuple(item.prediction_sha256 for item in bundle.items)
        if sample_maps.score_prediction_sha256s != prediction_sha256s:
            raise ValueError("calibration score input digests differ from the prediction bundle")
        first_identity = bundle.items[0].identity
        parameters = sample_maps.score.identity.operator_parameters
        if (
            parameters.get("lr_sha256") != first_identity.lr_sha256
            or parameters.get("source") != first_identity.source
        ):
            raise ValueError(
                "calibration score LR input identity differs from the prediction bundle"
            )
        validated.append((bundle, sample_maps))
    sample_ids = tuple(bundle.sample_id for bundle, _ in validated)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("calibration cache audit requires unique sample identities")
    return tuple(validated)


def _sample_entry(
    bundle: CalibrationPredictionBundle, maps: CalibrationMaps
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


def build_calibration_cache_audit(
    bundles: Sequence[CalibrationPredictionBundle],
    maps: Sequence[CalibrationMaps],
) -> dict[str, object]:
    """Enumerate the fixed cache identities without reading cache files or pixels."""

    validated = _validate_inputs(bundles, maps)
    samples = tuple(_sample_entry(bundle, sample_maps) for bundle, sample_maps in validated)
    return {
        "schema": "trustsr.phase2b3b-calibration-cache-audit.v1",
        "sample_count": len(samples),
        "prediction_count": sum(len(bundle.items) for bundle, _ in validated),
        "score_count": len(validated),
        "samples": samples,
    }
