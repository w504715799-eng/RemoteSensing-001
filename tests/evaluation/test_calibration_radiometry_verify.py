"""Independent parsed-receipt tests for Phase 2B3-B calibration radiometry."""

from __future__ import annotations

import copy
import hashlib
import importlib
from dataclasses import FrozenInstanceError
from types import MappingProxyType, ModuleType

import pytest

from trustsr.jsonio import canonical_json


def _module() -> ModuleType:
    return importlib.import_module("trustsr.evaluation.calibration_radiometry_verify")


def _saturation(
    minimum: object = 100,
    maximum: object = 9000,
    clipped: object = 0,
    by_band: object = None,
) -> dict[str, object]:
    return {
        "raw_crop_minimum": minimum,
        "raw_crop_maximum": maximum,
        "clipped_high_count": clipped,
        "clipped_high_by_band": [0, 0, 0, 0] if by_band is None else by_band,
    }


def _document() -> dict[str, object]:
    samples: list[dict[str, object]] = []
    for index, (day, bin_index, round_index) in enumerate(
        (day, bin_index, round_index)
        for day in (-1, 0, 1)
        for bin_index in range(4)
        for round_index in range(1, 11)
    ):
        if index == 0:
            lr = _saturation(208, 11968, 8, [4, 0, 0, 4])
            hr = _saturation(208, 11968, 117, [56, 0, 0, 61])
        else:
            lr = _saturation(100, 9000)
            hr = _saturation(50, 9500)
        samples.append(
            {
                "sample_id": f"calibration-{index:03d}",
                "days_between": day,
                "correlation_bin": bin_index,
                "selection_round": round_index,
                "radiometric_saturation": {"lr": lr, "hr": hr},
            }
        )
    return {
        "schema": "trustsr.phase2b3b-calibration-radiometry.v1",
        "split": "calibration",
        "ordered_sample_ids_sha256": hashlib.sha256(
            canonical_json([sample["sample_id"] for sample in samples])
        ).hexdigest(),
        "policy": {
            "normalization_policy": "uint16_saturate_10000_divide_10000_v2",
            "raw_radiometric_max": 32767,
            "saturation_threshold": 10000,
            "saturation_operation": "minimum(raw,10000)",
            "saturation_scope": "aligned_crop_only",
            "reflectance_divisor": 10000.0,
            "crop_policy": "center_crop_lr_1_hr_4_v1",
            "bands": ["B04", "B03", "B02", "B08"],
        },
        "sample_count": 120,
        "affected_sample_count": 1,
        "lr": _saturation(100, 11968, 8, [4, 0, 0, 4]),
        "hr": _saturation(50, 11968, 117, [56, 0, 0, 61]),
        "samples": samples,
    }


def _aggregate_payload(document: dict[str, object]) -> dict[str, object]:
    return {
        "sample_count": document["sample_count"],
        "affected_sample_count": document["affected_sample_count"],
        "lr": document["lr"],
        "hr": document["hr"],
    }


def test_independently_verifies_receipt_and_returns_immutable_host_free_digests() -> None:
    document = _document()

    verified = _module().verify_calibration_radiometry(document)

    assert verified.source_sha256 == hashlib.sha256(canonical_json(document)).hexdigest()
    assert verified.split == "calibration"
    assert verified.ordered_sample_ids_sha256 == document["ordered_sample_ids_sha256"]
    assert verified.sample_count == 120
    assert verified.affected_sample_count == 1
    assert verified.aggregate_sha256 == hashlib.sha256(
        canonical_json(_aggregate_payload(document))
    ).hexdigest()
    assert verified.aggregates == {
        "lr": {
            "raw_crop_minimum": 100,
            "raw_crop_maximum": 11968,
            "clipped_high_count": 8,
            "clipped_high_by_band": (4, 0, 0, 4),
        },
        "hr": {
            "raw_crop_minimum": 50,
            "raw_crop_maximum": 11968,
            "clipped_high_count": 117,
            "clipped_high_by_band": (56, 0, 0, 61),
        },
    }
    assert "calibration-" not in repr(verified)
    with pytest.raises(FrozenInstanceError):
        verified.sample_count = 0
    with pytest.raises(TypeError):
        verified.aggregates["lr"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        verified.aggregates["lr"]["clipped_high_by_band"] = ()  # type: ignore[index]


def test_sample_reordering_is_bound_by_source_digest_but_not_aggregate_digest() -> None:
    module = _module()
    first_document = _document()
    second_document = copy.deepcopy(first_document)
    second_document["samples"][0], second_document["samples"][1] = (  # type: ignore[index]
        second_document["samples"][1],  # type: ignore[index]
        second_document["samples"][0],  # type: ignore[index]
    )
    second_document["ordered_sample_ids_sha256"] = hashlib.sha256(
        canonical_json(
            [sample["sample_id"] for sample in second_document["samples"]]  # type: ignore[index]
        )
    ).hexdigest()

    first = module.verify_calibration_radiometry(first_document)
    second = module.verify_calibration_radiometry(second_document)

    assert first.source_sha256 != second.source_sha256
    assert first.aggregate_sha256 == second.aggregate_sha256


@pytest.mark.parametrize(
    "fault", ("extra", "missing", "schema", "split", "ordered_digest", "non_dict")
)
def test_rejects_wrong_top_level_schema_or_keys(fault: str) -> None:
    module = _module()
    document: object = _document()
    if fault == "extra":
        document["extra"] = True  # type: ignore[index]
    elif fault == "missing":
        document.pop("hr")  # type: ignore[union-attr]
    elif fault == "schema":
        document["schema"] = "wrong"  # type: ignore[index]
    elif fault == "split":
        document["split"] = "internal_test"  # type: ignore[index]
    elif fault == "ordered_digest":
        document["ordered_sample_ids_sha256"] = "0" * 64  # type: ignore[index]
    else:
        document = MappingProxyType(document)  # type: ignore[arg-type]

    with pytest.raises((TypeError, ValueError), match="receipt|schema|split|digest|JSON"):
        module.verify_calibration_radiometry(document)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("normalization_policy", "wrong"),
        ("raw_radiometric_max", True),
        ("saturation_threshold", 9999),
        ("saturation_operation", "clip"),
        ("saturation_scope", "full_asset"),
        ("reflectance_divisor", 10000),
        ("crop_policy", "wrong"),
        ("bands", ["B08", "B03", "B02", "B04"]),
    ),
)
def test_rejects_wrong_or_forged_policy(field: str, value: object) -> None:
    module = _module()
    document = _document()
    document["policy"][field] = value  # type: ignore[index]

    with pytest.raises((TypeError, ValueError), match="policy"):
        module.verify_calibration_radiometry(document)


def test_rejects_missing_or_extra_policy_key() -> None:
    module = _module()
    missing = _document()
    missing["policy"].pop("crop_policy")  # type: ignore[union-attr]
    extra = _document()
    extra["policy"]["extra"] = True  # type: ignore[index]

    with pytest.raises(ValueError, match="policy"):
        module.verify_calibration_radiometry(missing)
    with pytest.raises(ValueError, match="policy"):
        module.verify_calibration_radiometry(extra)


@pytest.mark.parametrize("fault", ("119", "duplicate", "stratum", "round", "sample_extra"))
def test_rejects_wrong_sample_count_identity_strata_or_keys(fault: str) -> None:
    module = _module()
    document = _document()
    samples = document["samples"]
    if fault == "119":
        samples.pop()  # type: ignore[union-attr]
    elif fault == "duplicate":
        samples[-1]["sample_id"] = samples[0]["sample_id"]  # type: ignore[index]
    elif fault == "stratum":
        samples[0]["days_between"] = 2  # type: ignore[index]
    elif fault == "round":
        samples[0]["selection_round"] = 2  # type: ignore[index]
    else:
        samples[0]["extra"] = True  # type: ignore[index]

    with pytest.raises((TypeError, ValueError), match="120|sample|stratum|round|unique"):
        module.verify_calibration_radiometry(document)


@pytest.mark.parametrize(
    "fault",
    (
        "integer_type",
        "band_container",
        "asset_extra",
        "maximum",
        "band_sum",
        "max_count",
        "minimum",
    ),
)
def test_rejects_forged_per_sample_saturation(fault: str) -> None:
    module = _module()
    document = _document()
    asset = document["samples"][0]["radiometric_saturation"]["lr"]  # type: ignore[index]
    if fault == "integer_type":
        asset["clipped_high_count"] = True
    elif fault == "band_container":
        asset["clipped_high_by_band"] = (4, 0, 0, 4)
    elif fault == "asset_extra":
        asset["extra"] = 0
    elif fault == "maximum":
        asset["raw_crop_maximum"] = 32768
    elif fault == "band_sum":
        asset["clipped_high_by_band"] = [3, 0, 0, 4]
    elif fault == "max_count":
        asset["raw_crop_maximum"] = 10000
    else:
        asset["raw_crop_minimum"] = 12000

    with pytest.raises((TypeError, ValueError), match="saturation|maximum|minimum|band|integer"):
        module.verify_calibration_radiometry(document)


@pytest.mark.parametrize("fault", ("affected", "lr_total", "hr_band", "lr_minimum"))
def test_rejects_claimed_aggregate_that_differs_from_samples(fault: str) -> None:
    module = _module()
    document = _document()
    if fault == "affected":
        document["affected_sample_count"] = 2
    elif fault == "lr_total":
        document["lr"]["clipped_high_count"] = 9  # type: ignore[index]
    elif fault == "hr_band":
        document["hr"]["clipped_high_by_band"][0] = 55  # type: ignore[index]
    else:
        document["lr"]["raw_crop_minimum"] = 99  # type: ignore[index]

    with pytest.raises((TypeError, ValueError), match="aggregate|affected"):
        module.verify_calibration_radiometry(document)


def test_does_not_read_files_or_import_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("verifier attempted file I/O")

    monkeypatch.setattr("builtins.open", forbidden)

    assert module.verify_calibration_radiometry(_document()).sample_count == 120
    assert "build_calibration_radiometry" not in module.__dict__
