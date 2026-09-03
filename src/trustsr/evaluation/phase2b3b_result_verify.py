"""Independent, metadata-only verification of a Phase 2B3-B final result."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from trustsr.data.calibration_pairs import validate_calibration_records
from trustsr.data.calibration_subset import load_calibration_records
from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    PHASE2B3A_NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
)
from trustsr.evaluation.calibration_cache_verify import verify_calibration_cache_audit
from trustsr.evaluation.calibration_fit import (
    FREEZE_CALIBRATION,
    STOP_INSUFFICIENT_COVERAGE,
)
from trustsr.evaluation.calibration_input_receipt import (
    SCHEMA as INPUT_RECEIPT_SCHEMA,
)
from trustsr.evaluation.calibration_input_receipt import (
    verify_calibration_input_receipt,
)
from trustsr.evaluation.calibration_radiometry_verify import (
    verify_calibration_radiometry,
)
from trustsr.evaluation.phase2b3b_preflight import (
    load_phase2b3b_preflight,
    ordered_sample_ids_sha256,
)
from trustsr.evaluation.phase2b3b_revision import (
    verify_recorded_phase2b3b_revision,
)
from trustsr.jsonio import canonical_json

SCHEMA = "trustsr.phase2b3b-calibration-result-metadata-verification.v1"
_RESULT_SCHEMA = "trustsr.phase2b3b-calibration.v1"
_RADIOMETRY_SCHEMA = "trustsr.phase2b3b-calibration-radiometry.v1"
_SOURCE = f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}"
_BANDS = ["B04", "B03", "B02", "B08"]
_SEEDS = [3407, 3408, 3409, 3410, 3411]
_REVISION = re.compile(r"[0-9a-f]{40}")
_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_ISO_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:")
_RESULT_KEYS = {
    "schema",
    "split",
    "upstream",
    "producer_revision",
    "frozen",
    "target",
    "threshold",
    "all_abstain",
    "risk_bound",
    "counts",
    "coverage",
    "radiometry",
    "samples",
    "input_receipt_sha256",
    "ordered_inputs_sha256",
    "cache_audit_sha256",
    "map_evidence_sha256",
    "phase_decision",
}
_SAMPLE_KEYS = {
    "sample_id",
    "split",
    "predictions",
    "score",
    "risk",
    "radiometric_saturation",
    "input",
}
_MEMBERSHIP_FIELDS = (
    "sample_id",
    "selection_sha256",
    "spatial_group_id",
    "days_between",
    "correlation_bin",
    "selection_round",
)


@dataclass(frozen=True)
class VerifiedPhase2B3BResult:
    """Metadata-consistency receipt; cannot prove cache pixels or score/risk calculations.

    This receipt cannot authorize acceptance on its own.
    """

    schema: str
    verification_scope: str
    cache_computation_verified: bool
    result_sha256: str
    cache_audit_sha256: str
    producer_revision: str
    ordered_sample_ids_sha256: str
    ordered_membership_sha256: str
    input_receipt_sha256: str
    ordered_inputs_sha256: str
    map_evidence_sha256: str
    radiometry_aggregate_sha256: str
    phase_decision: str


def _dict(value: object, keys: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{label} must be an exact JSON object")
    return value


def _list(value: object, length: int, label: str) -> list[object]:
    if type(value) is not list or len(value) != length:
        raise ValueError(f"{label} must be an exact {length}-item JSON array")
    return value


def _digest(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _number(value: object, label: str, *, lower: float, lower_open: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} is invalid")
    number = float(value)
    if (
        not math.isfinite(number)
        or number > 1.0
        or number < lower
        or (lower_open and number == lower)
    ):
        raise ValueError(f"{label} is invalid")
    return number


def _reject_non_json_or_leaks(value: object, *, key: str = "result") -> None:
    if type(value) is dict:
        for nested_key, nested in value.items():
            if type(nested_key) is not str:
                raise TypeError("result keys must be built-in strings")
            lowered = nested_key.casefold()
            if any(
                marker in lowered
                for marker in (
                    "path",
                    "endpoint",
                    "credential",
                    "timestamp",
                    "hostname",
                    "password",
                    "secret",
                    "token",
                )
            ):
                raise ValueError("result contains forbidden runtime or credential metadata")
            _reject_non_json_or_leaks(nested, key=nested_key)
        return
    if type(value) is list:
        for nested in value:
            _reject_non_json_or_leaks(nested, key=key)
        return
    if type(value) is str:
        lowered = value.casefold()
        if "internal_test" in lowered or "development" in lowered:
            raise ValueError("result contains a forbidden non-calibration split")
        if (
            value.startswith(("/", "\\\\"))
            or _WINDOWS_PATH.match(value)
            or "://" in value
            or _ISO_TIMESTAMP.match(value)
            or any(
                marker in lowered
                for marker in ("password=", "token=", "credential=", "secret=")
            )
        ):
            raise ValueError("result contains forbidden host or endpoint metadata")
        return
    if type(value) is float and not math.isfinite(value):
        raise ValueError("result numbers must be finite")
    if type(value) not in (int, float, bool, type(None)):
        raise TypeError(f"{key} must contain exact JSON built-in values")


def _loaded_json(value: object, label: str) -> dict[str, object]:
    try:
        normalized = json.loads(canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"trusted {label} is not canonical JSON data") from exc
    if type(normalized) is not dict:
        raise ValueError(f"trusted {label} is not a JSON object")
    return normalized


def _membership(
    raw_records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    records = validate_calibration_records(raw_records)
    membership = []
    for record in records:
        lr_asset = record["lr_asset"]
        hr_asset = record["hr_asset"]
        if not isinstance(lr_asset, Mapping) or not isinstance(hr_asset, Mapping):
            raise ValueError("authoritative calibration assets are invalid")
        membership.append(
            {
                "sample_id": record["sample_id"],
                "selection_sha256": record["selection_sha256"],
                "spatial_group_id": record["spatial_group_id"],
                "lr_asset_sha256": _digest(
                    lr_asset.get("sha256"), "authoritative LR asset digest"
                ),
                "hr_asset_sha256": _digest(
                    hr_asset.get("sha256"), "authoritative HR asset digest"
                ),
                "days_between": record["days_between"],
                "correlation_bin": record["correlation_bin"],
                "selection_round": record["selection_round"],
            }
        )
    return membership


def _validate_authority(
    preflight: dict[str, object], membership: list[dict[str, object]]
) -> tuple[dict[str, object], dict[str, object], list[str], str, str]:
    value = _dict(
        preflight,
        {"schema", "upstream", "calibration", "score", "risk", "input"},
        "trusted preflight",
    )
    if value["schema"] != "trustsr.phase2b3b-preflight.v1":
        raise ValueError("trusted preflight schema is invalid")
    upstream = _dict(
        value["upstream"],
        {
            "publication_commit",
            "producer_revision",
            "post_manifest_sha256",
            "input_audit_sha256",
            "evidence_sha256s",
        },
        "trusted preflight upstream",
    )
    calibration = _dict(
        value["calibration"],
        {
            "split",
            "sample_count",
            "ordered_sample_ids_sha256",
            "ordered_membership_sha256",
            "input_receipt_sha256s",
            "strata",
        },
        "trusted preflight calibration",
    )
    sample_ids = [record["sample_id"] for record in membership]
    ordered_ids = ordered_sample_ids_sha256(sample_ids)
    ordered_membership = _sha256(membership)
    per_record = [_sha256(record) for record in membership]
    counts = Counter(
        (record["days_between"], record["correlation_bin"]) for record in membership
    )
    expected_strata = [
        {"days_between": day, "correlation_bin": bin_index, "sample_count": counts[day, bin_index]}
        for day in (-1, 0, 1)
        for bin_index in range(4)
    ]
    if not (
        calibration["split"] == "calibration"
        and type(calibration["sample_count"]) is int
        and calibration["sample_count"] == 120
        and calibration["ordered_sample_ids_sha256"] == ordered_ids
        and calibration["ordered_membership_sha256"] == ordered_membership
        and calibration["input_receipt_sha256s"] == per_record
        and calibration["strata"] == expected_strata
    ):
        raise ValueError("trusted preflight differs from authoritative calibration metadata")
    result_upstream = {
        "post_manifest_sha256": upstream["post_manifest_sha256"],
        "input_audit_sha256": upstream["input_audit_sha256"],
        "phase2b3a_publication_commit": upstream["publication_commit"],
        "phase2b3a_calculation_revision": upstream["producer_revision"],
        "evidence_sha256s": upstream["evidence_sha256s"],
        "ordered_sample_ids_sha256": ordered_ids,
        "ordered_membership_sha256": ordered_membership,
    }
    frozen = {"score": value["score"], "risk": value["risk"], "input": value["input"]}
    return result_upstream, frozen, per_record, ordered_ids, ordered_membership


def _validate_summary(result: dict[str, object]) -> None:
    target = _dict(result["target"], {"alpha", "minimum_coverage"}, "target")
    alpha = _number(target["alpha"], "alpha", lower=0.0, lower_open=True)
    minimum_coverage = _number(
        target["minimum_coverage"], "minimum coverage", lower=0.0, lower_open=False
    )
    counts = _dict(
        result["counts"],
        {"calibration", "predictions", "scores", "trusted_pixels", "total_pixels"},
        "counts",
    )
    for name in counts:
        if type(counts[name]) is not int:
            raise TypeError(f"{name} must be an exact integer")
    trusted = counts["trusted_pixels"]
    total = counts["total_pixels"]
    if not (
        counts["calibration"] == 120
        and counts["predictions"] == 600
        and counts["scores"] == 120
        and total > 0
        and 0 <= trusted <= total
    ):
        raise ValueError("result counts are inconsistent")
    coverage = _number(result["coverage"], "coverage", lower=0.0, lower_open=False)
    if coverage != trusted / total:
        raise ValueError("coverage differs from exact pixel counts")
    threshold = result["threshold"]
    all_abstain = result["all_abstain"]
    if type(all_abstain) is not bool:
        raise TypeError("all_abstain must be an exact boolean")
    if threshold is None:
        if not all_abstain or trusted != 0:
            raise ValueError("null threshold requires exact all-abstain counts")
    elif (
        isinstance(threshold, bool)
        or not isinstance(threshold, int | float)
        or not math.isfinite(float(threshold))
        or not 0 <= threshold <= 0.25
        or all_abstain
    ):
        raise ValueError("threshold and all_abstain are inconsistent")
    risk_bound = _number(
        result["risk_bound"], "risk bound", lower=0.0, lower_open=True
    )
    if risk_bound < 1 / 121 or (threshold is not None and risk_bound > alpha):
        raise ValueError("risk bound is inconsistent with fixed calibration")
    expected_decision = (
        FREEZE_CALIBRATION
        if threshold is not None and coverage >= minimum_coverage
        else STOP_INSUFFICIENT_COVERAGE
    )
    if result["phase_decision"] != expected_decision:
        raise ValueError("phase decision is inconsistent")


def verify_phase2b3b_result(
    result: object,
    cache_audit: object,
    *,
    project_root: Path,
    evidence_dir: Path,
    storage_root: Path,
    manifest_path: Path,
) -> VerifiedPhase2B3BResult:
    """Verify metadata consistency only; does not prove cache pixels or calculations.

    This function does not replay cache entries or recompute score/risk maps, so its
    receipt cannot authorize acceptance on its own.
    """

    value = _dict(result, _RESULT_KEYS, "Phase 2B3-B result")
    _reject_non_json_or_leaks(value)
    canonical_result = canonical_json(value)
    if value["schema"] != _RESULT_SCHEMA or value["split"] != "calibration":
        raise ValueError("result schema or split is invalid")

    audit_verification = verify_calibration_cache_audit(cache_audit)
    audit = _dict(
        cache_audit,
        {
            "schema",
            "split",
            "ordered_sample_ids_sha256",
            "sample_count",
            "prediction_count",
            "score_count",
            "samples",
        },
        "cache audit",
    )
    preflight = _loaded_json(
        load_phase2b3b_preflight(evidence_dir, storage_root, manifest_path),
        "preflight",
    )
    raw_records = load_calibration_records(storage_root, manifest_path)
    membership = _membership(raw_records)
    expected_upstream, expected_frozen, per_record, ordered_ids, ordered_membership = (
        _validate_authority(preflight, membership)
    )
    if value["upstream"] != expected_upstream or value["frozen"] != expected_frozen:
        raise ValueError("result differs from frozen preflight authority")

    producer = value["producer_revision"]
    if type(producer) is not str or _REVISION.fullmatch(producer) is None:
        raise ValueError("result producer revision is invalid")
    if verify_recorded_phase2b3b_revision(project_root, producer) != producer:
        raise ValueError("recorded producer revision verification is inconsistent")

    result_samples = _list(value["samples"], 120, "result samples")
    audit_samples = _list(audit["samples"], 120, "audit samples")
    receipt_samples: list[dict[str, object]] = []
    radiometry_samples: list[dict[str, object]] = []
    map_evidence: list[dict[str, object]] = []
    for index, (raw_result_sample, raw_audit_sample, member) in enumerate(
        zip(result_samples, audit_samples, membership, strict=True)
    ):
        sample = _dict(raw_result_sample, _SAMPLE_KEYS, "result sample")
        audit_sample = _dict(
            raw_audit_sample,
            {"sample_id", "predictions", "score", "risk"},
            "audit sample",
        )
        input_projection = _dict(
            sample["input"], {"membership_sha256", "lr", "hr"}, "input"
        )
        if not (
            sample["sample_id"] == member["sample_id"] == audit_sample["sample_id"]
            and sample["split"] == "calibration"
            and input_projection["membership_sha256"] == per_record[index]
        ):
            raise ValueError("result samples differ from authoritative membership")
        expected_predictions = [
            {
                "seed": prediction["seed"],
                "cache_key": prediction["cache_key"],
                "prediction_sha256": prediction["prediction_sha256"],
            }
            for prediction in audit_sample["predictions"]
        ]
        if sample["predictions"] != expected_predictions:
            raise ValueError("result prediction projection differs from cache audit")
        expected_score = {
            "cache_key": audit_sample["score"]["cache_key"],
            "score_sha256": audit_sample["score"]["score_sha256"],
        }
        expected_risk = {"risk_sha256": audit_sample["risk"]["risk_sha256"]}
        if sample["score"] != expected_score or sample["risk"] != expected_risk:
            raise ValueError("result map projection differs from cache audit")
        lr = _dict(
            input_projection["lr"],
            {"asset_sha256", "tensor_sha256", "shape", "dtype"},
            "result LR input",
        )
        hr = _dict(
            input_projection["hr"],
            {"asset_sha256", "tensor_sha256", "shape", "dtype"},
            "result HR input",
        )
        audit_lr = audit_sample["predictions"][0]["identity"]["lr"]
        if {
            "shape": lr["shape"],
            "dtype": lr["dtype"],
            "sha256": lr["tensor_sha256"],
        } != audit_lr:
            raise ValueError("result LR input differs from cache audit")
        receipt_samples.append({"membership": member, "lr": dict(lr), "hr": dict(hr)})
        radiometry_samples.append(
            {
                "sample_id": member["sample_id"],
                "days_between": member["days_between"],
                "correlation_bin": member["correlation_bin"],
                "selection_round": member["selection_round"],
                "radiometric_saturation": sample["radiometric_saturation"],
            }
        )
        map_evidence.append(
            {
                "sample_id": member["sample_id"],
                "score_sha256": expected_score["score_sha256"],
                "risk_sha256": expected_risk["risk_sha256"],
            }
        )

    if audit_verification["ordered_sample_ids_sha256"] != ordered_ids:
        raise ValueError("cache audit order differs from authoritative membership")
    audit_sha256 = audit_verification["digests"]["audit_sha256"]
    if _digest(value["cache_audit_sha256"], "result cache audit digest") != audit_sha256:
        raise ValueError("result cache audit digest is inconsistent")
    map_sha256 = _sha256(map_evidence)
    if _digest(value["map_evidence_sha256"], "result map digest") != map_sha256:
        raise ValueError("result map evidence digest is inconsistent")

    input_receipt = {
        "schema": INPUT_RECEIPT_SCHEMA,
        "split": "calibration",
        "sample_count": 120,
        "input": {
            "manifest_sha256": POST_MANIFEST_SHA256,
            "source": _SOURCE,
            "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
            "crop_policy": CROP_POLICY,
            "bands": list(_BANDS),
            "scale": 4,
        },
        "ordered_sample_ids_sha256": ordered_ids,
        "ordered_membership_sha256": ordered_membership,
        "input_receipt_sha256s": per_record,
        "samples": receipt_samples,
        "ordered_inputs_sha256": value["ordered_inputs_sha256"],
    }
    input_verification = verify_calibration_input_receipt(input_receipt)
    if (
        _digest(value["input_receipt_sha256"], "result input receipt digest")
        != input_verification.source_sha256
    ):
        raise ValueError("result input receipt digest is inconsistent")

    radiometry_summary = _dict(
        value["radiometry"],
        {"policy", "affected_sample_count", "lr", "hr"},
        "result radiometry",
    )
    radiometry_receipt = {
        "schema": _RADIOMETRY_SCHEMA,
        "split": "calibration",
        "ordered_sample_ids_sha256": ordered_ids,
        "policy": radiometry_summary["policy"],
        "sample_count": 120,
        "affected_sample_count": radiometry_summary["affected_sample_count"],
        "lr": radiometry_summary["lr"],
        "hr": radiometry_summary["hr"],
        "samples": radiometry_samples,
    }
    radiometry_verification = verify_calibration_radiometry(radiometry_receipt)
    _validate_summary(value)
    return VerifiedPhase2B3BResult(
        schema=SCHEMA,
        verification_scope="metadata_consistency_only",
        cache_computation_verified=False,
        result_sha256=hashlib.sha256(canonical_result).hexdigest(),
        cache_audit_sha256=audit_sha256,
        producer_revision=producer,
        ordered_sample_ids_sha256=ordered_ids,
        ordered_membership_sha256=ordered_membership,
        input_receipt_sha256=input_verification.source_sha256,
        ordered_inputs_sha256=input_verification.ordered_inputs_sha256,
        map_evidence_sha256=map_sha256,
        radiometry_aggregate_sha256=radiometry_verification.aggregate_sha256,
        phase_decision=value["phase_decision"],
    )
