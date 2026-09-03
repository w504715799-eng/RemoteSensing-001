"""Read-only Phase 2B3-B trust anchor for published Phase 2B3-A evidence."""

from __future__ import annotations

import hashlib
import json
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from trustsr.jsonio import canonical_json

POST_MANIFEST_SHA256 = "c7f8ffa8415575d85daafe284a0796ec3f111442f0ac662f1d01311c4a851d4a"
INPUT_AUDIT_SHA256 = "fceb2ec04680ddf46bf4d0ed5a4a93edd33d58a09fc176d936bdef783114b44b"
PRODUCER_REVISION = "58694420c3c0e11d495953a1963c71b997261601"
NORMALIZATION_POLICY = "uint16_saturate_10000_divide_10000_v2"
CROP_POLICY = "center_crop_lr_1_hr_4_v1"
_MAX_FILE_BYTES = 5 * 1024**2

PUBLISHED_EVIDENCE_SHA256S = MappingProxyType(
    {
        "sen2naipv2-development-smoke-v2.json": (
            "2c962de9651f3d2cc65f321877564c3509d8d4414801fd5b445503aed5dbb947"
        ),
        "sen2naipv2-development-smoke-cache-audit-v2.json": (
            "88144cb6dcfc4d8fc68289188aa909fd2e597304b95e47d23f9d0f0c17127a47"
        ),
        "sen2naipv2-development-smoke-acceptance-v2.json": (
            "5ac7bd232ce2a0897b9b93a35f896de4f5641a0adc9f42ce3d1f6986f1a054d2"
        ),
        "sen2naipv2-development-score-audit-v1.json": (
            "5bb0e5138d6ed1df6c65744556be02ccd48b77d3288df39630d16fbd9cd2dce9"
        ),
        "sen2naipv2-development-score-cache-audit-v1.json": (
            "d61c36e2180a2dc3468d4d9aba083ac0925d163ac2bb910e0227138e9fa249f1"
        ),
        "sen2naipv2-development-score-acceptance-v1.json": (
            "34741fe788cac6e28c6d8b1ce2fd96335b608e1b3e6ffb29e82ac064a2118227"
        ),
    }
)

_SCHEMAS = {
    "sen2naipv2-development-smoke-v2.json": "trustsr.phase2b3a-development-smoke.v2",
    "sen2naipv2-development-smoke-cache-audit-v2.json": (
        "trustsr.phase2b3a-development-smoke-cache-audit.v2"
    ),
    "sen2naipv2-development-smoke-acceptance-v2.json": (
        "trustsr.phase2b3a-development-smoke-acceptance.v2"
    ),
    "sen2naipv2-development-score-audit-v1.json": (
        "trustsr.phase2b3a-development-score-audit.v1"
    ),
    "sen2naipv2-development-score-cache-audit-v1.json": (
        "trustsr.phase2b3a-development-score-cache-audit.v1"
    ),
    "sen2naipv2-development-score-acceptance-v1.json": (
        "trustsr.phase2b3a-development-score-acceptance.v1"
    ),
}

_A1_RECEIPT_DIGESTS = {
    "phase2b3a-a1-cache-audit.json": (
        "88144cb6dcfc4d8fc68289188aa909fd2e597304b95e47d23f9d0f0c17127a47"
    ),
    "phase2b3a-a1-replay.json": "11090c1c3940d3fd6f658861c9e9a56a901d97435db99543808813ff35c183fd",
    "phase2b3a-a1-result.json": (
        "2c962de9651f3d2cc65f321877564c3509d8d4414801fd5b445503aed5dbb947"
    ),
    "phase2b3a-a1-runtime.json": "94eed4dccc3113a8a9fcdfaea8d10a09c4329db25f4ab5d99a5e2b2d7eece23e",
    "phase2b3a-bundle-manifest.json": (
        "bd89914ff051bb0216d500856cc02fa98492ae20448377af5047b0b49f02b1f4"
    ),
}
_A2_RECEIPT_DIGESTS = {
    "phase2b3a-a2-cache-audit.json": (
        "d61c36e2180a2dc3468d4d9aba083ac0925d163ac2bb910e0227138e9fa249f1"
    ),
    "phase2b3a-a2-replay.json": "a0f6bcbb38df3bdee07bf6c1a657cc08018a5002628d544c692b430bd0929633",
    "phase2b3a-a2-result.json": (
        "5bb0e5138d6ed1df6c65744556be02ccd48b77d3288df39630d16fbd9cd2dce9"
    ),
    "phase2b3a-a2-runtime.json": "3473c65293e065f685dae18d7b5cf1121cf8feb4fd870d1bb9f55b13bff12bbb",
    "phase2b3a-bundle-manifest.json": (
        "0e2cfdf0ee880400ebea0f8098374677b84b18bae018f1f83829f72673928385"
    ),
}
_OPERATOR_PARAMETERS = {
    "algorithm": "ensemble_variance_score",
    "band_reduction": "mean",
    "correction": 0,
    "seed_count": 5,
    "seed_first": 3407,
    "seed_last": 3411,
}
_SEEDS = (3407, 3408, 3409, 3410, 3411)
_BANDS = ("B04", "B03", "B02", "B08")


@dataclass(frozen=True)
class FrozenPhase2B3AEvidence:
    """Small immutable B3-B input distilled from six verified publication receipts."""

    source_digests: Mapping[str, str]
    score_name: str
    operator_parameters: Mapping[str, object]
    seeds: tuple[int, ...]
    risk_name: str
    risk_window: int
    risk_upper_bound: float
    normalization_policy: str
    crop_policy: str
    bands: tuple[str, ...]
    scale: int
    post_manifest_sha256: str
    input_audit_sha256: str
    producer_revision: str
    candidate_eligibility_evidence: tuple[Mapping[str, object], ...]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_exact_keys(value: object, keys: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{label} schema is invalid")
    return value


def _read_canonical(path: Path, name: str) -> dict[str, object]:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ValueError(f"published evidence is unreadable: {name}") from exc
    if not stat.S_ISREG(mode) or path.is_symlink():
        raise ValueError(f"published evidence must be a regular non-symlink file: {name}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"published evidence is unreadable: {name}") from exc
    if len(raw) > _MAX_FILE_BYTES:
        raise ValueError(f"published evidence exceeds the 5 MiB limit: {name}")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"published evidence is not valid JSON: {name}") from exc
    if type(value) is not dict or canonical_json(value) != raw:
        raise ValueError(f"published evidence is not canonical JSON: {name}")
    if _sha256(raw) != PUBLISHED_EVIDENCE_SHA256S[name]:
        raise ValueError(f"published evidence SHA-256 mismatch: {name}")
    return value


def _validate_directory(evidence_dir: Path) -> None:
    if not isinstance(evidence_dir, Path) or evidence_dir.is_symlink() or not evidence_dir.is_dir():
        raise ValueError("evidence directory must be an existing non-symlink directory")
    try:
        if evidence_dir.resolve(strict=True) != evidence_dir.absolute():
            raise ValueError("evidence directory must not contain symlink components")
        observed = {entry.name for entry in evidence_dir.iterdir()}
    except OSError as exc:
        raise ValueError("evidence directory is unreadable") from exc
    if observed != set(PUBLISHED_EVIDENCE_SHA256S):
        raise ValueError("evidence directory must contain the exact six allowlisted files")


def _validate_schema(documents: Mapping[str, object]) -> dict[str, dict[str, object]]:
    if set(documents) != set(PUBLISHED_EVIDENCE_SHA256S):
        raise ValueError("published evidence file set is invalid")
    parsed: dict[str, dict[str, object]] = {}
    for name, schema in _SCHEMAS.items():
        document = documents[name]
        if type(document) is not dict or document.get("schema") != schema:
            raise ValueError(f"published evidence schema is invalid: {name}")
        parsed[name] = document
    return parsed


def _validate_receipt(
    receipt: object,
    *,
    schema: str,
    digests: Mapping[str, str],
    gates: set[str],
    extra_keys: set[str] | None = None,
    label: str,
) -> dict[str, object]:
    value = _require_exact_keys(
        receipt, {"schema", "digests", *gates, *(extra_keys or set())}, label
    )
    if value["schema"] != schema or value["digests"] != digests:
        raise ValueError(f"{label} schema or internal digest receipt is invalid")
    if any(value[gate] is not True for gate in gates):
        raise ValueError(f"{label} gate is not accepted")
    return value


def _validate_common_identity(result: Mapping[str, object], audit: Mapping[str, object]) -> None:
    upstream = {
        "post_manifest_sha256": POST_MANIFEST_SHA256,
        "input_audit_sha256": INPUT_AUDIT_SHA256,
    }
    if (
        result.get("upstream") != upstream
        or audit.get("post_manifest_sha256") != POST_MANIFEST_SHA256
        or audit.get("input_audit_sha256") != INPUT_AUDIT_SHA256
        or result.get("normalization_policy") != NORMALIZATION_POLICY
        or audit.get("normalization_policy") != NORMALIZATION_POLICY
        or result.get("bands") != list(_BANDS)
        or result.get("scale") != 4
    ):
        raise ValueError("published evidence has a mismatched frozen input identity")


def _validate_a1(documents: Mapping[str, dict[str, object]]) -> None:
    result = documents["sen2naipv2-development-smoke-v2.json"]
    audit = documents["sen2naipv2-development-smoke-cache-audit-v2.json"]
    receipt = documents["sen2naipv2-development-smoke-acceptance-v2.json"]
    _require_exact_keys(
        result,
        {
            "bands",
            "dataset_role",
            "include_ldsr_variance_k5",
            "k5_statistically_stable",
            "normalization_policy",
            "prediction_count",
            "radiometric_policy",
            "runtime_manifest_sha256",
            "sample_count",
            "samples",
            "scale",
            "schema",
            "score_count",
            "seed_sets",
            "stability_thresholds",
            "upstream",
        },
        "A1 result",
    )
    _require_exact_keys(
        audit,
        {
            "experiment_schema",
            "input_audit_sha256",
            "normalization_policy",
            "post_manifest_sha256",
            "prediction_count",
            "prediction_entries",
            "sample_count",
            "schema",
            "score_count",
            "score_entries",
        },
        "A1 cache audit",
    )
    _validate_common_identity(result, audit)
    _validate_receipt(
        receipt,
        schema="trustsr.phase2b3a-development-smoke-acceptance.v2",
        digests=_A1_RECEIPT_DIGESTS,
        gates={
            "bundle_integrity_pass",
            "replay_pass",
            "repeatability_pass",
            "resource_gate_pass",
            "include_ldsr_variance_k5",
        },
        label="A1 acceptance",
    )
    if (
        result.get("dataset_role") != "development_engineering_smoke_only"
        or result.get("include_ldsr_variance_k5") is not True
        or result.get("k5_statistically_stable") is not True
    ):
        raise ValueError("A1 K=5 gate is not accepted")


def _validate_a2(documents: Mapping[str, dict[str, object]]) -> dict[str, object]:
    result = documents["sen2naipv2-development-score-audit-v1.json"]
    audit = documents["sen2naipv2-development-score-cache-audit-v1.json"]
    receipt = documents["sen2naipv2-development-score-acceptance-v1.json"]
    _require_exact_keys(
        result,
        {
            "bands",
            "bootstrap",
            "candidate_names",
            "candidate_summaries",
            "code_revision",
            "dataset_role",
            "frozen_score",
            "include_ldsr_variance_k5",
            "normalization_policy",
            "phase_decision",
            "prediction_count",
            "radiometric_policy",
            "risk_configuration",
            "runtime_manifest_sha256",
            "sample_count",
            "samples",
            "scale",
            "schema",
            "score_configuration",
            "score_count",
            "selection_risk",
            "statistical_unit",
            "upstream",
        },
        "A2 result",
    )
    _require_exact_keys(
        audit,
        {
            "code_revision",
            "experiment_schema",
            "groups",
            "input_audit_sha256",
            "normalization_policy",
            "post_manifest_sha256",
            "prediction_count",
            "sample_count",
            "schema",
            "score_count",
        },
        "A2 cache audit",
    )
    _validate_common_identity(result, audit)
    _validate_receipt(
        receipt,
        schema="trustsr.phase2b3a-development-score-acceptance.v1",
        digests=_A2_RECEIPT_DIGESTS,
        gates={"bundle_integrity_pass", "replay_pass", "development_only_pass"},
        extra_keys={"frozen_score", "no_eligible_score"},
        label="A2 acceptance",
    )
    if receipt.get("no_eligible_score") is not False:
        raise ValueError("A2 acceptance incorrectly reports no eligible score")
    expected_names = ["lr_reprojection_l1", "three_model_disagreement", "ldsr_variance_k5"]
    if (
        result.get("dataset_role") != "development_score_selection_only"
        or result.get("include_ldsr_variance_k5") is not True
        or result.get("candidate_names") != expected_names
        or result.get("score_configuration", {}).get("ldsr_variance_k5") != _OPERATOR_PARAMETERS
        or result.get("risk_configuration")
        != {"name": "local_l1_risk", "primary_window": 9, "sensitivity_window": 1}
        or result.get("selection_risk") != "primary_window_9"
        or result.get("code_revision") != PRODUCER_REVISION
        or audit.get("code_revision") != PRODUCER_REVISION
    ):
        raise ValueError("A2 frozen score configuration is invalid")
    frozen = result.get("frozen_score")
    if type(frozen) is not dict or frozen != receipt.get("frozen_score"):
        raise ValueError("A2 result and acceptance frozen-score evidence differ")
    required_frozen = {
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
    }
    _require_exact_keys(frozen, required_frozen, "A2 frozen score")
    candidates = frozen["candidate_eligibility_evidence"]
    summaries = result.get("candidate_summaries")
    if (
        result.get("phase_decision") != "freeze_score"
        or frozen.get("name") != "ldsr_variance_k5"
        or frozen.get("operator_parameters") != _OPERATOR_PARAMETERS
        or frozen.get("seeds") != list(_SEEDS)
        or frozen.get("post_manifest_sha256") != POST_MANIFEST_SHA256
        or frozen.get("code_revision") != PRODUCER_REVISION
        or frozen.get("statistical_leader") != "ldsr_variance_k5"
        or frozen.get("indistinguishable_candidates") != ["ldsr_variance_k5"]
        or not isinstance(candidates, list)
        or not isinstance(summaries, list)
        or len(candidates) != len(expected_names)
        or len(summaries) != len(expected_names)
    ):
        raise ValueError("A2 frozen-score identity or eligibility evidence is invalid")
    candidate_names = [
        candidate.get("name") if type(candidate) is dict else None for candidate in candidates
    ]
    if candidate_names != expected_names:
        raise ValueError("A2 candidate eligibility order is invalid")
    expected_candidates = [
        summary.get("primary_window_9") if type(summary) is dict else None for summary in summaries
    ]
    if (
        candidates != expected_candidates
        or frozen.get("selected_candidate_evidence") != candidates[-1]
        or candidates[-1].get("eligible") is not True
    ):
        raise ValueError("A2 selected candidate eligibility evidence is invalid")
    return frozen


def _validate_semantics(documents: Mapping[str, object]) -> dict[str, object]:
    """Validate parsed documents separately so semantic mutations remain testable."""

    parsed = _validate_schema(documents)
    _validate_a1(parsed)
    return _validate_a2(parsed)


def _freeze(value: Any) -> object:
    if type(value) is dict:
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_freeze(item) for item in value)
    return value


def load_frozen_phase2b3a_evidence(evidence_dir: Path) -> FrozenPhase2B3AEvidence:
    """Load exactly the six pinned Phase 2B3-A receipts without opening non-JSON assets."""

    _validate_directory(evidence_dir)
    documents = {
        name: _read_canonical(evidence_dir / name, name) for name in PUBLISHED_EVIDENCE_SHA256S
    }
    frozen = _validate_semantics(documents)
    candidates = _freeze(frozen["candidate_eligibility_evidence"])
    if not isinstance(candidates, tuple) or not all(
        isinstance(item, Mapping) for item in candidates
    ):
        raise ValueError("A2 candidate eligibility evidence is invalid")
    return FrozenPhase2B3AEvidence(
        source_digests=MappingProxyType(dict(PUBLISHED_EVIDENCE_SHA256S)),
        score_name="ldsr_variance_k5",
        operator_parameters=MappingProxyType(dict(_OPERATOR_PARAMETERS)),
        seeds=_SEEDS,
        risk_name="local_l1_risk",
        risk_window=9,
        risk_upper_bound=1.0,
        normalization_policy=NORMALIZATION_POLICY,
        crop_policy=CROP_POLICY,
        bands=_BANDS,
        scale=4,
        post_manifest_sha256=POST_MANIFEST_SHA256,
        input_audit_sha256=INPUT_AUDIT_SHA256,
        producer_revision=PRODUCER_REVISION,
        candidate_eligibility_evidence=candidates,
    )
