"""Git-safe artifact and leakage policy for Phase 2B3-C publication."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from trustsr.jsonio import canonical_json

_DIRECTORY = Path("artifacts/phase2b3c")
_ALLOWED_SCHEMAS = {
    "sen2naipv2-internal-test-access-authorization-v1.json": (
        "trustsr.phase2b3c-internal-test-access-authorization.v1"
    ),
    "sen2naipv2-internal-test-evaluation-v1.json": (
        "trustsr.phase2b3c-evaluation.v1"
    ),
    "sen2naipv2-internal-test-evaluation-cache-audit-v1.json": (
        "trustsr.phase2b3c-evaluation-cache-audit.v1"
    ),
    "sen2naipv2-internal-test-evaluation-acceptance-v1.json": (
        "trustsr.phase2b3c-evaluation-acceptance.v1"
    ),
}
_MAXIMUM_BYTES = 5 * 1024**2
_FORBIDDEN_KEYS = {
    "path",
    "cache_path",
    "model_path",
    "project_root",
    "storage_root",
    "hostname",
    "host",
    "endpoint",
    "credential",
    "credentials",
    "token",
    "access_token",
    "secret",
    "password",
    "username",
    "raw_timestamp",
    "timestamp",
    "gpu_uuid",
}
_PER_SAMPLE_METRICS = {
    "loss",
    "coverage",
    "trusted_pixels",
    "total_pixels",
    "risk_ucb",
    "mean_loss",
    "maximum_loss",
}


def _git(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=False,
        capture_output=True,
        shell=False,
        stdin=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise ValueError("unable to inspect the local Git artifact index")
    return completed.stdout


def _indexed_entries(root: Path) -> tuple[tuple[Path, str, bytes], ...]:
    output = _git(root, "ls-files", "--stage", "-z", "--", str(_DIRECTORY))
    entries: list[tuple[Path, str, bytes]] = []
    for record in output.split(b"\0"):
        if not record:
            continue
        metadata, separator, encoded_path = record.partition(b"\t")
        fields = metadata.split()
        if not separator or len(fields) != 3 or fields[2] != b"0":
            raise ValueError("Phase 2B3-C Git index entry is malformed")
        mode = fields[0].decode("ascii")
        object_id = fields[1].decode("ascii")
        path = Path(os.fsdecode(encoded_path))
        payload = _git(root, "cat-file", "blob", object_id)
        entries.append((path, mode, payload))
    return tuple(entries)


def _scan_value(
    value: object,
    *,
    location: str,
    in_samples: bool,
    violations: list[str],
) -> None:
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                violations.append(f"{location}: JSON object key is not a string")
                continue
            child = f"{location}.{key}"
            if key.casefold() in _FORBIDDEN_KEYS:
                violations.append(f"{child}: forbidden operational or secret field {key}")
            if in_samples and key.casefold() in _PER_SAMPLE_METRICS:
                violations.append(f"{child}: forbidden per-sample numerical metric {key}")
            _scan_value(
                item,
                location=child,
                in_samples=in_samples or key == "samples",
                violations=violations,
            )
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _scan_value(
                item,
                location=f"{location}[{index}]",
                in_samples=in_samples,
                violations=violations,
            )
        return
    if type(value) is str and (
        value.startswith(("/", "~/", "file://", "http://", "https://"))
        or ":\\" in value
    ):
        violations.append(f"{location}: forbidden path or endpoint value")


def _scan_payload(
    name: str, payload: bytes, *, label: str, violations: list[str]
) -> None:
    if len(payload) > _MAXIMUM_BYTES:
        violations.append(f"{label}: artifact exceeds {_MAXIMUM_BYTES} bytes")
        return
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        violations.append(f"{label}: artifact is not canonical UTF-8 JSON")
        return
    if type(value) is not dict or canonical_json(value) != payload:
        violations.append(f"{label}: artifact is not canonical UTF-8 JSON")
        return
    if value.get("schema") != _ALLOWED_SCHEMAS[name]:
        violations.append(f"{label}: artifact schema is not the exact allowlisted schema")
    _scan_value(value, location=label, in_samples=False, violations=violations)


def phase2b3c_publication_policy_violations(
    project_root: Path,
) -> tuple[str, ...]:
    """Return deterministic violations for tracked Phase 2B3-C artifacts."""

    if not isinstance(project_root, Path):
        raise TypeError("project root must be a Path")
    root = project_root.resolve(strict=True)
    violations: list[str] = []
    for relative, mode, indexed_payload in _indexed_entries(root):
        label = str(relative)
        if (
            relative.parts[:2] != _DIRECTORY.parts
            or len(relative.parts) != 3
            or relative.name not in _ALLOWED_SCHEMAS
        ):
            violations.append(f"{label}: file is outside the exact Phase 2B3-C allowlist")
            continue
        if mode not in {"100644", "100755"}:
            violations.append(f"{label}: tracked Phase 2B3-C artifact is a symlink or non-file")
            continue
        _scan_payload(relative.name, indexed_payload, label=f"index:{label}", violations=violations)
        working = root / relative
        if working.is_symlink():
            violations.append(f"{label}: working Phase 2B3-C artifact is a symlink")
        elif working.exists():
            try:
                working_payload = working.read_bytes()
            except OSError:
                violations.append(f"{label}: working artifact is unreadable")
            else:
                _scan_payload(
                    relative.name,
                    working_payload,
                    label=f"worktree:{label}",
                    violations=violations,
                )
    return tuple(sorted(set(violations)))
