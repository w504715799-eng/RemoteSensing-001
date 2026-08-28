"""Tests for the strict, offline dataset provenance loader."""

import json
import socket
import urllib.request
from pathlib import Path

import pytest

from trustsr.data.provenance import DatasetSource, LfsObject, load_dataset_source


def source_payload() -> dict[str, object]:
    return {
        "schema": "trustsr.sen2naipv2-source.v1",
        "repository": "tacofoundation/SEN2NAIPv2",
        "revision": "c370504201072fdb1dd388013ab8c0fc7d00a57e",
        "license_claim": "cc0-1.0",
        "card_sha256": "5897aed9410fef305953ff5b34e83697b466901583b880158af2902a8267a58d",
        "bands": ["B04", "B03", "B02", "B08"],
        "scale": 4,
        "lr_shape": [130, 130],
        "hr_shape": [520, 520],
        "declared_total_bytes": 12,
        "objects": [
            {
                "path": "sen2naipv2-crosssensor.taco",
                "sha256": "c6f29d8e80dc5e856e2b4510c0e6830043d4b15c9228a9ca249a4f618e7475a5",
                "size_bytes": 9,
            },
            {
                "path": "sen2naipv2-unet.0000.part.taco",
                "sha256": "a276024df0f81ff53770cf1b415d0f86268bd2b090a467b80e2e8b3992d08acc",
                "size_bytes": 3,
            },
        ],
    }


def write_payload(tmp_path: Path, payload: dict[str, object], raw: str | None = None) -> Path:
    path = tmp_path / "source.json"
    path.write_text(raw if raw is not None else json.dumps(payload), encoding="utf-8")
    return path


def test_loads_source_without_network_and_decodes_immutable_types(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_network(*args: object, **kwargs: object) -> object:
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    monkeypatch.setattr(urllib.request, "urlopen", fail_network)

    source = load_dataset_source(write_payload(tmp_path, source_payload()))

    assert isinstance(source, DatasetSource)
    assert isinstance(source.objects[0], LfsObject)
    assert source.repository == "tacofoundation/SEN2NAIPv2"
    assert source.revision == "c370504201072fdb1dd388013ab8c0fc7d00a57e"
    assert source.bands == ("B04", "B03", "B02", "B08")
    assert source.lr_shape == (130, 130)
    assert source.hr_shape == (520, 520)
    assert source.total_bytes == 12
    assert source.objects[0].path == "sen2naipv2-crosssensor.taco"


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("schema", "wrong", "schema must be trustsr.sen2naipv2-source.v1"),
        ("revision", "ABC", "revision must be 40 lowercase hexadecimal characters"),
        ("card_sha256", "xyz", "card_sha256 must be 64 lowercase hexadecimal characters"),
        ("bands", ["B02", "B03", "B04", "B08"], "bands must equal"),
        ("scale", 2, "scale must equal 4"),
        ("declared_total_bytes", 99, "declared_total_bytes must equal object sizes"),
    ],
)
def test_rejects_invalid_declared_values(
    key: str, value: object, message: str, tmp_path: Path
) -> None:
    payload = source_payload()
    payload[key] = value

    with pytest.raises(ValueError, match=message):
        load_dataset_source(write_payload(tmp_path, payload))


def test_rejects_unknown_top_level_key(tmp_path: Path) -> None:
    payload = source_payload()
    payload["extra"] = True

    with pytest.raises(ValueError, match="unknown top-level keys"):
        load_dataset_source(write_payload(tmp_path, payload))


def test_rejects_missing_top_level_key(tmp_path: Path) -> None:
    payload = source_payload()
    del payload["repository"]

    with pytest.raises(ValueError, match="missing top-level keys"):
        load_dataset_source(write_payload(tmp_path, payload))


def test_rejects_duplicate_top_level_key(tmp_path: Path) -> None:
    payload = source_payload()
    raw = json.dumps(payload).replace(
        '"schema": "trustsr.sen2naipv2-source.v1",',
        '"schema": "trustsr.sen2naipv2-source.v1", "schema": "trustsr.sen2naipv2-source.v1",',
        1,
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_dataset_source(write_payload(tmp_path, payload, raw=raw))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["objects"].__setitem__(1, payload["objects"][0]),
            "duplicate object path",
        ),
        (lambda payload: payload["objects"][0].__setitem__("path", "/absolute.taco"), "relative"),
        (lambda payload: payload["objects"][0].__setitem__("path", "nested/../object.taco"), ".."),
        (
            lambda payload: payload["objects"][0].__setitem__("sha256", "A" * 64),
            "sha256 must be 64 lowercase hexadecimal characters",
        ),
        (
            lambda payload: payload["objects"][0].__setitem__("size_bytes", True),
            "size_bytes must be an integer",
        ),
        (
            lambda payload: payload["objects"][0].__setitem__("size_bytes", 0),
            "size_bytes must be positive",
        ),
        (lambda payload: payload.__setitem__("objects", []), "objects must not be empty"),
        (
            lambda payload: payload.__setitem__("hr_shape", [521, 520]),
            "hr_shape must equal lr_shape multiplied by scale",
        ),
    ],
)
def test_rejects_invalid_objects_and_shapes(
    mutation: object, message: str, tmp_path: Path
) -> None:
    payload = source_payload()
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        load_dataset_source(write_payload(tmp_path, payload))


def test_rejects_duplicate_object_key(tmp_path: Path) -> None:
    payload = source_payload()
    raw = json.dumps(payload).replace(
        '"path": "sen2naipv2-crosssensor.taco",',
        '"path": "sen2naipv2-crosssensor.taco", "path": "sen2naipv2-crosssensor.taco",',
        1,
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_dataset_source(write_payload(tmp_path, payload, raw=raw))


def test_rejects_unknown_object_key(tmp_path: Path) -> None:
    payload = source_payload()
    payload["objects"][0]["extra"] = 1

    with pytest.raises(ValueError, match="unknown object keys"):
        load_dataset_source(write_payload(tmp_path, payload))


def test_rejects_missing_object_key(tmp_path: Path) -> None:
    payload = source_payload()
    del payload["objects"][0]["sha256"]

    with pytest.raises(ValueError, match="missing object keys"):
        load_dataset_source(write_payload(tmp_path, payload))


def test_rejects_boolean_top_level_integer(tmp_path: Path) -> None:
    payload = source_payload()
    payload["scale"] = True

    with pytest.raises(ValueError, match="scale must be an integer"):
        load_dataset_source(write_payload(tmp_path, payload))


def test_rejects_non_positive_shapes(tmp_path: Path) -> None:
    payload = source_payload()
    payload["lr_shape"] = [0, 130]

    with pytest.raises(ValueError, match="lr_shape must contain positive integers"):
        load_dataset_source(write_payload(tmp_path, payload))
