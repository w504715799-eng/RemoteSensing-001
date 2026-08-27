import json
import socket
import sys
import urllib.request
from pathlib import Path

import pytest

from trustsr.cli.dataset_audit import build_payload, main
from trustsr.data.provenance import load_dataset_source


def _pinned_source_path() -> Path:
    return Path(__file__).parents[2] / "artifacts/datasets/sen2naipv2-source-v1.json"


def _network_forbidden(*args: object, **kwargs: object) -> None:
    raise AssertionError("dataset audit must not access the network")


def test_build_payload_reports_only_pinned_metadata() -> None:
    """Catch audit payloads that omit required local-only provenance evidence."""
    payload = build_payload(load_dataset_source(_pinned_source_path()))

    assert payload["schema"] == "trustsr.dataset-audit.v1"
    assert payload["metadata_only"] is True
    assert payload["network_accessed"] is False
    assert payload["pixel_data_downloaded"] is False
    assert payload["local_real_pixel_policy"] == "forbidden"
    assert payload["ready_for_phase2b1_schema_probe"] is True
    assert payload["object_count"] == 9
    assert payload["total_bytes"] == 149_356_128_592
    assert payload["variant_counts"] == {"crosssensor": 1, "histmatch": 4, "unet": 4}


def test_main_emits_byte_identical_strict_json_without_network_access(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch noncanonical output or any accidental attempt to fetch pixel data."""
    monkeypatch.setattr(socket, "create_connection", _network_forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", _network_forbidden)
    monkeypatch.setattr(sys, "argv", ["trustsr-dataset-audit"])

    assert main() == 0
    first = capsys.readouterr().out
    assert first.count("\n") == 1
    assert first == json.dumps(
        json.loads(first), sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n"

    monkeypatch.setattr(sys, "argv", ["trustsr-dataset-audit"])
    assert main() == 0
    second = capsys.readouterr().out

    assert second == first
