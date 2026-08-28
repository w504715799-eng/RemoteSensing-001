"""Offline tests for resumable, verified crosssensor acquisition."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from trustsr.data.crosssensor_source import SourceIntegrityError, acquire_crosssensor
from trustsr.data.provenance import DatasetSource, LfsObject


def _source(payload: bytes = b"synthetic-crosssensor") -> DatasetSource:
    object_spec = LfsObject(
        path="sen2naipv2-crosssensor.taco",
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )
    return DatasetSource(
        schema="trustsr.sen2naipv2-source.v1",
        repository="example/source",
        revision="a" * 40,
        license_claim="cc0-1.0",
        card_sha256="b" * 64,
        bands=("B04", "B03", "B02", "B08"),
        scale=4,
        lr_shape=(130, 130),
        hr_shape=(520, 520),
        declared_total_bytes=len(payload),
        objects=(object_spec,),
    )


class _WritingRunner:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.calls: list[tuple[tuple[str, ...], bool, bool]] = []

    def __call__(
        self, arguments: tuple[str, ...], *, check: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((arguments, check, text))
        output = Path(arguments[arguments.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(self.payload)
        return subprocess.CompletedProcess(arguments, 0, "", "")


def _ample_space(_: Path) -> shutil._ntuple_diskusage:
    return shutil._ntuple_diskusage(20 * 1024**3, 16 * 1024**3, 16 * 1024**3)


def test_acquires_a_verified_object_with_resumable_curl_and_atomic_promotion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = b"synthetic-crosssensor"
    source = _source(payload)
    runner = _WritingRunner(payload)
    monkeypatch.setattr(shutil, "disk_usage", _ample_space)

    acquired = acquire_crosssensor(
        source,
        tmp_path,
        "https://downloads.example.invalid/crosssensor.taco",
        confirmed_cloud_storage=True,
        runner=runner,
    )

    expected_final = (
        tmp_path
        / "trustsr"
        / "phase2b1a"
        / "source"
        / source.objects[0].sha256
        / "sen2naipv2-crosssensor.taco"
    )
    assert acquired.path == expected_final
    assert acquired.path.read_bytes() == payload
    assert acquired.size_bytes == len(payload)
    assert acquired.sha256 == hashlib.sha256(payload).hexdigest()
    assert not acquired.path.with_name(f"{acquired.path.name}.part").exists()
    assert len(runner.calls) == 1
    arguments, check, text = runner.calls[0]
    assert arguments == (
        "curl",
        "--fail",
        "--location",
        "--retry",
        "5",
        "--continue-at",
        "-",
        "--output",
        str(expected_final.with_name(f"{expected_final.name}.part")),
        "https://downloads.example.invalid/crosssensor.taco",
    )
    assert check and text


@pytest.mark.parametrize(
    "transport_url",
    [
        "http://downloads.example.invalid/crosssensor.taco",
        "https://user:password@downloads.example.invalid/crosssensor.taco",
        "https://downloads.example.invalid/crosssensor.taco#fragment",
    ],
)
def test_rejects_unsafe_transport_urls_before_runner_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, transport_url: str
) -> None:
    runner = _WritingRunner(b"unexpected")
    monkeypatch.setattr(shutil, "disk_usage", _ample_space)

    with pytest.raises(ValueError, match="HTTPS transport URL"):
        acquire_crosssensor(
            _source(), tmp_path, transport_url, confirmed_cloud_storage=True, runner=runner
        )

    assert runner.calls == []


def test_requires_explicit_cloud_confirmation_before_runner_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _WritingRunner(b"unexpected")
    monkeypatch.setattr(shutil, "disk_usage", _ample_space)

    with pytest.raises(ValueError, match="explicit cloud storage confirmation"):
        acquire_crosssensor(
            _source(),
            tmp_path,
            "https://downloads.example.invalid/crosssensor.taco",
            confirmed_cloud_storage=False,
            runner=runner,
        )

    assert runner.calls == []


def test_requires_more_than_fifteen_gibibytes_before_runner_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _WritingRunner(b"unexpected")
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _: shutil._ntuple_diskusage(20 * 1024**3, 15 * 1024**3, 15 * 1024**3),
    )

    with pytest.raises(ValueError, match="15 GiB"):
        acquire_crosssensor(
            _source(),
            tmp_path,
            "https://downloads.example.invalid/crosssensor.taco",
            confirmed_cloud_storage=True,
            runner=runner,
        )

    assert runner.calls == []


def test_rejects_a_source_without_the_crosssensor_inventory_entry_before_runner_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _WritingRunner(b"unexpected")
    source = _source()
    source = DatasetSource(
        **{**source.__dict__, "objects": (LfsObject("other.taco", "c" * 64, 1),)}
    )
    monkeypatch.setattr(shutil, "disk_usage", _ample_space)

    with pytest.raises(ValueError, match="crosssensor object"):
        acquire_crosssensor(
            source,
            tmp_path,
            "https://downloads.example.invalid/crosssensor.taco",
            confirmed_cloud_storage=True,
            runner=runner,
        )

    assert runner.calls == []


def test_existing_invalid_final_object_fails_without_overwrite_or_runner_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _source()
    runner = _WritingRunner(b"unexpected")
    final_path = (
        tmp_path
        / "trustsr"
        / "phase2b1a"
        / "source"
        / source.objects[0].sha256
        / source.objects[0].path
    )
    final_path.parent.mkdir(parents=True)
    final_path.write_bytes(b"invalid-real-object")
    monkeypatch.setattr(shutil, "disk_usage", _ample_space)

    with pytest.raises(SourceIntegrityError, match="size"):
        acquire_crosssensor(
            source,
            tmp_path,
            "https://downloads.example.invalid/crosssensor.taco",
            confirmed_cloud_storage=True,
            runner=runner,
        )

    assert final_path.read_bytes() == b"invalid-real-object"
    assert runner.calls == []


def test_interrupted_transfer_keeps_partial_file_for_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _source()
    partial_path = (
        tmp_path
        / "trustsr"
        / "phase2b1a"
        / "source"
        / source.objects[0].sha256
        / "sen2naipv2-crosssensor.taco.part"
    )

    def interrupted(
        arguments: tuple[str, ...], *, check: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        output = Path(arguments[arguments.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"interrupted")
        raise subprocess.CalledProcessError(18, arguments)

    monkeypatch.setattr(shutil, "disk_usage", _ample_space)

    with pytest.raises(subprocess.CalledProcessError):
        acquire_crosssensor(
            source,
            tmp_path,
            "https://downloads.example.invalid/crosssensor.taco",
            confirmed_cloud_storage=True,
            runner=interrupted,
        )

    assert partial_path.read_bytes() == b"interrupted"
    assert not partial_path.with_suffix("").exists()


def test_completed_invalid_partial_moves_to_non_overwriting_quarantine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _source()
    runner = _WritingRunner(b"wrong")
    monkeypatch.setattr(shutil, "disk_usage", _ample_space)

    with pytest.raises(SourceIntegrityError, match="size"):
        acquire_crosssensor(
            source,
            tmp_path,
            "https://downloads.example.invalid/crosssensor.taco",
            confirmed_cloud_storage=True,
            runner=runner,
        )

    final_path = (
        tmp_path
        / "trustsr"
        / "phase2b1a"
        / "source"
        / source.objects[0].sha256
        / source.objects[0].path
    )
    quarantine = tmp_path / "trustsr" / "phase2b1a" / "quarantine" / source.objects[0].sha256
    assert not final_path.exists()
    assert not final_path.with_name(f"{final_path.name}.part").exists()
    quarantined = tuple(quarantine.iterdir())
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"wrong"
