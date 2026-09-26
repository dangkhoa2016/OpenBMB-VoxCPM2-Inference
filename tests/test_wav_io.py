from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import pytest

from voxcpm_runtime.backend_types import AudioResult
from voxcpm_runtime.wav_io import (
    WavWriteError,
    validate_wav,
    write_wav,
)

_SAMPLE_RATE_HZ = 48_000


def _result(
    samples: tuple[float, ...] = (0.0, 0.5, -0.5, 1.0, -1.0),
    sample_rate_hz: int = _SAMPLE_RATE_HZ,
    channels: int = 1,
) -> AudioResult:
    return AudioResult(
        samples=samples,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
    )


def _raw_result(
    samples: tuple[float, ...],
    sample_rate_hz: int = _SAMPLE_RATE_HZ,
    channels: int = 1,
) -> AudioResult:
    """Build an AudioResult while bypassing its own field validation.

    The writer keeps its own defensive checks, so the tests need to be able to
    construct values that the frozen M3 contract would never allow.
    """

    result = AudioResult.__new__(AudioResult)
    object.__setattr__(result, "samples", samples)
    object.__setattr__(result, "sample_rate_hz", sample_rate_hz)
    object.__setattr__(result, "channels", channels)
    object.__setattr__(result, "metadata", ())
    return result


# --- writer -------------------------------------------------------------


def test_write_wav_produces_readable_pcm16_mono(tmp_path: Path) -> None:
    result = _result()
    written = write_wav(tmp_path / "out.wav", result)
    assert written.name == "out.wav"
    assert written.channel_count == 1
    assert written.sample_rate_hz == _SAMPLE_RATE_HZ
    assert written.sample_width_bytes == 2
    assert written.frame_count == 5
    assert written.duration_seconds == pytest.approx(5 / _SAMPLE_RATE_HZ)
    with wave.open(str(tmp_path / "out.wav"), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == _SAMPLE_RATE_HZ
        assert handle.getnframes() == 5


def test_write_wav_clips_samples_to_valid_pcm16_range(tmp_path: Path) -> None:
    written = write_wav(tmp_path / "clip.wav", _result((2.5, -2.5, 0.0)))
    assert written.frame_count == 3
    with wave.open(str(tmp_path / "clip.wav"), "rb") as handle:
        payload = handle.readframes(3)
    codes = struct.unpack("<3h", payload)
    assert codes[0] == 32_767
    assert codes[1] == -32_767
    assert codes[2] == 0


def test_write_wav_quantizes_little_endian(tmp_path: Path) -> None:
    write_wav(tmp_path / "le.wav", _result((1.0,)))
    payload = (tmp_path / "le.wav").read_bytes()
    assert payload[44:46] == struct.pack("<h", 32_767)


def test_write_wav_records_sha256_and_size(tmp_path: Path) -> None:
    import hashlib

    target = tmp_path / "hash.wav"
    written = write_wav(target, _result())
    assert written.sha256 == hashlib.sha256(target.read_bytes()).hexdigest()
    assert written.byte_size == target.stat().st_size
    assert written.byte_size > 44


def test_write_wav_creates_parent_directory_when_requested(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "deeper" / "out.wav"
    write_wav(target, _result(), create_parents=True)
    assert target.is_file()


def test_write_wav_refuses_missing_parent_by_default(tmp_path: Path) -> None:
    with pytest.raises(WavWriteError) as caught:
        write_wav(tmp_path / "absent" / "out.wav", _result())
    assert str(tmp_path) not in str(caught.value)


def test_write_wav_leaves_no_partial_file_on_failure(tmp_path: Path) -> None:
    target = tmp_path / "atomic.wav"
    with pytest.raises(WavWriteError):
        write_wav(target, object())  # type: ignore[arg-type]
    assert not target.exists()
    assert list(tmp_path.glob(".*")) == []


def test_write_wav_requires_audio_result(tmp_path: Path) -> None:
    with pytest.raises(WavWriteError):
        write_wav(tmp_path / "x.wav", None)  # type: ignore[arg-type]


def test_write_wav_rejects_empty_result(tmp_path: Path) -> None:
    with pytest.raises(WavWriteError) as caught:
        write_wav(tmp_path / "x.wav", _raw_result(()))
    assert "no samples" in str(caught.value)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_write_wav_rejects_non_finite_samples(
    tmp_path: Path, bad_value: float
) -> None:
    result = _raw_result((0.0, bad_value))
    with pytest.raises(WavWriteError) as caught:
        write_wav(tmp_path / "x.wav", result)
    assert "finite" in str(caught.value)


def test_write_wav_honours_channel_count(tmp_path: Path) -> None:
    written = write_wav(tmp_path / "stereo.wav", _result((0.1, 0.2, 0.3, 0.4), channels=2))
    assert written.channel_count == 2
    with wave.open(str(tmp_path / "stereo.wav"), "rb") as handle:
        assert handle.getnchannels() == 2
        assert handle.getnframes() == 2


# --- validation ---------------------------------------------------------


def test_validate_wav_accepts_written_file(tmp_path: Path) -> None:
    target = tmp_path / "ok.wav"
    write_wav(target, _result())
    validation = validate_wav(
        target,
        expected_sample_rate_hz=_SAMPLE_RATE_HZ,
        expected_channel_count=1,
    )
    assert validation.valid is True
    assert validation.failure_reason is None
    assert validation.exists is True
    assert validation.riff_header_ok is True
    assert validation.wave_format_pcm is True
    assert validation.frame_count == 5
    assert validation.sample_rate_matches is True
    assert validation.channel_count_matches is True
    assert validation.duration_seconds == pytest.approx(5 / _SAMPLE_RATE_HZ)
    assert validation.sha256 == written_sha(tmp_path)


def written_sha(tmp_path: Path) -> str:
    import hashlib

    return hashlib.sha256((tmp_path / "ok.wav").read_bytes()).hexdigest()


def test_validate_wav_reports_missing_file(tmp_path: Path) -> None:
    validation = validate_wav(tmp_path / "absent.wav")
    assert validation.valid is False
    assert validation.exists is False
    assert validation.failure_reason == "missing_file"


def test_validate_wav_reports_empty_file(tmp_path: Path) -> None:
    target = tmp_path / "empty.wav"
    target.write_bytes(b"")
    validation = validate_wav(target)
    assert validation.failure_reason == "empty_file"
    assert validation.byte_size == 0


def test_validate_wav_reports_truncated_file(tmp_path: Path) -> None:
    target = tmp_path / "truncated.wav"
    target.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    validation = validate_wav(target)
    assert validation.valid is False
    assert validation.failure_reason in {
        "riff_header_invalid",
        "wave_format_not_pcm",
        "wave_module_cannot_read",
    }


def test_validate_wav_reports_corrupt_riff_header(tmp_path: Path) -> None:
    target = tmp_path / "bad.wav"
    write_wav(target, _result())
    payload = bytearray(target.read_bytes())
    payload[0:4] = b"XXXX"
    target.write_bytes(bytes(payload))
    validation = validate_wav(target)
    assert validation.failure_reason == "riff_header_invalid"


def test_validate_wav_reports_inconsistent_riff_size(tmp_path: Path) -> None:
    target = tmp_path / "size.wav"
    write_wav(target, _result())
    payload = bytearray(target.read_bytes())
    struct.pack_into("<I", payload, 4, 999_999)
    target.write_bytes(bytes(payload))
    validation = validate_wav(target)
    assert validation.failure_reason == "riff_header_invalid"


def test_validate_wav_reports_sample_rate_mismatch(tmp_path: Path) -> None:
    target = tmp_path / "rate.wav"
    write_wav(target, _result())
    validation = validate_wav(target, expected_sample_rate_hz=16_000)
    assert validation.failure_reason == "sample_rate_mismatch"
    assert validation.sample_rate_matches is False
    assert validation.sample_rate_hz == _SAMPLE_RATE_HZ


def test_validate_wav_reports_channel_count_mismatch(tmp_path: Path) -> None:
    target = tmp_path / "ch.wav"
    write_wav(target, _result())
    validation = validate_wav(target, expected_channel_count=2)
    assert validation.failure_reason == "channel_count_mismatch"
    assert validation.channel_count_matches is False


def test_validate_wav_reports_unusable_path() -> None:
    validation = validate_wav(None)  # type: ignore[arg-type]
    assert validation.failure_reason == "unusable_path"
    assert validation.exists is False


def test_validation_dict_is_json_friendly(tmp_path: Path) -> None:
    import json

    target = tmp_path / "dict.wav"
    write_wav(target, _result())
    payload = validate_wav(target).to_dict()
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    for value in payload.values():
        if isinstance(value, float):
            assert math.isfinite(value)


def test_write_result_dict_excludes_parent_directory(tmp_path: Path) -> None:
    written = write_wav(tmp_path / "private" / "name.wav", _result(), create_parents=True)
    payload = written.to_dict()
    assert payload["name"] == "name.wav"
    assert str(tmp_path) not in str(payload)
