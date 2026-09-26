from __future__ import annotations

import array
import hashlib
import math
import os
import struct
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from voxcpm_runtime.backend_types import AudioResult

_PCM_SAMPLE_WIDTH_BYTES: Final = 2
_PCM_S16_MAX: Final = 32767
_PCM_S16_MIN: Final = -32768
_PCM_S16_SCALE: Final = 32767.0
_RIFF_MAGIC: Final = b"RIFF"
_WAVE_MAGIC: Final = b"WAVE"
_RIFF_HEADER_BYTES: Final = 12
_WAVE_FORMAT_PCM: Final = 1
_HASH_CHUNK_BYTES: Final = 1024 * 1024
_TEMP_SUFFIX: Final = ".part"


class WavWriteError(ValueError):
    """Raised when a WAV file cannot be written safely.

    The message is intentionally generic so that private filesystem paths are
    never surfaced to callers or to the public report.
    """


class WavValidationError(ValueError):
    """Raised when a written WAV file fails post-write validation."""


@dataclass(frozen=True, slots=True)
class WavWriteResult:
    """Project-owned description of a written WAV file.

    ``name`` holds the file name only. The parent directory is never recorded
    so that the public report cannot leak a private absolute path.
    """

    name: str
    byte_size: int
    sha256: str
    channel_count: int
    sample_rate_hz: int
    sample_width_bytes: int
    frame_count: int
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "byte_size": self.byte_size,
            "channel_count": self.channel_count,
            "duration_seconds": round(self.duration_seconds, 6),
            "frame_count": self.frame_count,
            "name": self.name,
            "sample_rate_hz": self.sample_rate_hz,
            "sample_width_bytes": self.sample_width_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class WavValidation:
    """Result of re-reading a written WAV file from disk."""

    exists: bool
    riff_header_ok: bool
    wave_format_pcm: bool
    channel_count: int
    sample_width_bytes: int
    sample_rate_hz: int
    frame_count: int
    duration_seconds: float
    byte_size: int
    sha256: str
    sample_rate_matches: bool
    channel_count_matches: bool
    failure_reason: str | None = None

    @property
    def valid(self) -> bool:
        return self.failure_reason is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "byte_size": self.byte_size,
            "channel_count": self.channel_count,
            "channel_count_matches": self.channel_count_matches,
            "duration_seconds": round(self.duration_seconds, 6),
            "exists": self.exists,
            "failure_reason": self.failure_reason,
            "frame_count": self.frame_count,
            "riff_header_ok": self.riff_header_ok,
            "sample_rate_hz": self.sample_rate_hz,
            "sample_rate_matches": self.sample_rate_matches,
            "sample_width_bytes": self.sample_width_bytes,
            "sha256": self.sha256,
            "valid": self.valid,
            "wave_format_pcm": self.wave_format_pcm,
        }


def _clip_and_quantize(samples: tuple[float, ...]) -> array.array[int]:
    """Clip to [-1.0, 1.0] and convert to little-endian signed PCM16."""

    buffer = array.array("h")
    append = buffer.append
    for sample in samples:
        if not isinstance(sample, (int, float)) or isinstance(sample, bool):
            raise WavWriteError("Audio samples must be finite real numbers.")
        value = float(sample)
        if not math.isfinite(value):
            raise WavWriteError("Audio samples must be finite real numbers.")
        if value > 1.0:
            value = 1.0
        elif value < -1.0:
            value = -1.0
        quantized = int(round(value * _PCM_S16_SCALE))
        if quantized > _PCM_S16_MAX:
            quantized = _PCM_S16_MAX
        elif quantized < _PCM_S16_MIN:
            quantized = _PCM_S16_MIN
        append(quantized)
    if sys.byteorder != "little":
        buffer.byteswap()
    return buffer


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_parent(path: Path, create_parents: bool) -> None:
    parent = path.parent
    if create_parents:
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            raise WavWriteError("The WAV output directory could not be created.") from None
    if not parent.is_dir():
        raise WavWriteError("The WAV output directory does not exist.")


def write_wav(
    path: str | os.PathLike[str],
    result: AudioResult,
    *,
    create_parents: bool = False,
) -> WavWriteResult:
    """Write a project-owned mono PCM16 WAV file atomically.

    The file is first written to a sibling temporary file and then moved into
    place, so a failed write never leaves a partially written target behind.
    """

    if not isinstance(result, AudioResult):
        raise WavWriteError("write_wav requires a project AudioResult.")
    if not result.samples:
        raise WavWriteError("The audio result contains no samples.")
    try:
        target = Path(path)
    except (TypeError, ValueError):
        raise WavWriteError("The WAV output path is not a usable filesystem path.") from None
    if not target.name:
        raise WavWriteError("The WAV output path is not a usable filesystem path.")
    _ensure_parent(target, create_parents)
    frames = _clip_and_quantize(result.samples)
    frame_count = len(frames)
    payload = frames.tobytes()
    temporary = target.with_name(f".{target.name}{_TEMP_SUFFIX}")
    try:
        with wave.open(str(temporary), "wb") as handle:
            handle.setnchannels(result.channels)
            handle.setsampwidth(_PCM_SAMPLE_WIDTH_BYTES)
            handle.setframerate(result.sample_rate_hz)
            handle.writeframes(payload)
    except (OSError, wave.Error, ValueError):
        _discard(temporary)
        raise WavWriteError("The WAV file could not be written.") from None
    try:
        os.replace(temporary, target)
    except OSError:
        _discard(temporary)
        raise WavWriteError("The WAV file could not be finalized.") from None
    try:
        byte_size = target.stat().st_size
        digest = _sha256_file(target)
    except OSError:
        raise WavWriteError("The written WAV file could not be inspected.") from None
    return WavWriteResult(
        name=target.name,
        byte_size=byte_size,
        sha256=digest,
        channel_count=result.channels,
        sample_rate_hz=result.sample_rate_hz,
        sample_width_bytes=_PCM_SAMPLE_WIDTH_BYTES,
        frame_count=frame_count,
        duration_seconds=frame_count / result.sample_rate_hz,
    )


def _discard(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _riff_header_ok(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            header = stream.read(_RIFF_HEADER_BYTES)
    except OSError:
        return False
    if len(header) < _RIFF_HEADER_BYTES:
        return False
    if header[:4] != _RIFF_MAGIC or header[8:12] != _WAVE_MAGIC:
        return False
    declared = struct.unpack("<I", header[4:8])[0]
    return declared == path.stat().st_size - 8


def _wave_format_is_pcm(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            stream.seek(12)
            chunk_id = stream.read(4)
            if chunk_id != b"fmt ":
                return False
            chunk_size = struct.unpack("<I", stream.read(4))[0]
            if chunk_size < 16:
                return False
            audio_format = struct.unpack("<H", stream.read(2))[0]
    except (OSError, struct.error):
        return False
    return audio_format == _WAVE_FORMAT_PCM


def _invalid(
    reason: str,
    *,
    exists: bool = True,
    riff_header_ok: bool = False,
    wave_format_pcm: bool = False,
    channel_count: int = 0,
    sample_width_bytes: int = 0,
    sample_rate_hz: int = 0,
    frame_count: int = 0,
    byte_size: int = 0,
    sha256: str = "",
    sample_rate_matches: bool = False,
    channel_count_matches: bool = False,
) -> WavValidation:
    duration = (
        frame_count / sample_rate_hz
        if sample_rate_hz > 0 and frame_count > 0
        else 0.0
    )
    return WavValidation(
        exists=exists,
        riff_header_ok=riff_header_ok,
        wave_format_pcm=wave_format_pcm,
        channel_count=channel_count,
        sample_width_bytes=sample_width_bytes,
        sample_rate_hz=sample_rate_hz,
        frame_count=frame_count,
        duration_seconds=duration,
        byte_size=byte_size,
        sha256=sha256,
        sample_rate_matches=sample_rate_matches,
        channel_count_matches=channel_count_matches,
        failure_reason=reason,
    )


def validate_wav(
    path: str | os.PathLike[str],
    *,
    expected_sample_rate_hz: int | None = None,
    expected_channel_count: int | None = None,
) -> WavValidation:
    """Re-read a written WAV file and return project-owned validation facts."""

    try:
        target = Path(path)
    except (TypeError, ValueError):
        return _invalid("unusable_path", exists=False)
    try:
        byte_size = target.stat().st_size
    except FileNotFoundError:
        return _invalid("missing_file", exists=False)
    except OSError:
        return _invalid("unreadable_file", exists=False)
    if byte_size == 0:
        return _invalid("empty_file", byte_size=0)
    riff_ok = _riff_header_ok(target)
    if not riff_ok:
        return _invalid("riff_header_invalid", byte_size=byte_size)
    format_pcm = _wave_format_is_pcm(target)
    if not format_pcm:
        return _invalid(
            "wave_format_not_pcm",
            riff_header_ok=True,
            byte_size=byte_size,
        )
    try:
        with wave.open(str(target), "rb") as handle:
            channel_count = handle.getnchannels()
            sample_width_bytes = handle.getsampwidth()
            sample_rate_hz = handle.getframerate()
            frame_count = handle.getnframes()
    except (OSError, wave.Error, EOFError):
        return _invalid(
            "wave_module_cannot_read",
            riff_header_ok=True,
            wave_format_pcm=True,
            byte_size=byte_size,
        )
    try:
        digest = _sha256_file(target)
    except OSError:
        digest = ""
    rate_ok = expected_sample_rate_hz is None or sample_rate_hz == expected_sample_rate_hz
    channels_ok = (
        expected_channel_count is None or channel_count == expected_channel_count
    )
    if channel_count <= 0:
        return _invalid(
            "channel_count_invalid",
            riff_header_ok=True,
            wave_format_pcm=True,
            channel_count=channel_count,
            sample_width_bytes=sample_width_bytes,
            sample_rate_hz=sample_rate_hz,
            frame_count=frame_count,
            byte_size=byte_size,
            sha256=digest,
            channel_count_matches=channels_ok,
        )
    if sample_width_bytes <= 0:
        return _invalid(
            "sample_width_invalid",
            riff_header_ok=True,
            wave_format_pcm=True,
            channel_count=channel_count,
            sample_width_bytes=sample_width_bytes,
            sample_rate_hz=sample_rate_hz,
            frame_count=frame_count,
            byte_size=byte_size,
            sha256=digest,
            sample_rate_matches=rate_ok,
            channel_count_matches=channels_ok,
        )
    if sample_rate_hz <= 0:
        return _invalid(
            "sample_rate_invalid",
            riff_header_ok=True,
            wave_format_pcm=True,
            channel_count=channel_count,
            sample_width_bytes=sample_width_bytes,
            sample_rate_hz=sample_rate_hz,
            frame_count=frame_count,
            byte_size=byte_size,
            sha256=digest,
            channel_count_matches=channels_ok,
        )
    if frame_count <= 0:
        return _invalid(
            "frame_count_empty",
            riff_header_ok=True,
            wave_format_pcm=True,
            channel_count=channel_count,
            sample_width_bytes=sample_width_bytes,
            sample_rate_hz=sample_rate_hz,
            frame_count=frame_count,
            byte_size=byte_size,
            sha256=digest,
            sample_rate_matches=rate_ok,
            channel_count_matches=channels_ok,
        )
    if not rate_ok:
        return _invalid(
            "sample_rate_mismatch",
            riff_header_ok=True,
            wave_format_pcm=True,
            channel_count=channel_count,
            sample_width_bytes=sample_width_bytes,
            sample_rate_hz=sample_rate_hz,
            frame_count=frame_count,
            byte_size=byte_size,
            sha256=digest,
            channel_count_matches=channels_ok,
        )
    if not channels_ok:
        return _invalid(
            "channel_count_mismatch",
            riff_header_ok=True,
            wave_format_pcm=True,
            channel_count=channel_count,
            sample_width_bytes=sample_width_bytes,
            sample_rate_hz=sample_rate_hz,
            frame_count=frame_count,
            byte_size=byte_size,
            sha256=digest,
            sample_rate_matches=rate_ok,
        )
    return WavValidation(
        exists=True,
        riff_header_ok=True,
        wave_format_pcm=True,
        channel_count=channel_count,
        sample_width_bytes=sample_width_bytes,
        sample_rate_hz=sample_rate_hz,
        frame_count=frame_count,
        duration_seconds=frame_count / sample_rate_hz,
        byte_size=byte_size,
        sha256=digest,
        sample_rate_matches=True,
        channel_count_matches=True,
        failure_reason=None,
    )


__all__: Final[tuple[str, ...]] = (
    "WavValidation",
    "WavValidationError",
    "WavWriteError",
    "WavWriteResult",
    "validate_wav",
    "write_wav",
)
