from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, TypeAlias

from voxcpm_runtime.errors import BackendRequestError, ErrorDetail

Metadata: TypeAlias = tuple[tuple[str, ErrorDetail], ...]


def _invalid(field: str) -> BackendRequestError:
    return BackendRequestError(
        "Backend request field is invalid.",
        details={"field": field},
    )


def _require_text(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise _invalid(field)
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        encodable = False
    else:
        encodable = True
    if not encodable:
        raise _invalid(field)


def _require_positive_integer(value: int, field: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise _invalid(field)


def _normalize_metadata(
    metadata: Metadata | None,
    field: str,
) -> Metadata:
    if metadata is None:
        return ()
    if not isinstance(metadata, tuple):
        raise _invalid(field)
    normalized: list[tuple[str, ErrorDetail]] = []
    seen: set[str] = set()
    for item in metadata:
        if not isinstance(item, tuple) or len(item) != 2:
            raise _invalid(field)
        key, value = item
        _require_text(key, f"{field}.key")
        if key in seen:
            raise _invalid(f"{field}.key")
        if value is not None and not isinstance(value, (str, bool, int, float)):
            raise _invalid(f"{field}.value")
        if isinstance(value, float) and not math.isfinite(value):
            raise _invalid(f"{field}.value")
        seen.add(key)
        normalized.append((key, value))
    return tuple(sorted(normalized))


def _normalize_samples(samples: tuple[float, ...]) -> tuple[float, ...]:
    if not isinstance(samples, tuple) or not samples:
        raise _invalid("samples")
    normalized: list[float] = []
    for sample in samples:
        if not isinstance(sample, (int, float)) or isinstance(sample, bool):
            raise _invalid("samples")
        value = float(sample)
        if not math.isfinite(value):
            raise _invalid("samples")
        normalized.append(0.0 if value == 0.0 else value)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class AudioReference:
    local_path: str

    def __post_init__(self) -> None:
        _require_text(self.local_path, "local_path")


@dataclass(frozen=True, slots=True)
class SpeechRequest:
    text: str

    def __post_init__(self) -> None:
        _require_text(self.text, "text")


@dataclass(frozen=True, slots=True)
class VoiceDesignRequest:
    text: str
    instruction: str

    def __post_init__(self) -> None:
        _require_text(self.text, "text")
        _require_text(self.instruction, "instruction")


@dataclass(frozen=True, slots=True)
class CloneRequest:
    text: str
    reference_audio: AudioReference

    def __post_init__(self) -> None:
        _require_text(self.text, "text")
        if not isinstance(self.reference_audio, AudioReference):
            raise _invalid("reference_audio")


@dataclass(frozen=True, slots=True)
class ContinuationRequest:
    text: str
    reference_audio: AudioReference
    reference_transcript: str

    def __post_init__(self) -> None:
        _require_text(self.text, "text")
        if not isinstance(self.reference_audio, AudioReference):
            raise _invalid("reference_audio")
        _require_text(self.reference_transcript, "reference_transcript")


OneShotRequest: TypeAlias = (
    SpeechRequest | VoiceDesignRequest | CloneRequest | ContinuationRequest
)


@dataclass(frozen=True, slots=True)
class StreamRequest:
    request: OneShotRequest

    def __post_init__(self) -> None:
        if not isinstance(
            self.request,
            (SpeechRequest, VoiceDesignRequest, CloneRequest, ContinuationRequest),
        ):
            raise _invalid("request")


@dataclass(frozen=True, slots=True)
class BackendInfo:
    backend: str
    implementation: str
    loaded: bool
    device: str
    model_id: str | None
    capabilities: tuple[str, ...]
    metadata: Metadata = ()

    def __post_init__(self) -> None:
        _require_text(self.backend, "backend")
        _require_text(self.implementation, "implementation")
        if not isinstance(self.loaded, bool):
            raise _invalid("loaded")
        _require_text(self.device, "device")
        if self.model_id is not None:
            _require_text(self.model_id, "model_id")
        if not isinstance(self.capabilities, tuple):
            raise _invalid("capabilities")
        capabilities: list[str] = []
        seen: set[str] = set()
        for capability in self.capabilities:
            _require_text(capability, "capabilities")
            if capability in seen:
                raise _invalid("capabilities")
            seen.add(capability)
            capabilities.append(capability)
        object.__setattr__(self, "capabilities", tuple(sorted(capabilities)))
        object.__setattr__(
            self,
            "metadata",
            _normalize_metadata(self.metadata, "metadata"),
        )


@dataclass(frozen=True, slots=True)
class AudioResult:
    samples: tuple[float, ...]
    sample_rate_hz: int
    channels: int
    metadata: Metadata = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "samples", _normalize_samples(self.samples))
        _require_positive_integer(self.sample_rate_hz, "sample_rate_hz")
        _require_positive_integer(self.channels, "channels")
        object.__setattr__(
            self,
            "metadata",
            _normalize_metadata(self.metadata, "metadata"),
        )


@dataclass(frozen=True, slots=True)
class AudioChunk:
    samples: tuple[float, ...]
    sample_rate_hz: int
    channels: int
    sequence: int
    is_final: bool
    metadata: Metadata = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "samples", _normalize_samples(self.samples))
        _require_positive_integer(self.sample_rate_hz, "sample_rate_hz")
        _require_positive_integer(self.channels, "channels")
        if not isinstance(self.sequence, int) or isinstance(self.sequence, bool):
            raise _invalid("sequence")
        if self.sequence < 0:
            raise _invalid("sequence")
        if not isinstance(self.is_final, bool):
            raise _invalid("is_final")
        object.__setattr__(
            self,
            "metadata",
            _normalize_metadata(self.metadata, "metadata"),
        )


__all__: Final[tuple[str, ...]] = (
    "AudioChunk",
    "AudioReference",
    "AudioResult",
    "BackendInfo",
    "CloneRequest",
    "ContinuationRequest",
    "OneShotRequest",
    "SpeechRequest",
    "StreamRequest",
    "VoiceDesignRequest",
)
