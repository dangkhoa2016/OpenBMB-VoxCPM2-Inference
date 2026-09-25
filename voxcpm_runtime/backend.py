from __future__ import annotations

from collections.abc import Iterator
from typing import Final, Protocol, runtime_checkable

from voxcpm_runtime.backend_types import (
    AudioChunk,
    AudioResult,
    BackendInfo,
    CloneRequest,
    ContinuationRequest,
    SpeechRequest,
    StreamRequest,
    VoiceDesignRequest,
)


@runtime_checkable
class InferenceBackend(Protocol):
    def load(self) -> BackendInfo:
        ...

    def synthesize(self, request: SpeechRequest) -> AudioResult:
        ...

    def design(self, request: VoiceDesignRequest) -> AudioResult:
        ...

    def clone(self, request: CloneRequest) -> AudioResult:
        ...

    def continue_audio(self, request: ContinuationRequest) -> AudioResult:
        ...

    def stream(self, request: StreamRequest) -> Iterator[AudioChunk]:
        ...

    def close(self) -> None:
        ...


__all__: Final[tuple[str, ...]] = ("InferenceBackend",)
