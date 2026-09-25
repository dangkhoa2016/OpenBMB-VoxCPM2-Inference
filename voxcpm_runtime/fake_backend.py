from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Final, Literal, TypeAlias, TypeVar, cast

from voxcpm_runtime.backend_types import (
    AudioChunk,
    AudioResult,
    BackendInfo,
    CloneRequest,
    ContinuationRequest,
    OneShotRequest,
    SpeechRequest,
    StreamRequest,
    VoiceDesignRequest,
)
from voxcpm_runtime.errors import (
    BackendError,
    BackendRequestError,
    BackendStateError,
    normalize_backend_error,
)

FakeOperation: TypeAlias = Literal[
    "load",
    "synthesize",
    "design",
    "clone",
    "continue_audio",
    "stream",
    "close",
]
FakeMetadata: TypeAlias = tuple[tuple[str, str], ...]
_ResultType = TypeVar("_ResultType")
_FAKE_OPERATIONS = frozenset(
    {
        "load",
        "synthesize",
        "design",
        "clone",
        "continue_audio",
        "stream",
        "close",
    }
)
_PRIVATE_DETAIL = "m3-private-backend-detail-placeholder"
_SAMPLE_RATE_HZ = 48_000
_SAMPLE_COUNT = 480
_DEFAULT_CHUNK_SIZE = 120


class _BackendState(Enum):
    CREATED = "created"
    LOADED = "loaded"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class FakeFailurePlan:
    operations: tuple[FakeOperation, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.operations, str) or not isinstance(
            self.operations, (tuple, list)
        ):
            raise ValueError("Fake failure operations must be provided as a sequence")
        operations = tuple(self.operations)
        if any(operation not in _FAKE_OPERATIONS for operation in operations):
            raise ValueError("Fake failure operation is not supported")
        if len(operations) != len(set(operations)):
            raise ValueError("Fake failure operations must be unique")
        object.__setattr__(self, "operations", tuple(sorted(operations)))


def _operation_for_request(request: OneShotRequest) -> str:
    if isinstance(request, SpeechRequest):
        return "synthesize"
    if isinstance(request, VoiceDesignRequest):
        return "design"
    if isinstance(request, CloneRequest):
        return "clone"
    if isinstance(request, ContinuationRequest):
        return "continue_audio"
    raise BackendRequestError(
        "Backend request is invalid.",
        details={"field": "request"},
    )


def _canonical_request(operation: str, request: OneShotRequest) -> bytes:
    payload: dict[str, object]
    if isinstance(request, SpeechRequest):
        payload = {"text": request.text}
    elif isinstance(request, VoiceDesignRequest):
        payload = {"instruction": request.instruction, "text": request.text}
    elif isinstance(request, CloneRequest):
        payload = {
            "reference_audio": request.reference_audio.local_path,
            "text": request.text,
        }
    elif isinstance(request, ContinuationRequest):
        payload = {
            "reference_audio": request.reference_audio.local_path,
            "reference_transcript": request.reference_transcript,
            "text": request.text,
        }
    else:
        raise BackendRequestError(
            "Backend request is invalid.",
            details={"field": "request"},
        )
    canonical = {
        "operation": operation,
        "payload": payload,
    }
    serialized = json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).digest()


def _result_metadata(operation: str) -> FakeMetadata:
    return (
        ("backend", "fake"),
        ("contract_scope", "test/fake contract only"),
        ("operation", operation),
        ("sample_rate_scope", "test/fake contract only"),
    )


def _call_normalized(operation: str, action: Callable[[], _ResultType]) -> _ResultType:
    try:
        return action()
    except BackendError as error:
        normalized: BackendError = error
    except Exception as error:
        normalized = normalize_backend_error(operation, error)
    raise normalized


def _synthesize_result(operation: str, request: OneShotRequest) -> AudioResult:
    digest = _canonical_request(operation, request)
    amplitude_nanos = 100_000_000 + (digest[-1] * 50_000_000 // 255)
    samples = tuple(
        (digest[index % len(digest)] - 128)
        * amplitude_nanos
        / 128_000_000_000
        for index in range(_SAMPLE_COUNT)
    )
    return AudioResult(
        samples=samples,
        sample_rate_hz=_SAMPLE_RATE_HZ,
        channels=1,
        metadata=_result_metadata(operation),
    )


class FakeVoxCPMBackend:
    def __init__(
        self,
        *,
        failure_plan: FakeFailurePlan | None = None,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
    ) -> None:
        if failure_plan is not None and not isinstance(failure_plan, FakeFailurePlan):
            raise ValueError("failure_plan must be a FakeFailurePlan")
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        self._failure_plan = failure_plan or FakeFailurePlan()
        self._chunk_size = chunk_size
        self._state = _BackendState.CREATED

    def load(self) -> BackendInfo:
        if self._state is _BackendState.CLOSED:
            raise BackendStateError(
                "Backend cannot reopen after it has been closed.",
                code="backend_closed",
                details={"state": self._state.value},
            )
        if self._state is _BackendState.LOADED:
            return self._info()
        _call_normalized("load", lambda: self._raise_injected_failure("load"))
        self._state = _BackendState.LOADED
        return self._info()

    def synthesize(self, request: SpeechRequest) -> AudioResult:
        return self._run("synthesize", SpeechRequest, request)

    def design(self, request: VoiceDesignRequest) -> AudioResult:
        return self._run("design", VoiceDesignRequest, request)

    def clone(self, request: CloneRequest) -> AudioResult:
        return self._run("clone", CloneRequest, request)

    def continue_audio(self, request: ContinuationRequest) -> AudioResult:
        return self._run("continue_audio", ContinuationRequest, request)

    def stream(self, request: StreamRequest) -> Iterator[AudioChunk]:
        self._ensure_ready()
        if not isinstance(request, StreamRequest):
            raise BackendRequestError(
                "Backend request is invalid.",
                details={"field": "request"},
            )
        operation = _operation_for_request(request.request)
        result = _call_normalized(
            "stream",
            lambda: self._stream_result(operation, request),
        )
        return self._chunks(result)

    def close(self) -> None:
        if self._state is _BackendState.CLOSED:
            return None
        _call_normalized("close", lambda: self._raise_injected_failure("close"))
        self._state = _BackendState.CLOSED
        return None

    def _run(
        self,
        operation: str,
        expected_type: type[object],
        request: object,
    ) -> AudioResult:
        self._ensure_ready()
        if not isinstance(request, expected_type):
            raise BackendRequestError(
                "Backend request is invalid.",
                details={"field": "request"},
            )
        return _call_normalized(
            operation,
            lambda: self._generation_result(operation, request),
        )

    def _generation_result(self, operation: str, request: object) -> AudioResult:
        self._raise_injected_failure(operation)
        return _synthesize_result(operation, cast(OneShotRequest, request))

    def _stream_result(
        self,
        operation: str,
        request: StreamRequest,
    ) -> AudioResult:
        self._raise_injected_failure("stream")
        return _synthesize_result(operation, request.request)

    def _ensure_ready(self) -> None:
        if self._state is _BackendState.CREATED:
            raise BackendStateError(
                "Backend must be loaded before generation.",
                code="backend_not_loaded",
                details={"state": self._state.value},
            )
        if self._state is _BackendState.CLOSED:
            raise BackendStateError(
                "Backend cannot generate after it has been closed.",
                code="backend_closed",
                details={"state": self._state.value},
            )

    def _raise_injected_failure(self, operation: str) -> None:
        if operation in self._failure_plan.operations:
            raise RuntimeError(_PRIVATE_DETAIL)

    def _chunks(self, result: AudioResult) -> Iterator[AudioChunk]:
        for start in range(0, len(result.samples), self._chunk_size):
            end = min(start + self._chunk_size, len(result.samples))
            yield AudioChunk(
                samples=result.samples[start:end],
                sample_rate_hz=result.sample_rate_hz,
                channels=result.channels,
                sequence=start // self._chunk_size,
                is_final=end == len(result.samples),
                metadata=result.metadata,
            )

    def _info(self) -> BackendInfo:
        return BackendInfo(
            backend="fake",
            implementation="deterministic-test",
            loaded=True,
            device="fake/cpu-test",
            model_id="m3-fake-deterministic",
            capabilities=(
                "clone",
                "continue_audio",
                "design",
                "stream",
                "synthesize",
            ),
            metadata=(
                ("contract_scope", "test/fake contract only"),
                ("sample_rate_scope", "test/fake contract only"),
            ),
        )


__all__: Final[tuple[str, ...]] = (
    "FakeFailurePlan",
    "FakeOperation",
    "FakeVoxCPMBackend",
)
