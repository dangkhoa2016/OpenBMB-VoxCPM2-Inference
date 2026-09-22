from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from voxcpm_runtime.backend_types import AudioChunk, AudioResult, OneShotRequest, StreamRequest
from voxcpm_runtime.errors import ErrorDetail

WorkerOperation: TypeAlias = Literal[
    "synthesize",
    "design",
    "clone",
    "continue_audio",
    "stream",
]
WorkerMessageKind: TypeAlias = Literal[
    "ready",
    "result",
    "stream_chunk",
    "stream_end",
    "error",
    "cancelled",
    "shutdown_ack",
]
WorkerPayload: TypeAlias = OneShotRequest | StreamRequest
WorkerResultPayload: TypeAlias = AudioResult | AudioChunk | None


def _require_token(value: str, field: str, *, max_length: int = 128) -> None:
    if not isinstance(value, str) or not value or len(value) > max_length:
        raise ValueError(f"{field} must be a non-empty bounded string")
    if any(ch in value for ch in ("/", "\\", "\x00", "\n", "\r")):
        raise ValueError(f"{field} contains unsupported characters")


@dataclass(frozen=True, slots=True)
class WorkerBootstrapSpec:
    worker_id: str
    backend_kind: Literal["fake", "real"] = "fake"
    physical_gpu_index: int | None = None
    fake_failure_operations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_token(self.worker_id, "worker_id")
        if self.backend_kind not in {"fake", "real"}:
            raise ValueError("backend_kind must be fake or real")
        if self.physical_gpu_index is not None:
            if (
                isinstance(self.physical_gpu_index, bool)
                or not isinstance(self.physical_gpu_index, int)
                or self.physical_gpu_index < 0
            ):
                raise ValueError("physical_gpu_index must be a non-negative integer")
        if not isinstance(self.fake_failure_operations, tuple):
            raise ValueError("fake_failure_operations must be a tuple")
        allowed = {"load", "synthesize", "design", "clone", "continue_audio", "stream", "close"}
        if any(item not in allowed for item in self.fake_failure_operations):
            raise ValueError("fake_failure_operations contains an unsupported operation")
        if len(set(self.fake_failure_operations)) != len(self.fake_failure_operations):
            raise ValueError("fake_failure_operations must be unique")
        object.__setattr__(self, "fake_failure_operations", tuple(sorted(self.fake_failure_operations)))


@dataclass(frozen=True, slots=True)
class WorkerRequest:
    request_id: str
    operation: WorkerOperation
    payload: WorkerPayload

    def __post_init__(self) -> None:
        _require_token(self.request_id, "request_id")
        if self.operation not in {
            "synthesize",
            "design",
            "clone",
            "continue_audio",
            "stream",
        }:
            raise ValueError("unsupported worker operation")
        if self.operation == "stream":
            if not isinstance(self.payload, StreamRequest):
                raise ValueError("stream operation requires StreamRequest")
        elif isinstance(self.payload, StreamRequest):
            raise ValueError("one-shot operation cannot carry StreamRequest")


@dataclass(frozen=True, slots=True)
class WorkerErrorData:
    code: str
    message: str
    retryable: bool
    details: tuple[tuple[str, ErrorDetail], ...] = ()

    def __post_init__(self) -> None:
        _require_token(self.code, "code", max_length=64)
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("message must be non-empty")
        if not isinstance(self.retryable, bool):
            raise ValueError("retryable must be boolean")
        object.__setattr__(self, "details", tuple(sorted(self.details)))


@dataclass(frozen=True, slots=True)
class WorkerMessage:
    kind: WorkerMessageKind
    worker_id: str
    request_id: str | None = None
    payload: WorkerResultPayload = None
    error: WorkerErrorData | None = None
    metadata: tuple[tuple[str, ErrorDetail], ...] = ()

    def __post_init__(self) -> None:
        _require_token(self.worker_id, "worker_id")
        if self.request_id is not None:
            _require_token(self.request_id, "request_id")
        if self.kind not in {
            "ready",
            "result",
            "stream_chunk",
            "stream_end",
            "error",
            "cancelled",
            "shutdown_ack",
        }:
            raise ValueError("unsupported worker message kind")
        object.__setattr__(self, "metadata", tuple(sorted(self.metadata)))


__all__: Final[tuple[str, ...]] = (
    "WorkerBootstrapSpec",
    "WorkerErrorData",
    "WorkerMessage",
    "WorkerMessageKind",
    "WorkerOperation",
    "WorkerPayload",
    "WorkerRequest",
    "WorkerResultPayload",
)
