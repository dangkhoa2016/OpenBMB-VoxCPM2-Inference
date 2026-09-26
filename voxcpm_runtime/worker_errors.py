from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping

from voxcpm_runtime.errors import ErrorDetail


class WorkerError(Exception):
    default_code = "worker_error"
    default_message = "Worker operation failed."

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        retryable: bool = False,
        details: Mapping[str, ErrorDetail] | None = None,
    ) -> None:
        public_message = self.default_message if message is None else message
        public_code = self.default_code if code is None else code
        if not isinstance(public_message, str) or not public_message.strip():
            raise ValueError("worker error message must be non-empty")
        if not isinstance(public_code, str) or not public_code.replace("_", "").isalnum():
            raise ValueError("worker error code must be a safe identifier")
        if not isinstance(retryable, bool):
            raise ValueError("retryable must be boolean")
        normalized = {} if details is None else dict(sorted(details.items()))
        super().__init__(public_message)
        self.message = public_message
        self.code = public_code
        self.retryable = retryable
        self.details = MappingProxyType(normalized)

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "details": dict(self.details),
        }


class WorkerStateError(WorkerError):
    default_code = "worker_state_error"
    default_message = "Worker lifecycle state does not allow this operation."


class WorkerStartError(WorkerError):
    default_code = "worker_start_failed"
    default_message = "Worker failed to start."


class WorkerTransportError(WorkerError):
    default_code = "worker_transport_failed"
    default_message = "Worker transport failed."


class WorkerExitedError(WorkerError):
    default_code = "worker_exited"
    default_message = "Worker process exited unexpectedly."


class WorkerCancelledError(WorkerError):
    default_code = "request_cancelled"
    default_message = "Worker request was cancelled."


class SchedulerAdmissionError(WorkerError):
    default_code = "scheduler_admission_failed"
    default_message = "Scheduler admission failed."


__all__: Final[tuple[str, ...]] = (
    "SchedulerAdmissionError",
    "WorkerCancelledError",
    "WorkerError",
    "WorkerExitedError",
    "WorkerStartError",
    "WorkerStateError",
    "WorkerTransportError",
)
