from __future__ import annotations

import math
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import ClassVar, Final, TypeAlias

ErrorDetail: TypeAlias = str | int | float | bool | None
ErrorDetails: TypeAlias = Mapping[str, ErrorDetail]
_SAFE_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")


def _safe_token(value: str, fallback: str) -> str:
    if not isinstance(value, str) or _SAFE_TOKEN.fullmatch(value) is None:
        return fallback
    return value


def _normalize_details(details: ErrorDetails | None) -> Mapping[str, ErrorDetail]:
    if details is None:
        normalized: dict[str, ErrorDetail] = {}
    else:
        values: dict[str, ErrorDetail] = {}
        for key, value in details.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("Backend error detail keys must be non-empty strings")
            if value is not None and not isinstance(value, (str, bool, int, float)):
                raise ValueError("Backend error details must contain safe scalar values")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Backend error details must contain finite values")
            values[key] = value
        normalized = dict(sorted(values.items()))
    return MappingProxyType(normalized)


class BackendError(Exception):
    default_code: ClassVar[str] = "backend_error"
    default_message: ClassVar[str] = "Backend operation failed."
    default_retryable: ClassVar[bool] = False

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        retryable: bool | None = None,
        details: ErrorDetails | None = None,
    ) -> None:
        public_message = self.default_message if message is None else message
        if not isinstance(public_message, str) or not public_message.strip():
            raise ValueError("Backend error messages must be non-empty strings")
        public_code = self.default_code if code is None else code
        if not isinstance(public_code, str) or _SAFE_TOKEN.fullmatch(public_code) is None:
            raise ValueError("Backend error codes must be safe identifiers")
        public_retryable = self.default_retryable if retryable is None else retryable
        if not isinstance(public_retryable, bool):
            raise ValueError("Backend error retryability must be boolean")
        super().__init__(public_message)
        self.message = public_message
        self.code = public_code
        self.retryable = public_retryable
        self.details = _normalize_details(details)

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "details": dict(sorted(self.details.items())),
            "message": self.message,
            "retryable": self.retryable,
        }


class BackendStateError(BackendError):
    default_code: ClassVar[str] = "backend_state_error"
    default_message: ClassVar[str] = "Backend lifecycle state does not allow this operation."


class BackendRequestError(BackendError):
    default_code: ClassVar[str] = "invalid_backend_request"
    default_message: ClassVar[str] = "Backend request is invalid."


class BackendUnsupportedError(BackendError):
    default_code: ClassVar[str] = "backend_operation_unsupported"
    default_message: ClassVar[str] = "Backend operation is not supported."


class BackendExecutionError(BackendError):
    default_code: ClassVar[str] = "backend_execution_failed"
    default_message: ClassVar[str] = "Backend execution failed."


def normalize_backend_error(operation: str, error: BaseException) -> BackendError:
    if isinstance(error, BackendError):
        return error
    safe_operation = _safe_token(operation, "backend_operation")
    exception_type = _safe_token(type(error).__name__, "UnknownBackendError")
    return BackendExecutionError(
        details={
            "exception_type": exception_type,
            "operation": safe_operation,
        }
    )


__all__: Final[tuple[str, ...]] = (
    "BackendError",
    "BackendExecutionError",
    "BackendRequestError",
    "BackendStateError",
    "BackendUnsupportedError",
    "ErrorDetail",
    "ErrorDetails",
    "normalize_backend_error",
)
