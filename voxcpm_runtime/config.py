from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class ConfigurationError(ValueError):
    def __init__(self, variable: str, reason: str) -> None:
        self.variable = variable
        self.reason = reason
        super().__init__(f"{variable}: {reason}")


ConfigError = ConfigurationError


class DeviceMode(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class Backend(str, Enum):
    PYTORCH_VOXCPM = "pytorch-voxcpm"


class OptimizationMode(str, Enum):
    AUTO = "auto"


class ReadinessMode(str, Enum):
    DEGRADED = "degraded"
    STRICT = "strict"


class StreamFormat(str, Enum):
    PCM_S16LE = "pcm_s16le"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    JSON = "json"


_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_INTEGER_PATTERN = re.compile(r"[0-9]+\Z", re.ASCII)
_UNSET = object()


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _trimmed(value: Any, variable: str) -> str | None:
    text = _text(value)
    if text is None:
        return None
    return text.strip()


def _mapping_value(mapping: Mapping[str, Any], key: str, default: Any = _UNSET) -> Any:
    value = mapping.get(key, _UNSET)
    if value is _UNSET or value is None:
        return default
    return value


def _lexical(
    mapping: Mapping[str, Any],
    key: str,
    default: str | None,
    optional: bool = False,
) -> str | None:
    raw = _mapping_value(mapping, key, _UNSET)
    if raw is _UNSET:
        return default
    value = _trimmed(raw, key)
    if not value:
        return None if optional else default
    return value


def _boolean(mapping: Mapping[str, Any], key: str, default: bool) -> bool:
    raw = _mapping_value(mapping, key, _UNSET)
    if raw is _UNSET:
        return default
    if isinstance(raw, bool):
        return raw
    value = _trimmed(raw, key)
    if not value:
        return default
    normalized = value.casefold()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ConfigurationError(key, "expected one of 1, true, yes, on, 0, false, no, off")


def _enum_value(
    mapping: Mapping[str, Any],
    key: str,
    default: str,
    allowed: tuple[str, ...],
    case_insensitive: bool = False,
) -> str:
    raw = _mapping_value(mapping, key, _UNSET)
    if raw is _UNSET:
        return default
    value = _trimmed(raw, key)
    if not value:
        return default
    normalized = value.casefold() if case_insensitive else value
    accepted = tuple(item.casefold() if case_insensitive else item for item in allowed)
    if normalized not in accepted:
        expected = ", ".join(allowed)
        raise ConfigurationError(key, f"expected one of: {expected}")
    for item in allowed:
        candidate = item.casefold() if case_insensitive else item
        if normalized == candidate:
            return item
    raise AssertionError("unreachable")


def _integer(
    mapping: Mapping[str, Any],
    key: str,
    default: int | None,
    positive: bool = False,
) -> int | None:
    raw = _mapping_value(mapping, key, _UNSET)
    if raw is _UNSET:
        return default
    if isinstance(raw, bool):
        raise ConfigurationError(key, "expected a decimal integer")
    value = _trimmed(raw, key)
    if not value:
        return default
    if _INTEGER_PATTERN.fullmatch(value) is None:
        raise ConfigurationError(key, "expected a non-negative decimal integer")
    try:
        parsed = int(value, 10)
    except (TypeError, ValueError):
        raise ConfigurationError(key, "expected a non-negative decimal integer") from None
    if positive and parsed == 0:
        raise ConfigurationError(key, "expected a positive integer")
    return parsed


def _port(mapping: Mapping[str, Any]) -> int:
    value = _integer(mapping, "VOXCPM_PORT", 8090, positive=True)
    assert value is not None
    if value > 65535:
        raise ConfigurationError("VOXCPM_PORT", "expected a value from 1 through 65535")
    return value


def _optional_float(
    mapping: Mapping[str, Any],
    key: str,
    positive: bool = False,
) -> float | None:
    raw = _mapping_value(mapping, key, _UNSET)
    if raw is _UNSET:
        return None
    if isinstance(raw, bool):
        raise ConfigurationError(key, "expected a finite number")
    value = _trimmed(raw, key)
    if not value:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ConfigurationError(key, "expected a finite number") from None
    if not math.isfinite(parsed):
        raise ConfigurationError(key, "expected a finite number")
    if positive and parsed <= 0:
        raise ConfigurationError(key, "expected a positive number")
    if not positive and parsed < 0:
        raise ConfigurationError(key, "expected a non-negative number")
    return parsed


def parse_bool(value: Any, default: bool | None = None) -> bool | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = _trimmed(value, "value")
    if not text:
        return default
    normalized = text.casefold()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ConfigurationError("value", "expected a boolean")


def parse_gpu_devices(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        members: list[Any] = list(value)
    else:
        text = _trimmed(value, "VOXCPM_GPU_DEVICES")
        if not text:
            return None
        if text == "auto":
            return None
        members = text.split(",")
    if not members:
        raise ConfigurationError("VOXCPM_GPU_DEVICES", "expected auto or a comma-separated GPU list")
    parsed: list[int] = []
    for member in members:
        if isinstance(member, bool):
            raise ConfigurationError("VOXCPM_GPU_DEVICES", "GPU indices must be non-negative integers")
        member_text = _trimmed(member, "VOXCPM_GPU_DEVICES")
        if member_text is None or not member_text or _INTEGER_PATTERN.fullmatch(member_text) is None:
            raise ConfigurationError(
                "VOXCPM_GPU_DEVICES",
                "expected auto or comma-separated non-negative integer indices",
            )
        parsed.append(int(member_text, 10))
    if len(set(parsed)) != len(parsed):
        raise ConfigurationError("VOXCPM_GPU_DEVICES", "duplicate GPU indices are not allowed")
    return tuple(parsed)


def parse_workers(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ConfigurationError("VOXCPM_WORKERS", "expected auto or a positive integer")
    if isinstance(value, int):
        if value <= 0:
            raise ConfigurationError("VOXCPM_WORKERS", "expected auto or a positive integer")
        return value
    text = _trimmed(value, "VOXCPM_WORKERS")
    if not text or text == "auto":
        return None
    if _INTEGER_PATTERN.fullmatch(text) is None:
        raise ConfigurationError("VOXCPM_WORKERS", "expected auto or a positive integer")
    parsed = int(text, 10)
    if parsed <= 0:
        raise ConfigurationError("VOXCPM_WORKERS", "expected auto or a positive integer")
    return parsed


def _token_value(mapping: Mapping[str, Any]) -> str | None:
    raw = _mapping_value(mapping, "VOXCPM_API_TOKEN", _UNSET)
    if raw is _UNSET:
        return None
    value = _text(raw)
    if value is None or value == "":
        return None
    return value


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    model_path: str | None = None
    backend: Backend = Backend.PYTORCH_VOXCPM
    offline: bool = True
    load_denoiser: bool = False
    optimize: OptimizationMode = OptimizationMode.AUTO
    upstream_revision: str | None = None
    device: DeviceMode = DeviceMode.AUTO
    gpu_devices: tuple[int, ...] | None = None
    workers: int | None = None
    host: str = "127.0.0.1"
    port: int = 8090
    api_token: str | None = field(default=None, repr=False)
    require_auth: bool = True
    allow_unauthenticated_external: bool = False
    max_text_chars: int | None = None
    max_reference_bytes: int | None = None
    max_reference_seconds: float | None = None
    max_prompt_audio_seconds: float | None = None
    max_output_seconds: float | None = None
    max_queue_size: int | None = None
    queue_timeout_seconds: float | None = None
    request_timeout_seconds: float | None = None
    max_inference_seconds: float | None = None
    max_concurrent_requests: int | None = None
    stream_format: StreamFormat = StreamFormat.PCM_S16LE
    stream_sample_rate: int = 48000
    stream_channels: int = 1
    stream_ipc_max_chunks: int | None = None
    stream_ipc_max_bytes: int | None = None
    stream_backpressure_timeout_seconds: float | None = None
    output_dir: str | None = None
    tmp_dir: str | None = None
    cache_dir: str | None = None
    min_tmp_free_bytes: int | None = None
    log_level: LogLevel = LogLevel.INFO
    log_format: LogFormat = LogFormat.JSON
    log_file: str | None = None
    readiness_mode: ReadinessMode = ReadinessMode.DEGRADED
    qualification_strict: bool = False

    def __post_init__(self) -> None:
        enum_fields = (
            ("backend", Backend, "VOXCPM_BACKEND"),
            ("optimize", OptimizationMode, "VOXCPM_OPTIMIZE"),
            ("device", DeviceMode, "VOXCPM_DEVICE"),
            ("stream_format", StreamFormat, "VOXCPM_STREAM_FORMAT"),
            ("log_level", LogLevel, "VOXCPM_LOG_LEVEL"),
            ("log_format", LogFormat, "VOXCPM_LOG_FORMAT"),
            ("readiness_mode", ReadinessMode, "VOXCPM_READINESS_MODE"),
        )
        for name, enum_type, variable in enum_fields:
            try:
                normalized = enum_type(getattr(self, name))
            except (TypeError, ValueError):
                expected = ", ".join(item.value for item in enum_type)
                raise ConfigurationError(variable, f"expected one of: {expected}") from None
            object.__setattr__(self, name, normalized)
        if isinstance(self.gpu_devices, str):
            object.__setattr__(self, "gpu_devices", parse_gpu_devices(self.gpu_devices))
        elif self.gpu_devices is not None:
            object.__setattr__(self, "gpu_devices", parse_gpu_devices(self.gpu_devices))
        if isinstance(self.workers, str):
            object.__setattr__(self, "workers", parse_workers(self.workers))
        elif self.workers is not None and not isinstance(self.workers, int):
            raise ConfigurationError("VOXCPM_WORKERS", "expected auto or a positive integer")
        if isinstance(self.workers, bool) or (
            self.workers is not None and (not isinstance(self.workers, int) or self.workers <= 0)
        ):
            raise ConfigurationError("VOXCPM_WORKERS", "expected auto or a positive integer")
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not 1 <= self.port <= 65535:
            raise ConfigurationError("VOXCPM_PORT", "expected a value from 1 through 65535")
        boolean_fields = (
            "offline",
            "load_denoiser",
            "require_auth",
            "allow_unauthenticated_external",
            "qualification_strict",
        )
        for name in boolean_fields:
            if not isinstance(getattr(self, name), bool):
                raise ConfigurationError(f"VOXCPM_{name.upper()}", "expected a boolean")
        positive_integer_fields = (
            "max_text_chars",
            "max_reference_bytes",
            "max_queue_size",
            "max_concurrent_requests",
            "stream_sample_rate",
            "stream_channels",
            "stream_ipc_max_chunks",
            "stream_ipc_max_bytes",
        )
        for name in positive_integer_fields:
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ConfigurationError(f"VOXCPM_{name.upper()}", "expected a positive integer")
        positive_float_fields = (
            "max_reference_seconds",
            "max_prompt_audio_seconds",
            "max_output_seconds",
            "request_timeout_seconds",
            "max_inference_seconds",
            "stream_backpressure_timeout_seconds",
        )
        for name in positive_float_fields:
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ConfigurationError(f"VOXCPM_{name.upper()}", "expected a positive finite number")
        nonnegative_float_fields = ("queue_timeout_seconds",)
        for name in nonnegative_float_fields:
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ConfigurationError(f"VOXCPM_{name.upper()}", "expected a non-negative finite number")
        if self.min_tmp_free_bytes is not None and (
            isinstance(self.min_tmp_free_bytes, bool)
            or not isinstance(self.min_tmp_free_bytes, int)
            or self.min_tmp_free_bytes < 0
        ):
            raise ConfigurationError("VOXCPM_MIN_TMP_FREE_BYTES", "expected a non-negative integer")

    @property
    def requested_device(self) -> str:
        return self.device.value

    @property
    def gpu_devices_spec(self) -> str:
        if self.gpu_devices is None:
            return "auto"
        return ",".join(str(index) for index in self.gpu_devices)

    @property
    def workers_spec(self) -> str:
        if self.workers is None:
            return "auto"
        return str(self.workers)

    @property
    def selected_gpu_indices(self) -> tuple[int, ...] | None:
        return self.gpu_devices

    @property
    def worker_limit(self) -> int | None:
        return self.workers

    @property
    def api_token_configured(self) -> bool:
        return self.api_token is not None and self.api_token != ""

    def to_redacted_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "backend": self.backend.value,
            "offline": self.offline,
            "load_denoiser": self.load_denoiser,
            "optimize": self.optimize.value,
            "upstream_revision": self.upstream_revision,
            "device": self.device.value,
            "gpu_devices": self.gpu_devices_spec,
            "workers": self.workers_spec,
            "host": self.host,
            "port": self.port,
            "api_token_configured": self.api_token_configured,
            "require_auth": self.require_auth,
            "allow_unauthenticated_external": self.allow_unauthenticated_external,
            "max_text_chars": self.max_text_chars,
            "max_reference_bytes": self.max_reference_bytes,
            "max_reference_seconds": self.max_reference_seconds,
            "max_prompt_audio_seconds": self.max_prompt_audio_seconds,
            "max_output_seconds": self.max_output_seconds,
            "max_queue_size": self.max_queue_size,
            "queue_timeout_seconds": self.queue_timeout_seconds,
            "request_timeout_seconds": self.request_timeout_seconds,
            "max_inference_seconds": self.max_inference_seconds,
            "max_concurrent_requests": self.max_concurrent_requests,
            "stream_format": self.stream_format.value,
            "stream_sample_rate": self.stream_sample_rate,
            "stream_channels": self.stream_channels,
            "stream_ipc_max_chunks": self.stream_ipc_max_chunks,
            "stream_ipc_max_bytes": self.stream_ipc_max_bytes,
            "stream_backpressure_timeout_seconds": self.stream_backpressure_timeout_seconds,
            "output_dir": self.output_dir,
            "tmp_dir": self.tmp_dir,
            "cache_dir": self.cache_dir,
            "min_tmp_free_bytes": self.min_tmp_free_bytes,
            "log_level": self.log_level.value,
            "log_format": self.log_format.value,
            "log_file": self.log_file,
            "readiness_mode": self.readiness_mode.value,
            "qualification_strict": self.qualification_strict,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.to_redacted_dict()

    def to_diagnostic_dict(self) -> dict[str, Any]:
        data = self.to_redacted_dict()
        for name in ("model_path", "output_dir", "tmp_dir", "cache_dir", "log_file"):
            data.pop(name, None)
            data[f"{name}_configured"] = bool(getattr(self, name))
        data["requested_device"] = self.requested_device
        return data

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None = None) -> "RuntimeConfig":
        values: Mapping[str, Any] = {} if mapping is None else mapping
        backend = _enum_value(values, "VOXCPM_BACKEND", Backend.PYTORCH_VOXCPM.value, (Backend.PYTORCH_VOXCPM.value,))
        optimize = _enum_value(values, "VOXCPM_OPTIMIZE", OptimizationMode.AUTO.value, (OptimizationMode.AUTO.value,))
        device = _enum_value(values, "VOXCPM_DEVICE", DeviceMode.AUTO.value, tuple(item.value for item in DeviceMode))
        stream_format = _enum_value(
            values,
            "VOXCPM_STREAM_FORMAT",
            StreamFormat.PCM_S16LE.value,
            (StreamFormat.PCM_S16LE.value,),
        )
        log_level = _enum_value(
            values,
            "VOXCPM_LOG_LEVEL",
            LogLevel.INFO.value,
            tuple(item.value for item in LogLevel),
            case_insensitive=True,
        )
        log_format = _enum_value(values, "VOXCPM_LOG_FORMAT", LogFormat.JSON.value, (LogFormat.JSON.value,))
        readiness = _enum_value(
            values,
            "VOXCPM_READINESS_MODE",
            ReadinessMode.DEGRADED.value,
            tuple(item.value for item in ReadinessMode),
        )
        return cls(
            model_path=_lexical(values, "VOXCPM_MODEL_PATH", None, optional=True),
            backend=Backend(backend),
            offline=_boolean(values, "VOXCPM_OFFLINE", True),
            load_denoiser=_boolean(values, "VOXCPM_LOAD_DENOISER", False),
            optimize=OptimizationMode(optimize),
            upstream_revision=_lexical(values, "VOXCPM_UPSTREAM_REVISION", None, optional=True),
            device=DeviceMode(device),
            gpu_devices=parse_gpu_devices(values.get("VOXCPM_GPU_DEVICES")),
            workers=parse_workers(values.get("VOXCPM_WORKERS")),
            host=_lexical(values, "VOXCPM_HOST", "127.0.0.1") or "127.0.0.1",
            port=_port(values),
            api_token=_token_value(values),
            require_auth=_boolean(values, "VOXCPM_REQUIRE_AUTH", True),
            allow_unauthenticated_external=_boolean(values, "VOXCPM_ALLOW_UNAUTHENTICATED_EXTERNAL", False),
            max_text_chars=_integer(values, "VOXCPM_MAX_TEXT_CHARS", None, positive=True),
            max_reference_bytes=_integer(values, "VOXCPM_MAX_REFERENCE_BYTES", None, positive=True),
            max_reference_seconds=_optional_float(values, "VOXCPM_MAX_REFERENCE_SECONDS", positive=True),
            max_prompt_audio_seconds=_optional_float(values, "VOXCPM_MAX_PROMPT_AUDIO_SECONDS", positive=True),
            max_output_seconds=_optional_float(values, "VOXCPM_MAX_OUTPUT_SECONDS", positive=True),
            max_queue_size=_integer(values, "VOXCPM_MAX_QUEUE_SIZE", None, positive=True),
            queue_timeout_seconds=_optional_float(values, "VOXCPM_QUEUE_TIMEOUT_SECONDS"),
            request_timeout_seconds=_optional_float(values, "VOXCPM_REQUEST_TIMEOUT_SECONDS", positive=True),
            max_inference_seconds=_optional_float(values, "VOXCPM_MAX_INFERENCE_SECONDS", positive=True),
            max_concurrent_requests=_integer(values, "VOXCPM_MAX_CONCURRENT_REQUESTS", None, positive=True),
            stream_format=StreamFormat(stream_format),
            stream_sample_rate=_integer(values, "VOXCPM_STREAM_SAMPLE_RATE", 48000, positive=True) or 48000,
            stream_channels=_integer(values, "VOXCPM_STREAM_CHANNELS", 1, positive=True) or 1,
            stream_ipc_max_chunks=_integer(values, "VOXCPM_STREAM_IPC_MAX_CHUNKS", None, positive=True),
            stream_ipc_max_bytes=_integer(values, "VOXCPM_STREAM_IPC_MAX_BYTES", None, positive=True),
            stream_backpressure_timeout_seconds=_optional_float(
                values,
                "VOXCPM_STREAM_BACKPRESSURE_TIMEOUT_SECONDS",
                positive=True,
            ),
            output_dir=_lexical(values, "VOXCPM_OUTPUT_DIR", None, optional=True),
            tmp_dir=_lexical(values, "VOXCPM_TMP_DIR", None, optional=True),
            cache_dir=_lexical(values, "VOXCPM_CACHE_DIR", None, optional=True),
            min_tmp_free_bytes=_integer(values, "VOXCPM_MIN_TMP_FREE_BYTES", None),
            log_level=LogLevel(log_level),
            log_format=LogFormat(log_format),
            log_file=_lexical(values, "VOXCPM_LOG_FILE", None, optional=True),
            readiness_mode=ReadinessMode(readiness),
            qualification_strict=_boolean(values, "VOXCPM_QUALIFICATION_STRICT", False),
        )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "RuntimeConfig":
        values = os.environ if environ is None else environ
        return cls.from_mapping(values)


def format_gpu_devices(indices: tuple[int, ...] | None) -> str:
    if indices is None:
        return "auto"
    return ",".join(str(index) for index in indices)


def format_workers(count: int | None) -> str:
    if count is None:
        return "auto"
    return str(count)
