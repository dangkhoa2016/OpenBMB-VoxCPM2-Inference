from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

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
from voxcpm_runtime.config import OptimizationMode
from voxcpm_runtime.device import ExecutionPlan
from voxcpm_runtime.errors import (
    BackendError,
    BackendExecutionError,
    BackendRequestError,
    BackendStateError,
    BackendUnsupportedError,
    ErrorDetail,
    normalize_backend_error,
)
from voxcpm_runtime.model_resolver import ResolvedModel

BACKEND_NAME: Final = "pytorch-voxcpm"
BACKEND_IMPLEMENTATION: Final = "pinned-upstream-local"
M4_CAPABILITIES: Final[tuple[str, ...]] = ("synthesize",)
_UPSTREAM_MODULE: Final = "voxcpm"
_UPSTREAM_SYMBOL: Final = "VoxCPM"


class _State(Enum):
    CREATED = "created"
    LOADED = "loaded"
    CLOSED = "closed"


def resolve_upstream_device(execution_plan: ExecutionPlan) -> str:
    """Map a project execution plan onto one explicit upstream device."""

    if not isinstance(execution_plan, ExecutionPlan):
        raise TypeError("execution_plan must be an ExecutionPlan")
    if execution_plan.device == "cpu":
        return "cpu"
    if execution_plan.device != "cuda":
        raise BackendUnsupportedError(
            "The execution device is not qualified by the real backend.",
            code="device_not_qualified",
            details={"device": execution_plan.device, "milestone": "M5"},
        )
    if (
        len(execution_plan.selected_gpu_indices) != 1
        or execution_plan.worker_count != 1
        or len(execution_plan.worker_gpu_indices) != 1
    ):
        raise BackendUnsupportedError(
            "Multi-GPU execution is not qualified by the real backend yet.",
            code="multi_gpu_not_qualified",
            details={"device": "cuda", "milestone": "M5"},
        )
    selected = execution_plan.selected_gpu_indices[0]
    if execution_plan.worker_gpu_indices[0] != selected:
        raise BackendUnsupportedError(
            "The CUDA worker mapping is not qualified by the real backend.",
            code="gpu_worker_mapping_not_qualified",
            details={"device": "cuda", "milestone": "M5"},
        )
    return f"cuda:{selected}"


def resolve_upstream_optimize(
    execution_plan: ExecutionPlan,
    optimization_mode: OptimizationMode,
) -> bool:
    """Map project AUTO onto the non-compiled M4/M5 baseline."""

    if not isinstance(execution_plan, ExecutionPlan):
        raise TypeError("execution_plan must be an ExecutionPlan")
    if not isinstance(optimization_mode, OptimizationMode):
        raise TypeError("optimization_mode must be an OptimizationMode")
    if execution_plan.device in {"cpu", "cuda"}:
        return False
    raise BackendUnsupportedError(
        "Upstream optimization semantics are not qualified for this device.",
        code="optimization_semantics_not_qualified",
        details={
            "device": execution_plan.device,
            "milestone": "M5",
            "optimization_mode": optimization_mode.value,
        },
    )


@dataclass(frozen=True, slots=True)
class ModelDeviceReport:
    """Device types observed on real upstream tensors after load.

    Reading ``tensor.device.type`` never allocates on the target device, so
    this inspection is safe to run as no-CUDA evidence.
    """

    parameter_device_types: tuple[str, ...]
    buffer_device_types: tuple[str, ...]
    parameter_count: int
    buffer_count: int
    expected_device: str | None = None

    @property
    def cpu_only(self) -> bool:
        return (
            bool(self.parameter_device_types)
            and set(self.parameter_device_types) == {"cpu"}
            and set(self.buffer_device_types) <= {"cpu"}
        )

    @property
    def all_on_expected_device(self) -> bool | None:
        if self.expected_device is None:
            return None
        observed = set(self.parameter_device_types) | set(self.buffer_device_types)
        return bool(self.parameter_device_types) and observed <= {self.expected_device}

    def to_dict(self) -> dict[str, Any]:
        return {
            "all_on_expected_device": self.all_on_expected_device,
            "buffer_count": self.buffer_count,
            "buffer_device_types": list(self.buffer_device_types),
            "cpu_only": self.cpu_only,
            "expected_device": self.expected_device,
            "parameter_count": self.parameter_count,
            "parameter_device_types": list(self.parameter_device_types),
        }


def _tensor_device_type(tensor: Any) -> str | None:
    device = getattr(tensor, "device", None)
    if device is None:
        return None
    device_type = getattr(device, "type", None)
    if isinstance(device_type, str) and device_type:
        return device_type
    text = str(device)
    return text.split(":", 1)[0] if text else None


def _collect_device_types(
    accessor: Any,
) -> tuple[set[str], int]:
    device_types: set[str] = set()
    count = 0
    if not callable(accessor):
        return device_types, count
    try:
        iterator = accessor()
    except Exception as error:
        raise normalize_backend_error("load", error) from None
    try:
        for item in iterator:
            count += 1
            tensor = item[1] if isinstance(item, tuple) else item
            device_type = _tensor_device_type(tensor)
            if device_type is not None:
                device_types.add(device_type)
            if count > 1_000_000:
                raise BackendExecutionError(
                    "Upstream model exposes an implausible tensor count.",
                    code="model_tensor_inventory_too_large",
                )
    except BackendError:
        raise
    except Exception as error:
        raise normalize_backend_error("load", error) from None
    return device_types, count


def inspect_model_devices(model: Any, *, expected_device: str | None = None) -> ModelDeviceReport:
    """Collect parameter and buffer device types from a loaded upstream model."""

    tts_model = getattr(model, "tts_model", None)
    if tts_model is None:
        raise BackendExecutionError(
            "The upstream model does not expose the expected tts_model attribute.",
            code="upstream_contract_violation",
            details={"attribute": "tts_model"},
        )
    parameter_types, parameter_count = _collect_device_types(
        getattr(tts_model, "named_parameters", None)
    )
    buffer_types, buffer_count = _collect_device_types(
        getattr(tts_model, "named_buffers", None)
    )
    return ModelDeviceReport(
        parameter_device_types=tuple(sorted(parameter_types)),
        buffer_device_types=tuple(sorted(buffer_types)),
        parameter_count=parameter_count,
        buffer_count=buffer_count,
        expected_device=expected_device,
    )


def _import_upstream() -> Any:
    """Import the pinned upstream wrapper lazily, inside the real load path."""

    import importlib

    try:
        module = importlib.import_module("voxcpm")
    except Exception as error:
        raise normalize_backend_error("load", error) from None
    symbol = getattr(module, _UPSTREAM_SYMBOL, None)
    if symbol is None:
        raise BackendExecutionError(
            "The pinned upstream package does not expose the expected VoxCPM class.",
            code="upstream_contract_violation",
            details={"module": _UPSTREAM_MODULE, "symbol": _UPSTREAM_SYMBOL},
        )
    return symbol


def _waveform_to_samples(waveform: Any) -> tuple[float, ...]:
    """Convert an upstream waveform into a validated 1-D tuple of floats."""

    if waveform is None:
        raise BackendExecutionError(
            "The upstream model returned no waveform.",
            code="empty_waveform",
        )
    ndim = getattr(waveform, "ndim", None)
    if isinstance(ndim, int) and not isinstance(ndim, bool) and ndim != 1:
        raise BackendExecutionError(
            "The upstream model returned a waveform that is not one-dimensional.",
            code="invalid_waveform_shape",
            details={"dimensions": ndim},
        )
    to_list = getattr(waveform, "tolist", None)
    try:
        raw = to_list() if callable(to_list) else list(waveform)
    except BackendError:
        raise
    except Exception as error:
        raise normalize_backend_error("synthesize", error) from None
    if not isinstance(raw, (list, tuple)):
        raise BackendExecutionError(
            "The upstream model returned a waveform that could not be read as samples.",
            code="invalid_waveform_type",
            details={"waveform_type": type(waveform).__name__},
        )
    if not raw:
        raise BackendExecutionError(
            "The upstream model returned an empty waveform.",
            code="empty_waveform",
        )
    samples: list[float] = []
    for value in raw:
        if isinstance(value, (list, tuple)):
            raise BackendExecutionError(
                "The upstream model returned a waveform that is not one-dimensional.",
                code="invalid_waveform_shape",
                details={"nested": True},
            )
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise BackendExecutionError(
                "The upstream model returned non-numeric audio samples.",
                code="invalid_waveform_sample",
                details={"sample_type": type(value).__name__},
            ) from None
        if not math.isfinite(number):
            raise BackendExecutionError(
                "The upstream model returned non-finite audio samples.",
                code="non_finite_waveform",
            )
        samples.append(number)
    return tuple(samples)


def _read_sample_rate_hz(model: Any) -> int:
    """Read the real output sample rate from the loaded upstream model."""

    tts_model = getattr(model, "tts_model", None)
    if tts_model is None:
        raise BackendExecutionError(
            "The upstream model does not expose the expected tts_model attribute.",
            code="upstream_contract_violation",
            details={"attribute": "tts_model"},
        )
    raw = getattr(tts_model, "sample_rate", None)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise BackendExecutionError(
            "The upstream model did not report a usable sample rate.",
            code="upstream_sample_rate_unavailable",
            details={"sample_rate_type": type(raw).__name__},
        )
    value = float(raw)
    if not math.isfinite(value) or value <= 0 or value != int(value):
        raise BackendExecutionError(
            "The upstream model did not report a positive integer sample rate.",
            code="upstream_sample_rate_invalid",
            details={"sample_rate_is_positive_integer": False},
        )
    return int(value)


class PytorchVoxCPMBackend:
    """Real local VoxCPM2 backend for qualified CPU and single-GPU TTS.

    The upstream package is imported only inside :meth:`load`, so ordinary CI
    never imports ``torch``, ``numpy`` or ``voxcpm`` through this module.
    """

    def __init__(
        self,
        *,
        resolved_model: ResolvedModel,
        execution_plan: ExecutionPlan,
        load_denoiser: bool = False,
        local_files_only: bool = True,
        optimize: bool = False,
        upstream_name: str = _UPSTREAM_MODULE,
    ) -> None:
        if not isinstance(resolved_model, ResolvedModel):
            raise TypeError("resolved_model must be a ResolvedModel")
        if not isinstance(execution_plan, ExecutionPlan):
            raise TypeError("execution_plan must be an ExecutionPlan")
        if not isinstance(load_denoiser, bool):
            raise TypeError("load_denoiser must be a boolean")
        if not isinstance(local_files_only, bool):
            raise TypeError("local_files_only must be a boolean")
        if not isinstance(optimize, bool):
            raise TypeError("optimize must be a boolean")
        if not isinstance(upstream_name, str) or not upstream_name:
            raise TypeError("upstream_name must be a non-empty string")
        self._resolved_model = resolved_model
        self._execution_plan = execution_plan
        self._upstream_device = resolve_upstream_device(execution_plan)
        self._load_denoiser = load_denoiser
        self._local_files_only = local_files_only
        self._optimize = optimize
        self._upstream_name = upstream_name
        self._model: Any = None
        self._state = _State.CREATED

    @property
    def execution_plan(self) -> ExecutionPlan:
        return self._execution_plan

    @property
    def resolved_model(self) -> ResolvedModel:
        return self._resolved_model

    @property
    def effective_optimize(self) -> bool:
        return self._optimize

    @property
    def upstream_device(self) -> str:
        return self._upstream_device

    @property
    def load_denoiser(self) -> bool:
        return self._load_denoiser

    @property
    def local_files_only(self) -> bool:
        return self._local_files_only

    @property
    def loaded_model(self) -> Any:
        """Return the loaded upstream model, or None before ``load()``."""

        return self._model

    def load(self) -> BackendInfo:
        if self._state is _State.CLOSED:
            raise BackendStateError(
                "Backend cannot reopen after it has been closed.",
                code="backend_closed",
                details={"state": self._state.value},
            )
        if self._state is _State.LOADED:
            return self._info()
        self._model = self._load_model()
        self._state = _State.LOADED
        return self._info()

    def synthesize(self, request: SpeechRequest) -> AudioResult:
        self._ensure_ready()
        if not isinstance(request, SpeechRequest):
            raise BackendRequestError(
                "Backend request is invalid.",
                details={"field": "request"},
            )
        return self._call_normalized("synthesize", lambda: self._synthesize(request))

    def design(self, request: VoiceDesignRequest) -> AudioResult:
        self._reject("design", request, (VoiceDesignRequest,))
        raise BackendUnsupportedError(
            "Voice design is not qualified by this backend yet.",
            code="design_not_qualified",
            details={"capability": "design", "milestone": "M6"},
        )

    def clone(self, request: CloneRequest) -> AudioResult:
        self._reject("clone", request, (CloneRequest,))
        raise BackendUnsupportedError(
            "Voice cloning is not qualified by this backend yet.",
            code="clone_not_qualified",
            details={"capability": "clone", "milestone": "M6"},
        )

    def continue_audio(self, request: ContinuationRequest) -> AudioResult:
        self._reject("continue_audio", request, (ContinuationRequest,))
        raise BackendUnsupportedError(
            "Audio continuation is not qualified by this backend yet.",
            code="continue_audio_not_qualified",
            details={"capability": "continue_audio", "milestone": "M6"},
        )

    def stream(self, request: StreamRequest) -> Iterator[AudioChunk]:
        self._reject("stream", request, (StreamRequest,))
        raise BackendUnsupportedError(
            "Streaming is not qualified by this backend yet.",
            code="stream_not_qualified",
            details={"capability": "stream", "milestone": "M7"},
        )

    def close(self) -> None:
        if self._state is _State.CLOSED:
            return None
        self._state = _State.CLOSED
        self._model = None
        return None

    def _reject(self, operation: str, request: object, expected: tuple[type, ...]) -> None:
        self._ensure_ready()
        if not isinstance(request, expected):
            raise BackendRequestError(
                "Backend request is invalid.",
                details={"field": "request"},
            )
        del operation

    def _ensure_ready(self) -> None:
        if self._state is _State.CREATED:
            raise BackendStateError(
                "Backend must be loaded before generation.",
                code="backend_not_loaded",
                details={"state": self._state.value},
            )
        if self._state is _State.CLOSED:
            raise BackendStateError(
                "Backend cannot generate after it has been closed.",
                code="backend_closed",
                details={"state": self._state.value},
            )

    def _call_normalized(self, operation: str, action: Any) -> Any:
        try:
            return action()
        except BackendError as error:
            raise error from None
        except Exception as error:
            raise normalize_backend_error(operation, error) from None

    def _load_model(self) -> Any:
        upstream_class = _import_upstream()
        try:
            return upstream_class.from_pretrained(
                str(self._resolved_model.path),
                load_denoiser=self._load_denoiser,
                local_files_only=self._local_files_only,
                optimize=self._optimize,
                device=self._upstream_device,
            )
        except BackendError:
            raise
        except Exception as error:
            raise normalize_backend_error("load", error) from None

    def _synthesize(self, request: SpeechRequest) -> AudioResult:
        model = self._model
        if model is None:
            raise BackendStateError(
                "Backend must be loaded before generation.",
                code="backend_not_loaded",
                details={"state": self._state.value},
            )
        waveform = model.generate(text=request.text)
        samples = _waveform_to_samples(waveform)
        sample_rate_hz = _read_sample_rate_hz(model)
        return AudioResult(
            samples=samples,
            sample_rate_hz=sample_rate_hz,
            channels=1,
            metadata=self._result_metadata(sample_rate_hz),
        )

    def _result_metadata(self, sample_rate_hz: int) -> tuple[tuple[str, ErrorDetail], ...]:
        revision = self._resolved_model.revision
        return (
            ("backend", BACKEND_NAME),
            ("model_id", self._resolved_model.model_id),
            ("model_revision", revision if revision is not None else "unrecorded"),
            ("model_source_kind", self._resolved_model.source_kind),
            ("sample_rate_source", "model.tts_model.sample_rate"),
            ("sample_rate_hz", sample_rate_hz),
            ("upstream_device", self._upstream_device),
            ("upstream_optimize", self._optimize),
        )

    def _info(self) -> BackendInfo:
        revision = self._resolved_model.revision
        return BackendInfo(
            backend=BACKEND_NAME,
            implementation=BACKEND_IMPLEMENTATION,
            loaded=True,
            device=self._execution_plan.device,
            model_id=self._resolved_model.model_id,
            capabilities=M4_CAPABILITIES,
            metadata=(
                ("load_denoiser", self._load_denoiser),
                ("local_files_only", self._local_files_only),
                ("model_revision", revision if revision is not None else "unrecorded"),
                ("model_source_kind", self._resolved_model.source_kind),
                ("upstream_device", self._upstream_device),
                ("upstream_optimize", self._optimize),
            ),
        )


__all__: Final[tuple[str, ...]] = (
    "BACKEND_IMPLEMENTATION",
    "BACKEND_NAME",
    "M4_CAPABILITIES",
    "ModelDeviceReport",
    "PytorchVoxCPMBackend",
    "inspect_model_devices",
    "resolve_upstream_device",
    "resolve_upstream_optimize",
)
