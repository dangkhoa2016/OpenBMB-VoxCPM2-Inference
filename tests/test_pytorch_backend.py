from __future__ import annotations

import inspect
import os
import sys
import types
from pathlib import Path
from typing import Any, get_type_hints

import pytest

from voxcpm_runtime.backend import InferenceBackend
from voxcpm_runtime.backend_types import (
    AudioChunk,
    AudioReference,
    AudioResult,
    CloneRequest,
    ContinuationRequest,
    SpeechRequest,
    StreamRequest,
    VoiceDesignRequest,
)
from voxcpm_runtime.config import OptimizationMode
from voxcpm_runtime.device import ExecutionPlan
from voxcpm_runtime.errors import (
    BackendExecutionError,
    BackendRequestError,
    BackendStateError,
    BackendUnsupportedError,
)
from voxcpm_runtime.model_resolver import ResolvedModel
from voxcpm_runtime.pytorch_backend import (
    REAL_BACKEND_CAPABILITIES,
    PytorchVoxCPMBackend,
    build_design_text,
    inspect_model_devices,
    resolve_upstream_device,
    resolve_upstream_optimize,
    validate_local_reference_audio,
)

_UPSTREAM_MODULE = "voxcpm"
_REAL_SAMPLE_RATE_HZ = 48_000
_PRIVATE_DETAIL = "m4-private-upstream-detail-placeholder"


class _StubTensor:
    def __init__(self, device_type: str = "cpu") -> None:
        self.device = types.SimpleNamespace(type=device_type)


class _StubTtsModel:
    def __init__(self, sample_rate: Any = _REAL_SAMPLE_RATE_HZ) -> None:
        self.sample_rate = sample_rate
        self._parameter_device = "cpu"
        self._buffer_device = "cpu"

    def named_parameters(self) -> list[tuple[str, _StubTensor]]:
        return [(f"weight_{index}", _StubTensor(self._parameter_device)) for index in range(3)]

    def named_buffers(self) -> list[tuple[str, _StubTensor]]:
        return [("cache", _StubTensor(self._buffer_device))]


class _StubStream:
    def __init__(
        self,
        chunks: tuple[Any, ...],
        *,
        error_at: int | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self._chunks = chunks
        self._error_at = error_at
        self._close_error = close_error
        self._index = 0
        self.closed = False
        self.next_calls = 0

    def __iter__(self) -> "_StubStream":
        return self

    def __next__(self) -> Any:
        self.next_calls += 1
        if self._error_at is not None and self._index == self._error_at:
            raise RuntimeError("private-stream-error")
        if self._index >= len(self._chunks):
            raise StopIteration
        value = self._chunks[self._index]
        self._index += 1
        return value

    def close(self) -> None:
        self.closed = True
        if self._close_error is not None:
            raise self._close_error


class _StubModel:
    def __init__(
        self,
        waveform: Any = (0.0, 0.5, -0.5, 1.0, -1.0),
        tts_model: Any = None,
        generate_error: BaseException | None = None,
        *,
        stream_chunks: tuple[Any, ...] = ((0.0, 0.1), (0.2, 0.3), (0.4,)),
        stream_error_at: int | None = None,
        stream_close_error: BaseException | None = None,
    ) -> None:
        self.tts_model = _StubTtsModel() if tts_model is None else tts_model
        self.waveform = waveform
        self.generate_error = generate_error
        self.calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []
        self.streams: list[_StubStream] = []
        self.stream_chunks = stream_chunks
        self.stream_error_at = stream_error_at
        self.stream_close_error = stream_close_error

    def generate(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        if self.generate_error is not None:
            raise self.generate_error
        return self.waveform

    def generate_streaming(self, **kwargs: Any) -> _StubStream:
        self.stream_calls.append(dict(kwargs))
        stream = _StubStream(
            self.stream_chunks,
            error_at=self.stream_error_at,
            close_error=self.stream_close_error,
        )
        self.streams.append(stream)
        return stream


class _StubUpstream:
    """Stand-in for ``voxcpm.VoxCPM`` recording ``from_pretrained`` kwargs."""

    instances: list["_StubUpstream"] = []

    def __init__(
        self,
        model: Any = None,
        error: BaseException | None = None,
    ) -> None:
        self.model = _StubModel() if model is None else model
        self.error = error
        self.pretrained_calls: list[dict[str, Any]] = []
        _StubUpstream.instances.append(self)

    def from_pretrained(self, hf_model_id: str, **kwargs: Any) -> Any:
        self.pretrained_calls.append({"hf_model_id": hf_model_id, **kwargs})
        if self.error is not None:
            raise self.error
        return self.model


@pytest.fixture(autouse=True)
def _reset_stub_registry() -> None:
    _StubUpstream.instances.clear()


@pytest.fixture
def upstream_stub(monkeypatch: pytest.MonkeyPatch) -> _StubUpstream:
    stub = _StubUpstream()
    module = types.ModuleType(_UPSTREAM_MODULE)
    module.VoxCPM = stub  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _UPSTREAM_MODULE, module)
    return stub


def _resolved(tmp_path: Path, model_factory: Any) -> ResolvedModel:
    root = model_factory(tmp_path / "model")
    from voxcpm_runtime.config import RuntimeConfig
    from voxcpm_runtime.model_resolver import ModelResolver

    return ModelResolver().resolve(
        RuntimeConfig(model_path=str(root), offline=True),
    )


def _cpu_plan() -> ExecutionPlan:
    return ExecutionPlan(effective_device="cpu", selected_gpu_indices=(), worker_count=1)


def _cuda_plan(index: int = 0) -> ExecutionPlan:
    return ExecutionPlan(
        effective_device="cuda",
        selected_gpu_indices=(index,),
        worker_count=1,
        worker_gpu_indices=(index,),
    )


def _backend(
    tmp_path: Path,
    model_factory: Any,
    *,
    optimize: bool = False,
    load_denoiser: bool = False,
    local_files_only: bool = True,
    execution_plan: ExecutionPlan | None = None,
) -> PytorchVoxCPMBackend:
    return PytorchVoxCPMBackend(
        resolved_model=_resolved(tmp_path, model_factory),
        execution_plan=_cpu_plan() if execution_plan is None else execution_plan,
        load_denoiser=load_denoiser,
        local_files_only=local_files_only,
        optimize=optimize,
    )


# --- protocol and construction ------------------------------------------


def test_real_backend_satisfies_protocol_surface(
    tmp_path: Path, model_factory: Any
) -> None:
    assert isinstance(_backend(tmp_path, model_factory), InferenceBackend)


@pytest.mark.parametrize(
    "method_name",
    [
        "load",
        "synthesize",
        "design",
        "clone",
        "continue_audio",
        "stream",
        "close",
    ],
)
def test_operation_signatures_are_stable(method_name: str) -> None:
    protocol_method = getattr(InferenceBackend, method_name)
    real_method = getattr(PytorchVoxCPMBackend, method_name)
    assert list(inspect.signature(protocol_method).parameters) == list(
        inspect.signature(real_method).parameters
    )
    assert get_type_hints(protocol_method) == get_type_hints(real_method)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"resolved_model": object()},
        {"execution_plan": object()},
        {"load_denoiser": "yes"},
        {"local_files_only": 1},
        {"optimize": "no"},
        {"upstream_name": ""},
    ],
)
def test_constructor_rejects_invalid_inputs(
    tmp_path: Path,
    model_factory: Any,
    kwargs: dict[str, Any],
) -> None:
    base: dict[str, Any] = {
        "resolved_model": _resolved(tmp_path, model_factory),
        "execution_plan": _cpu_plan(),
        "load_denoiser": False,
        "local_files_only": True,
        "optimize": False,
        "upstream_name": "voxcpm",
    }
    base.update(kwargs)
    with pytest.raises(TypeError):
        PytorchVoxCPMBackend(**base)


# --- CPU optimization mapping (F3) -------------------------------------


def test_cpu_auto_optimization_maps_to_upstream_false() -> None:
    assert resolve_upstream_optimize(_cpu_plan(), OptimizationMode.AUTO) is False


def test_cuda_auto_optimization_maps_to_upstream_false() -> None:
    assert resolve_upstream_optimize(_cuda_plan(), OptimizationMode.AUTO) is False


def test_single_cuda_plan_maps_to_indexed_upstream_device() -> None:
    assert resolve_upstream_device(_cuda_plan(0)) == "cuda:0"
    assert resolve_upstream_device(_cuda_plan(1)) == "cuda:1"


def test_multi_gpu_plan_is_refused_by_real_backend() -> None:
    plan = ExecutionPlan(
        effective_device="cuda",
        selected_gpu_indices=(0, 1),
        worker_count=2,
        worker_gpu_indices=(0, 1),
    )
    with pytest.raises(BackendUnsupportedError) as caught:
        resolve_upstream_device(plan)
    assert caught.value.code == "multi_gpu_not_qualified"
    assert caught.value.details["milestone"] == "M5"


def test_resolve_upstream_optimize_validates_types() -> None:
    with pytest.raises(TypeError):
        resolve_upstream_optimize(object(), OptimizationMode.AUTO)
    with pytest.raises(TypeError):
        resolve_upstream_optimize(_cpu_plan(), "auto")


# --- load mapping (F4) --------------------------------------------------


def test_load_passes_canonical_m4_arguments(
    tmp_path: Path, model_factory: Any, upstream_stub: _StubUpstream
) -> None:
    backend = _backend(tmp_path, model_factory, optimize=False, load_denoiser=False)
    backend.load()
    assert upstream_stub.pretrained_calls == [
        {
            "device": "cpu",
            "hf_model_id": str(backend.resolved_model.path),
            "load_denoiser": False,
            "local_files_only": True,
            "optimize": False,
        }
    ]


def test_load_passes_indexed_cuda_device(
    tmp_path: Path, model_factory: Any, upstream_stub: _StubUpstream
) -> None:
    backend = _backend(
        tmp_path,
        model_factory,
        execution_plan=_cuda_plan(0),
        optimize=False,
        load_denoiser=False,
    )
    backend.load()
    assert upstream_stub.pretrained_calls == [
        {
            "device": "cuda:0",
            "hf_model_id": str(backend.resolved_model.path),
            "load_denoiser": False,
            "local_files_only": True,
            "optimize": False,
        }
    ]
    assert backend.upstream_device == "cuda:0"


def test_load_uses_resolved_path_without_hard_coding(
    tmp_path: Path, model_factory: Any, upstream_stub: _StubUpstream
) -> None:
    backend = _backend(tmp_path, model_factory)
    backend.load()
    call = upstream_stub.pretrained_calls[0]
    assert call["hf_model_id"] == str(backend.resolved_model.path)
    assert call["hf_model_id"] == str((tmp_path / "model").resolve())
    assert "/kaggle/input" not in call["hf_model_id"]


def test_upstream_import_is_lazy_until_load(
    tmp_path: Path, model_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delitem(sys.modules, _UPSTREAM_MODULE, raising=False)
    backend = _backend(tmp_path, model_factory)
    with pytest.raises(BackendExecutionError) as caught:
        backend.load()
    assert caught.value.code == "backend_execution_failed"
    assert caught.value.details == {
        "exception_type": "ModuleNotFoundError",
        "operation": "load",
    }
    assert _UPSTREAM_MODULE not in sys.modules


# --- lifecycle (F6) -----------------------------------------------------


def test_load_is_idempotent(
    tmp_path: Path, model_factory: Any, upstream_stub: _StubUpstream
) -> None:
    backend = _backend(tmp_path, model_factory)
    first = backend.load()
    second = backend.load()
    assert first == second
    assert len(upstream_stub.pretrained_calls) == 1


def test_close_is_idempotent(
    tmp_path: Path, model_factory: Any, upstream_stub: _StubUpstream
) -> None:
    backend = _backend(tmp_path, model_factory)
    backend.load()
    assert backend.close() is None
    assert backend.close() is None
    assert backend.loaded_model is None


def test_generation_before_load_is_rejected(
    tmp_path: Path, model_factory: Any
) -> None:
    backend = _backend(tmp_path, model_factory)
    with pytest.raises(BackendStateError) as caught:
        backend.synthesize(SpeechRequest(text="hello"))
    assert caught.value.code == "backend_not_loaded"


def test_generation_after_close_is_rejected(
    tmp_path: Path, model_factory: Any, upstream_stub: _StubUpstream
) -> None:
    backend = _backend(tmp_path, model_factory)
    backend.load()
    backend.close()
    with pytest.raises(BackendStateError) as caught:
        backend.synthesize(SpeechRequest(text="hello"))
    assert caught.value.code == "backend_closed"


def test_reopen_after_close_is_rejected(
    tmp_path: Path, model_factory: Any, upstream_stub: _StubUpstream
) -> None:
    backend = _backend(tmp_path, model_factory)
    backend.load()
    backend.close()
    with pytest.raises(BackendStateError) as caught:
        backend.load()
    assert caught.value.code == "backend_closed"


# --- BackendInfo / capabilities (F5) ------------------------------------


def test_backend_info_advertises_qualified_operations(
    tmp_path: Path, model_factory: Any, upstream_stub: _StubUpstream
) -> None:
    info = _backend(tmp_path, model_factory).load()
    assert info.capabilities == (
        "clone",
        "continue_audio",
        "design",
        "stream",
        "synthesize",
    )
    assert info.capabilities == REAL_BACKEND_CAPABILITIES
    assert info.loaded is True
    assert info.device == "cpu"
    assert info.backend == "pytorch-voxcpm"


@pytest.mark.parametrize(
    ("operation", "request_factory"),
    [
        ("design", lambda: object()),
        ("clone", lambda: object()),
        ("continue_audio", lambda: object()),
    ],
)
def test_qualified_capabilities_reject_wrong_request_type(
    tmp_path: Path,
    model_factory: Any,
    upstream_stub: _StubUpstream,
    operation: str,
    request_factory: Any,
) -> None:
    backend = _backend(tmp_path, model_factory)
    backend.load()
    with pytest.raises(BackendRequestError):
        getattr(backend, operation)(request_factory())


# --- synthesis mapping (Phase G) ----------------------------------------


def test_synthesize_maps_speech_request_to_upstream_generate(
    tmp_path: Path, model_factory: Any, upstream_stub: _StubUpstream
) -> None:
    from voxcpm_runtime.backend_types import SpeechRequest

    backend = _backend(tmp_path, model_factory)
    backend.load()
    backend.synthesize(SpeechRequest(text="Xin chào từ VoxCPM2."))
    assert upstream_stub.model.calls == [{"text": "Xin chào từ VoxCPM2."}]


def test_synthesize_reads_real_sample_rate_from_model(
    tmp_path: Path, model_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _StubModel(tts_model=_StubTtsModel(sample_rate=24_000))
    upstream = _StubUpstream(model=model)
    module = types.ModuleType(_UPSTREAM_MODULE)
    module.VoxCPM = upstream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _UPSTREAM_MODULE, module)
    backend = _backend(tmp_path, model_factory)
    backend.load()
    result = backend.synthesize(SpeechRequest(text="rate probe"))
    assert result.sample_rate_hz == 24_000
    assert result.metadata == (
        ("backend", "pytorch-voxcpm"),
        ("conditioning", "none"),
        ("model_id", backend.resolved_model.model_id),
        ("model_revision", "unrecorded"),
        ("model_source_kind", "explicit-path"),
        ("operation", "synthesize"),
        ("sample_rate_hz", 24_000),
        ("sample_rate_source", "model.tts_model.sample_rate"),
        ("upstream_device", "cpu"),
        ("upstream_optimize", False),
    )


def test_synthesize_converts_waveform_to_audio_result(
    tmp_path: Path, model_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _StubModel(waveform=(0.0, 0.25, -0.75))
    upstream = _StubUpstream(model=model)
    module = types.ModuleType(_UPSTREAM_MODULE)
    module.VoxCPM = upstream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _UPSTREAM_MODULE, module)
    backend = _backend(tmp_path, model_factory)
    backend.load()
    result = backend.synthesize(SpeechRequest(text="convert"))
    assert isinstance(result, AudioResult)
    assert result.samples == (0.0, 0.25, -0.75)
    assert result.channels == 1
    assert result.sample_rate_hz == _REAL_SAMPLE_RATE_HZ


def test_synthesize_preserves_clipping_range_without_clipping(
    tmp_path: Path, model_factory: Any, upstream_stub: _StubUpstream
) -> None:
    from voxcpm_runtime.backend_types import SpeechRequest

    backend = _backend(tmp_path, model_factory)
    backend.load()
    result = backend.synthesize(SpeechRequest(text="clip"))
    assert result.samples == (0.0, 0.5, -0.5, 1.0, -1.0)


def test_synthesize_rejects_non_speech_request(
    tmp_path: Path, model_factory: Any, upstream_stub: _StubUpstream
) -> None:
    backend = _backend(tmp_path, model_factory)
    backend.load()
    with pytest.raises(BackendRequestError):
        backend.synthesize(object())  # type: ignore[arg-type]


# --- waveform validation (Phase G) --------------------------------------


@pytest.mark.parametrize(
    ("waveform", "code"),
    [
        ((), "empty_waveform"),
        (None, "empty_waveform"),
        ((0.1, float("nan")), "non_finite_waveform"),
        ((0.1, float("inf")), "non_finite_waveform"),
        ((("nested", 0.1),), "invalid_waveform_shape"),
        ((("only-nested",),), "invalid_waveform_shape"),
        ((0.1, "not-a-number"), "invalid_waveform_sample"),
    ],
)
def test_invalid_waveforms_are_rejected(
    tmp_path: Path,
    model_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
    waveform: Any,
    code: str,
) -> None:
    model = _StubModel(waveform=waveform)
    upstream = _StubUpstream(model=model)
    module = types.ModuleType(_UPSTREAM_MODULE)
    module.VoxCPM = upstream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _UPSTREAM_MODULE, module)
    backend = _backend(tmp_path, model_factory)
    backend.load()
    with pytest.raises(BackendExecutionError) as caught:
        backend.synthesize(SpeechRequest(text="invalid"))
    assert caught.value.code == code


def test_wrong_dimensional_ndarray_is_rejected(
    tmp_path: Path, model_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _TwoDimensional:
        ndim = 2

        def tolist(self) -> Any:
            return [[0.1, 0.2]]

    model = _StubModel(waveform=_TwoDimensional())
    upstream = _StubUpstream(model=model)
    module = types.ModuleType(_UPSTREAM_MODULE)
    module.VoxCPM = upstream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _UPSTREAM_MODULE, module)
    backend = _backend(tmp_path, model_factory)
    backend.load()
    with pytest.raises(BackendExecutionError) as caught:
        backend.synthesize(SpeechRequest(text="2d"))
    assert caught.value.code == "invalid_waveform_shape"
    assert caught.value.details["dimensions"] == 2


@pytest.mark.parametrize(
    "sample_rate",
    [None, 0, -1, 48_000.5, "48000", True, float("nan")],
)
def test_invalid_upstream_sample_rate_is_rejected(
    tmp_path: Path,
    model_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
    sample_rate: Any,
) -> None:
    model = _StubModel(tts_model=_StubTtsModel(sample_rate=sample_rate))
    upstream = _StubUpstream(model=model)
    module = types.ModuleType(_UPSTREAM_MODULE)
    module.VoxCPM = upstream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _UPSTREAM_MODULE, module)
    backend = _backend(tmp_path, model_factory)
    backend.load()
    with pytest.raises(BackendExecutionError) as caught:
        backend.synthesize(SpeechRequest(text="rate"))
    assert caught.value.code in {
        "upstream_sample_rate_unavailable",
        "upstream_sample_rate_invalid",
    }


# --- error normalization (F7) -------------------------------------------


def test_unknown_load_exception_is_normalized(
    tmp_path: Path, model_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream = _StubUpstream(error=RuntimeError(f"{_PRIVATE_DETAIL} /secret/model/path"))
    module = types.ModuleType(_UPSTREAM_MODULE)
    module.VoxCPM = upstream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _UPSTREAM_MODULE, module)
    backend = _backend(tmp_path, model_factory)
    with pytest.raises(BackendExecutionError) as caught:
        backend.load()
    assert caught.value.code == "backend_execution_failed"
    assert caught.value.details == {
        "exception_type": "RuntimeError",
        "operation": "load",
    }
    assert _PRIVATE_DETAIL not in caught.value.message
    assert "/secret/model/path" not in caught.value.message


def test_unknown_generate_exception_is_normalized(
    tmp_path: Path, model_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _StubModel(generate_error=ValueError(f"{_PRIVATE_DETAIL} https://token.example"))
    upstream = _StubUpstream(model=model)
    module = types.ModuleType(_UPSTREAM_MODULE)
    module.VoxCPM = upstream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _UPSTREAM_MODULE, module)
    backend = _backend(tmp_path, model_factory)
    backend.load()
    with pytest.raises(BackendExecutionError) as caught:
        backend.synthesize(SpeechRequest(text="boom"))
    assert caught.value.code == "backend_execution_failed"
    assert caught.value.details["operation"] == "synthesize"
    assert _PRIVATE_DETAIL not in caught.value.message
    assert "token.example" not in caught.value.message


def test_missing_upstream_class_is_normalized(
    tmp_path: Path, model_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = types.ModuleType(_UPSTREAM_MODULE)
    monkeypatch.setitem(sys.modules, _UPSTREAM_MODULE, module)
    backend = _backend(tmp_path, model_factory)
    with pytest.raises(BackendExecutionError) as caught:
        backend.load()
    assert caught.value.code == "upstream_contract_violation"
    assert caught.value.details["symbol"] == "VoxCPM"


def test_upstream_import_failure_is_normalized(
    tmp_path: Path, model_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    def _explode(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == _UPSTREAM_MODULE:
            raise ImportError(f"{_PRIVATE_DETAIL}")
        return original(name, *args, **kwargs)

    original = importlib.import_module
    monkeypatch.setattr(importlib, "import_module", _explode)
    monkeypatch.delitem(sys.modules, _UPSTREAM_MODULE, raising=False)
    backend = _backend(tmp_path, model_factory)
    with pytest.raises(BackendExecutionError) as caught:
        backend.load()
    assert caught.value.details["exception_type"] == "ImportError"
    assert _PRIVATE_DETAIL not in caught.value.message


def test_model_path_never_appears_in_public_error_envelope(
    tmp_path: Path, model_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream = _StubUpstream(
        error=RuntimeError(f"failed loading {tmp_path / 'model'}")
    )
    module = types.ModuleType(_UPSTREAM_MODULE)
    module.VoxCPM = upstream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _UPSTREAM_MODULE, module)
    backend = _backend(tmp_path, model_factory)
    with pytest.raises(BackendExecutionError) as caught:
        backend.load()
    envelope = caught.value.to_dict()
    serialized = str(envelope)
    assert str(tmp_path) not in serialized
    assert "model" not in str(envelope.get("details", {}))


# --- device inspection (Phase L) ---------------------------------------


def test_inspect_model_devices_reports_cpu_only(
    tmp_path: Path, model_factory: Any, upstream_stub: _StubUpstream
) -> None:
    backend = _backend(tmp_path, model_factory)
    backend.load()
    report = inspect_model_devices(backend.loaded_model)
    assert report.parameter_device_types == ("cpu",)
    assert report.buffer_device_types == ("cpu",)
    assert report.cpu_only is True
    assert report.parameter_count == 3


def test_inspect_model_devices_detects_non_cpu_tensors() -> None:
    tts_model = _StubTtsModel()
    tts_model._parameter_device = "cuda:0"  # type: ignore[attr-defined]
    model = _StubModel(tts_model=tts_model)
    report = inspect_model_devices(model)
    assert report.parameter_device_types == ("cuda:0",)
    assert report.cpu_only is False


def test_inspect_model_devices_matches_expected_cuda() -> None:
    tts_model = _StubTtsModel()
    tts_model._parameter_device = "cuda"  # type: ignore[attr-defined]
    tts_model._buffer_device = "cuda"  # type: ignore[attr-defined]
    model = _StubModel(tts_model=tts_model)
    report = inspect_model_devices(model, expected_device="cuda")
    assert report.parameter_device_types == ("cuda",)
    assert report.buffer_device_types == ("cuda",)
    assert report.expected_device == "cuda"
    assert report.all_on_expected_device is True
    assert report.to_dict()["all_on_expected_device"] is True


def test_inspect_model_devices_rejects_mixed_expected_cuda() -> None:
    tts_model = _StubTtsModel()
    tts_model._parameter_device = "cuda"  # type: ignore[attr-defined]
    tts_model._buffer_device = "cpu"  # type: ignore[attr-defined]
    model = _StubModel(tts_model=tts_model)
    report = inspect_model_devices(model, expected_device="cuda")
    assert report.all_on_expected_device is False


def test_inspect_model_devices_requires_tts_model() -> None:
    with pytest.raises(BackendExecutionError) as caught:
        inspect_model_devices(types.SimpleNamespace())
    assert caught.value.code == "upstream_contract_violation"
    assert caught.value.details == {"attribute": "tts_model"}


def test_synthesize_requires_loaded_model_object(
    tmp_path: Path, model_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _StubModel()
    upstream = _StubUpstream(model=model)
    module = types.ModuleType(_UPSTREAM_MODULE)
    module.VoxCPM = upstream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _UPSTREAM_MODULE, module)
    backend = _backend(tmp_path, model_factory)
    backend.load()
    object.__setattr__(backend, "_model", None)
    with pytest.raises(BackendStateError) as caught:
        backend.synthesize(SpeechRequest(text="x"))
    assert caught.value.code == "backend_not_loaded"


# --- M6 reference audio policy (Section 5) -----------------------------


def _reference_file(tmp_path: Path, name: str = "reference.wav") -> Path:
    target = tmp_path / name
    target.write_bytes(b"RIFF0000WAVEfmt ")
    return target


def test_missing_reference_audio_is_a_project_owned_error(
    tmp_path: Path, model_factory: Any, upstream_stub: _StubUpstream
) -> None:
    backend = _backend(tmp_path, model_factory)
    backend.load()
    request = CloneRequest(
        text="a",
        reference_audio=AudioReference(
            local_path=str(tmp_path / "absent" / "private-voice.wav")
        ),
    )
    with pytest.raises(BackendRequestError) as caught:
        backend.clone(request)
    assert caught.value.code == "reference_audio_not_found"
    assert "private-voice.wav" not in caught.value.message
    assert str(tmp_path) not in str(caught.value.to_dict())


def test_reference_audio_directory_is_refused(
    tmp_path: Path, model_factory: Any, upstream_stub: _StubUpstream
) -> None:
    backend = _backend(tmp_path, model_factory)
    backend.load()
    request = CloneRequest(
        text="a",
        reference_audio=AudioReference(local_path=str(tmp_path)),
    )
    with pytest.raises(BackendRequestError) as caught:
        backend.clone(request)
    assert caught.value.code == "reference_audio_not_file"
    assert str(tmp_path) not in str(caught.value.to_dict())


def test_unreadable_reference_audio_is_refused(
    tmp_path: Path, model_factory: Any, upstream_stub: _StubUpstream
) -> None:
    if os.geteuid() == 0:
        pytest.skip("root bypasses filesystem read permissions")
    reference = _reference_file(tmp_path)
    reference.chmod(0o000)
    backend = _backend(tmp_path, model_factory)
    backend.load()
    with pytest.raises(BackendRequestError) as caught:
        backend.clone(
            CloneRequest(
                text="a", reference_audio=AudioReference(local_path=str(reference))
            )
        )
    assert caught.value.code == "reference_audio_unreadable"


def test_continuation_rejects_missing_reference_audio(
    tmp_path: Path, model_factory: Any, upstream_stub: _StubUpstream
) -> None:
    backend = _backend(tmp_path, model_factory)
    backend.load()
    with pytest.raises(BackendRequestError) as caught:
        backend.continue_audio(
            ContinuationRequest(
                text="a",
                reference_audio=AudioReference(local_path=str(tmp_path / "absent.wav")),
                reference_transcript="b",
            )
        )
    assert caught.value.code == "reference_audio_not_found"


def test_reference_validation_rejects_non_audio_reference() -> None:
    with pytest.raises(BackendRequestError) as caught:
        validate_local_reference_audio("reference.wav")  # type: ignore[arg-type]
    assert caught.value.code == "invalid_backend_request"


def test_reference_validation_returns_resolved_local_path(tmp_path: Path) -> None:
    reference = _reference_file(tmp_path)
    assert validate_local_reference_audio(
        AudioReference(local_path=str(reference))
    ) == str(reference)


# --- M6 design mapping --------------------------------------------------


def test_design_maps_instruction_and_text_exactly_once(
    tmp_path: Path, model_factory: Any, upstream_stub: _StubUpstream
) -> None:
    backend = _backend(tmp_path, model_factory)
    backend.load()
    backend.design(
        VoiceDesignRequest(
            text="Xin chào từ VoxCPM2.",
            instruction="giọng nữ ấm áp, bình tĩnh",
        )
    )
    assert upstream_stub.model.calls == [
        {"text": "(giọng nữ ấm áp, bình tĩnh)Xin chào từ VoxCPM2."}
    ]


def test_design_strips_instruction_whitespace(
    tmp_path: Path, model_factory: Any, upstream_stub: _StubUpstream
) -> None:
    backend = _backend(tmp_path, model_factory)
    backend.load()
    backend.design(VoiceDesignRequest(text="target", instruction="  calm  "))
    assert upstream_stub.model.calls == [{"text": "(calm)target"}]


def test_design_passes_no_reference_or_prompt_audio(
    tmp_path: Path, model_factory: Any, upstream_stub: _StubUpstream
) -> None:
    backend = _backend(tmp_path, model_factory)
    backend.load()
    backend.design(VoiceDesignRequest(text="a", instruction="b"))
    call = upstream_stub.model.calls[0]
    assert set(call) == {"text"}
    assert "reference_wav_path" not in call
    assert "prompt_wav_path" not in call
    assert "prompt_text" not in call


def test_design_returns_audio_result_with_real_sample_rate(
    tmp_path: Path, model_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _StubModel(waveform=(0.0, 0.25, -0.75))
    upstream = _StubUpstream(model=model)
    module = types.ModuleType(_UPSTREAM_MODULE)
    module.VoxCPM = upstream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _UPSTREAM_MODULE, module)
    backend = _backend(tmp_path, model_factory)
    backend.load()
    result = backend.design(VoiceDesignRequest(text="a", instruction="b"))
    assert isinstance(result, AudioResult)
    assert result.samples == (0.0, 0.25, -0.75)
    assert result.channels == 1
    assert result.sample_rate_hz == _REAL_SAMPLE_RATE_HZ
    assert dict(result.metadata)["operation"] == "design"
    assert dict(result.metadata)["conditioning"] == "instruction"


def test_design_reads_real_sample_rate(
    tmp_path: Path, model_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _StubModel(tts_model=_StubTtsModel(sample_rate=24_000))
    upstream = _StubUpstream(model=model)
    module = types.ModuleType(_UPSTREAM_MODULE)
    module.VoxCPM = upstream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _UPSTREAM_MODULE, module)
    backend = _backend(tmp_path, model_factory)
    backend.load()
    assert backend.design(
        VoiceDesignRequest(text="a", instruction="b")
    ).sample_rate_hz == 24_000


def test_design_failure_does_not_expose_instruction(
    tmp_path: Path, model_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "m6-private-instruction-please-hide"
    model = _StubModel(generate_error=ValueError(f"{secret} /secret/ref.wav"))
    upstream = _StubUpstream(model=model)
    module = types.ModuleType(_UPSTREAM_MODULE)
    module.VoxCPM = upstream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _UPSTREAM_MODULE, module)
    backend = _backend(tmp_path, model_factory)
    backend.load()
    with pytest.raises(BackendExecutionError) as caught:
        backend.design(VoiceDesignRequest(text="target", instruction=secret))
    envelope = str(caught.value.to_dict())
    assert caught.value.code == "backend_execution_failed"
    assert caught.value.details["operation"] == "design"
    assert secret not in envelope
    assert "/secret/ref.wav" not in envelope


def test_design_requires_loaded_backend(
    tmp_path: Path, model_factory: Any
) -> None:
    backend = _backend(tmp_path, model_factory)
    with pytest.raises(BackendStateError) as caught:
        backend.design(VoiceDesignRequest(text="a", instruction="b"))
    assert caught.value.code == "backend_not_loaded"


# --- M6 clone mapping ---------------------------------------------------


def test_clone_maps_local_path_to_reference_wav_path(
    tmp_path: Path, model_factory: Any, upstream_stub: _StubUpstream
) -> None:
    reference = _reference_file(tmp_path)
    backend = _backend(tmp_path, model_factory)
    backend.load()
    backend.clone(
        CloneRequest(
            text="Xin chào, đây là phép thử clone giọng.",
            reference_audio=AudioReference(local_path=str(reference)),
        )
    )
    assert upstream_stub.model.calls == [
        {
            "text": "Xin chào, đây là phép thử clone giọng.",
            "reference_wav_path": str(reference),
        }
    ]


def test_clone_passes_no_prompt_audio_or_text(
    tmp_path: Path, model_factory: Any, upstream_stub: _StubUpstream
) -> None:
    reference = _reference_file(tmp_path)
    backend = _backend(tmp_path, model_factory)
    backend.load()
    backend.clone(
        CloneRequest(
            text="a", reference_audio=AudioReference(local_path=str(reference))
        )
    )
    call = upstream_stub.model.calls[0]
    assert set(call) == {"text", "reference_wav_path"}
    assert "prompt_wav_path" not in call
    assert "prompt_text" not in call


def test_clone_returns_audio_result(
    tmp_path: Path, model_factory: Any, upstream_stub: _StubUpstream
) -> None:
    reference = _reference_file(tmp_path)
    backend = _backend(tmp_path, model_factory)
    backend.load()
    result = backend.clone(
        CloneRequest(
            text="a", reference_audio=AudioReference(local_path=str(reference))
        )
    )
    assert isinstance(result, AudioResult)
    assert result.sample_rate_hz == _REAL_SAMPLE_RATE_HZ
    assert dict(result.metadata)["operation"] == "clone"
    assert dict(result.metadata)["conditioning"] == "reference"


def test_clone_never_leaks_local_path_on_failure(
    tmp_path: Path, model_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = _reference_file(tmp_path)
    model = _StubModel(
        generate_error=RuntimeError(f"could not decode {reference} at /secret/place")
    )
    upstream = _StubUpstream(model=model)
    module = types.ModuleType(_UPSTREAM_MODULE)
    module.VoxCPM = upstream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _UPSTREAM_MODULE, module)
    backend = _backend(tmp_path, model_factory)
    backend.load()
    with pytest.raises(BackendExecutionError) as caught:
        backend.clone(
            CloneRequest(
                text="a",
                reference_audio=AudioReference(local_path=str(reference)),
            )
        )
    envelope = str(caught.value.to_dict())
    assert caught.value.details["operation"] == "clone"
    assert str(reference) not in envelope
    assert "reference.wav" not in envelope
    assert str(tmp_path) not in envelope
    assert "/secret/place" not in envelope


def test_clone_requires_loaded_backend(
    tmp_path: Path, model_factory: Any
) -> None:
    backend = _backend(tmp_path, model_factory)
    with pytest.raises(BackendStateError) as caught:
        backend.clone(
            CloneRequest(
                text="a", reference_audio=AudioReference(local_path="reference.wav")
            )
        )
    assert caught.value.code == "backend_not_loaded"


# --- M6 continuation mapping --------------------------------------------


def test_continue_audio_maps_local_path_to_prompt_wav_path(
    tmp_path: Path, model_factory: Any, upstream_stub: _StubUpstream
) -> None:
    reference = _reference_file(tmp_path)
    backend = _backend(tmp_path, model_factory)
    backend.load()
    backend.continue_audio(
        ContinuationRequest(
            text="Và đây là phần tiếp theo.",
            reference_audio=AudioReference(local_path=str(reference)),
            reference_transcript="Xin chào, đây là giọng nói tham chiếu.",
        )
    )
    assert upstream_stub.model.calls == [
        {
            "text": "Và đây là phần tiếp theo.",
            "prompt_text": "Xin chào, đây là giọng nói tham chiếu.",
            "prompt_wav_path": str(reference),
        }
    ]


def test_continue_audio_passes_no_reference_wav_path(
    tmp_path: Path, model_factory: Any, upstream_stub: _StubUpstream
) -> None:
    reference = _reference_file(tmp_path)
    backend = _backend(tmp_path, model_factory)
    backend.load()
    backend.continue_audio(
        ContinuationRequest(
            text="a",
            reference_audio=AudioReference(local_path=str(reference)),
            reference_transcript="b",
        )
    )
    call = upstream_stub.model.calls[0]
    assert set(call) == {"text", "prompt_text", "prompt_wav_path"}
    assert "reference_wav_path" not in call


def test_continue_audio_returns_audio_result(
    tmp_path: Path, model_factory: Any, upstream_stub: _StubUpstream
) -> None:
    reference = _reference_file(tmp_path)
    backend = _backend(tmp_path, model_factory)
    backend.load()
    result = backend.continue_audio(
        ContinuationRequest(
            text="a",
            reference_audio=AudioReference(local_path=str(reference)),
            reference_transcript="b",
        )
    )
    assert isinstance(result, AudioResult)
    assert dict(result.metadata)["operation"] == "continue_audio"
    assert dict(result.metadata)["conditioning"] == "continuation"


def test_continuation_does_not_leak_path_or_transcript_on_failure(
    tmp_path: Path, model_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = _reference_file(tmp_path)
    transcript = "m6-private-reference-transcript"
    model = _StubModel(generate_error=ValueError(f"{transcript} {reference}"))
    upstream = _StubUpstream(model=model)
    module = types.ModuleType(_UPSTREAM_MODULE)
    module.VoxCPM = upstream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _UPSTREAM_MODULE, module)
    backend = _backend(tmp_path, model_factory)
    backend.load()
    with pytest.raises(BackendExecutionError) as caught:
        backend.continue_audio(
            ContinuationRequest(
                text="target",
                reference_audio=AudioReference(local_path=str(reference)),
                reference_transcript=transcript,
            )
        )
    envelope = str(caught.value.to_dict())
    assert caught.value.details["operation"] == "continue_audio"
    assert transcript not in envelope
    assert str(reference) not in envelope
    assert str(tmp_path) not in envelope


def test_continue_audio_requires_loaded_backend(
    tmp_path: Path, model_factory: Any
) -> None:
    backend = _backend(tmp_path, model_factory)
    with pytest.raises(BackendStateError) as caught:
        backend.continue_audio(
            ContinuationRequest(
                text="a",
                reference_audio=AudioReference(local_path="reference.wav"),
                reference_transcript="b",
            )
        )
    assert caught.value.code == "backend_not_loaded"


# --- M6 design text rule ------------------------------------------------


def test_build_design_text_matches_pinned_upstream_rule() -> None:
    assert build_design_text("target", "calm") == "(calm)target"
    assert build_design_text("target", "  calm  ") == "(calm)target"
    assert build_design_text("target", "") == "target"
    assert build_design_text("target", "   ") == "target"


def test_build_design_text_rejects_non_strings() -> None:
    with pytest.raises(TypeError):
        build_design_text(object(), "calm")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "operation",
    ["synthesize", "design", "clone", "continue_audio"],
)
def test_waveform_conversion_failure_preserves_operation(
    tmp_path: Path,
    model_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    class _ExplodingWaveform:
        ndim = 1

        def tolist(self) -> list[float]:
            raise RuntimeError("private waveform conversion detail")

    upstream = _StubUpstream(model=_StubModel(waveform=_ExplodingWaveform()))
    module = types.ModuleType(_UPSTREAM_MODULE)
    module.VoxCPM = upstream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _UPSTREAM_MODULE, module)
    backend = _backend(tmp_path, model_factory)
    backend.load()
    reference = _reference_file(tmp_path)

    requests = {
        "synthesize": SpeechRequest(text="a"),
        "design": VoiceDesignRequest(text="a", instruction="b"),
        "clone": CloneRequest(
            text="a",
            reference_audio=AudioReference(local_path=str(reference)),
        ),
        "continue_audio": ContinuationRequest(
            text="a",
            reference_audio=AudioReference(local_path=str(reference)),
            reference_transcript="b",
        ),
    }

    with pytest.raises(BackendExecutionError) as caught:
        getattr(backend, operation)(requests[operation])
    assert caught.value.code == "backend_execution_failed"
    assert caught.value.details["operation"] == operation
    assert "private waveform conversion detail" not in str(caught.value.to_dict())


# --- native streaming (M7) ----------------------------------------------


@pytest.mark.parametrize(
    ("stream_request", "expected_kwargs"),
    [
        (
            SpeechRequest(text="stream speech"),
            {"text": "stream speech"},
        ),
        (
            VoiceDesignRequest(text="stream design", instruction="calm"),
            {"text": "(calm)stream design"},
        ),
    ],
)
def test_stream_maps_unconditioned_and_design_requests_to_native_upstream(
    tmp_path: Path,
    model_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
    stream_request: Any,
    expected_kwargs: dict[str, Any],
) -> None:
    model = _StubModel()
    upstream = _StubUpstream(model=model)
    module = types.ModuleType(_UPSTREAM_MODULE)
    module.VoxCPM = upstream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _UPSTREAM_MODULE, module)

    backend = _backend(tmp_path, model_factory)
    backend.load()
    iterator = backend.stream(StreamRequest(stream_request))
    assert model.stream_calls == []
    first = next(iterator)
    assert isinstance(first, AudioChunk)
    assert model.calls == []
    assert model.stream_calls == [expected_kwargs]
    iterator.close()


def test_stream_maps_clone_request_to_reference_wav_path(
    tmp_path: Path,
    model_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _StubModel()
    upstream = _StubUpstream(model=model)
    module = types.ModuleType(_UPSTREAM_MODULE)
    module.VoxCPM = upstream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _UPSTREAM_MODULE, module)
    reference = _reference_file(tmp_path)

    backend = _backend(tmp_path, model_factory)
    backend.load()
    list(
        backend.stream(
            StreamRequest(
                CloneRequest(
                    text="clone stream",
                    reference_audio=AudioReference(local_path=str(reference)),
                )
            )
        )
    )
    assert model.stream_calls == [
        {
            "text": "clone stream",
            "reference_wav_path": str(reference),
        }
    ]
    assert model.calls == []


def test_stream_maps_continuation_request_to_prompt_pair(
    tmp_path: Path,
    model_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _StubModel()
    upstream = _StubUpstream(model=model)
    module = types.ModuleType(_UPSTREAM_MODULE)
    module.VoxCPM = upstream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _UPSTREAM_MODULE, module)
    reference = _reference_file(tmp_path)

    backend = _backend(tmp_path, model_factory)
    backend.load()
    list(
        backend.stream(
            StreamRequest(
                ContinuationRequest(
                    text="continuation stream",
                    reference_audio=AudioReference(local_path=str(reference)),
                    reference_transcript="prefix transcript",
                )
            )
        )
    )
    assert model.stream_calls == [
        {
            "text": "continuation stream",
            "prompt_text": "prefix transcript",
            "prompt_wav_path": str(reference),
        }
    ]
    assert model.calls == []


def test_stream_chunks_are_contiguous_non_empty_and_have_one_final_chunk(
    tmp_path: Path,
    model_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _StubModel(
        stream_chunks=((0.0, 0.1), (0.2,), (0.3, 0.4, 0.5)),
        tts_model=_StubTtsModel(sample_rate=24_000),
    )
    upstream = _StubUpstream(model=model)
    module = types.ModuleType(_UPSTREAM_MODULE)
    module.VoxCPM = upstream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _UPSTREAM_MODULE, module)

    backend = _backend(tmp_path, model_factory)
    backend.load()
    chunks = list(backend.stream(StreamRequest(SpeechRequest(text="chunks"))))

    assert [chunk.sequence for chunk in chunks] == [0, 1, 2]
    assert [chunk.is_final for chunk in chunks] == [False, False, True]
    assert all(chunk.samples for chunk in chunks)
    assert all(chunk.sample_rate_hz == 24_000 for chunk in chunks)
    assert all(chunk.channels == 1 for chunk in chunks)
    assert tuple(sample for chunk in chunks for sample in chunk.samples) == (
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
    )
    assert all(("operation", "stream") in chunk.metadata for chunk in chunks)
    assert model.streams[0].closed is True


def test_stream_uses_one_chunk_lookahead_lazily(
    tmp_path: Path,
    model_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _StubModel(stream_chunks=((0.0,), (0.1,), (0.2,)))
    upstream = _StubUpstream(model=model)
    module = types.ModuleType(_UPSTREAM_MODULE)
    module.VoxCPM = upstream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _UPSTREAM_MODULE, module)

    backend = _backend(tmp_path, model_factory)
    backend.load()
    iterator = backend.stream(StreamRequest(SpeechRequest(text="lazy")))

    assert model.stream_calls == []
    first = next(iterator)
    stream = model.streams[0]
    assert first.sequence == 0
    assert first.is_final is False
    assert stream.next_calls == 2
    iterator.close()
    assert stream.closed is True


def test_stream_empty_upstream_is_project_error(
    tmp_path: Path,
    model_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _StubModel(stream_chunks=())
    upstream = _StubUpstream(model=model)
    module = types.ModuleType(_UPSTREAM_MODULE)
    module.VoxCPM = upstream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _UPSTREAM_MODULE, module)

    backend = _backend(tmp_path, model_factory)
    backend.load()
    with pytest.raises(BackendExecutionError) as caught:
        next(backend.stream(StreamRequest(SpeechRequest(text="empty"))))
    assert caught.value.code == "empty_stream"
    assert caught.value.details["operation"] == "stream"
    assert model.streams[0].closed is True


def test_stream_failure_before_first_chunk_is_normalized(
    tmp_path: Path,
    model_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _StubModel(stream_error_at=0)
    upstream = _StubUpstream(model=model)
    module = types.ModuleType(_UPSTREAM_MODULE)
    module.VoxCPM = upstream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _UPSTREAM_MODULE, module)

    backend = _backend(tmp_path, model_factory)
    backend.load()
    with pytest.raises(BackendExecutionError) as caught:
        next(backend.stream(StreamRequest(SpeechRequest(text="failure"))))
    assert caught.value.code == "backend_execution_failed"
    assert caught.value.details["operation"] == "stream"
    assert "private-stream-error" not in str(caught.value.to_dict())
    assert model.streams[0].closed is True


def test_stream_failure_after_emitted_chunk_is_normalized(
    tmp_path: Path,
    model_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _StubModel(
        stream_chunks=((0.0,), (0.1,), (0.2,)),
        stream_error_at=2,
    )
    upstream = _StubUpstream(model=model)
    module = types.ModuleType(_UPSTREAM_MODULE)
    module.VoxCPM = upstream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _UPSTREAM_MODULE, module)

    backend = _backend(tmp_path, model_factory)
    backend.load()
    iterator = backend.stream(StreamRequest(SpeechRequest(text="partial")))
    first = next(iterator)
    assert first.sequence == 0
    assert first.is_final is False

    with pytest.raises(BackendExecutionError) as caught:
        next(iterator)
    assert caught.value.details["operation"] == "stream"
    assert "private-stream-error" not in str(caught.value.to_dict())
    assert model.streams[0].closed is True


def test_stream_waveform_conversion_failure_preserves_stream_operation(
    tmp_path: Path,
    model_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ExplodingWaveform:
        ndim = 1

        def tolist(self) -> list[float]:
            raise RuntimeError("private-stream-waveform-detail")

    model = _StubModel(
        stream_chunks=(_ExplodingWaveform(), (0.1,)),
    )
    upstream = _StubUpstream(model=model)
    module = types.ModuleType(_UPSTREAM_MODULE)
    module.VoxCPM = upstream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _UPSTREAM_MODULE, module)

    backend = _backend(tmp_path, model_factory)
    backend.load()
    with pytest.raises(BackendExecutionError) as caught:
        next(backend.stream(StreamRequest(SpeechRequest(text="convert"))))
    assert caught.value.details["operation"] == "stream"
    assert "private-stream-waveform-detail" not in str(caught.value.to_dict())


def test_stream_early_close_closes_upstream_and_backend_remains_reusable(
    tmp_path: Path,
    model_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _StubModel(stream_chunks=((0.0,), (0.1,), (0.2,)))
    upstream = _StubUpstream(model=model)
    module = types.ModuleType(_UPSTREAM_MODULE)
    module.VoxCPM = upstream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _UPSTREAM_MODULE, module)

    backend = _backend(tmp_path, model_factory)
    backend.load()
    iterator = backend.stream(StreamRequest(SpeechRequest(text="cancel")))
    assert next(iterator).sequence == 0
    iterator.close()
    assert model.streams[0].closed is True

    result = backend.synthesize(SpeechRequest(text="after cancel"))
    assert result.samples
    assert model.calls == [{"text": "after cancel"}]


def test_stream_close_failure_is_normalized(
    tmp_path: Path,
    model_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _StubModel(
        stream_chunks=((0.0,),),
        stream_close_error=RuntimeError("private-close-detail"),
    )
    upstream = _StubUpstream(model=model)
    module = types.ModuleType(_UPSTREAM_MODULE)
    module.VoxCPM = upstream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _UPSTREAM_MODULE, module)

    backend = _backend(tmp_path, model_factory)
    backend.load()
    with pytest.raises(BackendExecutionError) as caught:
        list(backend.stream(StreamRequest(SpeechRequest(text="close failure"))))
    assert caught.value.details["operation"] == "stream"
    assert "private-close-detail" not in str(caught.value.to_dict())


def test_stream_rejects_wrong_request_type_eagerly(
    tmp_path: Path,
    model_factory: Any,
    upstream_stub: _StubUpstream,
) -> None:
    backend = _backend(tmp_path, model_factory)
    backend.load()
    with pytest.raises(BackendRequestError):
        backend.stream(object())  # type: ignore[arg-type]


def test_stream_lifecycle_is_enforced_eagerly(
    tmp_path: Path,
    model_factory: Any,
    upstream_stub: _StubUpstream,
) -> None:
    backend = _backend(tmp_path, model_factory)
    with pytest.raises(BackendStateError) as before:
        backend.stream(StreamRequest(SpeechRequest(text="before load")))
    assert before.value.code == "backend_not_loaded"

    backend.load()
    backend.close()
    with pytest.raises(BackendStateError) as after:
        backend.stream(StreamRequest(SpeechRequest(text="after close")))
    assert after.value.code == "backend_closed"
