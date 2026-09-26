from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path
from typing import Any, get_type_hints

import pytest

from voxcpm_runtime.backend import InferenceBackend
from voxcpm_runtime.backend_types import (
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
    M4_CAPABILITIES,
    PytorchVoxCPMBackend,
    inspect_model_devices,
    resolve_upstream_optimize,
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


class _StubModel:
    def __init__(
        self,
        waveform: Any = (0.0, 0.5, -0.5, 1.0, -1.0),
        tts_model: Any = None,
        generate_error: BaseException | None = None,
    ) -> None:
        self.tts_model = _StubTtsModel() if tts_model is None else tts_model
        self.waveform = waveform
        self.generate_error = generate_error
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        if self.generate_error is not None:
            raise self.generate_error
        return self.waveform


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


def _backend(
    tmp_path: Path,
    model_factory: Any,
    *,
    optimize: bool = False,
    load_denoiser: bool = False,
    local_files_only: bool = True,
) -> PytorchVoxCPMBackend:
    return PytorchVoxCPMBackend(
        resolved_model=_resolved(tmp_path, model_factory),
        execution_plan=_cpu_plan(),
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


def test_non_cpu_auto_optimization_is_refused_not_invented() -> None:
    cuda_plan = ExecutionPlan(
        effective_device="cuda",
        selected_gpu_indices=(0,),
        worker_count=1,
        worker_gpu_indices=(0,),
    )
    with pytest.raises(BackendUnsupportedError) as caught:
        resolve_upstream_optimize(cuda_plan, OptimizationMode.AUTO)
    assert caught.value.code == "optimization_semantics_not_qualified"
    assert caught.value.details["device"] == "cuda"


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


def test_backend_info_advertises_only_synthesize(
    tmp_path: Path, model_factory: Any, upstream_stub: _StubUpstream
) -> None:
    info = _backend(tmp_path, model_factory).load()
    assert info.capabilities == ("synthesize",)
    assert info.capabilities == M4_CAPABILITIES
    assert info.loaded is True
    assert info.device == "cpu"
    assert info.backend == "pytorch-voxcpm"


@pytest.mark.parametrize(
    ("operation", "request_factory", "code"),
    [
        (
            "design",
            lambda: VoiceDesignRequest(text="a", instruction="b"),
            "design_not_qualified",
        ),
        (
            "clone",
            lambda: CloneRequest(
                text="a", reference_audio=AudioReference(local_path="reference.wav")
            ),
            "clone_not_qualified",
        ),
        (
            "continue_audio",
            lambda: ContinuationRequest(
                text="a",
                reference_audio=AudioReference(local_path="reference.wav"),
                reference_transcript="b",
            ),
            "continue_audio_not_qualified",
        ),
        ("stream", lambda: StreamRequest(SpeechRequest(text="a")), "stream_not_qualified"),
    ],
)
def test_unqualified_capabilities_raise(
    tmp_path: Path,
    model_factory: Any,
    upstream_stub: _StubUpstream,
    operation: str,
    request_factory: Any,
    code: str,
) -> None:
    backend = _backend(tmp_path, model_factory)
    backend.load()
    with pytest.raises(BackendUnsupportedError) as caught:
        getattr(backend, operation)(request_factory())
    assert caught.value.code == code


def test_unqualified_capability_rejects_wrong_request_type(
    tmp_path: Path, model_factory: Any, upstream_stub: _StubUpstream
) -> None:
    backend = _backend(tmp_path, model_factory)
    backend.load()
    with pytest.raises(BackendRequestError):
        backend.design(object())  # type: ignore[arg-type]


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
        ("model_id", backend.resolved_model.model_id),
        ("model_revision", "unrecorded"),
        ("model_source_kind", "explicit-path"),
        ("sample_rate_hz", 24_000),
        ("sample_rate_source", "model.tts_model.sample_rate"),
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
