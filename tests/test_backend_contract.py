import inspect
from typing import get_type_hints

import pytest

import voxcpm_runtime
from voxcpm_runtime.backend import InferenceBackend
from voxcpm_runtime.backend_types import (
    AudioChunk,
    AudioReference,
    AudioResult,
    BackendInfo,
    CloneRequest,
    ContinuationRequest,
    SpeechRequest,
    StreamRequest,
    VoiceDesignRequest,
)
from voxcpm_runtime.fake_backend import FakeVoxCPMBackend


def _loaded_backend() -> FakeVoxCPMBackend:
    backend = FakeVoxCPMBackend()
    backend.load()
    return backend


def test_fake_backend_satisfies_protocol_surface() -> None:
    assert isinstance(FakeVoxCPMBackend(), InferenceBackend)


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
    fake_method = getattr(FakeVoxCPMBackend, method_name)
    assert list(inspect.signature(protocol_method).parameters) == list(
        inspect.signature(fake_method).parameters
    )
    assert get_type_hints(protocol_method) == get_type_hints(fake_method)


def test_load_returns_deterministic_backend_info() -> None:
    backend = FakeVoxCPMBackend()
    info = backend.load()
    assert info == backend.load()
    assert isinstance(info, BackendInfo)
    assert info.backend == "fake"
    assert info.implementation == "deterministic-test"
    assert info.loaded is True
    assert info.device == "fake/cpu-test"
    assert info.model_id == "m3-fake-deterministic"
    assert info.capabilities == (
        "clone",
        "continue_audio",
        "design",
        "stream",
        "synthesize",
    )
    assert ("contract_scope", "test/fake contract only") in info.metadata
    assert ("sample_rate_scope", "test/fake contract only") in info.metadata


@pytest.mark.parametrize(
    ("method_name", "operation_request"),
    [
        ("synthesize", SpeechRequest("hello")),
        (
            "design",
            VoiceDesignRequest("hello", "a calm synthetic voice"),
        ),
        (
            "clone",
            CloneRequest("hello", AudioReference("/request/reference.wav")),
        ),
        (
            "continue_audio",
            ContinuationRequest(
                "hello",
                AudioReference("/request/prefix.wav"),
                "prefix transcript",
            ),
        ),
    ],
)
def test_one_shot_operations_return_audio_result(
    method_name,
    operation_request,
) -> None:
    result = getattr(_loaded_backend(), method_name)(operation_request)
    assert isinstance(result, AudioResult)
    assert result.samples
    assert result.sample_rate_hz == 48_000
    assert result.channels == 1
    assert ("backend", "fake") in result.metadata
    assert ("sample_rate_scope", "test/fake contract only") in result.metadata


def test_stream_yields_audio_chunks_and_close_returns_none() -> None:
    backend = _loaded_backend()
    chunks = backend.stream(StreamRequest(SpeechRequest("hello")))
    assert all(isinstance(chunk, AudioChunk) for chunk in chunks)
    assert backend.close() is None


def test_contract_equality_and_repr_are_deterministic() -> None:
    request = SpeechRequest("same text")
    result = AudioResult(
        samples=(0.0, 0.25, -0.25),
        sample_rate_hz=48_000,
        channels=1,
        metadata=(("z", "last"), ("a", "first")),
    )
    assert request == SpeechRequest("same text")
    assert request != SpeechRequest("other text")
    assert result == AudioResult(
        samples=(0.0, 0.25, -0.25),
        sample_rate_hz=48_000,
        channels=1,
        metadata=(("a", "first"), ("z", "last")),
    )
    assert repr(request) == repr(SpeechRequest("same text"))
    assert repr(result) == repr(result)


def test_m3_symbols_are_not_root_package_exports() -> None:
    assert voxcpm_runtime.__all__ == ["__version__"]
    assert voxcpm_runtime.__version__ == "1.0.0"
