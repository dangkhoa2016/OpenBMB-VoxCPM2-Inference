import ast
import builtins
import hashlib
import math
import os
import socket
import struct
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

import deploy.kaggle.model_mounts as model_mounts
import voxcpm_runtime.model_resolver as model_resolver
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
from voxcpm_runtime.fake_backend import FakeVoxCPMBackend

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_RUNTIME_DIR = REPOSITORY_ROOT / "voxcpm_runtime"
_M3_RUNTIME_FILES = (
    _RUNTIME_DIR / "backend.py",
    _RUNTIME_DIR / "backend_types.py",
    _RUNTIME_DIR / "errors.py",
    _RUNTIME_DIR / "fake_backend.py",
)
_ALLOWED_INTERNAL_IMPORTS = {
    "voxcpm_runtime.backend_types",
    "voxcpm_runtime.errors",
}
_GOLDEN_SAMPLE_FORMAT = "<d"
_GOLDEN_DIGEST = "75b074a436b54ec3947272220ab2bb35223c712ea43985ce3610cf512387cd7e"


def _requests():
    return (
        SpeechRequest("same deterministic request"),
        VoiceDesignRequest("same deterministic request", "calm voice"),
        CloneRequest(
            "same deterministic request",
            AudioReference("/request/reference.wav"),
        ),
        ContinuationRequest(
            "same deterministic request",
            AudioReference("/request/prefix.wav"),
            "prefix transcript",
        ),
    )


def _loaded_backend() -> FakeVoxCPMBackend:
    backend = FakeVoxCPMBackend()
    backend.load()
    return backend


def test_all_fake_operations_produce_deterministic_finite_audio() -> None:
    backend = _loaded_backend()
    methods = (
        backend.synthesize,
        backend.design,
        backend.clone,
        backend.continue_audio,
    )
    for method, request in zip(methods, _requests(), strict=True):
        first = method(request)
        second = method(request)
        assert isinstance(first, AudioResult)
        assert first == second
        assert first.samples
        assert all(math.isfinite(sample) for sample in first.samples)
        assert all(-0.2 <= sample <= 0.2 for sample in first.samples)


def test_fresh_instances_produce_identical_results() -> None:
    first_backend = _loaded_backend()
    second_backend = _loaded_backend()
    for method_name, request in zip(
        (
            "synthesize",
            "design",
            "clone",
            "continue_audio",
        ),
        _requests(),
        strict=True,
    ):
        first = getattr(first_backend, method_name)(request)
        second = getattr(second_backend, method_name)(request)
        assert first == second


def test_different_text_and_operations_produce_different_audio() -> None:
    backend = _loaded_backend()
    first = backend.synthesize(SpeechRequest("first text"))
    second = backend.synthesize(SpeechRequest("second text"))
    assert first.samples != second.samples
    results = [
        backend.synthesize(SpeechRequest("same text")),
        backend.design(VoiceDesignRequest("same text", "calm voice")),
        backend.clone(
            CloneRequest("same text", AudioReference("/request/same.wav"))
        ),
        backend.continue_audio(
            ContinuationRequest(
                "same text",
                AudioReference("/request/same.wav"),
                "same transcript",
            )
        ),
    ]
    assert len({result.samples for result in results}) == 4


def test_reference_descriptor_changes_deterministic_audio() -> None:
    backend = _loaded_backend()
    first = backend.clone(
        CloneRequest("hello", AudioReference("/request/first.wav"))
    )
    second = backend.clone(
        CloneRequest("hello", AudioReference("/request/second.wav"))
    )
    assert first.samples != second.samples


def test_stream_chunks_are_deterministic_and_reassemble_full_result() -> None:
    backend = _loaded_backend()
    request = SpeechRequest("stream reassembly")
    result = backend.synthesize(request)
    first = list(backend.stream(StreamRequest(request)))
    second = list(backend.stream(StreamRequest(request)))
    assert first == second
    assert len(first) == 4
    assert [chunk.sequence for chunk in first] == [0, 1, 2, 3]
    assert [chunk.is_final for chunk in first] == [False, False, False, True]
    assert all(isinstance(chunk, AudioChunk) for chunk in first)
    assert all(chunk.samples for chunk in first)
    assert all(chunk.sample_rate_hz == result.sample_rate_hz for chunk in first)
    assert all(chunk.channels == result.channels for chunk in first)
    assert first[-1].is_final is True
    assert result.samples == tuple(
        sample for chunk in first for sample in chunk.samples
    )


def test_stream_supports_every_one_shot_request_shape() -> None:
    backend = _loaded_backend()
    methods = (
        backend.synthesize,
        backend.design,
        backend.clone,
        backend.continue_audio,
    )
    for method, request in zip(methods, _requests(), strict=True):
        expected = method(request)
        chunks = list(backend.stream(StreamRequest(request)))
        assert chunks[-1].is_final is True
        assert sum(chunk.is_final for chunk in chunks) == 1
        assert all(not chunk.is_final for chunk in chunks[:-1])
        assert expected.samples == tuple(
            sample for chunk in chunks for sample in chunk.samples
        )


@pytest.mark.parametrize("chunk_size", [1, 7, 119, 120, 121, 480, 1000])
def test_stream_chunk_sizes_preserve_invariants(chunk_size) -> None:
    backend = FakeVoxCPMBackend(chunk_size=chunk_size)
    backend.load()
    request = SpeechRequest("bounded chunk sizes")
    expected = backend.synthesize(request)
    chunks = list(backend.stream(StreamRequest(request)))
    assert [chunk.sequence for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.samples for chunk in chunks)
    assert [chunk.is_final for chunk in chunks] == [False] * (len(chunks) - 1) + [True]
    assert all(chunk.sample_rate_hz == expected.sample_rate_hz for chunk in chunks)
    assert all(chunk.channels == expected.channels for chunk in chunks)
    assert expected.samples == tuple(
        sample for chunk in chunks for sample in chunk.samples
    )


@pytest.mark.parametrize("chunk_size", [0, -1, True, 1.5, "120"])
def test_invalid_chunk_sizes_fail_closed(chunk_size) -> None:
    with pytest.raises(ValueError):
        FakeVoxCPMBackend(chunk_size=chunk_size)


def test_fake_audio_has_frozen_binary64_digest() -> None:
    backend = _loaded_backend()
    result = backend.synthesize(SpeechRequest("M3 deterministic audio"))
    serialized = b"".join(
        struct.pack(_GOLDEN_SAMPLE_FORMAT, sample) for sample in result.samples
    )
    assert hashlib.sha256(serialized).hexdigest() == _GOLDEN_DIGEST


def test_fake_operations_are_hermetic_under_exploding_guards(monkeypatch) -> None:
    attempts = 0

    def forbidden(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise AssertionError("Fake backend attempted an isolated operation")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(os, "getenv", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket.socket, "sendto", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)

    backend = _loaded_backend()
    results = [
        backend.synthesize(SpeechRequest("hello")),
        backend.design(VoiceDesignRequest("hello", "calm voice")),
        backend.clone(
            CloneRequest("hello", AudioReference("/request/nonexistent.wav"))
        ),
        backend.continue_audio(
            ContinuationRequest(
                "hello",
                AudioReference("/request/nonexistent.wav"),
                "prefix transcript",
            )
        ),
    ]
    chunks = list(backend.stream(StreamRequest(SpeechRequest("hello"))))
    assert all(isinstance(result, AudioResult) for result in results)
    assert all(isinstance(chunk, AudioChunk) for chunk in chunks)
    assert backend.close() is None
    assert attempts == 0


def test_fake_backend_does_not_resolve_models_or_touch_kaggle(
    monkeypatch,
    tmp_path,
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Fake backend touched model resolution")

    for name in (
        "VOXCPM_MODEL_PATH",
        "VOXCPM_CACHE_DIR",
        "VOXCPM_OFFLINE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(model_resolver.ModelResolver, "resolve", forbidden)
    monkeypatch.setattr(
        model_mounts,
        "discover_kaggle_model_candidates",
        forbidden,
    )
    backend = _loaded_backend()
    assert backend.synthesize(SpeechRequest("hello")).samples
    assert backend.close() is None


def test_m3_runtime_contains_no_forbidden_imports_or_randomized_hash() -> None:
    forbidden_roots = {
        "aiohttp",
        "huggingface_hub",
        "httpx",
        "numpy",
        "requests",
        "torch",
        "transformers",
        "voxcpm",
    }
    forbidden_m3_roots = {
        "random",
        "secrets",
        "time",
    }
    for path in sorted(_RUNTIME_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                modules = {node.module or ""}
            else:
                continue
            roots = {module.split(".", 1)[0] for module in modules}
            assert roots.isdisjoint(forbidden_roots)
    for path in _M3_RUNTIME_FILES:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        assert "/kaggle/input" not in source
        assert "VOXCPM_MODEL_PATH" not in source
        assert "voxcpm-source-lock" not in source
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                modules = {node.module or ""}
            else:
                modules = set()
            roots = {module.split(".", 1)[0] for module in modules}
            assert roots.isdisjoint(forbidden_m3_roots)
            for module in modules:
                if module.startswith("voxcpm_runtime"):
                    assert module in _ALLOWED_INTERNAL_IMPORTS
                else:
                    assert not module.startswith("deploy")
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    assert node.func.id != "hash"
                if isinstance(node.func, ast.Attribute):
                    assert node.func.attr != "hash"


def test_fake_digest_is_stable_across_python_hash_seeds() -> None:
    script = """
import hashlib
import struct
from voxcpm_runtime.backend_types import SpeechRequest
from voxcpm_runtime.fake_backend import FakeVoxCPMBackend
backend = FakeVoxCPMBackend()
backend.load()
result = backend.synthesize(SpeechRequest("hash seed invariant"))
payload = b"".join(struct.pack("<d", sample) for sample in result.samples)
print(hashlib.sha256(payload).hexdigest())
"""
    digests = []
    for seed in ("1", "987654"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            cwd=REPOSITORY_ROOT,
            env=environment,
        )
        assert completed.returncode == 0, completed.stderr
        digests.append(completed.stdout.strip())
    assert digests[0] == digests[1]
