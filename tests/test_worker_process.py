import os
import signal
import time

import pytest

from voxcpm_runtime.backend_types import (
    AudioReference,
    CloneRequest,
    ContinuationRequest,
    SpeechRequest,
    StreamRequest,
    VoiceDesignRequest,
)
from voxcpm_runtime.worker_client import WorkerClient, WorkerState
from voxcpm_runtime.worker_errors import WorkerError, WorkerExitedError, WorkerStartError
from voxcpm_runtime.worker_types import WorkerBootstrapSpec, WorkerRequest


def _client(**kwargs):
    return WorkerClient(
        WorkerBootstrapSpec(worker_id="worker0", **kwargs),
        startup_timeout_seconds=5,
        shutdown_timeout_seconds=2,
        stream_buffer_chunks=2,
    )


def test_spawn_worker_starts_ready_and_shuts_down_without_orphan():
    client = _client()
    ready = client.start()
    pid = client.pid
    assert ready.kind == "ready"
    assert client.state is WorkerState.READY
    assert pid is not None and client.is_alive
    client.shutdown()
    assert client.state is WorkerState.STOPPED
    assert not client.is_alive


@pytest.mark.parametrize(
    ("operation", "payload"),
    [
        ("synthesize", SpeechRequest("hello")),
        ("design", VoiceDesignRequest("hello", "calm")),
        ("clone", CloneRequest("hello", AudioReference("/tmp/reference.wav"))),
        (
            "continue_audio",
            ContinuationRequest(
                "hello",
                AudioReference("/tmp/reference.wav"),
                "prefix",
            ),
        ),
    ],
)
def test_spawn_worker_executes_all_one_shot_operations(operation, payload):
    client = _client()
    client.start()
    result = client.execute(WorkerRequest(f"req_{operation}", operation, payload))
    assert result.samples
    assert result.sample_rate_hz == 48_000
    assert result.channels == 1
    assert client.state is WorkerState.READY
    client.shutdown()


def test_spawn_worker_stream_preserves_chunk_semantics():
    client = _client()
    client.start()
    request = WorkerRequest(
        "req_stream",
        "stream",
        StreamRequest(SpeechRequest("stream me")),
    )
    chunks = list(client.stream(request))
    assert chunks
    assert [chunk.sequence for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.samples for chunk in chunks)
    assert sum(chunk.is_final for chunk in chunks) == 1
    assert chunks[-1].is_final
    assert client.state is WorkerState.READY
    client.shutdown()


def test_stream_buffer_is_bounded_and_active_cancel_kills_worker():
    client = _client()
    client.start()
    pid = client.pid
    client.submit(
        WorkerRequest(
            "req_cancel",
            "stream",
            StreamRequest(SpeechRequest("long enough for fake chunks")),
        )
    )
    time.sleep(0.1)
    assert client.state is WorkerState.BUSY
    cancelled = client.cancel_active()
    assert cancelled == "req_cancel"
    assert client.state is WorkerState.STOPPED
    assert not client.is_alive
    assert pid is not None


def test_backend_error_is_sanitized_across_process_boundary():
    client = _client(fake_failure_operations=("synthesize",))
    client.start()
    with pytest.raises(WorkerError) as caught:
        client.execute(
            WorkerRequest("req_error", "synthesize", SpeechRequest("secret text"))
        )
    payload = caught.value.to_dict()
    assert payload["code"] == "backend_execution_failed"
    assert "m3-private-backend-detail-placeholder" not in str(payload)
    client.shutdown()


def test_startup_failure_is_reported_and_process_is_reaped():
    client = _client(fake_failure_operations=("load",))
    with pytest.raises(WorkerStartError):
        client.start()
    assert client.state is WorkerState.FAILED
    assert not client.is_alive
    client.shutdown()
    assert client.state is WorkerState.STOPPED


@pytest.mark.skipif(os.name == "nt", reason="signal-based crash test is POSIX-only")
def test_unexpected_worker_exit_is_detected():
    client = _client()
    client.start()
    assert client.pid is not None
    client.submit(
        WorkerRequest("req_crash", "synthesize", SpeechRequest("hello"))
    )
    os.kill(client.pid, signal.SIGKILL)
    with pytest.raises(WorkerExitedError):
        client.receive(timeout_seconds=1)
    client.shutdown()
    assert client.state is WorkerState.STOPPED


def test_worker_client_rejects_non_spawn_context():
    import multiprocessing as mp

    methods = mp.get_all_start_methods()
    non_spawn = next((name for name in methods if name != "spawn"), None)
    if non_spawn is None:
        pytest.skip("no non-spawn context available")
    with pytest.raises(ValueError, match="spawn"):
        WorkerClient(
            WorkerBootstrapSpec(worker_id="worker0"),
            context=mp.get_context(non_spawn),
        )
