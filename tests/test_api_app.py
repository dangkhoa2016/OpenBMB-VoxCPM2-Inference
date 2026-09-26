import io
import os
import wave

import pytest
from fastapi.testclient import TestClient

from voxcpm_runtime.api_app import create_app
from voxcpm_runtime.api_auth import is_loopback_host
from voxcpm_runtime.api_errors import ApiError
from voxcpm_runtime.config import RuntimeConfig
from voxcpm_runtime.worker_client import WorkerClient
from voxcpm_runtime.worker_types import WorkerBootstrapSpec


def _config(**overrides):
    values = dict(
        device="cpu",
        host="127.0.0.1",
        api_token="test-token",
        require_auth=True,
        allow_unauthenticated_external=False,
        max_text_chars=64,
        max_queue_size=2,
    )
    values.update(overrides)
    return RuntimeConfig(**values)


def _fake_worker(_config):
    return WorkerClient(
        WorkerBootstrapSpec(worker_id="api-test", backend_kind="fake"),
        startup_timeout_seconds=5,
        shutdown_timeout_seconds=2,
        stream_buffer_chunks=2,
    )


def _auth(token="test-token"):
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_loopback_host_policy(host):
    assert is_loopback_host(host)


def test_external_unauthenticated_bind_is_rejected():
    with pytest.raises(ApiError) as caught:
        create_app(
            _config(
                host="0.0.0.0",
                require_auth=False,
                api_token=None,
                allow_unauthenticated_external=False,
            ),
            worker_factory=_fake_worker,
        )
    assert caught.value.code == "unauthenticated_external_bind_forbidden"


def test_auth_required_without_token_is_rejected():
    with pytest.raises(ApiError) as caught:
        create_app(
            _config(api_token=None, require_auth=True),
            worker_factory=_fake_worker,
        )
    assert caught.value.code == "api_token_required"


def test_health_readiness_auth_and_one_shot_wav():
    app = create_app(_config(), worker_factory=_fake_worker)
    with TestClient(app) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}

        ready = client.get("/readyz")
        assert ready.status_code == 200
        assert ready.json()["status"] == "ready"

        missing = client.post("/v1/tts", json={"text": "hello"})
        assert missing.status_code == 401
        assert "test-token" not in missing.text

        wrong = client.post(
            "/v1/tts",
            json={"text": "hello"},
            headers=_auth("wrong-token"),
        )
        assert wrong.status_code == 401
        assert "wrong-token" not in wrong.text

        response = client.post(
            "/v1/tts",
            json={"text": "Xin chào từ VoxCPM2."},
            headers={**_auth(), "X-Request-ID": "request-1"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("audio/wav")
        assert response.headers["x-request-id"] == "request-1"
        assert response.headers["cache-control"] == "no-store"
        assert "access-control-allow-origin" not in response.headers
        with wave.open(io.BytesIO(response.content), "rb") as wav:
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
            assert wav.getframerate() == 48_000
            assert wav.getnframes() > 0


def test_generated_request_id_and_duplicate_id_conflict():
    app = create_app(_config(), worker_factory=_fake_worker)
    with TestClient(app) as client:
        generated = client.post("/v1/tts", json={"text": "hello"}, headers=_auth())
        assert generated.status_code == 200
        assert generated.headers["x-request-id"]

        first = client.post(
            "/v1/tts",
            json={"text": "hello"},
            headers={**_auth(), "X-Request-ID": "same-id"},
        )
        assert first.status_code == 200
        duplicate = client.post(
            "/v1/tts",
            json={"text": "hello again"},
            headers={**_auth(), "X-Request-ID": "same-id"},
        )
        assert duplicate.status_code == 409
        assert duplicate.json()["error"]["code"] == "duplicate_request_id"


def test_text_validation_is_normalized_by_endpoint_rules():
    app = create_app(_config(max_text_chars=5), worker_factory=_fake_worker)
    with TestClient(app) as client:
        blank = client.post("/v1/tts", json={"text": "   "}, headers=_auth())
        assert blank.status_code == 422
        assert blank.json()["error"]["code"] == "invalid_text"

        too_long = client.post("/v1/tts", json={"text": "123456"}, headers=_auth())
        assert too_long.status_code == 422
        assert too_long.json()["error"]["code"] == "text_too_long"


def test_ready_becomes_503_when_only_worker_stops():
    app = create_app(_config(), worker_factory=_fake_worker)
    with TestClient(app) as client:
        assert client.get("/readyz").status_code == 200
        app.state.runtime.worker.shutdown()
        ready = client.get("/readyz")
        assert ready.status_code == 503
        assert ready.json()["status"] == "not_ready"


def test_openapi_contains_no_token_value():
    app = create_app(_config(), worker_factory=_fake_worker)
    with TestClient(app) as client:
        schema = client.get("/openapi.json")
        assert schema.status_code == 200
        assert "test-token" not in schema.text


def test_http_stream_returns_pcm16_with_public_headers():
    app = create_app(
        _config(stream_ipc_max_chunks=2),
        worker_factory=_fake_worker,
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/tts/stream",
            json={"text": "stream hello"},
            headers={**_auth(), "X-Request-ID": "stream-http-1"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/octet-stream")
        assert response.headers["x-request-id"] == "stream-http-1"
        assert response.headers["x-audio-format"] == "pcm_s16le"
        assert response.headers["x-audio-sample-rate"] == "48000"
        assert response.headers["x-audio-channels"] == "1"
        assert len(response.content) == 480 * 2
        assert len(response.content) % 2 == 0


def test_http_queue_full_maps_real_scheduler_admission_to_429():
    import time

    from voxcpm_runtime.backend_types import SpeechRequest, StreamRequest
    from voxcpm_runtime.worker_types import WorkerRequest

    app = create_app(
        _config(max_queue_size=1, stream_ipc_max_chunks=1),
        worker_factory=_fake_worker,
    )
    with TestClient(app) as client:
        runtime = app.state.runtime
        runtime.submit_stream(
            WorkerRequest(
                "hold-stream",
                "stream",
                StreamRequest(SpeechRequest("hold worker busy")),
            )
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if runtime.snapshot().busy_workers == 1:
                break
            time.sleep(0.01)
        assert runtime.snapshot().busy_workers == 1
        runtime.submit(
            WorkerRequest(
                "pending-one",
                "synthesize",
                SpeechRequest("pending"),
            )
        )
        response = client.post(
            "/v1/tts",
            json={"text": "rejected"},
            headers={**_auth(), "X-Request-ID": "queue-full-http"},
        )
        assert response.status_code == 429
        payload = response.json()["error"]
        assert payload["code"] == "queue_full"
        assert payload["retryable"] is True
        runtime.cancel("hold-stream")


def _two_fake_workers(_config):
    return (
        WorkerClient(
            WorkerBootstrapSpec(worker_id="worker0", backend_kind="fake"),
            startup_timeout_seconds=5,
            shutdown_timeout_seconds=2,
            stream_buffer_chunks=2,
        ),
        WorkerClient(
            WorkerBootstrapSpec(worker_id="worker1", backend_kind="fake"),
            startup_timeout_seconds=5,
            shutdown_timeout_seconds=2,
            stream_buffer_chunks=2,
        ),
    )


def test_ready_stays_200_with_one_of_two_workers_stopped_then_503_with_zero():
    app = create_app(_config(), worker_factory=_two_fake_workers)
    with TestClient(app) as client:
        initial = client.get("/readyz")
        assert initial.status_code == 200
        assert initial.json()["workers"]["ready"] == 2

        worker0, worker1 = app.state.runtime.workers
        worker0.shutdown()
        degraded = client.get("/readyz")
        assert degraded.status_code == 200
        assert degraded.json()["workers"]["ready"] == 1

        worker1.shutdown()
        unavailable = client.get("/readyz")
        assert unavailable.status_code == 503
        assert unavailable.json()["workers"]["ready"] == 0


@pytest.mark.skipif(os.name == "nt", reason="signal-based crash test is POSIX-only")
def test_ready_stays_200_after_one_active_worker_crashes_when_survivor_is_healthy():
    import signal
    import time

    from voxcpm_runtime.backend_types import SpeechRequest, StreamRequest
    from voxcpm_runtime.worker_types import WorkerRequest

    app = create_app(_config(), worker_factory=_two_fake_workers)
    with TestClient(app) as client:
        runtime = app.state.runtime
        runtime.submit_stream(
            WorkerRequest(
                "crash-ready-a",
                "stream",
                StreamRequest(SpeechRequest("hold worker0")),
            )
        )
        worker0, worker1 = runtime.workers
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and worker0.state.value != "busy":
            time.sleep(0.01)
        assert worker0.pid is not None
        os.kill(worker0.pid, signal.SIGKILL)

        deadline = time.monotonic() + 3
        response = None
        while time.monotonic() < deadline:
            response = client.get("/readyz")
            body = response.json()
            if body.get("workers", {}).get("failed") == 1:
                break
            time.sleep(0.05)
        assert response is not None
        assert response.status_code == 200
        body = response.json()
        assert body["workers"]["failed"] == 1
        assert body["workers"]["ready"] == 1
        assert worker1.is_alive
