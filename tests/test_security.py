import json

import pytest
from fastapi.testclient import TestClient

from voxcpm_runtime.api_app import create_app
from voxcpm_runtime.config import RuntimeConfig
from voxcpm_runtime.runtime_metrics import redact_text
from voxcpm_runtime.worker_client import WorkerClient
from voxcpm_runtime.worker_types import WorkerBootstrapSpec


def _config(**overrides):
    values = dict(
        device="cpu",
        host="127.0.0.1",
        api_token="security-test-token",
        require_auth=True,
        allow_unauthenticated_external=False,
        max_text_chars=8,
        max_queue_size=2,
    )
    values.update(overrides)
    return RuntimeConfig(**values)


def _worker(_config):
    return WorkerClient(
        WorkerBootstrapSpec(worker_id="security-worker", backend_kind="fake"),
        startup_timeout_seconds=5,
        shutdown_timeout_seconds=2,
        stream_buffer_chunks=2,
    )


def _auth(value="security-test-token"):
    return {"Authorization": f"Bearer {value}"}

@pytest.mark.parametrize(
    "authorization",
    [
        None,
        "Basic abc",
        "Bearer ",
        "Bearer wrong-token",
        "Bearer    security-test-token",
        "Bearer\tsecurity-test-token",
        "Bearer " + ("x" * 10_000),
    ],
)
def test_bearer_negative_matrix_rejects_without_leaking(authorization):
    app = create_app(_config(), worker_factory=_worker)
    with TestClient(app) as client:
        headers = {} if authorization is None else {"Authorization": authorization}
        response = client.post("/v1/tts", json={"text": "hello"}, headers=headers)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"
    assert "security-test-token" not in response.text
    assert "wrong-token" not in response.text

def test_duplicate_authorization_headers_fail_closed():
    app = create_app(_config(), worker_factory=_worker)
    headers = [
        ("Authorization", "Bearer security-test-token"),
        ("Authorization", "Bearer wrong-token"),
    ]
    with TestClient(app) as client:
        response = client.post("/v1/tts", json={"text": "hello"}, headers=headers)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"
    assert "security-test-token" not in response.text


def test_validation_errors_are_normalized_and_do_not_echo_input():
    secret = "ghp_SECURITYSHOULDNOTLEAK123456"
    app = create_app(_config(), worker_factory=_worker)
    with TestClient(app) as client:
        response = client.post(
            "/v1/tts",
            json={"text": "hello", "extra": secret},
            headers=_auth(),
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
    assert secret not in response.text

@pytest.mark.parametrize(
    ("body", "content_type"),
    [
        ("{bad", "application/json"),
        ("{}", "application/json"),
        (json.dumps({"text": 123}), "application/json"),
        (json.dumps({"text": "hello"}), "text/plain"),
        (json.dumps({"text": "hello"}), None),
    ],
)
def test_invalid_body_shapes_have_deterministic_public_error(body, content_type):
    app = create_app(_config(), worker_factory=_worker)
    headers = _auth()
    if content_type is not None:
        headers["Content-Type"] = content_type
    with TestClient(app) as client:
        response = client.post("/v1/tts", content=body, headers=headers)
    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "code": "invalid_request",
            "message": "Request payload is invalid.",
            "retryable": False,
            "request_id": None,
        }
    }

def test_text_limit_accepts_boundary_and_rejects_one_over():
    app = create_app(_config(max_text_chars=8), worker_factory=_worker)
    with TestClient(app) as client:
        accepted = client.post("/v1/tts", json={"text": "12345678"}, headers=_auth())
        rejected = client.post("/v1/tts", json={"text": "123456789"}, headers=_auth())
    assert accepted.status_code == 200
    assert rejected.status_code == 422
    assert rejected.json()["error"]["code"] == "text_too_long"


def test_openapi_and_central_redaction_do_not_expose_tokens():
    app = create_app(_config(), worker_factory=_worker)
    with TestClient(app) as client:
        schema = client.get("/openapi.json")
    assert schema.status_code == 200
    assert "security-test-token" not in schema.text
    for secret in (
        "Bearer security-test-token",
        "hf_SECURITYSECRET123456",
        "ghp_SECURITYSECRET123456",
        "github_pat_SECURITYSECRET123456",
    ):
        assert secret not in redact_text(f"credential={secret}")


def test_request_timeout_cancels_and_returns_normalized_504(monkeypatch):
    from concurrent.futures import Future

    app = create_app(
        _config(request_timeout_seconds=0.05),
        worker_factory=_worker,
    )
    with TestClient(app) as client:
        runtime = app.state.runtime
        pending = Future()
        cancelled = []

        monkeypatch.setattr(runtime, "submit", lambda _request: pending)
        monkeypatch.setattr(runtime, "cancel", lambda request_id: cancelled.append(request_id))

        response = client.post(
            "/v1/tts",
            json={"text": "hello"},
            headers={**_auth(), "X-Request-ID": "timeout-one-shot"},
        )

    assert response.status_code == 504
    assert response.json() == {
        "error": {
            "code": "request_timeout",
            "message": "Request timed out.",
            "retryable": True,
            "request_id": None,
        }
    }
    assert cancelled == ["timeout-one-shot"]


def test_stream_backpressure_timeout_releases_dispatcher_and_worker():
    import time

    from voxcpm_runtime.backend_types import SpeechRequest, StreamRequest
    from voxcpm_runtime.worker_types import WorkerRequest

    app = create_app(
        _config(
            stream_ipc_max_chunks=1,
            stream_backpressure_timeout_seconds=0.05,
        ),
        worker_factory=_worker,
    )
    with TestClient(app):
        runtime = app.state.runtime
        stream_queue = runtime.submit_stream(
            WorkerRequest(
                "slow-consumer",
                "stream",
                StreamRequest(SpeechRequest("slow client")),
            )
        )

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if runtime.snapshot().ready_workers == 1:
                break
            time.sleep(0.02)

        assert runtime.snapshot().ready_workers == 1
        message = stream_queue.get(timeout=0.5)
        assert message.kind == "error"
        assert message.error is not None
        assert message.error.code == "stream_backpressure_timeout"
        assert message.error.retryable is True
