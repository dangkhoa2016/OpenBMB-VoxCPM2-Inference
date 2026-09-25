import json
import math

import pytest

import voxcpm_runtime.fake_backend as fake_backend_module
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
from voxcpm_runtime.errors import (
    BackendError,
    BackendExecutionError,
    BackendRequestError,
    BackendStateError,
    BackendUnsupportedError,
    normalize_backend_error,
)
from voxcpm_runtime.fake_backend import FakeFailurePlan, FakeVoxCPMBackend

_PRIVATE_DETAIL = "m3-private-backend-detail-placeholder"


def test_existing_backend_error_is_preserved_without_double_wrapping() -> None:
    error = BackendUnsupportedError()
    normalized = normalize_backend_error("design", error)
    assert normalized is error
    assert error.code == "backend_operation_unsupported"
    assert error.message == "Backend operation is not supported."
    assert error.retryable is False


def test_unknown_error_normalizes_without_raw_message_or_traceback() -> None:
    private_message = "secret-token-and-private-path"
    error = normalize_backend_error(
        "synthesize",
        RuntimeError(private_message),
    )
    assert isinstance(error, BackendExecutionError)
    assert error.code == "backend_execution_failed"
    assert error.message == "Backend execution failed."
    assert error.details == {
        "exception_type": "RuntimeError",
        "operation": "synthesize",
    }
    serialized = json.dumps(error.to_dict(), sort_keys=True)
    assert private_message not in str(error)
    assert private_message not in repr(error)
    assert private_message not in serialized
    assert "Traceback" not in serialized


def test_error_envelope_is_deterministic_and_details_are_read_only() -> None:
    error = BackendExecutionError(
        details={"operation": "stream", "attempt": 1},
        retryable=True,
    )
    assert error.to_dict() == {
        "code": "backend_execution_failed",
        "details": {"attempt": 1, "operation": "stream"},
        "message": "Backend execution failed.",
        "retryable": True,
    }
    assert error.to_dict() == error.to_dict()
    with pytest.raises(TypeError):
        error.details["operation"] = "clone"


def test_generation_before_load_is_a_state_error() -> None:
    backend = FakeVoxCPMBackend()
    operations = (
        lambda: backend.synthesize(SpeechRequest("hello")),
        lambda: backend.design(VoiceDesignRequest("hello", "calm")),
        lambda: backend.clone(
            CloneRequest("hello", AudioReference("/request/reference.wav"))
        ),
        lambda: backend.continue_audio(
            ContinuationRequest(
                "hello",
                AudioReference("/request/prefix.wav"),
                "prefix transcript",
            )
        ),
        lambda: backend.stream(StreamRequest(SpeechRequest("hello"))),
    )
    for operation in operations:
        with pytest.raises(BackendStateError) as raised:
            operation()
        assert raised.value.code == "backend_not_loaded"
        assert raised.value.details == {"state": "created"}


def test_generation_after_close_is_a_state_error() -> None:
    backend = FakeVoxCPMBackend()
    backend.load()
    assert backend.close() is None
    assert backend.close() is None
    with pytest.raises(BackendStateError) as raised:
        backend.synthesize(SpeechRequest("hello"))
    assert raised.value.code == "backend_closed"
    with pytest.raises(BackendStateError) as load_error:
        backend.load()
    assert load_error.value.code == "backend_closed"


def test_close_before_load_is_idempotent_and_permanent() -> None:
    backend = FakeVoxCPMBackend()
    assert backend.close() is None
    assert backend.close() is None
    with pytest.raises(BackendStateError) as raised:
        backend.load()
    assert raised.value.code == "backend_closed"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SpeechRequest(" "),
        lambda: VoiceDesignRequest("text", ""),
        lambda: AudioReference(" "),
        lambda: CloneRequest("text", "not-a-reference"),
        lambda: ContinuationRequest("text", AudioReference("path"), " "),
        lambda: StreamRequest("not-a-request"),
        lambda: SpeechRequest("\ud800"),
        lambda: VoiceDesignRequest("text", "\udfff"),
        lambda: AudioReference("\ud800"),
    ],
)
def test_invalid_request_values_raise_normalized_request_errors(factory) -> None:
    with pytest.raises(BackendRequestError) as raised:
        factory()
    assert raised.value.code == "invalid_backend_request"
    assert raised.value.message == "Backend request field is invalid."


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BackendInfo("", "fake", True, "fake", None, ()),
        lambda: AudioResult((), 48_000, 1),
        lambda: AudioResult((math.nan,), 48_000, 1),
        lambda: AudioResult((0.0,), 0, 1),
        lambda: AudioResult((0.0,), 48_000, True),
        lambda: AudioChunk((0.0,), 48_000, 1, -1, False),
        lambda: AudioChunk((0.0,), 48_000, 1, 0, 1),
    ],
)
def test_invalid_contract_values_raise_normalized_request_errors(factory) -> None:
    with pytest.raises(BackendRequestError):
        factory()


def test_runtime_request_type_mismatch_is_normalized() -> None:
    backend = FakeVoxCPMBackend()
    backend.load()
    with pytest.raises(BackendRequestError) as raised:
        backend.synthesize(VoiceDesignRequest("hello", "calm"))
    assert raised.value.details == {"field": "request"}


def test_controlled_fake_failure_is_normalized_and_redacted() -> None:
    backend = FakeVoxCPMBackend(
        failure_plan=FakeFailurePlan(("synthesize",))
    )
    backend.load()
    with pytest.raises(BackendExecutionError) as raised:
        backend.synthesize(SpeechRequest("hello"))
    error = raised.value
    assert error.code == "backend_execution_failed"
    assert error.message == "Backend execution failed."
    assert error.to_dict()["details"] == {
        "exception_type": "RuntimeError",
        "operation": "synthesize",
    }
    public_output = f"{error!s}\n{error!r}\n{json.dumps(error.to_dict())}"
    assert error.__context__ is None
    assert error.__cause__ is None
    assert _PRIVATE_DETAIL not in public_output


def test_controlled_stream_failure_happens_before_iteration() -> None:
    backend = FakeVoxCPMBackend(failure_plan=FakeFailurePlan(("stream",)))
    backend.load()
    with pytest.raises(BackendExecutionError):
        backend.stream(StreamRequest(SpeechRequest("hello")))


@pytest.mark.parametrize(
    "operation",
    ["load", "synthesize", "design", "clone", "continue_audio", "stream", "close"],
)
def test_failure_injection_normalizes_every_operation(operation) -> None:
    backend = FakeVoxCPMBackend(failure_plan=FakeFailurePlan((operation,)))
    if operation == "load":
        action = backend.load
    else:
        backend.load()
        requests = {
            "synthesize": SpeechRequest("hello"),
            "design": VoiceDesignRequest("hello", "calm"),
            "clone": CloneRequest(
                "hello",
                AudioReference("/request/reference.wav"),
            ),
            "continue_audio": ContinuationRequest(
                "hello",
                AudioReference("/request/prefix.wav"),
                "prefix transcript",
            ),
            "stream": StreamRequest(SpeechRequest("hello")),
            "close": None,
        }
        request = requests[operation]
        if request is None:
            action = backend.close
        else:

            def action():
                return getattr(backend, operation)(request)
    with pytest.raises(BackendExecutionError) as raised:
        action()
    assert raised.value.details == {
        "exception_type": "RuntimeError",
        "operation": operation,
    }
    assert raised.value.__context__ is None
    assert raised.value.__cause__ is None


def test_unknown_internal_failure_is_normalized(monkeypatch) -> None:
    private_message = "m3-unexpected-private-detail"
    backend = FakeVoxCPMBackend()
    backend.load()

    def fail(_operation, _request):
        raise RuntimeError(private_message)

    monkeypatch.setattr(fake_backend_module, "_synthesize_result", fail)
    with pytest.raises(BackendExecutionError) as raised:
        backend.synthesize(SpeechRequest("hello"))
    assert raised.value.__context__ is None
    assert raised.value.__cause__ is None
    assert private_message not in json.dumps(raised.value.to_dict())


@pytest.mark.parametrize("operations", [("unknown",), ("load", "load"), "stream"])
def test_invalid_failure_plans_fail_closed(operations) -> None:
    with pytest.raises(ValueError):
        FakeFailurePlan(operations)


def test_failure_plan_is_disabled_by_default() -> None:
    backend = FakeVoxCPMBackend()
    backend.load()
    assert isinstance(
        backend.synthesize(SpeechRequest("hello")),
        AudioResult,
    )


def test_base_error_rejects_unsafe_detail_values() -> None:
    with pytest.raises(ValueError):
        BackendError(details={"unsafe": object()})
    with pytest.raises(ValueError):
        BackendError(details={"unsafe": math.inf})
