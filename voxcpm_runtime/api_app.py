from __future__ import annotations

import asyncio
import io
import uuid
import wave
from array import array
from contextlib import asynccontextmanager
from typing import Callable, Final

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from voxcpm_runtime.api_auth import validate_bind_auth_policy, verify_bearer_token
from voxcpm_runtime.api_errors import ApiError
from voxcpm_runtime.api_runtime import ApiRuntime
from voxcpm_runtime.backend_types import AudioChunk, AudioResult, SpeechRequest, StreamRequest
from voxcpm_runtime.config import RuntimeConfig
from voxcpm_runtime.scheduler import SchedulerSnapshot
from voxcpm_runtime.worker_client import WorkerClient
from voxcpm_runtime.worker_errors import SchedulerAdmissionError
from voxcpm_runtime.worker_factory import build_worker_clients, start_worker_clients
from voxcpm_runtime.worker_types import WorkerMessage, WorkerRequest


class TtsRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)


def _safe_request_id(value: str | None) -> str:
    if value is None:
        return uuid.uuid4().hex
    candidate = value.strip()
    if not candidate or len(candidate) > 128:
        raise ApiError("invalid_request_id", "Request ID is invalid.", status_code=422)
    if any(ch in candidate for ch in ("/", "\\", "\x00", "\n", "\r")):
        raise ApiError("invalid_request_id", "Request ID is invalid.", status_code=422)
    return candidate


def _pcm16_bytes(samples: tuple[float, ...]) -> bytes:
    pcm = array("h")
    for sample in samples:
        value = min(1.0, max(-1.0, float(sample)))
        pcm.append(max(-32768, min(32767, int(round(value * 32767.0)))))
    if pcm.itemsize != 2:
        raise ApiError("pcm_encoding_failed", "Audio encoding failed.", status_code=500)
    import sys

    if sys.byteorder != "little":
        pcm.byteswap()
    return pcm.tobytes()


def _wav_bytes(result: AudioResult) -> bytes:
    if result.channels != 1:
        raise ApiError("unsupported_audio_channels", "Audio channel layout is unsupported.", status_code=500)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(result.channels)
        output.setsampwidth(2)
        output.setframerate(result.sample_rate_hz)
        output.writeframes(_pcm16_bytes(result.samples))
    return buffer.getvalue()


def _default_workers(config: RuntimeConfig) -> tuple[WorkerClient, ...]:
    return build_worker_clients(config)


def _public_error(error: ApiError, request_id: str | None = None) -> JSONResponse:
    payload = {
        "error": {
            "code": error.code,
            "message": error.message,
            "retryable": error.retryable,
            "request_id": request_id,
        }
    }
    return JSONResponse(payload, status_code=error.status_code)


def _map_scheduler_error(error: SchedulerAdmissionError) -> ApiError:
    mapping = {
        "duplicate_request_id": (409, False, "Request ID is already in use."),
        "queue_full": (429, True, "Request could not be admitted."),
        "scheduler_closed": (503, True, "Service is unavailable."),
        "no_healthy_worker": (503, True, "Service is unavailable."),
    }
    status, retryable, message = mapping.get(
        error.code,
        (503, error.retryable, "Service is unavailable."),
    )
    return ApiError(error.code, message, status_code=status, retryable=retryable)


def _message_to_result(message: WorkerMessage) -> AudioResult:
    if message.kind == "result" and isinstance(message.payload, AudioResult):
        return message.payload
    if message.kind == "error" and message.error is not None:
        status = 503 if message.error.retryable else 500
        raise ApiError(
            message.error.code,
            message.error.message,
            status_code=status,
            retryable=message.error.retryable,
        )
    raise ApiError("invalid_worker_result", "Worker returned an invalid result.", status_code=500)


def create_app(
    config: RuntimeConfig | None = None,
    *,
    worker_factory: Callable[[RuntimeConfig], WorkerClient | tuple[WorkerClient, ...]] | None = None,
) -> FastAPI:
    runtime_config = RuntimeConfig.from_env() if config is None else config
    validate_bind_auth_policy(runtime_config)
    if runtime_config.max_queue_size is None:
        raise ApiError(
            "max_queue_size_required",
            "VOXCPM_MAX_QUEUE_SIZE must be configured for the API runtime.",
            status_code=500,
        )
    factory = _default_workers if worker_factory is None else worker_factory

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        created = factory(runtime_config)
        workers = created if isinstance(created, tuple) else (created,)
        start_worker_clients(workers)
        runtime = ApiRuntime(
            workers,
            max_pending_requests=runtime_config.max_queue_size,
            stream_queue_chunks=runtime_config.stream_ipc_max_chunks or 4,
        )
        app.state.runtime = runtime
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False
            await asyncio.to_thread(runtime.close)

    app = FastAPI(
        title="OpenBMB VoxCPM2 Inference",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.ready = False

    @app.exception_handler(ApiError)
    async def api_error_handler(_request: Request, error: ApiError):
        return _public_error(error)

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        runtime: ApiRuntime | None = getattr(app.state, "runtime", None)
        if not app.state.ready or runtime is None:
            return JSONResponse({"status": "not_ready"}, status_code=503)
        snapshot: SchedulerSnapshot = runtime.snapshot()
        ready = snapshot.ready_workers > 0 or snapshot.busy_workers > 0
        if not ready:
            return JSONResponse(
                {
                    "status": "not_ready",
                    "workers": {
                        "ready": snapshot.ready_workers,
                        "busy": snapshot.busy_workers,
                        "failed": snapshot.failed_workers,
                    },
                },
                status_code=503,
            )
        return {
            "status": "ready",
            "workers": {
                "ready": snapshot.ready_workers,
                "busy": snapshot.busy_workers,
                "failed": snapshot.failed_workers,
            },
        }

    @app.post("/v1/tts")
    async def tts(
        body: TtsRequestModel,
        authorization: str | None = Header(default=None),
        x_request_id: str | None = Header(default=None),
    ):
        request_id = _safe_request_id(x_request_id)
        verify_bearer_token(runtime_config, authorization)
        text = body.text.strip()
        if not text:
            raise ApiError("invalid_text", "Text must not be blank.", status_code=422)
        if runtime_config.max_text_chars is not None and len(text) > runtime_config.max_text_chars:
            raise ApiError("text_too_long", "Text exceeds the configured limit.", status_code=422)
        runtime: ApiRuntime = app.state.runtime
        try:
            future = runtime.submit(
                WorkerRequest(request_id, "synthesize", SpeechRequest(text))
            )
        except SchedulerAdmissionError as error:
            raise _map_scheduler_error(error) from None
        try:
            message = await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            try:
                await asyncio.to_thread(runtime.cancel, request_id)
            except Exception:
                pass
            raise
        result = _message_to_result(message)
        return Response(
            content=_wav_bytes(result),
            media_type="audio/wav",
            headers={"X-Request-ID": request_id, "Cache-Control": "no-store"},
        )

    @app.post("/v1/tts/stream")
    async def tts_stream(
        body: TtsRequestModel,
        authorization: str | None = Header(default=None),
        x_request_id: str | None = Header(default=None),
    ):
        request_id = _safe_request_id(x_request_id)
        verify_bearer_token(runtime_config, authorization)
        text = body.text.strip()
        if not text:
            raise ApiError("invalid_text", "Text must not be blank.", status_code=422)
        if runtime_config.max_text_chars is not None and len(text) > runtime_config.max_text_chars:
            raise ApiError("text_too_long", "Text exceeds the configured limit.", status_code=422)
        runtime: ApiRuntime = app.state.runtime
        try:
            stream_queue = runtime.submit_stream(
                WorkerRequest(
                    request_id,
                    "stream",
                    StreamRequest(SpeechRequest(text)),
                )
            )
        except SchedulerAdmissionError as error:
            raise _map_scheduler_error(error) from None

        async def audio_bytes():
            completed = False
            try:
                while True:
                    message = await asyncio.to_thread(stream_queue.get)
                    if message.kind == "stream_chunk":
                        chunk = message.payload
                        if not isinstance(chunk, AudioChunk):
                            raise ApiError(
                                "invalid_stream_chunk",
                                "Worker returned an invalid stream chunk.",
                                status_code=500,
                            )
                        yield _pcm16_bytes(chunk.samples)
                        continue
                    if message.kind == "stream_end":
                        completed = True
                        return
                    if message.kind == "error" and message.error is not None:
                        raise ApiError(
                            message.error.code,
                            message.error.message,
                            status_code=503 if message.error.retryable else 500,
                            retryable=message.error.retryable,
                        )
                    raise ApiError(
                        "invalid_stream_message",
                        "Worker returned an invalid stream message.",
                        status_code=500,
                    )
            finally:
                if not completed:
                    try:
                        await asyncio.to_thread(runtime.cancel, request_id)
                    except Exception:
                        pass

        return StreamingResponse(
            audio_bytes(),
            media_type="application/octet-stream",
            headers={
                "X-Request-ID": request_id,
                "X-Audio-Format": runtime_config.stream_format.value,
                "X-Audio-Sample-Rate": str(runtime_config.stream_sample_rate),
                "X-Audio-Channels": str(runtime_config.stream_channels),
                "Cache-Control": "no-store",
            },
        )

    return app


__all__: Final[tuple[str, ...]] = ("TtsRequestModel", "create_app")
