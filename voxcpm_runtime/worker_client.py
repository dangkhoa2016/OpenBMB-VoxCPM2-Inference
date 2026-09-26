from __future__ import annotations

import multiprocessing as mp
import queue
from collections.abc import Iterator
from enum import Enum
from typing import Final

from voxcpm_runtime.backend_types import AudioChunk, AudioResult
from voxcpm_runtime.worker_errors import (
    WorkerError,
    WorkerExitedError,
    WorkerStartError,
    WorkerStateError,
    WorkerTransportError,
)
from voxcpm_runtime.worker_process import worker_process_main
from voxcpm_runtime.worker_types import WorkerBootstrapSpec, WorkerMessage, WorkerRequest


class WorkerState(Enum):
    CREATED = "created"
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class WorkerClient:
    def __init__(
        self,
        spec: WorkerBootstrapSpec,
        *,
        startup_timeout_seconds: float = 10.0,
        shutdown_timeout_seconds: float = 5.0,
        stream_buffer_chunks: int = 4,
        context: mp.context.BaseContext | None = None,
    ) -> None:
        if not isinstance(spec, WorkerBootstrapSpec):
            raise TypeError("spec must be WorkerBootstrapSpec")
        if startup_timeout_seconds <= 0 or shutdown_timeout_seconds <= 0:
            raise ValueError("worker timeouts must be positive")
        if (
            isinstance(stream_buffer_chunks, bool)
            or not isinstance(stream_buffer_chunks, int)
            or stream_buffer_chunks <= 0
        ):
            raise ValueError("stream_buffer_chunks must be a positive integer")
        self._spec = spec
        self._startup_timeout = float(startup_timeout_seconds)
        self._shutdown_timeout = float(shutdown_timeout_seconds)
        self._stream_buffer_chunks = stream_buffer_chunks
        self._ctx = mp.get_context("spawn") if context is None else context
        if self._ctx.get_start_method() != "spawn":
            raise ValueError("WorkerClient requires the spawn multiprocessing context")
        self._state = WorkerState.CREATED
        self._process: mp.Process | None = None
        self._control = None
        self._results = None
        self._active_request_id: str | None = None
        self._ready_message: WorkerMessage | None = None

    @property
    def worker_id(self) -> str:
        return self._spec.worker_id

    @property
    def state(self) -> WorkerState:
        return self._state

    @property
    def pid(self) -> int | None:
        process = self._process
        return None if process is None else process.pid

    @property
    def active_request_id(self) -> str | None:
        return self._active_request_id

    @property
    def ready_message(self) -> WorkerMessage | None:
        return self._ready_message

    @property
    def is_alive(self) -> bool:
        process = self._process
        return process is not None and process.is_alive()

    def start(self) -> WorkerMessage:
        if self._state is not WorkerState.CREATED:
            raise WorkerStateError(code="worker_already_started")
        parent_control, child_control = self._ctx.Pipe(duplex=True)
        results = self._ctx.Queue(maxsize=self._stream_buffer_chunks)
        process = self._ctx.Process(
            target=worker_process_main,
            args=(child_control, results, self._spec),
            name=f"voxcpm-worker-{self.worker_id}",
            daemon=False,
        )
        self._state = WorkerState.STARTING
        self._control = parent_control
        self._results = results
        self._process = process
        process.start()
        child_control.close()
        try:
            message = self._receive_message(self._startup_timeout)
        except WorkerError:
            self._state = WorkerState.FAILED
            self._terminate_and_join()
            raise
        if message.kind == "error":
            self._state = WorkerState.FAILED
            self._terminate_and_join()
            raise self._error_from_message(message, WorkerStartError)
        if message.kind != "ready":
            self._state = WorkerState.FAILED
            self._terminate_and_join()
            raise WorkerStartError(details={"kind": message.kind})
        self._ready_message = message
        self._state = WorkerState.READY
        return message

    def submit(self, request: WorkerRequest) -> None:
        if self._state is not WorkerState.READY:
            raise WorkerStateError(
                code="worker_not_ready",
                details={"state": self._state.value},
            )
        if not isinstance(request, WorkerRequest):
            raise TypeError("request must be WorkerRequest")
        control = self._control
        if control is None:
            raise WorkerTransportError()
        try:
            control.send(request)
        except (BrokenPipeError, EOFError, OSError):
            self._mark_failed()
            raise WorkerExitedError() from None
        self._active_request_id = request.request_id
        self._state = WorkerState.BUSY

    def receive(self, *, timeout_seconds: float | None = None) -> WorkerMessage:
        if self._state not in {WorkerState.BUSY, WorkerState.STOPPING}:
            raise WorkerStateError(
                code="worker_not_busy",
                details={"state": self._state.value},
            )
        message = self._receive_message(timeout_seconds)
        active = self._active_request_id
        if message.request_id is not None and active is not None:
            if message.request_id != active:
                self._mark_failed()
                raise WorkerTransportError(code="request_id_mismatch")
        if message.kind in {"result", "error", "stream_end", "cancelled"}:
            self._active_request_id = None
            if self._state is not WorkerState.STOPPING:
                self._state = WorkerState.READY
        return message

    def execute(
        self,
        request: WorkerRequest,
        *,
        timeout_seconds: float | None = None,
    ) -> AudioResult:
        if request.operation == "stream":
            raise ValueError("execute does not accept stream requests")
        self.submit(request)
        message = self.receive(timeout_seconds=timeout_seconds)
        if message.kind == "error":
            raise self._error_from_message(message, WorkerError)
        if message.kind != "result" or not isinstance(message.payload, AudioResult):
            self._mark_failed()
            raise WorkerTransportError(code="invalid_worker_result")
        return message.payload

    def stream(
        self,
        request: WorkerRequest,
        *,
        timeout_seconds: float | None = None,
    ) -> Iterator[AudioChunk]:
        if request.operation != "stream":
            raise ValueError("stream requires a stream WorkerRequest")
        self.submit(request)
        try:
            while True:
                message = self.receive(timeout_seconds=timeout_seconds)
                if message.kind == "stream_chunk":
                    if not isinstance(message.payload, AudioChunk):
                        self._mark_failed()
                        raise WorkerTransportError(code="invalid_stream_chunk")
                    yield message.payload
                    continue
                if message.kind == "stream_end":
                    return
                if message.kind == "error":
                    raise self._error_from_message(message, WorkerError)
                self._mark_failed()
                raise WorkerTransportError(
                    code="unexpected_stream_message",
                    details={"kind": message.kind},
                )
        finally:
            if self._state is WorkerState.BUSY:
                self.cancel_active()

    def cancel_active(self) -> str:
        if self._state is not WorkerState.BUSY or self._active_request_id is None:
            raise WorkerStateError(code="no_active_request")
        request_id = self._active_request_id
        self._terminate_and_join()
        self._active_request_id = None
        self._state = WorkerState.STOPPED
        self._cleanup_transport()
        return request_id

    def shutdown(self) -> None:
        if self._state is WorkerState.STOPPED:
            return
        if self._state is WorkerState.CREATED:
            self._state = WorkerState.STOPPED
            return
        if self._state is WorkerState.BUSY:
            self.cancel_active()
            return
        if self._state is WorkerState.FAILED:
            self._terminate_and_join()
            self._cleanup_transport()
            self._state = WorkerState.STOPPED
            return
        control = self._control
        process = self._process
        if control is None or process is None:
            self._state = WorkerState.STOPPED
            self._cleanup_transport()
            return
        self._state = WorkerState.STOPPING
        try:
            control.send("shutdown")
            message = self._receive_message(self._shutdown_timeout)
            if message.kind != "shutdown_ack":
                raise WorkerTransportError(
                    code="invalid_shutdown_ack",
                    details={"kind": message.kind},
                )
        except WorkerError:
            self._terminate_and_join()
        else:
            process.join(self._shutdown_timeout)
            if process.is_alive():
                self._terminate_and_join()
        finally:
            self._state = WorkerState.STOPPED
            self._cleanup_transport()

    close = shutdown

    def _receive_message(self, timeout_seconds: float | None) -> WorkerMessage:
        results = self._results
        process = self._process
        if results is None:
            raise WorkerTransportError()
        timeout = self._startup_timeout if timeout_seconds is None else timeout_seconds
        try:
            message = results.get(timeout=timeout)
        except queue.Empty:
            if process is not None and not process.is_alive():
                self._mark_failed()
                raise WorkerExitedError(details={"exitcode": process.exitcode}) from None
            raise WorkerTransportError(
                code="worker_receive_timeout",
                retryable=True,
            ) from None
        except (EOFError, OSError, ValueError):
            self._mark_failed()
            raise WorkerTransportError() from None
        if not isinstance(message, WorkerMessage):
            self._mark_failed()
            raise WorkerTransportError(code="invalid_worker_message")
        return message

    @staticmethod
    def _error_from_message(
        message: WorkerMessage,
        error_type: type[WorkerError],
    ) -> WorkerError:
        data = message.error
        if data is None:
            return error_type()
        return error_type(
            data.message,
            code=data.code,
            retryable=data.retryable,
            details=dict(data.details),
        )

    def _terminate_and_join(self) -> None:
        process = self._process
        if process is None:
            return
        if process.is_alive():
            process.terminate()
        process.join(self._shutdown_timeout)
        if process.is_alive():
            process.kill()
            process.join(self._shutdown_timeout)

    def _mark_failed(self) -> None:
        self._state = WorkerState.FAILED

    def _cleanup_transport(self) -> None:
        control = self._control
        results = self._results
        self._control = None
        self._results = None
        if control is not None:
            try:
                control.close()
            except OSError:
                pass
        if results is not None:
            try:
                results.close()
                results.join_thread()
            except (OSError, ValueError):
                pass


__all__: Final[tuple[str, ...]] = ("WorkerClient", "WorkerState")
