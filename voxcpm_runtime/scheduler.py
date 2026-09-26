from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Final, Literal

from voxcpm_runtime.worker_client import WorkerClient, WorkerState
from voxcpm_runtime.worker_errors import (
    SchedulerAdmissionError,
    WorkerError,
    WorkerExitedError,
    WorkerStateError,
    WorkerTransportError,
)
from voxcpm_runtime.worker_types import WorkerErrorData, WorkerMessage, WorkerRequest

SubmissionState = Literal["dispatched", "queued"]


@dataclass(frozen=True, slots=True)
class SchedulerSnapshot:
    state: str
    worker_count: int
    ready_workers: int
    busy_workers: int
    failed_workers: int
    stopped_workers: int
    pending_requests: int
    max_pending_requests: int


class Scheduler:
    def __init__(
        self,
        workers: tuple[WorkerClient, ...],
        *,
        max_pending_requests: int,
    ) -> None:
        if not isinstance(workers, tuple) or not workers:
            raise ValueError("workers must be a non-empty tuple")
        if len({worker.worker_id for worker in workers}) != len(workers):
            raise ValueError("worker ids must be unique")
        if any(worker.state is not WorkerState.READY for worker in workers):
            raise ValueError("all scheduler workers must be ready")
        if (
            isinstance(max_pending_requests, bool)
            or not isinstance(max_pending_requests, int)
            or max_pending_requests <= 0
        ):
            raise ValueError("max_pending_requests must be a positive integer")
        self._workers = tuple(sorted(workers, key=lambda item: item.worker_id))
        self._max_pending = max_pending_requests
        self._pending: deque[WorkerRequest] = deque()
        self._active: dict[str, str] = {}
        self._submitted_ids: set[str] = set()
        self._events: deque[WorkerMessage] = deque()
        self._closed = False

    @property
    def workers(self) -> tuple[WorkerClient, ...]:
        return self._workers

    def snapshot(self) -> SchedulerSnapshot:
        states = [worker.state for worker in self._workers]
        return SchedulerSnapshot(
            state="closed" if self._closed else "open",
            worker_count=len(self._workers),
            ready_workers=sum(state is WorkerState.READY for state in states),
            busy_workers=sum(state is WorkerState.BUSY for state in states),
            failed_workers=sum(state is WorkerState.FAILED for state in states),
            stopped_workers=sum(state is WorkerState.STOPPED for state in states),
            pending_requests=len(self._pending),
            max_pending_requests=self._max_pending,
        )

    def submit(self, request: WorkerRequest) -> SubmissionState:
        if self._closed:
            raise SchedulerAdmissionError(code="scheduler_closed")
        if not isinstance(request, WorkerRequest):
            raise TypeError("request must be WorkerRequest")
        if request.request_id in self._submitted_ids:
            raise SchedulerAdmissionError(
                code="duplicate_request_id",
                details={"request_id": request.request_id},
            )
        ready = self._ready_workers()
        if ready:
            worker = ready[0]
            worker.submit(request)
            self._active[worker.worker_id] = request.request_id
            self._submitted_ids.add(request.request_id)
            return "dispatched"
        if len(self._pending) >= self._max_pending:
            raise SchedulerAdmissionError(
                code="queue_full",
                retryable=True,
                details={
                    "max_pending_requests": self._max_pending,
                    "pending_requests": len(self._pending),
                },
            )
        if not self._healthy_workers():
            raise SchedulerAdmissionError(code="no_healthy_worker", retryable=True)
        self._pending.append(request)
        self._submitted_ids.add(request.request_id)
        return "queued"

    def poll(self, *, timeout_seconds: float = 0.0) -> tuple[WorkerMessage, ...]:
        if self._events:
            return self._drain_events()
        for worker in self._workers:
            request_id = self._active.get(worker.worker_id)
            if request_id is None:
                continue
            try:
                message = worker.receive(timeout_seconds=timeout_seconds)
            except WorkerTransportError as error:
                if error.code == "worker_receive_timeout":
                    continue
                self._worker_failed(worker, request_id, error.code)
                continue
            except WorkerExitedError:
                self._worker_failed(worker, request_id, "worker_exited")
                continue
            self._events.append(message)
            if message.kind in {"result", "error", "stream_end", "cancelled"}:
                self._active.pop(worker.worker_id, None)
                self._dispatch_pending()
        self._fail_pending_without_healthy_workers()
        return self._drain_events()

    def cancel(self, request_id: str) -> WorkerMessage:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be non-empty")
        for index, request in enumerate(self._pending):
            if request.request_id == request_id:
                del self._pending[index]
                return WorkerMessage(
                    kind="cancelled",
                    worker_id="scheduler",
                    request_id=request_id,
                    metadata=(("stage", "pending"),),
                )
        for worker in self._workers:
            if self._active.get(worker.worker_id) != request_id:
                continue
            cancelled = worker.cancel_active()
            self._active.pop(worker.worker_id, None)
            event = WorkerMessage(
                kind="cancelled",
                worker_id=worker.worker_id,
                request_id=cancelled,
                metadata=(("stage", "active"),),
            )
            self._fail_pending_without_healthy_workers()
            return event
        raise SchedulerAdmissionError(
            code="request_not_found",
            details={"request_id": request_id},
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        while self._pending:
            request = self._pending.popleft()
            self._events.append(
                WorkerMessage(
                    kind="cancelled",
                    worker_id="scheduler",
                    request_id=request.request_id,
                    metadata=(("stage", "shutdown"),),
                )
            )
        for worker in self._workers:
            try:
                worker.shutdown()
            except WorkerError:
                pass
        self._active.clear()

    def _ready_workers(self) -> list[WorkerClient]:
        return [worker for worker in self._workers if worker.state is WorkerState.READY]

    def _healthy_workers(self) -> list[WorkerClient]:
        return [
            worker
            for worker in self._workers
            if worker.state
            not in {WorkerState.FAILED, WorkerState.STOPPED, WorkerState.STOPPING}
        ]

    def _dispatch_pending(self) -> None:
        while self._pending:
            ready = self._ready_workers()
            if not ready:
                return
            worker = ready[0]
            request = self._pending.popleft()
            try:
                worker.submit(request)
            except WorkerStateError:
                self._pending.appendleft(request)
                return
            self._active[worker.worker_id] = request.request_id

    def _worker_failed(
        self,
        worker: WorkerClient,
        request_id: str,
        code: str,
    ) -> None:
        self._active.pop(worker.worker_id, None)
        self._events.append(
            WorkerMessage(
                kind="error",
                worker_id=worker.worker_id,
                request_id=request_id,
                error=WorkerErrorData(
                    code=code,
                    message="Worker process is unavailable.",
                    retryable=True,
                ),
            )
        )

    def _fail_pending_without_healthy_workers(self) -> None:
        if self._healthy_workers():
            return
        while self._pending:
            request = self._pending.popleft()
            self._events.append(
                WorkerMessage(
                    kind="error",
                    worker_id="scheduler",
                    request_id=request.request_id,
                    error=WorkerErrorData(
                        code="no_healthy_worker",
                        message="No healthy worker is available.",
                        retryable=True,
                    ),
                )
            )

    def _drain_events(self) -> tuple[WorkerMessage, ...]:
        events = tuple(self._events)
        self._events.clear()
        return events


__all__: Final[tuple[str, ...]] = (
    "Scheduler",
    "SchedulerSnapshot",
    "SubmissionState",
)
