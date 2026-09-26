from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from typing import Final

from voxcpm_runtime.scheduler import Scheduler
from voxcpm_runtime.worker_client import WorkerClient
from voxcpm_runtime.worker_errors import SchedulerAdmissionError, WorkerError
from voxcpm_runtime.worker_types import WorkerMessage, WorkerRequest


class ApiRuntime:
    def __init__(self, worker: WorkerClient, *, max_pending_requests: int) -> None:
        self._worker = worker
        self._scheduler = Scheduler((worker,), max_pending_requests=max_pending_requests)
        self._lock = threading.RLock()
        self._futures: dict[str, Future[WorkerMessage]] = {}
        self._closed = False
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._dispatch_loop,
            name="voxcpm-api-dispatcher",
            daemon=False,
        )
        self._thread.start()

    @property
    def worker(self) -> WorkerClient:
        return self._worker

    def snapshot(self):
        with self._lock:
            return self._scheduler.snapshot()

    def submit(self, request: WorkerRequest) -> Future[WorkerMessage]:
        future: Future[WorkerMessage] = Future()
        with self._lock:
            if self._closed:
                raise SchedulerAdmissionError(code="scheduler_closed")
            self._scheduler.submit(request)
            self._futures[request.request_id] = future
        return future

    def cancel(self, request_id: str) -> WorkerMessage:
        with self._lock:
            event = self._scheduler.cancel(request_id)
            future = self._futures.pop(request_id, None)
            if future is not None and not future.done():
                future.cancel()
            return event

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._stop.set()
        self._thread.join(timeout=2.0)
        with self._lock:
            self._scheduler.close()
            for future in self._futures.values():
                if not future.done():
                    future.cancel()
            self._futures.clear()

    def _dispatch_loop(self) -> None:
        while not self._stop.is_set():
            try:
                with self._lock:
                    events = self._scheduler.poll(timeout_seconds=0.05)
            except WorkerError:
                events = ()
            for event in events:
                if event.request_id is None:
                    continue
                if event.kind not in {"result", "error", "cancelled"}:
                    continue
                with self._lock:
                    future = self._futures.pop(event.request_id, None)
                if future is not None and not future.done():
                    future.set_result(event)
            if not events:
                time.sleep(0.005)


__all__: Final[tuple[str, ...]] = ("ApiRuntime",)
