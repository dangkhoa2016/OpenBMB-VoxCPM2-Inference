from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from typing import Final

from voxcpm_runtime.scheduler import Scheduler
from voxcpm_runtime.worker_client import WorkerClient
from voxcpm_runtime.worker_errors import SchedulerAdmissionError, WorkerError
from voxcpm_runtime.worker_types import WorkerMessage, WorkerRequest


class ApiRuntime:
    def __init__(
        self,
        workers: tuple[WorkerClient, ...],
        *,
        max_pending_requests: int,
        stream_queue_chunks: int = 4,
    ) -> None:
        if (
            isinstance(stream_queue_chunks, bool)
            or not isinstance(stream_queue_chunks, int)
            or stream_queue_chunks <= 0
        ):
            raise ValueError("stream_queue_chunks must be a positive integer")
        if not isinstance(workers, tuple) or not workers:
            raise ValueError("workers must be a non-empty tuple")
        self._workers = workers
        self._scheduler = Scheduler(workers, max_pending_requests=max_pending_requests)
        self._stream_queue_chunks = stream_queue_chunks
        self._lock = threading.RLock()
        self._futures: dict[str, Future[WorkerMessage]] = {}
        self._streams: dict[str, queue.Queue[WorkerMessage]] = {}
        self._closed = False
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._dispatch_loop,
            name="voxcpm-api-dispatcher",
            daemon=False,
        )
        self._thread.start()

    @property
    def workers(self) -> tuple[WorkerClient, ...]:
        return self._workers

    @property
    def worker(self) -> WorkerClient:
        if len(self._workers) != 1:
            raise RuntimeError("worker property requires exactly one worker")
        return self._workers[0]

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

    def submit_stream(self, request: WorkerRequest) -> queue.Queue[WorkerMessage]:
        if request.operation != "stream":
            raise ValueError("submit_stream requires a stream request")
        stream_queue: queue.Queue[WorkerMessage] = queue.Queue(
            maxsize=self._stream_queue_chunks
        )
        with self._lock:
            if self._closed:
                raise SchedulerAdmissionError(code="scheduler_closed")
            self._scheduler.submit(request)
            self._streams[request.request_id] = stream_queue
        return stream_queue

    def cancel(self, request_id: str) -> WorkerMessage:
        with self._lock:
            event = self._scheduler.cancel(request_id)
            future = self._futures.pop(request_id, None)
            self._streams.pop(request_id, None)
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
            self._streams.clear()

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
                if event.kind in {"stream_chunk", "stream_end", "error", "cancelled"}:
                    if self._route_stream_event(event):
                        continue
                if event.kind not in {"result", "error", "cancelled"}:
                    continue
                with self._lock:
                    future = self._futures.pop(event.request_id, None)
                if future is not None and not future.done():
                    future.set_result(event)
            if not events:
                time.sleep(0.005)

    def _route_stream_event(self, event: WorkerMessage) -> bool:
        request_id = event.request_id
        if request_id is None:
            return False
        with self._lock:
            stream_queue = self._streams.get(request_id)
        if stream_queue is None:
            return False
        while not self._stop.is_set():
            with self._lock:
                if request_id not in self._streams:
                    return True
            try:
                stream_queue.put(event, timeout=0.05)
                break
            except queue.Full:
                continue
        if event.kind in {"stream_end", "error", "cancelled"}:
            with self._lock:
                self._streams.pop(request_id, None)
        return True


__all__: Final[tuple[str, ...]] = ("ApiRuntime",)
