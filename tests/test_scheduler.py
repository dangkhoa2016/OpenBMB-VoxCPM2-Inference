import os
import signal
import time

import pytest

from voxcpm_runtime.backend_types import SpeechRequest, StreamRequest
from voxcpm_runtime.scheduler import Scheduler
from voxcpm_runtime.worker_client import WorkerClient, WorkerState
from voxcpm_runtime.worker_errors import SchedulerAdmissionError
from voxcpm_runtime.worker_types import WorkerBootstrapSpec, WorkerRequest


def _worker(worker_id="worker0", *, buffer_chunks=1):
    client = WorkerClient(
        WorkerBootstrapSpec(worker_id=worker_id),
        startup_timeout_seconds=5,
        shutdown_timeout_seconds=2,
        stream_buffer_chunks=buffer_chunks,
    )
    client.start()
    return client


def _stream_request(request_id):
    return WorkerRequest(
        request_id,
        "stream",
        StreamRequest(SpeechRequest(f"stream {request_id}")),
    )


def _tts_request(request_id):
    return WorkerRequest(
        request_id,
        "synthesize",
        SpeechRequest(f"speech {request_id}"),
    )


def test_scheduler_requires_ready_workers():
    client = WorkerClient(WorkerBootstrapSpec(worker_id="worker0"))
    with pytest.raises(ValueError, match="ready"):
        Scheduler((client,), max_pending_requests=1)


def test_bounded_admission_and_duplicate_ids():
    worker = _worker()
    scheduler = Scheduler((worker,), max_pending_requests=1)
    assert scheduler.submit(_stream_request("a")) == "dispatched"
    assert scheduler.submit(_tts_request("b")) == "queued"
    with pytest.raises(SchedulerAdmissionError) as full:
        scheduler.submit(_tts_request("c"))
    assert full.value.code == "queue_full"
    with pytest.raises(SchedulerAdmissionError) as duplicate:
        scheduler.submit(_tts_request("b"))
    assert duplicate.value.code == "duplicate_request_id"
    scheduler.close()
    assert not worker.is_alive


def test_pending_cancel_removes_request_without_touching_active_worker():
    worker = _worker()
    scheduler = Scheduler((worker,), max_pending_requests=2)
    scheduler.submit(_stream_request("a"))
    scheduler.submit(_tts_request("b"))
    event = scheduler.cancel("b")
    assert event.kind == "cancelled"
    assert event.request_id == "b"
    assert dict(event.metadata)["stage"] == "pending"
    assert scheduler.snapshot().pending_requests == 0
    assert worker.state is WorkerState.BUSY
    scheduler.close()


def test_active_cancel_terminates_worker_and_fails_pending():
    worker = _worker()
    scheduler = Scheduler((worker,), max_pending_requests=2)
    scheduler.submit(_stream_request("a"))
    scheduler.submit(_tts_request("b"))
    event = scheduler.cancel("a")
    assert event.kind == "cancelled"
    assert dict(event.metadata)["stage"] == "active"
    assert worker.state is WorkerState.STOPPED
    assert not worker.is_alive
    events = scheduler.poll()
    assert len(events) == 1
    assert events[0].request_id == "b"
    assert events[0].kind == "error"
    assert events[0].error is not None
    assert events[0].error.code == "no_healthy_worker"
    scheduler.close()


def test_fifo_dispatch_order_with_one_worker():
    worker = _worker(buffer_chunks=1)
    scheduler = Scheduler((worker,), max_pending_requests=3)
    scheduler.submit(_stream_request("a"))
    scheduler.submit(_tts_request("b"))
    scheduler.submit(_tts_request("c"))

    terminal = []
    deadline = time.monotonic() + 5
    while len(terminal) < 3 and time.monotonic() < deadline:
        for event in scheduler.poll(timeout_seconds=0.2):
            if event.kind in {"stream_end", "result", "error", "cancelled"}:
                terminal.append(event.request_id)
    assert terminal == ["a", "b", "c"]
    scheduler.close()


def test_completed_request_id_cannot_be_reused():
    worker = _worker()
    scheduler = Scheduler((worker,), max_pending_requests=1)
    scheduler.submit(_tts_request("once"))
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        events = scheduler.poll(timeout_seconds=0.2)
        if any(event.kind == "result" for event in events):
            break
    with pytest.raises(SchedulerAdmissionError) as caught:
        scheduler.submit(_tts_request("once"))
    assert caught.value.code == "duplicate_request_id"
    scheduler.close()


@pytest.mark.skipif(os.name == "nt", reason="signal-based crash test is POSIX-only")
def test_worker_crash_is_contained_and_pending_is_failed():
    worker = _worker(buffer_chunks=1)
    scheduler = Scheduler((worker,), max_pending_requests=2)
    scheduler.submit(_stream_request("a"))
    scheduler.submit(_tts_request("b"))
    assert worker.pid is not None
    time.sleep(0.05)
    os.kill(worker.pid, signal.SIGKILL)

    codes = {}
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline and len(codes) < 2:
        for event in scheduler.poll(timeout_seconds=0.2):
            if event.kind == "error" and event.error is not None:
                codes[event.request_id] = event.error.code
    assert codes["a"] == "worker_exited"
    assert codes["b"] == "no_healthy_worker"
    scheduler.close()
    assert not worker.is_alive


def test_snapshot_and_close_are_deterministic_and_idempotent():
    worker = _worker()
    scheduler = Scheduler((worker,), max_pending_requests=2)
    before = scheduler.snapshot()
    assert before.state == "open"
    assert before.ready_workers == 1
    assert before.pending_requests == 0
    scheduler.close()
    scheduler.close()
    after = scheduler.snapshot()
    assert after.state == "closed"
    assert after.stopped_workers == 1
    assert not worker.is_alive
    with pytest.raises(SchedulerAdmissionError) as caught:
        scheduler.submit(_tts_request("late"))
    assert caught.value.code == "scheduler_closed"


def test_two_workers_accept_two_active_requests_and_queue_third():
    worker0 = _worker("worker0", buffer_chunks=1)
    worker1 = _worker("worker1", buffer_chunks=1)
    scheduler = Scheduler((worker0, worker1), max_pending_requests=2)
    assert scheduler.submit(_stream_request("a")) == "dispatched"
    assert scheduler.submit(_stream_request("b")) == "dispatched"
    assert scheduler.submit(_tts_request("c")) == "queued"
    snapshot = scheduler.snapshot()
    assert snapshot.busy_workers == 2
    assert snapshot.pending_requests == 1
    scheduler.close()
    assert not worker0.is_alive
    assert not worker1.is_alive


def test_single_worker_cancel_preserves_other_worker_and_pending_work():
    worker0 = _worker("worker0", buffer_chunks=1)
    worker1 = _worker("worker1", buffer_chunks=1)
    scheduler = Scheduler((worker0, worker1), max_pending_requests=2)
    scheduler.submit(_stream_request("a"))
    scheduler.submit(_stream_request("b"))
    scheduler.submit(_tts_request("c"))

    cancelled = scheduler.cancel("a")
    assert cancelled.worker_id == "worker0"
    assert worker0.state is WorkerState.STOPPED
    assert worker1.state is WorkerState.BUSY

    terminal = []
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and "c" not in terminal:
        for event in scheduler.poll(timeout_seconds=0.2):
            if event.kind in {"stream_end", "result", "error", "cancelled"}:
                terminal.append(event.request_id)
    assert "b" in terminal
    assert "c" in terminal
    assert scheduler.snapshot().failed_workers == 0
    scheduler.close()


@pytest.mark.skipif(os.name == "nt", reason="signal-based crash test is POSIX-only")
def test_single_worker_crash_does_not_fail_pending_when_survivor_is_healthy():
    worker0 = _worker("worker0", buffer_chunks=1)
    worker1 = _worker("worker1", buffer_chunks=1)
    scheduler = Scheduler((worker0, worker1), max_pending_requests=2)
    scheduler.submit(_stream_request("a"))
    scheduler.submit(_stream_request("b"))
    scheduler.submit(_tts_request("c"))
    assert worker0.pid is not None
    os.kill(worker0.pid, signal.SIGKILL)

    events = []
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        events.extend(scheduler.poll(timeout_seconds=0.2))
        if any(event.request_id == "c" and event.kind == "result" for event in events):
            break

    by_request = {}
    for event in events:
        if event.request_id is not None:
            by_request.setdefault(event.request_id, []).append(event)
    assert any(
        event.kind == "error"
        and event.error is not None
        and event.error.code == "worker_exited"
        for event in by_request["a"]
    )
    assert not any(
        event.kind == "error"
        and event.error is not None
        and event.error.code == "no_healthy_worker"
        for event in by_request.get("c", [])
    )
    assert any(event.kind == "result" for event in by_request["c"])
    scheduler.close()
