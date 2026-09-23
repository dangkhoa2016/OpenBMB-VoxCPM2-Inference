import pytest

from voxcpm_runtime.api_errors import ApiError
from voxcpm_runtime.config import RuntimeConfig
from voxcpm_runtime.worker_client import WorkerClient, WorkerState
from voxcpm_runtime.worker_factory import (
    build_worker_clients,
    parent_execution_plan,
    start_worker_clients,
)
from voxcpm_runtime.worker_types import WorkerBootstrapSpec


def _config(**overrides):
    values = dict(
        device="cuda",
        gpu_devices=(0, 1),
        workers=2,
        max_queue_size=2,
    )
    values.update(overrides)
    return RuntimeConfig(**values)


def test_parent_execution_plan_preserves_explicit_gpu_order_without_detection():
    plan = parent_execution_plan(_config(gpu_devices=(1, 0)))
    assert plan.effective_device == "cuda"
    assert plan.selected_gpu_indices == (1, 0)
    assert plan.worker_count == 2
    assert plan.worker_gpu_indices == (1, 0)


def test_parent_execution_plan_requires_explicit_gpu_selection_for_cuda_or_auto():
    for device in ("cuda", "auto"):
        with pytest.raises(ApiError) as caught:
            parent_execution_plan(_config(device=device, gpu_devices=None, workers=None))
        assert caught.value.code == "explicit_gpu_selection_required"


def test_build_worker_clients_maps_one_worker_per_physical_gpu_in_order():
    captured = []

    def builder(spec, **kwargs):
        captured.append((spec, kwargs))
        return WorkerClient(spec, **kwargs)

    workers = build_worker_clients(
        _config(gpu_devices=(1, 0), workers=2, stream_ipc_max_chunks=3),
        client_builder=builder,
    )
    assert [worker.worker_id for worker in workers] == ["worker0", "worker1"]
    assert [item[0].physical_gpu_index for item in captured] == [1, 0]
    assert all(item[1]["stream_buffer_chunks"] == 3 for item in captured)


def test_build_worker_clients_respects_worker_limit():
    captured = []

    def builder(spec, **kwargs):
        captured.append(spec)
        return WorkerClient(spec, **kwargs)

    workers = build_worker_clients(
        _config(gpu_devices=(1, 0), workers=1),
        client_builder=builder,
    )
    assert len(workers) == 1
    assert captured[0].physical_gpu_index == 1


def test_partial_startup_failure_cleans_up_already_started_workers():
    good = WorkerClient(
        WorkerBootstrapSpec(worker_id="worker0", backend_kind="fake"),
        startup_timeout_seconds=5,
        shutdown_timeout_seconds=2,
    )
    bad = WorkerClient(
        WorkerBootstrapSpec(
            worker_id="worker1",
            backend_kind="fake",
            fake_failure_operations=("load",),
        ),
        startup_timeout_seconds=5,
        shutdown_timeout_seconds=2,
    )
    with pytest.raises(Exception):
        start_worker_clients((good, bad))
    assert good.state is WorkerState.STOPPED
    assert bad.state is WorkerState.STOPPED
    assert not good.is_alive
    assert not bad.is_alive


def test_cpu_factory_stays_single_worker():
    workers = build_worker_clients(
        RuntimeConfig(device="cpu", max_queue_size=1),
    )
    assert len(workers) == 1
    assert workers[0].worker_id == "worker0"
