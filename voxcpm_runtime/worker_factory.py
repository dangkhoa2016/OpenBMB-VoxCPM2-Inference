from __future__ import annotations

from collections.abc import Callable
from typing import Final

from voxcpm_runtime.api_errors import ApiError
from voxcpm_runtime.config import RuntimeConfig
from voxcpm_runtime.device import ExecutionPlan
from voxcpm_runtime.profiles import ProfileResolutionError, materialize_execution_profile
from voxcpm_runtime.worker_client import WorkerClient
from voxcpm_runtime.worker_types import WorkerBootstrapSpec


def parent_execution_plan(config: RuntimeConfig) -> ExecutionPlan:
    """Build the API-parent topology without importing Torch."""
    if not isinstance(config, RuntimeConfig):
        raise TypeError("config must be RuntimeConfig")

    try:
        config = materialize_execution_profile(config)
    except ProfileResolutionError as error:
        raise ApiError(
            "invalid_execution_profile",
            "API execution profile could not be resolved.",
            status_code=500,
        ) from error

    device = config.device.value
    if device == "cpu":
        if config.gpu_devices:
            raise ApiError(
                "cpu_gpu_selection_conflict",
                "CPU API runtime cannot select GPUs.",
                status_code=500,
            )
        if config.workers not in (None, 1):
            raise ApiError(
                "cpu_worker_count_invalid",
                "CPU API runtime supports exactly one worker.",
                status_code=500,
            )
        return ExecutionPlan(effective_device="cpu", worker_count=1)

    if device not in {"cuda", "auto"}:
        raise ApiError("invalid_device", "Unsupported API runtime device.", status_code=500)

    selected = config.gpu_devices
    if selected is None or not selected:
        raise ApiError(
            "explicit_gpu_selection_required",
            "API CUDA runtime requires an explicit GPU selection.",
            status_code=500,
        )
    workers = len(selected) if config.workers is None else config.workers
    try:
        return ExecutionPlan(
            effective_device="cuda",
            selected_gpu_indices=selected,
            worker_count=workers,
            worker_gpu_indices=selected[:workers],
        )
    except ValueError as error:
        raise ApiError(
            "invalid_worker_topology",
            "API worker topology is invalid.",
            status_code=500,
        ) from error


def build_worker_clients(
    config: RuntimeConfig,
    *,
    client_builder: Callable[..., WorkerClient] = WorkerClient,
) -> tuple[WorkerClient, ...]:
    plan = parent_execution_plan(config)
    stream_buffer_chunks = config.stream_ipc_max_chunks or 4

    if plan.effective_device == "cpu":
        specs = (
            WorkerBootstrapSpec(
                worker_id="worker0",
                backend_kind="real",
                physical_gpu_index=None,
            ),
        )
    else:
        specs = tuple(
            WorkerBootstrapSpec(
                worker_id=f"worker{index}",
                backend_kind="real",
                physical_gpu_index=physical_gpu,
            )
            for index, physical_gpu in enumerate(plan.worker_gpu_indices)
        )

    return tuple(
        client_builder(
            spec,
            startup_timeout_seconds=90.0,
            shutdown_timeout_seconds=10.0,
            stream_buffer_chunks=stream_buffer_chunks,
        )
        for spec in specs
    )


def start_worker_clients(workers: tuple[WorkerClient, ...]) -> tuple[WorkerClient, ...]:
    if not isinstance(workers, tuple) or not workers:
        raise ValueError("workers must be a non-empty tuple")
    started: list[WorkerClient] = []
    try:
        for worker in workers:
            worker.start()
            started.append(worker)
    except BaseException:
        for worker in reversed(started):
            try:
                worker.shutdown()
            except BaseException:
                pass
        for worker in workers[len(started) :]:
            try:
                worker.shutdown()
            except BaseException:
                pass
        raise
    return workers


__all__: Final[tuple[str, ...]] = (
    "build_worker_clients",
    "parent_execution_plan",
    "start_worker_clients",
)
