from __future__ import annotations

import os
from collections.abc import Iterator
from multiprocessing.connection import Connection
from queue import Full
from typing import Any, Final

from voxcpm_runtime.backend import InferenceBackend
from voxcpm_runtime.backend_types import AudioChunk, AudioResult, BackendInfo
from voxcpm_runtime.errors import BackendError
from voxcpm_runtime.worker_types import (
    WorkerBootstrapSpec,
    WorkerErrorData,
    WorkerMessage,
    WorkerRequest,
)


def _backend_error_data(error: BackendError) -> WorkerErrorData:
    payload = error.to_dict()
    details = tuple(sorted(payload["details"].items()))
    return WorkerErrorData(
        code=str(payload["code"]),
        message=str(payload["message"]),
        retryable=bool(payload["retryable"]),
        details=details,
    )


def _unexpected_error_data(error: BaseException) -> WorkerErrorData:
    name = type(error).__name__
    safe_name = name if name.isidentifier() and len(name) <= 64 else "UnknownWorkerError"
    return WorkerErrorData(
        code="worker_execution_failed",
        message="Worker execution failed.",
        retryable=False,
        details=(("exception_type", safe_name),),
    )


def _create_fake_backend(spec: WorkerBootstrapSpec) -> InferenceBackend:
    from voxcpm_runtime.fake_backend import FakeFailurePlan, FakeVoxCPMBackend

    return FakeVoxCPMBackend(
        failure_plan=FakeFailurePlan(spec.fake_failure_operations),
    )


def _create_real_backend(spec: WorkerBootstrapSpec) -> InferenceBackend:
    if spec.physical_gpu_index is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(spec.physical_gpu_index)
        os.environ["VOXCPM_DEVICE"] = "cuda"
        os.environ["VOXCPM_GPU_DEVICES"] = "0"
        os.environ["VOXCPM_WORKERS"] = "1"

    from voxcpm_runtime.config import RuntimeConfig
    from voxcpm_runtime.device import DeviceManager
    from voxcpm_runtime.model_resolver import ModelResolver
    from voxcpm_runtime.pytorch_backend import (
        PytorchVoxCPMBackend,
        resolve_upstream_optimize,
    )

    config = RuntimeConfig.from_env()
    manager = DeviceManager()
    inventory = manager.detect()
    plan = manager.resolve(
        config,
        gpu_devices=config.gpu_devices,
        workers=config.workers,
        inventory=inventory,
    )
    resolved = ModelResolver().resolve(config)
    return PytorchVoxCPMBackend(
        resolved_model=resolved,
        execution_plan=plan,
        load_denoiser=config.load_denoiser,
        local_files_only=config.offline,
        optimize=resolve_upstream_optimize(plan, config.optimize),
    )


def _create_backend(spec: WorkerBootstrapSpec) -> InferenceBackend:
    if spec.backend_kind == "fake":
        return _create_fake_backend(spec)
    return _create_real_backend(spec)


def _ready_metadata(info: BackendInfo) -> tuple[tuple[str, Any], ...]:
    return (
        ("backend", info.backend),
        ("implementation", info.implementation),
        ("device", info.device),
        ("model_id", info.model_id),
        ("capabilities", ",".join(info.capabilities)),
    )


def _put_message(result_queue: Any, message: WorkerMessage) -> None:
    try:
        result_queue.put(message, block=True)
    except (BrokenPipeError, EOFError, OSError, ValueError, Full):
        raise SystemExit(1) from None


def _execute_request(
    backend: InferenceBackend,
    request: WorkerRequest,
    result_queue: Any,
    worker_id: str,
) -> None:
    if request.operation == "stream":
        iterator: Iterator[AudioChunk] | None = None
        try:
            iterator = backend.stream(request.payload)
            for chunk in iterator:
                _put_message(
                    result_queue,
                    WorkerMessage(
                        kind="stream_chunk",
                        worker_id=worker_id,
                        request_id=request.request_id,
                        payload=chunk,
                    ),
                )
            _put_message(
                result_queue,
                WorkerMessage(
                    kind="stream_end",
                    worker_id=worker_id,
                    request_id=request.request_id,
                ),
            )
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
        return

    method = getattr(backend, request.operation)
    result = method(request.payload)
    if not isinstance(result, AudioResult):
        raise TypeError("worker backend returned an invalid result")
    _put_message(
        result_queue,
        WorkerMessage(
            kind="result",
            worker_id=worker_id,
            request_id=request.request_id,
            payload=result,
        ),
    )


def worker_process_main(
    control: Connection,
    result_queue: Any,
    spec: WorkerBootstrapSpec,
) -> None:
    backend: InferenceBackend | None = None
    try:
        backend = _create_backend(spec)
        info = backend.load()
        _put_message(
            result_queue,
            WorkerMessage(
                kind="ready",
                worker_id=spec.worker_id,
                metadata=_ready_metadata(info),
            ),
        )
        while True:
            command = control.recv()
            if command == "shutdown":
                backend.close()
                _put_message(
                    result_queue,
                    WorkerMessage(kind="shutdown_ack", worker_id=spec.worker_id),
                )
                return
            if not isinstance(command, WorkerRequest):
                raise TypeError("worker control message is invalid")
            try:
                _execute_request(backend, command, result_queue, spec.worker_id)
            except BackendError as error:
                _put_message(
                    result_queue,
                    WorkerMessage(
                        kind="error",
                        worker_id=spec.worker_id,
                        request_id=command.request_id,
                        error=_backend_error_data(error),
                    ),
                )
            except BaseException as error:
                _put_message(
                    result_queue,
                    WorkerMessage(
                        kind="error",
                        worker_id=spec.worker_id,
                        request_id=command.request_id,
                        error=_unexpected_error_data(error),
                    ),
                )
    except EOFError:
        return
    except BaseException as error:
        try:
            _put_message(
                result_queue,
                WorkerMessage(
                    kind="error",
                    worker_id=spec.worker_id,
                    error=_unexpected_error_data(error),
                ),
            )
        except BaseException:
            pass
    finally:
        if backend is not None:
            try:
                backend.close()
            except BaseException:
                pass
        try:
            control.close()
        except OSError:
            pass


__all__: Final[tuple[str, ...]] = ("worker_process_main",)
