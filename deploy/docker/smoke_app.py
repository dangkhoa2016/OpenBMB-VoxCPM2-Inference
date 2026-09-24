from __future__ import annotations

import os
from pathlib import Path

from voxcpm_runtime.api_app import create_app
from voxcpm_runtime.config import RuntimeConfig
from voxcpm_runtime.worker_client import WorkerClient
from voxcpm_runtime.worker_types import WorkerBootstrapSpec


def _fake_worker(_config: RuntimeConfig) -> WorkerClient:
    return WorkerClient(
        WorkerBootstrapSpec(worker_id="docker-contract-smoke", backend_kind="fake"),
        startup_timeout_seconds=5,
        shutdown_timeout_seconds=2,
        stream_buffer_chunks=2,
    )


def _config() -> RuntimeConfig:
    model_path = Path(os.environ.get("VOXCPM_MODEL_PATH", "/models/VoxCPM2"))
    if not model_path.is_dir():
        raise RuntimeError(f"VOXCPM_MODEL_PATH must be a mounted directory: {model_path}")

    token = os.environ.get("VOXCPM_API_TOKEN")
    if not token:
        raise RuntimeError("VOXCPM_API_TOKEN is required for the Docker contract smoke app")

    return RuntimeConfig(
        model_path=str(model_path),
        profile="cpu",
        device="cpu",
        host="0.0.0.0",
        port=int(os.environ.get("VOXCPM_PORT", "8090")),
        api_token=token,
        require_auth=True,
        allow_unauthenticated_external=False,
        max_text_chars=4096,
        max_queue_size=2,
        stream_ipc_max_chunks=2,
    )


app = create_app(_config(), worker_factory=_fake_worker)
