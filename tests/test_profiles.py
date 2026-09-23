import io
import json
import subprocess
import wave

import pytest
from fastapi.testclient import TestClient

from voxcpm_runtime.api_app import create_app
from voxcpm_runtime.config import ExecutionProfile, RuntimeConfig
from voxcpm_runtime.device import GPUInfo, HardwareInventory, resolve_execution_plan
from voxcpm_runtime.profiles import (
    ProfileResolutionError,
    materialize_execution_profile,
    probe_runtime_gpu_indices,
)
from voxcpm_runtime.worker_client import WorkerClient
from voxcpm_runtime.worker_types import WorkerBootstrapSpec


def _gpu_probe(*indices):
    return lambda: tuple(indices)


def _inventory(count):
    return HardwareInventory(
        cuda_available=count > 0,
        gpus=tuple(
            GPUInfo(index=index, name=f"gpu-{index}", total_memory_bytes=16 * 1024**3)
            for index in range(count)
        ),
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("cpu", ExecutionProfile.CPU),
        ("cuda-single", ExecutionProfile.CUDA_SINGLE),
        ("gpu", ExecutionProfile.CUDA_SINGLE),
        ("cuda-replica", ExecutionProfile.CUDA_REPLICA),
        ("multi-gpu", ExecutionProfile.CUDA_REPLICA),
        ("auto", ExecutionProfile.AUTO),
    ],
)
def test_profile_names_and_aliases_are_frozen(value, expected):
    config = RuntimeConfig.from_mapping({"VOXCPM_PROFILE": value})
    assert config.profile is expected
    assert config.to_redacted_dict()["profile"] == expected.value


@pytest.mark.parametrize("value", ["cuda", "multi", "GPU0", "replica", "banana"])
def test_unknown_profile_fails(value):
    with pytest.raises(ValueError, match="VOXCPM_PROFILE"):
        RuntimeConfig.from_mapping({"VOXCPM_PROFILE": value})


def test_cpu_profile_materializes_single_cpu_worker():
    config = materialize_execution_profile(
        RuntimeConfig(profile="cpu"),
        gpu_probe=_gpu_probe(0, 1),
    )
    assert config.device == "cpu"
    assert config.gpu_devices is None
    assert config.workers == 1


def test_cuda_single_profile_selects_one_gpu_and_never_falls_back():
    config = materialize_execution_profile(
        RuntimeConfig(profile="cuda-single"),
        gpu_probe=_gpu_probe(0, 1),
    )
    assert config.device == "cuda"
    assert config.gpu_devices == (0,)
    assert config.workers == 1

    with pytest.raises(ProfileResolutionError, match="usable CUDA"):
        materialize_execution_profile(
            RuntimeConfig(profile="cuda-single"),
            gpu_probe=_gpu_probe(),
        )


def test_cuda_replica_profile_maps_one_worker_per_visible_gpu():
    config = materialize_execution_profile(
        RuntimeConfig(profile="cuda-replica"),
        gpu_probe=_gpu_probe(0, 1),
    )
    assert config.device == "cuda"
    assert config.gpu_devices == (0, 1)
    assert config.workers == 2

    with pytest.raises(ProfileResolutionError, match="at least two"):
        materialize_execution_profile(
            RuntimeConfig(profile="cuda-replica"),
            gpu_probe=_gpu_probe(0),
        )


def test_auto_profile_resolves_cpu_single_and_multi_gpu_deterministically():
    cpu = materialize_execution_profile(
        RuntimeConfig(profile="auto"),
        gpu_probe=_gpu_probe(),
    )
    one = materialize_execution_profile(
        RuntimeConfig(profile="auto"),
        gpu_probe=_gpu_probe(0),
    )
    two = materialize_execution_profile(
        RuntimeConfig(profile="auto"),
        gpu_probe=_gpu_probe(0, 1),
    )
    assert (cpu.device.value, cpu.gpu_devices, cpu.workers) == ("cpu", None, 1)
    assert (one.device.value, one.gpu_devices, one.workers) == ("cuda", (0,), 1)
    assert (two.device.value, two.gpu_devices, two.workers) == ("cuda", (0, 1), 2)


def test_profile_conflicts_are_rejected_instead_of_silently_overridden():
    with pytest.raises(ProfileResolutionError, match="conflicts"):
        materialize_execution_profile(
            RuntimeConfig(profile="cpu", device="cuda"),
            gpu_probe=_gpu_probe(),
        )
    with pytest.raises(ProfileResolutionError, match="exactly one selected GPU"):
        materialize_execution_profile(
            RuntimeConfig(profile="cuda-single", gpu_devices=(0, 1)),
            gpu_probe=_gpu_probe(0, 1),
        )
    with pytest.raises(ProfileResolutionError, match="one worker per selected GPU"):
        materialize_execution_profile(
            RuntimeConfig(profile="cuda-replica", gpu_devices=(0, 1), workers=1),
            gpu_probe=_gpu_probe(0, 1),
        )


@pytest.mark.parametrize(
    ("profile", "gpu_count", "expected"),
    [
        ("cpu", 2, ("cpu", (), 1, ())),
        ("cuda-single", 2, ("cuda", (0,), 1, (0,))),
        ("cuda-replica", 2, ("cuda", (0, 1), 2, (0, 1))),
        ("auto", 0, ("cpu", (), 1, ())),
        ("auto", 1, ("cuda", (0,), 1, (0,))),
        ("auto", 2, ("cuda", (0, 1), 2, (0, 1))),
    ],
)
def test_execution_plan_uses_profile_semantics(profile, gpu_count, expected):
    plan = resolve_execution_plan(RuntimeConfig(profile=profile), _inventory(gpu_count))
    assert (
        plan.effective_device,
        plan.selected_gpu_indices,
        plan.worker_count,
        plan.worker_gpu_indices,
    ) == expected


def test_probe_does_not_forward_api_token(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"count": 2}), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = probe_runtime_gpu_indices(
        environ={
            "PATH": "/bin",
            "VOXCPM_API_TOKEN": "PROFILE_SECRET",
            "CUDA_VISIBLE_DEVICES": "0,1",
        }
    )
    assert result == (0, 1)
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert "VOXCPM_API_TOKEN" not in captured["env"]


def _fake_worker(_config):
    return WorkerClient(
        WorkerBootstrapSpec(worker_id="profile-api", backend_kind="fake"),
        startup_timeout_seconds=5,
        shutdown_timeout_seconds=2,
    )


@pytest.mark.parametrize(
    ("profile", "gpus"),
    [
        ("cpu", ()),
        ("cuda-single", (0,)),
        ("cuda-replica", (0, 1)),
        ("auto", (0, 1)),
    ],
)
def test_same_api_request_contract_across_profiles(profile, gpus):
    config = RuntimeConfig(
        profile=profile,
        host="127.0.0.1",
        api_token="profile-token",
        max_queue_size=1,
    )
    normalized = materialize_execution_profile(config, gpu_probe=lambda: gpus)
    app = create_app(normalized, worker_factory=_fake_worker)
    with TestClient(app) as client:
        response = client.post(
            "/v1/tts",
            json={"text": "same client request"},
            headers={"Authorization": "Bearer profile-token", "X-Request-ID": "profile-request"},
        )
        assert response.status_code == 200
        assert response.headers["x-request-id"] == "profile-request"
        with wave.open(io.BytesIO(response.content), "rb") as wav:
            assert wav.getframerate() == 48_000
            assert wav.getnchannels() == 1
            assert wav.getnframes() > 0
