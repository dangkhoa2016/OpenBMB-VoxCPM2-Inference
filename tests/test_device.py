import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from voxcpm_runtime import device as device_module

from voxcpm_runtime.config import RuntimeConfig
from voxcpm_runtime.device import (
    CudaUnavailableError,
    DeviceManager,
    ExecutionPlan,
    GPUInfo,
    HardwareInventory,
    InvalidGpuSelectionError,
    WorkerOversubscriptionError,
    detect_hardware,
    resolve,
    resolve_execution_plan,
)


def cpu_inventory():
    return HardwareInventory(
        cpu_logical_count=4,
        cpu_physical_count=2,
        host_ram_bytes=16 * 1024**3,
        cuda_available=False,
    )


def cuda_inventory(count=2):
    return HardwareInventory(
        cpu_logical_count=4,
        cpu_physical_count=2,
        host_ram_bytes=16 * 1024**3,
        cuda_available=True,
        gpus=tuple(GPUInfo(index=index, name=f"fake-gpu-{index}", total_memory_bytes=8 * 1024**3) for index in range(count)),
    )


def test_explicit_cpu_never_resolves_cuda():
    plan = resolve("cpu", "auto", "auto", cuda_inventory())
    assert plan == ExecutionPlan(effective_device="cpu", selected_gpu_indices=(), worker_count=1, worker_gpu_indices=())


def test_explicit_cuda_without_cuda_fails_without_fallback():
    with pytest.raises(CudaUnavailableError):
        resolve("cuda", "auto", "auto", cpu_inventory())


def test_auto_without_cuda_resolves_cpu():
    plan = resolve("auto", "auto", "auto", cpu_inventory())
    assert plan.effective_device == "cpu"
    assert plan.selected_gpu_indices == ()
    assert plan.worker_count == 1


def test_auto_with_one_gpu_resolves_cuda_with_one_worker():
    plan = resolve("auto", "auto", "auto", cuda_inventory(1))
    assert plan.effective_device == "cuda"
    assert plan.selected_gpu_indices == (0,)
    assert plan.worker_count == 1
    assert plan.worker_gpu_indices == (0,)


def test_auto_with_two_gpus_resolves_deterministically():
    first = resolve("auto", "auto", "auto", cuda_inventory(2))
    second = resolve("auto", "auto", "auto", cuda_inventory(2))
    assert first == second
    assert first.effective_device == "cuda"
    assert first.selected_gpu_indices == (0, 1)
    assert first.worker_count == 2
    assert first.worker_gpu_indices == (0, 1)


def test_explicit_gpu_list_preserves_order():
    plan = resolve("cuda", "1,0", "auto", cuda_inventory(2))
    assert plan.selected_gpu_indices == (1, 0)
    assert plan.worker_count == 2


@pytest.mark.parametrize("selection", ["-1", "0,,1", "0,a", "0,0", "2"])
def test_invalid_or_out_of_range_gpu_selection_fails(selection):
    with pytest.raises(ValueError):
        config = RuntimeConfig.from_mapping({"VOXCPM_DEVICE": "cuda", "VOXCPM_GPU_DEVICES": selection})
        resolve_execution_plan(config, cuda_inventory(2))


def test_explicit_worker_limit_is_honored():
    plan = resolve("cuda", "0,1", "1", cuda_inventory(2))
    assert plan.selected_gpu_indices == (0, 1)
    assert plan.worker_count == 1
    assert plan.worker_gpu_indices == (0,)


def test_explicit_worker_oversubscription_is_rejected():
    with pytest.raises(WorkerOversubscriptionError):
        resolve("cuda", "0", "2", cuda_inventory(1))


def test_cpu_rejects_multiple_explicit_workers():
    with pytest.raises(WorkerOversubscriptionError):
        resolve("cpu", "auto", "2", cpu_inventory())


def test_cpu_rejects_explicit_gpu_list():
    with pytest.raises(InvalidGpuSelectionError):
        resolve("cpu", "0", "auto", cuda_inventory(1))


def test_invalid_device_fails_in_resolver():
    with pytest.raises(ValueError):
        resolve("gpu", "auto", "auto", cpu_inventory())


def test_manager_resolves_with_injected_inventory():
    manager = DeviceManager(cuda_inventory(2))
    plan = manager.resolve("auto", "auto", "auto")
    assert plan.effective_device == "cuda"
    assert plan.worker_count == 2


def test_inventory_and_plan_are_json_serializable():
    inventory = cuda_inventory(1)
    plan = resolve("auto", "auto", "auto", inventory)
    json.dumps(inventory.to_dict())
    json.dumps(plan.to_dict())
    assert inventory.to_dict()["gpu_count"] == 1
    assert inventory.to_dict()["gpus"][0]["name"] == "fake-gpu-0"


def test_fake_torch_cuda_probe_reports_runtime_inventory(monkeypatch):
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 2

        @staticmethod
        def get_device_name(index):
            return f"fake-cuda-{index}"

        @staticmethod
        def get_device_properties(index):
            return SimpleNamespace(total_memory=4 * 1024**3)

    fake_torch = SimpleNamespace(__version__="fake-torch", version=SimpleNamespace(cuda="12.4"), cuda=FakeCuda())
    monkeypatch.setattr(device_module, "_probe_driver_version", lambda: "fake-driver")
    inventory = detect_hardware(environ={}, torch_module=fake_torch)
    assert inventory.cuda_available is True
    assert inventory.gpu_count == 2
    assert inventory.torch_version == "fake-torch"
    assert inventory.cuda_runtime_version == "12.4"
    assert inventory.cuda_driver_version == "fake-driver"
    assert inventory.gpus[1].name == "fake-cuda-1"
    assert inventory.gpus[1].total_memory_bytes == 4 * 1024**3


def test_real_detection_is_safe_on_cpu_only_hosts():
    inventory = detect_hardware()
    assert inventory.cpu_logical_count is None or inventory.cpu_logical_count >= 1
    assert inventory.host_ram_bytes is None or inventory.host_ram_bytes > 0
    assert isinstance(inventory.cuda_available, bool)
    assert inventory.gpu_count == len(inventory.gpus)


def test_inventory_rejects_duplicate_gpu_indices():
    with pytest.raises(ValueError, match="unique"):
        HardwareInventory(gpus=(GPUInfo(index=0), GPUInfo(index=0)))


def test_gpu_info_accepts_vram_alias():
    gpu = GPUInfo(index=0, vram_bytes=8 * 1024**3)
    assert gpu.total_memory_bytes == 8 * 1024**3
    assert gpu.vram_bytes == 8 * 1024**3


def test_cuda_plan_derives_worker_mapping_when_omitted():
    plan = ExecutionPlan(effective_device="cuda", selected_gpu_indices=(0, 1), worker_count=1)
    assert plan.worker_gpu_indices == (0,)


def test_cpu_plan_rejects_multiple_workers_directly():
    with pytest.raises(ValueError, match="one worker"):
        ExecutionPlan(effective_device="cpu", worker_count=2)


def test_cuda_plan_rejects_shared_or_unselected_worker_gpu():
    with pytest.raises(ValueError, match="share"):
        ExecutionPlan(
            effective_device="cuda",
            selected_gpu_indices=(0, 1),
            worker_count=2,
            worker_gpu_indices=(0, 0),
        )
    with pytest.raises(ValueError, match="selected"):
        ExecutionPlan(
            effective_device="cuda",
            selected_gpu_indices=(0,),
            worker_count=1,
            worker_gpu_indices=(1,),
        )


def test_visible_mapping_is_preserved_without_exposing_raw_values():
    inventory = HardwareInventory(
        cuda_available=True,
        gpu_count=2,
        visible_device_mapping={
            "runtime_indices": [0, 1],
            "raw": "MAPPING_SECRET",
            "nested": {"value": "NESTED_SECRET"},
        },
    )
    encoded = json.dumps(inventory.to_dict(), sort_keys=True)
    mapping = inventory.to_dict()["visible_device_mapping"]
    assert mapping["runtime_indices"] == [0, 1]
    assert mapping["raw"] == "redacted"
    assert mapping["nested"]["value"] == "redacted"
    assert "MAPPING_SECRET" not in encoded
    assert "NESTED_SECRET" not in encoded


def test_visible_mapping_redacts_unknown_string_values():
    inventory = HardwareInventory(
        visible_device_mapping={"metadata": "UNKNOWN_MAPPING_SECRET"},
    )
    encoded = json.dumps(inventory.to_dict(), sort_keys=True)
    assert "UNKNOWN_MAPPING_SECRET" not in encoded
    assert inventory.to_dict()["visible_device_mapping"]["metadata"] == "redacted"


def test_physical_cpu_count_requires_complete_records(monkeypatch):
    original_read_text = Path.read_text

    def read_text(path, encoding=None):
        if str(path) == "/proc/cpuinfo":
            return "processor: 0\nphysical id: 0\ncore id: 0\n\nprocessor: 1\nphysical id: 0\ncore id: 1\n"
        return original_read_text(path, encoding=encoding)

    monkeypatch.setattr(Path, "read_text", read_text)
    assert device_module._read_physical_cpu_count() == 2

    def incomplete_read_text(path, encoding=None):
        if str(path) == "/proc/cpuinfo":
            return "processor: 0\nphysical id: 0\n\nprocessor: 1\nphysical id: 0\ncore id: 1\n"
        return original_read_text(path, encoding=encoding)

    monkeypatch.setattr(Path, "read_text", incomplete_read_text)
    assert device_module._read_physical_cpu_count() is None


def test_driver_probe_does_not_forward_secret_environment(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="535.104.05\n", stderr="")

    monkeypatch.setattr(device_module.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(device_module.subprocess, "run", fake_run)
    monkeypatch.setenv("VOXCPM_API_TOKEN", "PROBE_SECRET")
    assert device_module._probe_driver_version() == "535.104.05"
    assert "VOXCPM_API_TOKEN" not in captured["env"]
