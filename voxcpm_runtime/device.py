from __future__ import annotations

import importlib
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import DeviceMode, RuntimeConfig, parse_gpu_devices, parse_workers
from .profiles import materialize_execution_profile


class DeviceResolutionError(ValueError):
    pass


class CudaUnavailableError(DeviceResolutionError):
    pass


class InvalidDeviceError(DeviceResolutionError):
    pass


class InvalidGpuSelectionError(DeviceResolutionError):
    pass


class WorkerOversubscriptionError(DeviceResolutionError):
    pass


@dataclass(frozen=True, slots=True)
class GPUInfo:
    index: int | None = None
    name: str | None = None
    total_memory_bytes: int | None = None
    runtime_index: int | None = None
    device_index: int | None = None
    vram_bytes: int | None = None

    def __post_init__(self) -> None:
        if (
            self.total_memory_bytes is not None
            and self.vram_bytes is not None
            and self.total_memory_bytes != self.vram_bytes
        ):
            raise ValueError("GPU memory aliases must agree")
        if self.total_memory_bytes is None:
            object.__setattr__(self, "total_memory_bytes", self.vram_bytes)
        elif self.vram_bytes is None:
            object.__setattr__(self, "vram_bytes", self.total_memory_bytes)
        values = [value for value in (self.index, self.runtime_index, self.device_index) if value is not None]
        if not values:
            raise ValueError("a GPU index is required")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise ValueError("GPU index must be a non-negative integer")
        if any(value != values[0] for value in values[1:]):
            raise ValueError("GPU index aliases must agree")
        object.__setattr__(self, "index", values[0])
        object.__setattr__(self, "runtime_index", values[0])
        object.__setattr__(self, "device_index", values[0])
        if self.total_memory_bytes is not None and (
            isinstance(self.total_memory_bytes, bool)
            or not isinstance(self.total_memory_bytes, int)
            or self.total_memory_bytes < 0
        ):
            raise ValueError("GPU memory must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "runtime_index": self.runtime_index,
            "name": self.name,
            "total_memory_bytes": self.total_memory_bytes,
        }


@dataclass(frozen=True, slots=True)
class HardwareInventory:
    cpu_logical_count: int | None = None
    cpu_physical_count: int | None = None
    host_ram_bytes: int | None = None
    cuda_available: bool = False
    gpus: tuple[GPUInfo, ...] = ()
    gpu_count: int | None = None
    torch_version: str | None = None
    cuda_runtime_version: str | None = None
    cuda_driver_version: str | None = None
    cuda_probe_status: str = "not_probed"
    visible_device_mapping: Any = None
    notes: tuple[str, ...] = ()
    platform: str | None = None

    def __post_init__(self) -> None:
        normalized_input = tuple(self.gpus or ())
        normalized_gpus: list[GPUInfo] = []
        for gpu in normalized_input:
            if isinstance(gpu, GPUInfo):
                normalized_gpus.append(gpu)
                continue
            if isinstance(gpu, Mapping):
                values = dict(gpu)
                if "index" not in values and "runtime_index" in values:
                    values["index"] = values["runtime_index"]
                if "total_memory_bytes" not in values and "vram_bytes" in values:
                    values["total_memory_bytes"] = values["vram_bytes"]
                allowed = {"index", "name", "total_memory_bytes", "runtime_index", "device_index"}
                normalized_gpus.append(GPUInfo(**{key: values[key] for key in values if key in allowed}))
                continue
            raise TypeError("gpus must contain GPUInfo values or mappings")
        indices = [gpu.index for gpu in normalized_gpus]
        if len(set(indices)) != len(indices):
            raise ValueError("GPU indices must be unique")
        if self.gpu_count is None:
            count = len(normalized_gpus)
        else:
            if isinstance(self.gpu_count, bool) or not isinstance(self.gpu_count, int) or self.gpu_count < 0:
                raise ValueError("gpu_count must be a non-negative integer")
            count = self.gpu_count
        if count < len(normalized_gpus):
            raise ValueError("gpu_count cannot be smaller than the GPU list")
        if count > len(normalized_gpus):
            existing = set(indices)
            normalized_gpus.extend(
                GPUInfo(index=index) for index in range(count) if index not in existing
            )
        object.__setattr__(self, "gpus", tuple(normalized_gpus))
        object.__setattr__(self, "gpu_count", count)
        object.__setattr__(self, "notes", tuple(self.notes or ()))
        if not isinstance(self.cuda_available, bool):
            raise ValueError("cuda_available must be a boolean")
        for name in ("cpu_logical_count", "cpu_physical_count", "host_ram_bytes"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer")

    @property
    def gpu_indices(self) -> tuple[int, ...]:
        if self.gpus:
            return tuple(gpu.index for gpu in self.gpus)
        return tuple(range(self.gpu_count or 0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_logical_count": self.cpu_logical_count,
            "cpu_physical_count": self.cpu_physical_count,
            "host_ram_bytes": self.host_ram_bytes,
            "cuda_available": self.cuda_available,
            "cuda_probe_status": self.cuda_probe_status,
            "torch_version": self.torch_version,
            "cuda_runtime_version": self.cuda_runtime_version,
            "cuda_driver_version": self.cuda_driver_version,
            "gpu_count": self.gpu_count,
            "gpus": [gpu.to_dict() for gpu in self.gpus],
            "visible_device_mapping": _safe_visible_mapping(self.visible_device_mapping),
            "notes": list(self.notes),
            "platform": self.platform,
        }


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    effective_device: str
    selected_gpu_indices: tuple[int, ...] = ()
    worker_count: int = 1
    worker_gpu_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        effective_device = self.effective_device.value if isinstance(self.effective_device, DeviceMode) else str(self.effective_device)
        selected = () if self.selected_gpu_indices is None else tuple(self.selected_gpu_indices)
        worker_gpus = () if self.worker_gpu_indices is None else tuple(self.worker_gpu_indices)
        if effective_device not in {"cpu", "cuda"}:
            raise ValueError("effective device must be cpu or cuda")
        if isinstance(self.worker_count, bool) or not isinstance(self.worker_count, int) or self.worker_count <= 0:
            raise ValueError("worker_count must be positive")
        if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in selected):
            raise ValueError("selected GPU indices must be non-negative integers")
        if len(set(selected)) != len(selected):
            raise ValueError("selected GPU indices must be unique")
        if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in worker_gpus):
            raise ValueError("worker GPU indices must be non-negative integers")
        if effective_device == "cuda" and not worker_gpus:
            worker_gpus = selected[: self.worker_count]
        if effective_device == "cpu" and (selected or worker_gpus):
            raise ValueError("CPU plans cannot select GPUs")
        if effective_device == "cpu" and self.worker_count != 1:
            raise ValueError("CPU plans support one worker")
        if effective_device == "cuda":
            if not selected:
                raise ValueError("CUDA plans require selected GPUs")
            if self.worker_count > len(selected):
                raise ValueError("CUDA workers cannot exceed selected GPUs")
            if len(worker_gpus) != self.worker_count:
                raise ValueError("each CUDA worker must map to one selected GPU")
            if len(set(worker_gpus)) != len(worker_gpus):
                raise ValueError("CUDA workers cannot share a selected GPU")
            if any(index not in selected for index in worker_gpus):
                raise ValueError("CUDA worker GPUs must be selected")
        object.__setattr__(self, "effective_device", effective_device)
        object.__setattr__(self, "selected_gpu_indices", selected)
        object.__setattr__(self, "worker_gpu_indices", worker_gpus)

    @property
    def device(self) -> str:
        return self.effective_device

    @property
    def selected_gpus(self) -> tuple[int, ...]:
        return self.selected_gpu_indices

    @property
    def workers(self) -> int:
        return self.worker_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "effective_device": self.effective_device,
            "selected_gpu_indices": list(self.selected_gpu_indices),
            "worker_count": self.worker_count,
            "worker_gpu_indices": list(self.worker_gpu_indices),
        }


def _safe_visible_mapping(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text in {"raw", "value", "token", "uuid"}:
                safe[key_text] = "redacted"
            elif key_text == "runtime_indices" and isinstance(item, (list, tuple)):
                safe[key_text] = [
                    index if isinstance(index, int) and not isinstance(index, bool) else "redacted"
                    for index in item
                ]
            elif isinstance(item, (int, float, bool)) or item is None:
                safe[key_text] = item
            elif isinstance(item, (list, tuple, dict)):
                safe[key_text] = _safe_visible_mapping(item)
            else:
                safe[key_text] = "redacted"
        return safe
    if isinstance(value, (list, tuple)):
        return [_safe_visible_mapping(item) if isinstance(item, (dict, list, tuple)) else (
            item if isinstance(item, int) and not isinstance(item, bool) else "redacted"
        ) for item in value]
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return "configured"


def _read_physical_cpu_count() -> int | None:
    path = Path("/proc/cpuinfo")
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    pairs: set[tuple[str, str]] = set()
    record_seen = False
    physical_id: str | None = None
    core_id: str | None = None

    def finish_record() -> bool:
        if not record_seen:
            return True
        if physical_id is None or core_id is None:
            return False
        pairs.add((physical_id, core_id))
        return True

    for raw_line in content.splitlines() + [""]:
        line = raw_line.strip()
        if not line:
            if not finish_record():
                return None
            record_seen = False
            physical_id = None
            core_id = None
            continue
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        if key == "processor":
            record_seen = True
        elif key == "physical id":
            physical_id = value
        elif key == "core id":
            core_id = value
    return len(pairs) if pairs else None


def _read_host_ram_bytes() -> int | None:
    path = Path("/proc/meminfo")
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            if raw_line.startswith("MemTotal:"):
                fields = raw_line.split()
                if len(fields) >= 2 and fields[1].isdigit():
                    return int(fields[1], 10) * 1024
                break
    except (OSError, UnicodeError, ValueError):
        pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if isinstance(pages, int) and isinstance(page_size, int) and pages >= 0 and page_size > 0:
            return pages * page_size
    except (AttributeError, OSError, ValueError):
        pass
    return None


def _probe_environment() -> dict[str, str]:
    allowed = ("PATH", "HOME", "LANG", "LC_ALL", "SYSTEMROOT")
    return {key: os.environ[key] for key in allowed if key in os.environ}


def _probe_driver_version() -> str | None:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
            env=_probe_environment(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        value = line.strip()
        if value and re.fullmatch(r"[0-9A-Za-z._-]{1,64}", value):
            return value
    return None


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def detect_hardware(
    environ: Mapping[str, str] | None = None,
    torch_module: Any = None,
) -> HardwareInventory:
    environment = os.environ if environ is None else environ
    notes: list[str] = []
    cpu_logical_count = os.cpu_count()
    cpu_physical_count = _read_physical_cpu_count()
    host_ram_bytes = _read_host_ram_bytes()
    if host_ram_bytes is None:
        notes.append("host RAM could not be inspected")
    if cpu_physical_count is None:
        notes.append("physical CPU count is unavailable")
    cuda_available = False
    gpu_infos: tuple[GPUInfo, ...] = ()
    torch_version: str | None = None
    cuda_runtime_version: str | None = None
    cuda_probe_status: str
    if torch_module is None:
        try:
            torch_module = importlib.import_module("torch")
        except Exception as exc:
            cuda_probe_status = "torch_unavailable"
            notes.append(f"PyTorch probe unavailable: {type(exc).__name__}")
            torch_module = None
    if torch_module is not None:
        try:
            torch_version_value = getattr(torch_module, "__version__", None)
            torch_version = str(torch_version_value) if torch_version_value is not None else None
            version_module = getattr(torch_module, "version", None)
            runtime_value = getattr(version_module, "cuda", None) if version_module is not None else None
            cuda_runtime_version = str(runtime_value) if runtime_value else None
        except Exception as exc:
            notes.append(f"PyTorch version probe failed: {type(exc).__name__}")
        try:
            is_available = bool(torch_module.cuda.is_available())
            device_count = int(torch_module.cuda.device_count())
        except Exception as exc:
            is_available = False
            device_count = 0
            cuda_probe_status = "probe_error"
            notes.append(f"PyTorch CUDA probe failed: {type(exc).__name__}")
        if "cuda_probe_status" not in locals() or cuda_probe_status not in {
            "torch_unavailable",
            "probe_error",
        }:
            cuda_probe_status = "available" if is_available and device_count > 0 else "cuda_unavailable"
        if is_available and device_count > 0:
            discovered: list[GPUInfo] = []
            for index in range(device_count):
                name: str | None = None
                memory: int | None = None
                try:
                    raw_name = torch_module.cuda.get_device_name(index)
                    name = str(raw_name) if raw_name is not None else None
                except Exception:
                    notes.append(f"GPU {index} name unavailable")
                try:
                    properties = torch_module.cuda.get_device_properties(index)
                    raw_memory = getattr(properties, "total_memory", None)
                    if raw_memory is not None:
                        memory = int(raw_memory)
                except Exception:
                    notes.append(f"GPU {index} memory unavailable")
                discovered.append(GPUInfo(index=index, name=name, total_memory_bytes=memory))
            gpu_infos = tuple(discovered)
            cuda_available = True
        elif is_available and device_count == 0:
            notes.append("PyTorch reported CUDA available with zero visible devices")
        if not is_available:
            notes.append("CUDA is unavailable through PyTorch")
    if torch_version is None:
        torch_version = _package_version("torch")
    cuda_driver_version = _probe_driver_version()
    visible_device_mapping: Any = None
    if "CUDA_VISIBLE_DEVICES" in environment:
        visible_device_mapping = {
            "configured": True,
            "runtime_indices": list(range(len(gpu_infos))),
        }
        notes.append("CUDA_VISIBLE_DEVICES is configured; raw value redacted")
    return HardwareInventory(
        cpu_logical_count=cpu_logical_count,
        cpu_physical_count=cpu_physical_count,
        host_ram_bytes=host_ram_bytes,
        cuda_available=cuda_available,
        gpus=gpu_infos,
        gpu_count=len(gpu_infos),
        torch_version=torch_version,
        cuda_runtime_version=cuda_runtime_version,
        cuda_driver_version=cuda_driver_version,
        cuda_probe_status=cuda_probe_status,
        visible_device_mapping=visible_device_mapping,
        notes=tuple(notes),
        platform=platform.platform(),
    )


def _coerce_device(value: Any) -> str:
    if isinstance(value, DeviceMode):
        return value.value
    if isinstance(value, RuntimeConfig):
        return value.device.value
    if value is None:
        return DeviceMode.AUTO.value
    text = str(value)
    if text not in {item.value for item in DeviceMode}:
        raise InvalidDeviceError("requested device must be one of: auto, cpu, cuda")
    return text


def _coerce_gpu_selection(value: Any) -> tuple[int, ...] | None:
    if isinstance(value, RuntimeConfig):
        return value.gpu_devices
    return parse_gpu_devices(value)


def _coerce_workers(value: Any) -> int | None:
    if isinstance(value, RuntimeConfig):
        return value.workers
    return parse_workers(value)


def _visible_gpu_indices(inventory: HardwareInventory) -> tuple[int, ...]:
    if inventory.gpus:
        return tuple(gpu.index for gpu in inventory.gpus)
    return tuple(range(inventory.gpu_count or 0))


def _validate_gpu_selection(
    selection: tuple[int, ...] | None,
    inventory: HardwareInventory,
) -> tuple[int, ...]:
    visible = _visible_gpu_indices(inventory)
    if selection is None:
        return visible
    for index in selection:
        if index not in visible:
            raise InvalidGpuSelectionError(
                f"GPU index {index} is not visible; visible indices are {list(visible)}"
            )
    return selection


def _resolve_values(
    requested_device: Any,
    gpu_devices: Any,
    workers: Any,
    inventory: HardwareInventory,
) -> ExecutionPlan:
    device = _coerce_device(requested_device)
    selected_spec = _coerce_gpu_selection(gpu_devices)
    worker_limit = _coerce_workers(workers)
    has_explicit_gpu_selection = selected_spec is not None
    if device == DeviceMode.CPU.value:
        if has_explicit_gpu_selection:
            raise InvalidGpuSelectionError("an explicit GPU list cannot be combined with CPU execution")
        if worker_limit is not None and worker_limit > 1:
            raise WorkerOversubscriptionError(
                "CPU execution supports one worker; explicit oversubscription is not allowed"
            )
        return ExecutionPlan(
            effective_device=DeviceMode.CPU.value,
            selected_gpu_indices=(),
            worker_count=1,
            worker_gpu_indices=(),
        )
    usable_cuda = inventory.cuda_available and inventory.gpu_count > 0
    if device == DeviceMode.CUDA.value:
        if not usable_cuda:
            raise CudaUnavailableError("CUDA was requested but no usable CUDA device is available")
    elif not usable_cuda:
        if has_explicit_gpu_selection:
            raise InvalidGpuSelectionError("an explicit GPU list requires a usable CUDA device")
        if worker_limit is not None and worker_limit > 1:
            raise WorkerOversubscriptionError(
                "CPU execution supports one worker; explicit oversubscription is not allowed"
            )
        return ExecutionPlan(
            effective_device=DeviceMode.CPU.value,
            selected_gpu_indices=(),
            worker_count=1,
            worker_gpu_indices=(),
        )
    selected = _validate_gpu_selection(selected_spec, inventory)
    if not selected:
        raise CudaUnavailableError("CUDA was requested but no visible GPU is available")
    if worker_limit is None:
        resolved_workers = len(selected)
    else:
        resolved_workers = worker_limit
        if resolved_workers > len(selected):
            raise WorkerOversubscriptionError(
                f"requested {resolved_workers} workers for {len(selected)} selected GPUs; "
                "explicit oversubscription is not allowed"
            )
    return ExecutionPlan(
        effective_device=DeviceMode.CUDA.value,
        selected_gpu_indices=selected,
        worker_count=resolved_workers,
        worker_gpu_indices=selected[:resolved_workers],
    )


def resolve_execution_plan(config: RuntimeConfig, inventory: HardwareInventory) -> ExecutionPlan:
    if not isinstance(config, RuntimeConfig):
        raise TypeError("config must be a RuntimeConfig")
    if not isinstance(inventory, HardwareInventory):
        raise TypeError("inventory must be a HardwareInventory")
    normalized = materialize_execution_profile(
        config,
        gpu_probe=lambda: _visible_gpu_indices(inventory),
    )
    return _resolve_values(
        normalized.device,
        normalized.gpu_devices,
        normalized.workers,
        inventory,
    )


def resolve(
    requested_device: Any,
    gpu_devices: Any = "auto",
    workers: Any = "auto",
    inventory: HardwareInventory | None = None,
) -> ExecutionPlan:
    if inventory is None:
        raise ValueError("inventory is required for deterministic resolution")
    if not isinstance(inventory, HardwareInventory):
        raise TypeError("inventory must be a HardwareInventory")
    return _resolve_values(requested_device, gpu_devices, workers, inventory)


class DeviceManager:
    def __init__(self, inventory: HardwareInventory | None = None) -> None:
        self.inventory = inventory

    def detect(self) -> HardwareInventory:
        return detect_hardware()

    def resolve(
        self,
        requested_device: Any,
        gpu_devices: Any = "auto",
        workers: Any = "auto",
        inventory: HardwareInventory | None = None,
    ) -> ExecutionPlan:
        selected_inventory = inventory if inventory is not None else self.inventory
        if selected_inventory is None:
            selected_inventory = self.detect()
        if not isinstance(selected_inventory, HardwareInventory):
            raise TypeError("inventory must be a HardwareInventory")
        if isinstance(requested_device, RuntimeConfig):
            return resolve_execution_plan(requested_device, selected_inventory)
        return _resolve_values(requested_device, gpu_devices, workers, selected_inventory)


__all__ = [
    "CudaUnavailableError",
    "DeviceManager",
    "DeviceResolutionError",
    "ExecutionPlan",
    "GPUInfo",
    "HardwareInventory",
    "InvalidDeviceError",
    "InvalidGpuSelectionError",
    "WorkerOversubscriptionError",
    "detect_hardware",
    "resolve",
    "resolve_execution_plan",
]
