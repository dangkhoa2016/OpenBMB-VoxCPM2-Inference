from __future__ import annotations

import os
import platform
import re
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

_STATUS_PATH: Final = "/proc/self/status"
_MEMINFO_PATH: Final = "/proc/meminfo"
_RSS_FIELD: Final = "VmRSS:"
_MEM_TOTAL_PREFIX: Final = "MemTotal:"
_MEM_AVAILABLE_PREFIX: Final = "MemAvailable:"
_CGROUP_V2_LIMIT: Final = "/sys/fs/cgroup/memory.max"
_CGROUP_V1_LIMIT: Final = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
_CGROUP_V2_EVENTS: Final = "/sys/fs/cgroup/memory.events"
_CGROUP_V1_EVENTS: Final = "/sys/fs/cgroup/memory/memory.failcnt"
_UNLIMITED_CGROUP_VALUE: Final = "max"
_URL_PATTERN: Final = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://\S+")
_HOME_PATTERN: Final = re.compile(r"~(?:(?=/)|(?=$))")
_ABSOLUTE_PATH_PATTERN: Final = re.compile(r"(?<![\w.~])/(?:[^\s/'\"]+/)*[^\s/'\"]*")
_TOKEN_PATTERN: Final = re.compile(
    r"\b(?:hf_|huggingface_|ghp_|gho_|ghu_|ghs_|github_pat_|glpat-|xox[abprs]-|sk-)[A-Za-z0-9_-]{6,}"
)
_REDACTED_URL: Final = "<redacted-url>"
_REDACTED_PATH: Final = "<redacted-path>"
_REDACTED_TOKEN: Final = "<redacted-token>"


class RuntimeMetricsError(RuntimeError):
    """Raised when a runtime metric cannot be produced safely."""


def redact_text(value: str) -> str:
    """Remove URLs, absolute filesystem paths, home paths and token shapes."""

    redacted = _TOKEN_PATTERN.sub(_REDACTED_TOKEN, value)
    redacted = _URL_PATTERN.sub(_REDACTED_URL, redacted)
    redacted = _HOME_PATTERN.sub(_REDACTED_PATH, redacted)
    return _ABSOLUTE_PATH_PATTERN.sub(_REDACTED_PATH, redacted)


def redact_value(value: Any) -> Any:
    """Recursively redact strings inside JSON-compatible containers."""

    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {str(key): redact_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_value(item) for item in value]
    return value


def _read_status_kib(field_prefix: str) -> int | None:
    try:
        content = Path(_MEMINFO_PATH).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in content.splitlines():
        if line.startswith(field_prefix):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1], 10)
    return None


def _read_self_status_rss_kib() -> int | None:
    try:
        content = Path(_STATUS_PATH).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in content.splitlines():
        if line.startswith(_RSS_FIELD):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1], 10)
    return None


def current_rss_bytes() -> int | None:
    """Return the current resident set size in bytes, or None when unavailable."""

    kib = _read_self_status_rss_kib()
    if kib is None:
        return None
    return kib * 1024


def peak_rss_bytes() -> int | None:
    """Return the process peak RSS in bytes with Linux units normalized."""

    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
    except (AttributeError, OSError, ValueError):
        return None
    raw = int(usage.ru_maxrss)
    if raw < 0:
        return None
    if sys.platform == "darwin":
        return raw
    return raw * 1024


def host_total_ram_bytes() -> int | None:
    """Return installed host RAM in bytes, or None when unavailable."""

    kib = _read_status_kib(_MEM_TOTAL_PREFIX)
    if kib is not None:
        return kib * 1024
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    if isinstance(pages, int) and isinstance(page_size, int) and pages >= 0 and page_size > 0:
        return pages * page_size
    return None


def host_available_ram_bytes() -> int | None:
    """Return currently available host RAM in bytes, or None when unavailable."""

    kib = _read_status_kib(_MEM_AVAILABLE_PREFIX)
    if kib is None:
        return None
    return kib * 1024


def cgroup_memory_limit_bytes() -> int | None:
    """Return the cgroup v2/v1 memory ceiling in bytes, or None when unlimited."""

    for candidate in (_CGROUP_V2_LIMIT, _CGROUP_V1_LIMIT):
        try:
            text = Path(candidate).read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if not text or text == _UNLIMITED_CGROUP_VALUE:
            continue
        if text.isdigit():
            return int(text, 10)
    return None


def cgroup_memory_events() -> dict[str, int] | None:
    """Return cgroup OOM counters when the kernel exposes them."""

    try:
        content = Path(_CGROUP_V2_EVENTS).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    counters: dict[str, int] = {}
    for line in content.splitlines():
        parts = line.split()
        if len(parts) != 2 or not parts[0].isalnum() or not parts[1].isdigit():
            continue
        counters[parts[0]] = int(parts[1], 10)
    return counters or None


@dataclass(slots=True)
class Stopwatch:
    """Monotonic phase timer that never reports a negative duration."""

    name: str
    started_at: float = field(default_factory=time.monotonic)
    ended_at: float | None = None

    def stop(self) -> "Stopwatch":
        if self.ended_at is None:
            self.ended_at = time.monotonic()
        return self

    @property
    def duration_seconds(self) -> float | None:
        if self.ended_at is None:
            return None
        return max(0.0, self.ended_at - self.started_at)

    def to_dict(self) -> dict[str, Any]:
        duration = self.duration_seconds
        return {
            "duration_seconds": None if duration is None else round(duration, 6),
            "name": self.name,
            "started": True,
        }


@dataclass(frozen=True, slots=True)
class TorchCudaObservation:
    """Non-allocating CUDA visibility facts observed after importing torch."""

    torch_imported: bool
    cuda_available: bool | None = None
    cuda_initialized: bool | None = None
    cuda_device_count: int | None = None
    error_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cuda_available": self.cuda_available,
            "cuda_device_count": self.cuda_device_count,
            "cuda_initialized": self.cuda_initialized,
            "error_type": self.error_type,
            "torch_imported": self.torch_imported,
        }


def observe_torch_cuda(torch_module: Any) -> TorchCudaObservation:
    """Record CUDA visibility without touching any CUDA memory API.

    ``torch.cuda.is_available`` and ``torch.cuda.is_initialized`` are pure
    state queries. Free-memory and allocation APIs are deliberately avoided
    because they can create a CUDA context, which would invalidate the
    no-CUDA evidence this milestone depends on.
    """

    if torch_module is None:
        return TorchCudaObservation(torch_imported=False)
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        return TorchCudaObservation(
            torch_imported=True,
            error_type="AttributeError",
        )
    try:
        available = bool(cuda.is_available())
        initialized = bool(cuda.is_initialized())
        device_count = int(cuda.device_count())
    except Exception as error:
        return TorchCudaObservation(
            torch_imported=True,
            error_type=type(error).__name__,
        )
    return TorchCudaObservation(
        torch_imported=True,
        cuda_available=available,
        cuda_initialized=initialized,
        cuda_device_count=device_count,
    )


@dataclass(frozen=True, slots=True)
class GpuMemoryObservation:
    """CUDA memory facts for one already-selected runtime GPU."""

    torch_imported: bool
    cuda_available: bool | None = None
    device_index: int | None = None
    device_name: str | None = None
    device_total_memory_bytes: int | None = None
    compute_capability: str | None = None
    memory_allocated_bytes: int | None = None
    memory_reserved_bytes: int | None = None
    max_memory_allocated_bytes: int | None = None
    max_memory_reserved_bytes: int | None = None
    error_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "compute_capability": self.compute_capability,
            "cuda_available": self.cuda_available,
            "device_index": self.device_index,
            "device_name": self.device_name,
            "device_total_memory_bytes": self.device_total_memory_bytes,
            "error_type": self.error_type,
            "max_memory_allocated_bytes": self.max_memory_allocated_bytes,
            "max_memory_reserved_bytes": self.max_memory_reserved_bytes,
            "memory_allocated_bytes": self.memory_allocated_bytes,
            "memory_reserved_bytes": self.memory_reserved_bytes,
            "torch_imported": self.torch_imported,
        }


def observe_gpu_memory(torch_module: Any, device_index: int) -> GpuMemoryObservation:
    """Record allocation and device facts for a CUDA qualification run."""

    if torch_module is None:
        return GpuMemoryObservation(torch_imported=False)
    if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
        raise ValueError("device_index must be a non-negative integer")
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        return GpuMemoryObservation(torch_imported=True, error_type="AttributeError")
    try:
        available = bool(cuda.is_available())
        if not available:
            return GpuMemoryObservation(torch_imported=True, cuda_available=False)
        props = cuda.get_device_properties(device_index)
        major = int(getattr(props, "major"))
        minor = int(getattr(props, "minor"))
        return GpuMemoryObservation(
            torch_imported=True,
            cuda_available=True,
            device_index=device_index,
            device_name=str(getattr(props, "name")),
            device_total_memory_bytes=int(getattr(props, "total_memory")),
            compute_capability=f"{major}.{minor}",
            memory_allocated_bytes=int(cuda.memory_allocated(device_index)),
            memory_reserved_bytes=int(cuda.memory_reserved(device_index)),
            max_memory_allocated_bytes=int(cuda.max_memory_allocated(device_index)),
            max_memory_reserved_bytes=int(cuda.max_memory_reserved(device_index)),
        )
    except Exception as error:
        return GpuMemoryObservation(
            torch_imported=True,
            cuda_available=True,
            device_index=device_index,
            error_type=type(error).__name__,
        )


@dataclass(frozen=True, slots=True)
class HostFacts:
    """Host and process memory facts captured without external dependencies."""

    cpu_count: int | None
    host_total_ram_bytes: int | None
    host_available_ram_bytes: int | None
    cgroup_memory_limit_bytes: int | None
    peak_rss_bytes: int | None
    platform: str
    python_version: str
    python_implementation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "cgroup_memory_limit_bytes": self.cgroup_memory_limit_bytes,
            "cpu_count": self.cpu_count,
            "host_available_ram_bytes": self.host_available_ram_bytes,
            "host_total_ram_bytes": self.host_total_ram_bytes,
            "peak_rss_bytes": self.peak_rss_bytes,
            "platform": self.platform,
            "python_implementation": self.python_implementation,
            "python_version": self.python_version,
        }


def collect_host_facts() -> HostFacts:
    """Capture host, platform and peak-memory facts for the M4 report."""

    return HostFacts(
        cpu_count=os.cpu_count(),
        host_total_ram_bytes=host_total_ram_bytes(),
        host_available_ram_bytes=host_available_ram_bytes(),
        cgroup_memory_limit_bytes=cgroup_memory_limit_bytes(),
        peak_rss_bytes=peak_rss_bytes(),
        platform=platform.platform(),
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
    )


def dependency_versions(names: tuple[str, ...]) -> dict[str, str]:
    """Return installed versions for the selected distribution names."""

    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _version

    resolved: dict[str, str] = {}
    for name in names:
        try:
            resolved[name] = _version(name)
        except PackageNotFoundError:
            resolved[name] = "NOT_INSTALLED"
        except Exception:
            resolved[name] = "UNAVAILABLE"
    return resolved


__all__: Final[tuple[str, ...]] = (
    "GpuMemoryObservation",
    "HostFacts",
    "RuntimeMetricsError",
    "Stopwatch",
    "TorchCudaObservation",
    "cgroup_memory_events",
    "cgroup_memory_limit_bytes",
    "collect_host_facts",
    "current_rss_bytes",
    "dependency_versions",
    "host_available_ram_bytes",
    "host_total_ram_bytes",
    "observe_gpu_memory",
    "observe_torch_cuda",
    "peak_rss_bytes",
    "redact_text",
    "redact_value",
)
