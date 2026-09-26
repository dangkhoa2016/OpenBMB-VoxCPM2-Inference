from __future__ import annotations

import json
import math
import resource
import sys
import types
from pathlib import Path

import pytest

from voxcpm_runtime.runtime_metrics import (
    HostFacts,
    Stopwatch,
    TorchCudaObservation,
    cgroup_memory_events,
    cgroup_memory_limit_bytes,
    collect_host_facts,
    current_rss_bytes,
    dependency_versions,
    host_available_ram_bytes,
    host_total_ram_bytes,
    observe_torch_cuda,
    peak_rss_bytes,
    redact_text,
    redact_value,
)


# --- RSS helpers and units (Phase J) ------------------------------------


@pytest.mark.skipif(
    not Path("/proc/self/status").is_file(), reason="Linux /proc is required"
)
def test_current_rss_is_positive_and_in_bytes() -> None:
    value = current_rss_bytes()
    assert value is not None
    assert value > 0
    assert value % 1024 == 0


def test_peak_rss_normalizes_linux_units_to_bytes() -> None:
    value = peak_rss_bytes()
    assert value is not None
    assert value > 0
    if sys.platform != "darwin":
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        assert value == raw * 1024
    else:
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        assert value == raw


def test_host_ram_helpers_are_positive() -> None:
    total = host_total_ram_bytes()
    assert total is not None
    assert total > 0
    available = host_available_ram_bytes()
    if available is not None:
        assert available > 0


def test_host_facts_are_json_serializable() -> None:
    facts = collect_host_facts()
    assert isinstance(facts, HostFacts)
    payload = facts.to_dict()
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    assert payload["python_version"] == ".".join(str(part) for part in sys.version_info[:3])
    assert isinstance(payload["cpu_count"], int)
    assert payload["peak_rss_bytes"] is None or payload["peak_rss_bytes"] > 0


def test_cgroup_helpers_do_not_raise() -> None:
    limit = cgroup_memory_limit_bytes()
    assert limit is None or limit > 0
    events = cgroup_memory_events()
    assert events is None or all(
        isinstance(key, str) and isinstance(value, int) and value >= 0
        for key, value in events.items()
    )


# --- stopwatch ----------------------------------------------------------


def test_stopwatch_reports_monotonic_duration() -> None:
    watch = Stopwatch("load")
    assert watch.duration_seconds is None
    watch.stop()
    duration = watch.duration_seconds
    assert duration is not None
    assert duration >= 0.0
    assert math.isfinite(duration)


def test_stopwatch_stop_is_idempotent() -> None:
    watch = Stopwatch("load")
    watch.stop()
    first = watch.duration_seconds
    watch.stop()
    assert watch.duration_seconds == first


def test_stopwatch_dict_is_json_friendly() -> None:
    payload = Stopwatch("synthesize").stop().to_dict()
    assert payload["name"] == "synthesize"
    assert payload["started"] is True
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


# --- CUDA observation (Phase L) -----------------------------------------


class _FakeCuda:
    def __init__(self, available: bool, initialized: bool, count: int) -> None:
        self._available = available
        self._initialized = initialized
        self._count = count
        self.queried: list[str] = []

    def is_available(self) -> bool:
        self.queried.append("is_available")
        return self._available

    def is_initialized(self) -> bool:
        self.queried.append("is_initialized")
        return self._initialized

    def device_count(self) -> int:
        self.queried.append("device_count")
        return self._count


def _torch_stub(cuda: object | None) -> object:
    return types.SimpleNamespace(cuda=cuda, __version__="0.0-stub")


def test_observe_torch_cuda_reports_canonical_cpu_facts() -> None:
    observation = observe_torch_cuda(_torch_stub(_FakeCuda(False, False, 0)))
    assert observation == TorchCudaObservation(
        torch_imported=True,
        cuda_available=False,
        cuda_initialized=False,
        cuda_device_count=0,
    )
    assert observation.to_dict()["cuda_available"] is False
    assert observation.to_dict()["cuda_initialized"] is False


def test_observe_torch_cuda_uses_no_allocation_api() -> None:
    cuda = _FakeCuda(False, False, 0)
    observe_torch_cuda(_torch_stub(cuda))
    assert cuda.queried == ["is_available", "is_initialized", "device_count"]
    for forbidden in ("mem_get_info", "memory_allocated", "synchronize", "init"):
        assert forbidden not in cuda.queried


def test_observe_torch_cuda_handles_missing_module() -> None:
    observation = observe_torch_cuda(None)
    assert observation.torch_imported is False
    assert observation.cuda_available is None
    assert observation.cuda_initialized is None


def test_observe_torch_cuda_handles_missing_cuda_attribute() -> None:
    observation = observe_torch_cuda(types.SimpleNamespace())
    assert observation.torch_imported is True
    assert observation.error_type == "AttributeError"


def test_observe_torch_cuda_handles_probe_error() -> None:
    class _Broken:
        def is_available(self) -> bool:
            raise RuntimeError("no driver")

    observation = observe_torch_cuda(_torch_stub(_Broken()))
    assert observation.error_type == "RuntimeError"
    assert observation.cuda_available is None


def test_observe_torch_cuda_dict_is_json_friendly() -> None:
    payload = observe_torch_cuda(_torch_stub(_FakeCuda(False, False, 0))).to_dict()
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


# --- redaction (Phase S) ------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "/kaggle/input/models/dangkhoa2016/openbmb-voxcpm2/pytorch/default/1",
        "/root/OpenBMB-VoxCPM2-Inference/voxcpm_runtime/pytorch_backend.py",
        "/tmp/voxcpm-m4-cpu.wav",
    ],
)
def test_redact_text_removes_absolute_paths(value: str) -> None:
    redacted = redact_text(f"failed to load {value} on host")
    assert value not in redacted
    assert "<redacted-path>" in redacted


def test_redact_text_removes_urls() -> None:
    redacted = redact_text("see https://huggingface.co/openbmb/VoxCPM2 for details")
    assert "huggingface.co" not in redacted
    assert "<redacted-url>" in redacted


def test_redact_text_removes_home_paths() -> None:
    redacted = redact_text("config lives under ~/models/voxcpm2")
    assert "~/models" not in redacted
    assert "<redacted-path>" in redacted


def test_redact_text_removes_token_shapes() -> None:
    for token in ("hf_abcdef123456", "ghp_abcdef123456", "github_pat_abcdef123456"):
        redacted = redact_text(f"using {token} to authenticate")
        assert token not in redacted
        assert "<redacted-token>" in redacted


def test_redact_text_preserves_safe_values() -> None:
    for value in (
        "openbmb/VoxCPM2",
        "pytorch-voxcpm",
        "voxcpm2",
        "32279effe8c19989596f05d353d1447f51d9e915",
        "f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69",
        "2.14.0+cpu",
        "pcm_s16le",
    ):
        assert redact_text(value) == value


def test_redact_value_walks_containers() -> None:
    payload = {
        "model": {"path": "/kaggle/input/models/x", "model_id": "openbmb/VoxCPM2"},
        "notes": ["/tmp/out.wav", "safe"],
        "count": 3,
        "enabled": True,
    }
    redacted = redact_value(payload)
    serialized = json.dumps(redacted)
    assert "/kaggle/input" not in serialized
    assert "/tmp/out.wav" not in serialized
    assert redacted["model"]["model_id"] == "openbmb/VoxCPM2"
    assert redacted["count"] == 3
    assert redacted["enabled"] is True


def test_redact_value_leaves_non_strings_untouched() -> None:
    assert redact_value(7) == 7
    assert redact_value(None) is None


# --- dependency versions ------------------------------------------------


def test_dependency_versions_marks_missing_distributions() -> None:
    versions = dependency_versions(("voxcpm_runtime_missing_distribution_xyz",))
    assert versions == {"voxcpm_runtime_missing_distribution_xyz": "NOT_INSTALLED"}


def test_dependency_versions_reports_installed_distribution() -> None:
    versions = dependency_versions(("pytest",))
    assert versions["pytest"] not in {"NOT_INSTALLED", "UNAVAILABLE"}
