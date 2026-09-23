from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from typing import Callable, Final, Mapping

from voxcpm_runtime.config import DeviceMode, ExecutionProfile, RuntimeConfig


class ProfileResolutionError(ValueError):
    pass


GpuProbe = Callable[[], tuple[int, ...]]


def _compatible_device(config: RuntimeConfig, expected: DeviceMode) -> None:
    if config.device not in {DeviceMode.AUTO, expected}:
        raise ProfileResolutionError(
            f"profile {config.profile.value!r} conflicts with VOXCPM_DEVICE={config.device.value!r}"
        )


def _probe_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if environ is None else environ
    allowed = (
        "PATH",
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "NVIDIA_VISIBLE_DEVICES",
    )
    return {key: source[key] for key in allowed if key in source}


def probe_runtime_gpu_indices(
    *,
    python_executable: str | None = None,
    environ: Mapping[str, str] | None = None,
    timeout_seconds: float = 20.0,
) -> tuple[int, ...]:
    executable = sys.executable if python_executable is None else python_executable
    script = (
        "import json\n"
        "try:\n"
        " import torch\n"
        " count=torch.cuda.device_count() if torch.cuda.is_available() else 0\n"
        "except Exception:\n"
        " count=0\n"
        "print(json.dumps({'count': int(count)}))\n"
    )
    try:
        completed = subprocess.run(
            [executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=_probe_environment(environ),
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if completed.returncode != 0:
        return ()
    try:
        payload = json.loads(completed.stdout.strip())
        count = int(payload["count"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return ()
    if count <= 0:
        return ()
    return tuple(range(count))


def materialize_execution_profile(
    config: RuntimeConfig,
    *,
    gpu_probe: GpuProbe = probe_runtime_gpu_indices,
) -> RuntimeConfig:
    if not isinstance(config, RuntimeConfig):
        raise TypeError("config must be RuntimeConfig")
    profile = config.profile
    if profile is None:
        return config

    if profile is ExecutionProfile.CPU:
        _compatible_device(config, DeviceMode.CPU)
        if config.gpu_devices is not None:
            raise ProfileResolutionError("cpu profile cannot select GPUs")
        if config.workers not in (None, 1):
            raise ProfileResolutionError("cpu profile supports exactly one worker")
        return replace(config, device=DeviceMode.CPU, gpu_devices=None, workers=1)

    if profile is ExecutionProfile.CUDA_SINGLE:
        _compatible_device(config, DeviceMode.CUDA)
        selected = config.gpu_devices
        if selected is None:
            visible = gpu_probe()
            if not visible:
                raise ProfileResolutionError("cuda-single profile requires one usable CUDA device")
            selected = (visible[0],)
        if len(selected) != 1:
            raise ProfileResolutionError("cuda-single profile requires exactly one selected GPU")
        if config.workers not in (None, 1):
            raise ProfileResolutionError("cuda-single profile supports exactly one worker")
        return replace(config, device=DeviceMode.CUDA, gpu_devices=selected, workers=1)

    if profile is ExecutionProfile.CUDA_REPLICA:
        _compatible_device(config, DeviceMode.CUDA)
        selected = config.gpu_devices
        if selected is None:
            selected = gpu_probe()
        if len(selected) < 2:
            raise ProfileResolutionError("cuda-replica profile requires at least two usable GPUs")
        if config.workers not in (None, len(selected)):
            raise ProfileResolutionError(
                "cuda-replica profile requires one worker per selected GPU"
            )
        return replace(
            config,
            device=DeviceMode.CUDA,
            gpu_devices=selected,
            workers=len(selected),
        )

    if profile is ExecutionProfile.AUTO:
        if config.device is DeviceMode.CPU:
            if config.gpu_devices is not None:
                raise ProfileResolutionError("auto profile with CPU execution cannot select GPUs")
            if config.workers not in (None, 1):
                raise ProfileResolutionError("auto CPU execution supports exactly one worker")
            return replace(config, workers=1)

        if config.gpu_devices is not None:
            selected = config.gpu_devices
        else:
            selected = gpu_probe()

        if selected:
            if config.workers is not None and config.workers > len(selected):
                raise ProfileResolutionError(
                    "auto profile cannot oversubscribe selected GPUs"
                )
            workers = len(selected) if config.workers is None else config.workers
            return replace(
                config,
                device=DeviceMode.CUDA,
                gpu_devices=selected,
                workers=workers,
            )

        if config.device is DeviceMode.CUDA:
            raise ProfileResolutionError("auto profile was constrained to CUDA but CUDA is unavailable")
        if config.workers not in (None, 1):
            raise ProfileResolutionError("auto CPU execution supports exactly one worker")
        return replace(config, device=DeviceMode.CPU, gpu_devices=None, workers=1)

    raise AssertionError("unreachable")


__all__: Final[tuple[str, ...]] = (
    "ProfileResolutionError",
    "materialize_execution_profile",
    "probe_runtime_gpu_indices",
)
