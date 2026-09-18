from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import RuntimeConfig
from .device import ExecutionPlan, HardwareInventory, detect_hardware, resolve_execution_plan


_PACKAGE_NAMES = (
    "torch",
    "transformers",
    "numpy",
    "soundfile",
    "librosa",
    "torchaudio",
    "scipy",
    "av",
    "fastapi",
    "pydantic",
    "voxcpm",
)


@dataclass(frozen=True, slots=True)
class EnvironmentInfo:
    python_version: str
    platform: str
    torch_version: str | None = None
    cuda_runtime_version: str | None = None
    cuda_driver_version: str | None = None
    packages: Mapping[str, str | None] = None
    source_git_commit: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "packages", dict(self.packages or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "python": self.python_version,
            "platform": self.platform,
            "torch": self.torch_version,
            "cuda_runtime_version": self.cuda_runtime_version,
            "cuda_driver_version": self.cuda_driver_version,
            "packages": dict(self.packages),
            "source_git_commit": self.source_git_commit,
        }


@dataclass(frozen=True, slots=True)
class SourceLockInfo:
    locked_upstream_commit: str | None = None
    configured_upstream_revision: str | None = None
    upstream_revision_matches: bool | None = None
    source_lock_status: str = "unavailable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "locked_upstream_commit": self.locked_upstream_commit,
            "configured_upstream_revision": self.configured_upstream_revision,
            "upstream_revision_matches": self.upstream_revision_matches,
            "source_lock_status": self.source_lock_status,
        }


@dataclass(frozen=True, slots=True)
class DoctorReport:
    project_name: str
    project_version: str
    config: Mapping[str, Any]
    environment: Mapping[str, Any]
    hardware: Mapping[str, Any] | None
    execution: Mapping[str, Any] | None
    provenance: Mapping[str, Any]
    warnings: tuple[str, ...] = ()
    errors: tuple[Mapping[str, str], ...] = ()
    status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": self.status,
            "project": {
                "name": self.project_name,
                "version": self.project_version,
            },
            "config": dict(self.config),
            "environment": dict(self.environment),
            "hardware": dict(self.hardware) if self.hardware is not None else None,
            "execution": dict(self.execution) if self.execution is not None else None,
            "provenance": dict(self.provenance),
            "warnings": list(self.warnings),
            "errors": [dict(error) for error in self.errors],
        }


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _project_version() -> str:
    from . import __version__

    return __version__


def _source_lock_candidates() -> tuple[Path, ...]:
    package_root = Path(__file__).resolve().parents[1]
    candidates = [package_root / "provenance" / "voxcpm-source-lock.json"]
    candidates.append(Path(sys.prefix) / "provenance" / "voxcpm-source-lock.json")
    try:
        candidates.append(Path.cwd() / "provenance" / "voxcpm-source-lock.json")
    except OSError:
        pass
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return tuple(unique)


def read_source_lock() -> tuple[str | None, str]:
    for path in _source_lock_candidates():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None, "invalid"
        if not isinstance(payload, dict):
            return None, "invalid"
        upstream = payload.get("upstream_code")
        if not isinstance(upstream, dict):
            return None, "invalid"
        commit = upstream.get("commit")
        if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40,64}", commit) is None:
            return None, "invalid"
        return commit, "loaded"
    return None, "unavailable"


def _git_environment() -> dict[str, str]:
    allowed = ("PATH", "HOME", "LANG", "LC_ALL", "SYSTEMROOT", "GIT_CONFIG_NOSYSTEM")
    return {key: os.environ[key] for key in allowed if key in os.environ}


def discover_git_commit() -> str | None:
    executable = shutil.which("git")
    if executable is None:
        return None
    package_root = Path(__file__).resolve().parents[1]
    try:
        roots = (package_root, Path.cwd())
    except OSError:
        roots = (package_root,)
    for root in roots:
        try:
            result = subprocess.run(
                [executable, "-C", str(root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
                env=_git_environment(),
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode == 0:
            value = result.stdout.strip()
            if re.fullmatch(r"[0-9a-f]{40,64}", value):
                return value
    return None


def collect_environment(inventory: HardwareInventory | None = None) -> EnvironmentInfo:
    packages = {name: _package_version(name) for name in _PACKAGE_NAMES}
    torch_version = (inventory.torch_version or packages.get("torch")) if inventory is not None else packages.get("torch")
    return EnvironmentInfo(
        python_version=platform.python_version(),
        platform=platform.platform(),
        torch_version=torch_version,
        cuda_runtime_version=inventory.cuda_runtime_version if inventory is not None else None,
        cuda_driver_version=inventory.cuda_driver_version if inventory is not None else None,
        packages=packages,
        source_git_commit=discover_git_commit(),
    )


def collect_provenance(config: RuntimeConfig) -> tuple[SourceLockInfo, tuple[str, ...]]:
    locked_commit, status = read_source_lock()
    configured = config.upstream_revision
    matches = locked_commit == configured if configured is not None and locked_commit is not None else None
    warnings: list[str] = []
    if configured is not None and locked_commit is not None and not matches:
        warnings.append("configured upstream revision does not match the source lock")
    return (
        SourceLockInfo(
            locked_upstream_commit=locked_commit,
            configured_upstream_revision=configured,
            upstream_revision_matches=matches,
            source_lock_status=status,
        ),
        tuple(warnings),
    )


def _config_summary(config: RuntimeConfig) -> dict[str, Any]:
    return config.to_diagnostic_dict()


def build_doctor_report(
    config: RuntimeConfig,
    inventory: HardwareInventory | None = None,
) -> DoctorReport:
    selected_inventory = inventory if inventory is not None else detect_hardware()
    plan: ExecutionPlan = resolve_execution_plan(config, selected_inventory)
    provenance, warnings = collect_provenance(config)
    return DoctorReport(
        project_name="openbmb-voxcpm2-inference",
        project_version=_project_version(),
        config=_config_summary(config),
        environment=collect_environment(selected_inventory).to_dict(),
        hardware=selected_inventory.to_dict(),
        execution=plan.to_dict(),
        provenance=provenance.to_dict(),
        warnings=warnings,
        errors=(),
        status="ok",
    )


__all__ = [
    "DoctorReport",
    "EnvironmentInfo",
    "SourceLockInfo",
    "build_doctor_report",
    "collect_environment",
    "collect_provenance",
    "discover_git_commit",
    "read_source_lock",
]
