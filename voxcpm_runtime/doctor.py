from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from .config import ConfigurationError, RuntimeConfig
from .device import DeviceManager, DeviceResolutionError, HardwareInventory
from .diagnostics import collect_environment, collect_provenance, read_source_lock


def _project_version() -> str:
    from . import __version__

    return __version__


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _base_error_report(
    message: str,
    error_type: str,
    config: RuntimeConfig | None = None,
    inventory: HardwareInventory | None = None,
) -> dict[str, Any]:
    config_data = config.to_diagnostic_dict() if config is not None else None
    if inventory is not None:
        environment = collect_environment(inventory).to_dict()
        hardware = inventory.to_dict()
    else:
        environment = collect_environment().to_dict()
        hardware = None
    if config is not None:
        provenance, _ = collect_provenance(config)
        provenance_data = provenance.to_dict()
    else:
        locked_commit, status = read_source_lock()
        provenance_data = {
            "locked_upstream_commit": locked_commit,
            "configured_upstream_revision": None,
            "upstream_revision_matches": None,
            "source_lock_status": status,
        }
    return {
        "schema_version": 1,
        "status": "error",
        "project": {
            "name": "openbmb-voxcpm2-inference",
            "version": _project_version(),
        },
        "config": config_data,
        "environment": environment,
        "hardware": hardware,
        "execution": None,
        "provenance": provenance_data,
        "warnings": [],
        "errors": [{"type": error_type, "message": message}],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="voxcpm-doctor",
        description="Inspect VoxCPM2 runtime configuration and hardware without loading a model.",
        allow_abbrev=False,
    )
    parser.parse_args(argv)
    config: RuntimeConfig | None = None
    inventory: HardwareInventory | None = None
    try:
        config = RuntimeConfig.from_env()
        inventory = DeviceManager().detect()
        from .diagnostics import build_doctor_report

        report = build_doctor_report(config, inventory)
        _emit(report.to_dict())
        return 0
    except ConfigurationError as error:
        message = f"{error.variable}: invalid configuration value"
        _emit(_base_error_report(message, type(error).__name__, config, inventory))
        sys.stderr.write(f"voxcpm-doctor: {message}\n")
        return 2
    except DeviceResolutionError as error:
        message = str(error)
        _emit(_base_error_report(message, type(error).__name__, config, inventory))
        sys.stderr.write(f"voxcpm-doctor: {message}\n")
        return 2
    except Exception as error:
        _emit(_base_error_report("unexpected internal error", type(error).__name__, config, inventory))
        sys.stderr.write("voxcpm-doctor: unexpected internal error\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
