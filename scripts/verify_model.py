from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _emit_error(code: str, message: str, error_type: str) -> int:
    _emit(
        {
            "error": {
                "code": code,
                "message": message,
                "type": error_type,
            },
            "schema_version": 1,
            "status": "error",
        }
    )
    return 2


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    root_text = str(repository_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    try:
        from deploy.kaggle.model_mounts import (
            KaggleDiscoveryError,
            discover_kaggle_model_candidates,
        )
        from voxcpm_runtime.config import ConfigurationError, RuntimeConfig
        from voxcpm_runtime.model_resolver import ModelResolutionError, ModelResolver
    except Exception as error:
        _emit(
            {
                "error": {
                    "code": "verification_import_error",
                    "message": "Model verification dependencies could not be imported",
                    "type": type(error).__name__,
                },
                "schema_version": 1,
                "status": "error",
            }
        )
        return 1

    try:
        config = RuntimeConfig.from_env()
        candidates = (
            ()
            if config.model_path is not None
            else discover_kaggle_model_candidates()
        )
        resolved = ModelResolver().resolve(
            config,
            deployment_candidates=candidates,
        )
    except ConfigurationError as error:
        return _emit_error(
            "configuration_error",
            str(error),
            type(error).__name__,
        )
    except KaggleDiscoveryError as error:
        return _emit_error(
            "kaggle_discovery_error",
            str(error),
            type(error).__name__,
        )
    except ModelResolutionError as error:
        _emit(
            {
                "error": error.to_dict(),
                "schema_version": 1,
                "status": "error",
            }
        )
        return 2
    except Exception as error:
        _emit(
            {
                "error": {
                    "code": "unexpected_verification_error",
                    "message": "Model verification failed unexpectedly",
                    "type": type(error).__name__,
                },
                "schema_version": 1,
                "status": "error",
            }
        )
        return 1
    payload = resolved.to_dict()
    payload.update(
        {
            "inference_performed": False,
            "offline": config.offline,
            "operation": "metadata-only",
            "remote_acquisition_count": 0,
            "schema_version": 1,
            "status": "ok",
        }
    )
    _emit(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
