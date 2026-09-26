from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from voxcpm_runtime.backend_types import (
    AudioReference,
    BackendInfo,
    CloneRequest,
    ContinuationRequest,
    OneShotRequest,
    SpeechRequest,
    VoiceDesignRequest,
)
from voxcpm_runtime.config import ConfigurationError, OptimizationMode, RuntimeConfig
from voxcpm_runtime.device import DeviceManager, DeviceResolutionError
from voxcpm_runtime.diagnostics import read_source_lock
from voxcpm_runtime.errors import BackendError, BackendRequestError
from voxcpm_runtime.model_resolver import ModelResolutionError, ModelResolver
from voxcpm_runtime.pytorch_backend import (
    PytorchVoxCPMBackend,
    inspect_model_devices,
    resolve_upstream_optimize,
)
from voxcpm_runtime.runtime_metrics import (
    Stopwatch,
    cgroup_memory_events,
    collect_host_facts,
    current_rss_bytes,
    dependency_versions,
    observe_gpu_memory,
    observe_torch_cuda,
    redact_value,
)
from voxcpm_runtime.wav_io import WavWriteError, validate_wav, write_wav

_TRACKED_DISTRIBUTIONS: tuple[str, ...] = (
    "voxcpm",
    "torch",
    "torchaudio",
    "transformers",
    "huggingface-hub",
    "numpy",
    "einops",
    "pydantic",
    "tqdm",
    "librosa",
    "safetensors",
)


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(
        json.dumps(redact_value(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voxcpm-generate",
        description=(
            "Synthesize speech with a local VoxCPM2 model through the "
            "project-owned backend boundary."
        ),
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Text to synthesize. Required unless --load-only is used.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Destination path for the generated mono PCM16 WAV file.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional path for a machine-readable JSON runtime report.",
    )
    parser.add_argument(
        "--voice-instruction",
        default=None,
        help=(
            "Voice design instruction. Selects the voice-design operation and "
            "cannot be combined with reference or prompt audio."
        ),
    )
    parser.add_argument(
        "--reference-audio",
        default=None,
        help=(
            "Local reference WAV for the voice-clone operation. It is read "
            "from the local filesystem and is never fetched remotely."
        ),
    )
    parser.add_argument(
        "--prompt-audio",
        default=None,
        help=(
            "Local prefix WAV for the audio-continuation operation. It must be "
            "used together with --prompt-text."
        ),
    )
    parser.add_argument(
        "--prompt-text",
        default=None,
        help="Transcript of --prompt-audio for the audio-continuation operation.",
    )
    parser.add_argument(
        "--load-only",
        action="store_true",
        help=(
            "Load the real model, record device and memory facts, then exit "
            "without generating audio."
        ),
    )
    return parser


_OPERATION_STANDARD: str = "standard-tts"
_OPERATION_DESIGN: str = "voice-design"
_OPERATION_CLONE: str = "voice-clone"
_OPERATION_CONTINUATION: str = "audio-continuation"
_OPERATION_LOAD_ONLY: str = "load-only"


def _present(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _build_request(
    namespace: argparse.Namespace,
) -> tuple[OneShotRequest | None, str, str, str | None]:
    """Resolve CLI flags into one project request plus its report labels.

    The fourth element is a public validation problem, or None when the flag
    combination selects exactly one qualified operation.
    """

    instruction = namespace.voice_instruction
    reference = namespace.reference_audio
    prompt_audio = namespace.prompt_audio
    prompt_text = namespace.prompt_text
    text = namespace.text

    if _present(instruction) and (
        _present(reference) or _present(prompt_audio) or _present(prompt_text)
    ):
        return None, "", "", (
            "--voice-instruction cannot be combined with --reference-audio, "
            "--prompt-audio, or --prompt-text."
        )
    if _present(reference) and (_present(prompt_audio) or _present(prompt_text)):
        return None, "", "", (
            "--reference-audio cannot be combined with --prompt-audio or "
            "--prompt-text."
        )
    if _present(prompt_audio) != _present(prompt_text):
        return None, "", "", (
            "--prompt-audio and --prompt-text must be used together."
        )
    if _present(instruction):
        return (
            VoiceDesignRequest(text=text or "", instruction=instruction),
            _OPERATION_DESIGN,
            "instruction",
            None,
        )
    if _present(reference):
        return (
            CloneRequest(
                text=text or "",
                reference_audio=AudioReference(local_path=reference),
            ),
            _OPERATION_CLONE,
            "reference",
            None,
        )
    if _present(prompt_audio):
        return (
            ContinuationRequest(
                text=text or "",
                reference_audio=AudioReference(local_path=prompt_audio),
                reference_transcript=prompt_text,
            ),
            _OPERATION_CONTINUATION,
            "continuation",
            None,
        )
    return SpeechRequest(text=text or ""), _OPERATION_STANDARD, "none", None


def _request_summary(
    request: OneShotRequest,
    text: str | None,
    instruction: str | None,
) -> dict[str, Any]:
    """Summarize a request using counts only, never content or paths."""

    uses_reference = isinstance(request, (CloneRequest, ContinuationRequest))
    return {
        "input_text_characters": len(text or ""),
        "instruction_characters": len(instruction.strip()) if _present(instruction) else 0,
        "reference_audio_used": uses_reference,
        "reference_input_kind": "local-file" if uses_reference else "none",
        "reference_transcript_characters": (
            len(request.reference_transcript)
            if isinstance(request, ContinuationRequest)
            else 0
        ),
    }


def _dispatch_one_shot(
    backend: PytorchVoxCPMBackend,
    request: OneShotRequest,
) -> Any:
    """Route one project request to its qualified backend operation."""

    if isinstance(request, VoiceDesignRequest):
        return backend.design(request)
    if isinstance(request, CloneRequest):
        return backend.clone(request)
    if isinstance(request, ContinuationRequest):
        return backend.continue_audio(request)
    if isinstance(request, SpeechRequest):
        return backend.synthesize(request)
    raise BackendRequestError(
        "Backend request is invalid.",
        details={"field": "request"},
    )


def _resolve_model(config: RuntimeConfig) -> Any:
    candidates: tuple[Any, ...] = ()
    if config.model_path is None:
        from deploy.kaggle.model_mounts import discover_kaggle_model_candidates

        candidates = discover_kaggle_model_candidates()
    return ModelResolver().resolve(config, deployment_candidates=candidates)


def _model_report(resolved: Any) -> dict[str, Any]:
    return {
        "architecture": resolved.config_identity.architecture,
        "config_sha256": resolved.config_identity.sha256,
        "file_count": resolved.file_count,
        "model_id": resolved.model_id,
        "path_configured": True,
        "revision": resolved.revision,
        "source_kind": resolved.source_kind,
        "total_bytes": resolved.total_bytes,
    }


def _backend_report(info: BackendInfo, backend: PytorchVoxCPMBackend) -> dict[str, Any]:
    return {
        "backend": info.backend,
        "capabilities": list(info.capabilities),
        "device": info.device,
        "implementation": info.implementation,
        "load_denoiser": backend.load_denoiser,
        "local_files_only": backend.local_files_only,
        "model_id": info.model_id,
        "upstream_device": backend.upstream_device,
        "upstream_optimize": backend.effective_optimize,
    }


def _write_report(destination: str, payload: dict[str, Any]) -> None:
    target = Path(destination)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(redact_value(payload), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
    except OSError:
        sys.stderr.write("The runtime report could not be written.\n")


def _finalize(report: dict[str, Any], destination: str | None) -> None:
    report["cgroup_memory_events"] = cgroup_memory_events()
    _emit(report)
    if destination is not None:
        _write_report(destination, report)


def main(arguments: Sequence[str] | None = None) -> int:
    namespace = _parser().parse_args(arguments)
    text: str | None = namespace.text
    load_only: bool = bool(namespace.load_only)
    report_path: str | None = namespace.report

    if not load_only and (not isinstance(text, str) or not text.strip()):
        return _emit_error(
            "invalid_request",
            "--text must be a non-empty string unless --load-only is used.",
            "ArgumentError",
        )
    if not load_only and not isinstance(namespace.output, str):
        return _emit_error(
            "invalid_request",
            "--output is required unless --load-only is used.",
            "ArgumentError",
        )

    request: OneShotRequest | None = None
    operation = _OPERATION_LOAD_ONLY if load_only else _OPERATION_STANDARD
    conditioning = "none"
    if not load_only:
        request, operation, conditioning, problem = _build_request(namespace)
        if request is None:
            return _emit_error(
                "invalid_request",
                problem or "The requested operation combination is not supported.",
                "ArgumentError",
            )

    try:
        config = RuntimeConfig.from_env()
    except ConfigurationError as error:
        return _emit_error("configuration_error", str(error), type(error).__name__)

    try:
        resolved = _resolve_model(config)
    except ModelResolutionError as error:
        return _emit_error(
            error.code,
            "The local model could not be resolved.",
            type(error).__name__,
        )
    except Exception as error:
        return _emit_error(
            "model_discovery_error",
            "Local model discovery failed unexpectedly.",
            type(error).__name__,
        )

    device_manager = DeviceManager()
    try:
        inventory = device_manager.detect()
        execution_plan = device_manager.resolve(
            config,
            gpu_devices=config.gpu_devices,
            workers=config.workers,
            inventory=inventory,
        )
    except DeviceResolutionError as error:
        return _emit_error(
            "device_resolution_error",
            "The execution plan could not be resolved.",
            type(error).__name__,
        )
    except Exception as error:
        return _emit_error(
            "device_discovery_error",
            "Device discovery failed unexpectedly.",
            type(error).__name__,
        )

    try:
        effective_optimize = resolve_upstream_optimize(
            execution_plan,
            OptimizationMode(config.optimize),
        )
    except BackendError as error:
        return _emit_error(error.code, error.message, type(error).__name__)
    except TypeError as error:
        return _emit_error("invalid_configuration", str(error), type(error).__name__)

    backend = PytorchVoxCPMBackend(
        resolved_model=resolved,
        execution_plan=execution_plan,
        load_denoiser=config.load_denoiser,
        local_files_only=config.offline,
        optimize=effective_optimize,
    )

    locked_commit, lock_status = read_source_lock()
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "ok",
        "project": {"name": "openbmb-voxcpm2-inference", "milestone": "M6"},
        "operation": operation,
        "request": _request_summary(request, text, namespace.voice_instruction) if request else {},
        "config": {
            "backend": config.backend.value,
            "device_requested": config.requested_device,
            "gpu_devices": config.gpu_devices_spec,
            "load_denoiser": config.load_denoiser,
            "offline": config.offline,
            "optimize_mode": config.optimize.value,
            "workers": config.workers_spec,
        },
        "model": _model_report(resolved),
        "hardware": inventory.to_dict(),
        "execution_plan": execution_plan.to_dict(),
        "host": collect_host_facts().to_dict(),
        "upstream": {
            "dependency_versions": dependency_versions(_TRACKED_DISTRIBUTIONS),
            "locked_source_commit": locked_commit,
            "source_lock_status": lock_status,
        },
        "timings": {},
        "memory": {"rss_before_load_bytes": current_rss_bytes()},
        "cuda": {},
        "gpu_memory": None,
        "backend": {},
        "audio": None,
        "wav": None,
        "wav_validation": None,
        "notes": [
            "Absolute filesystem paths are withheld from this public report.",
            "Model weights are never committed to this repository.",
        ],
    }

    rss_before_load = current_rss_bytes()
    load_timer = Stopwatch("load")
    try:
        info = backend.load()
    except BackendError as error:
        load_timer.stop()
        report["status"] = "error"
        report["error"] = error.to_dict()
        report["timings"]["load"] = load_timer.to_dict()
        report["memory"] = {
            "rss_after_load_bytes": current_rss_bytes(),
            "rss_before_load_bytes": rss_before_load,
        }
        report["host"] = collect_host_facts().to_dict()
        _finalize(report, report_path)
        return 3
    load_timer.stop()

    import torch

    report["cuda"] = observe_torch_cuda(torch).to_dict()
    if execution_plan.device == "cuda":
        report["gpu_memory"] = observe_gpu_memory(
            torch, execution_plan.selected_gpu_indices[0]
        ).to_dict()
    report["timings"]["load"] = load_timer.to_dict()
    report["memory"]["rss_after_load_bytes"] = current_rss_bytes()
    report["host"] = collect_host_facts().to_dict()
    report["backend"] = _backend_report(info, backend)
    try:
        report["model_devices"] = inspect_model_devices(
            backend.loaded_model, expected_device=execution_plan.device
        ).to_dict()
    except BackendError as error:
        report["model_devices"] = {"error": error.to_dict()}

    if load_only:
        backend.close()
        _finalize(report, report_path)
        return 0

    generate_timer = Stopwatch(operation)
    try:
        result = _dispatch_one_shot(backend, request)
    except BackendError as error:
        generate_timer.stop()
        backend.close()
        report["status"] = "error"
        report["error"] = error.to_dict()
        report["timings"][operation] = generate_timer.to_dict()
        report["memory"]["rss_after_generation_bytes"] = current_rss_bytes()
        report["host"] = collect_host_facts().to_dict()
        _finalize(report, report_path)
        return 3
    generate_timer.stop()

    report["timings"][operation] = generate_timer.to_dict()
    report["memory"]["rss_after_generation_bytes"] = current_rss_bytes()
    if execution_plan.device == "cuda":
        report["gpu_memory"] = observe_gpu_memory(
            torch, execution_plan.selected_gpu_indices[0]
        ).to_dict()
    report["audio"] = {
        "channels": result.channels,
        "conditioning": conditioning,
        "duration_seconds": round(len(result.samples) / result.sample_rate_hz, 6),
        "input_text_characters": len(text or ""),
        "sample_count": len(result.samples),
        "sample_rate_hz": result.sample_rate_hz,
    }
    report["host"] = collect_host_facts().to_dict()

    try:
        written = write_wav(namespace.output, result, create_parents=True)
    except WavWriteError:
        backend.close()
        report["status"] = "error"
        report["error"] = {
            "code": "wav_write_failed",
            "message": "The WAV output could not be written.",
            "retryable": False,
        }
        _finalize(report, report_path)
        return 4
    report["wav"] = written.to_dict()

    validation = validate_wav(
        namespace.output,
        expected_sample_rate_hz=result.sample_rate_hz,
        expected_channel_count=result.channels,
    )
    report["wav_validation"] = validation.to_dict()
    backend.close()

    if not validation.valid:
        report["status"] = "error"
        report["error"] = {
            "code": "wav_validation_failed",
            "message": "The written WAV file did not pass validation.",
            "retryable": False,
        }
        _finalize(report, report_path)
        return 4

    _finalize(report, report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
