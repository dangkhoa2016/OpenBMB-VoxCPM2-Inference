from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import pytest

from scripts.generate import (
    _build_request,
    _dispatch_one_shot,
    _parser,
    _request_summary,
    main,
)
from voxcpm_runtime.backend_types import (
    AudioReference,
    CloneRequest,
    ContinuationRequest,
    SpeechRequest,
    VoiceDesignRequest,
)

_REFERENCE = "/private/qualification/workspace/reference.wav"
_TRANSCRIPT = "Xin chào, đây là giọng nói tham chiếu."
_INSTRUCTION = "giọng nữ ấm áp, bình tĩnh"


def _namespace(**flags: Any) -> Any:
    """Parse a CLI namespace from keyword flags plus a fixed output path."""

    return _parser().parse_args(["--output", "/tmp/out.wav", *_argv(flags)])


def _argv(flags: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for name, value in flags.items():
        argv.append(f"--{name.replace('_', '-')}")
        argv.append(str(value))
    return argv


# --- operation selection -----------------------------------------------


def test_no_feature_flags_selects_standard_tts() -> None:
    request, operation, conditioning, problem = _build_request(
        _namespace(text="Xin chào từ VoxCPM2.")
    )
    assert problem is None
    assert operation == "standard-tts"
    assert conditioning == "none"
    assert request == SpeechRequest(text="Xin chào từ VoxCPM2.")


def test_voice_instruction_selects_voice_design() -> None:
    request, operation, conditioning, problem = _build_request(
        _namespace(text="Xin chào từ VoxCPM2.", voice_instruction=_INSTRUCTION)
    )
    assert problem is None
    assert operation == "voice-design"
    assert conditioning == "instruction"
    assert request == VoiceDesignRequest(
        text="Xin chào từ VoxCPM2.", instruction=_INSTRUCTION
    )


def test_reference_audio_selects_voice_clone() -> None:
    request, operation, conditioning, problem = _build_request(
        _namespace(
            text="Xin chào, đây là phép thử clone giọng.",
            reference_audio=_REFERENCE,
        )
    )
    assert problem is None
    assert operation == "voice-clone"
    assert conditioning == "reference"
    assert isinstance(request, CloneRequest)
    assert request.reference_audio.local_path == _REFERENCE


def test_prompt_pair_selects_audio_continuation() -> None:
    request, operation, conditioning, problem = _build_request(
        _namespace(
            text="Và đây là phần tiếp theo.",
            prompt_audio=_REFERENCE,
            prompt_text=_TRANSCRIPT,
        )
    )
    assert problem is None
    assert operation == "audio-continuation"
    assert conditioning == "continuation"
    assert isinstance(request, ContinuationRequest)
    assert request.reference_transcript == _TRANSCRIPT
    assert request.reference_audio.local_path == _REFERENCE


# --- mutually exclusive validation -------------------------------------


@pytest.mark.parametrize(
    "flags",
    [
        {"text": "a", "voice_instruction": "b", "reference_audio": _REFERENCE},
        {"text": "a", "voice_instruction": "b", "prompt_audio": _REFERENCE},
        {"text": "a", "voice_instruction": "b", "prompt_text": _TRANSCRIPT},
        {"text": "a", "voice_instruction": "b", "reference_audio": _REFERENCE, "prompt_text": _TRANSCRIPT},
    ],
)
def test_voice_instruction_cannot_combine_with_audio(flags: dict[str, str]) -> None:
    request, _, _, problem = _build_request(_namespace(**flags))
    assert request is None
    assert problem is not None
    assert "--voice-instruction" in problem


@pytest.mark.parametrize(
    "flags",
    [
        {"text": "a", "reference_audio": _REFERENCE, "prompt_audio": _REFERENCE},
        {"text": "a", "reference_audio": _REFERENCE, "prompt_text": _TRANSCRIPT},
    ],
)
def test_reference_audio_cannot_combine_with_prompt(flags: dict[str, str]) -> None:
    request, _, _, problem = _build_request(_namespace(**flags))
    assert request is None
    assert problem is not None
    assert "--reference-audio" in problem


@pytest.mark.parametrize(
    "flags",
    [
        {"text": "a", "prompt_audio": _REFERENCE},
        {"text": "a", "prompt_text": _TRANSCRIPT},
        {"text": "a", "prompt_audio": "   ", "prompt_text": _TRANSCRIPT},
    ],
)
def test_prompt_audio_and_prompt_text_must_travel_together(
    flags: dict[str, str],
) -> None:
    request, _, _, problem = _build_request(_namespace(**flags))
    assert request is None
    assert problem is not None
    assert "--prompt-audio" in problem


@pytest.mark.parametrize(
    "flags",
    [
        {"text": "a", "voice_instruction": "   "},
        {"text": "a", "reference_audio": "   "},
        {"text": "a", "prompt_audio": "   ", "prompt_text": "   "},
    ],
)
def test_blank_feature_flags_do_not_select_an_operation(flags: dict[str, str]) -> None:
    request, operation, conditioning, problem = _build_request(_namespace(**flags))
    assert problem is None
    assert operation == "standard-tts"
    assert conditioning == "none"
    assert isinstance(request, SpeechRequest)


# --- public failure -----------------------------------------------------


@pytest.mark.parametrize(
    "flags",
    [
        {
            "text": "a",
            "voice_instruction": "b",
            "reference_audio": _REFERENCE,
        },
        {"text": "a", "reference_audio": _REFERENCE, "prompt_text": _TRANSCRIPT},
        {"text": "a", "prompt_audio": _REFERENCE},
    ],
)
def test_invalid_combination_exits_two_without_loading_a_model(
    flags: dict[str, str],
) -> None:
    assert main(["--output", "/tmp/unused.wav", *_argv(flags)]) == 2


def test_invalid_combination_reports_a_structured_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "--output",
            "/tmp/unused.wav",
            "--text",
            "a",
            "--voice-instruction",
            "b",
            "--reference-audio",
            _REFERENCE,
        ]
    )
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "invalid_request"
    assert "--voice-instruction" in payload["error"]["message"]
    assert _REFERENCE not in json.dumps(payload)


def test_missing_text_exits_two() -> None:
    assert main(["--output", "/tmp/unused.wav"]) == 2


def test_missing_output_exits_two() -> None:
    assert main(["--text", "a"]) == 2


# --- safe request summary ----------------------------------------------


def test_request_summary_never_leaks_content_or_paths() -> None:
    request, _, _, _ = _build_request(
        _namespace(
            text="Xin chào, đây là phép thử clone giọng.",
            reference_audio=_REFERENCE,
        )
    )
    assert request is not None
    summary = _request_summary(request, "Xin chào, đây là phép thử clone giọng.", None)
    serialized = json.dumps(summary)
    assert summary["reference_audio_used"] is True
    assert summary["reference_input_kind"] == "local-file"
    assert summary["reference_transcript_characters"] == 0
    assert summary["instruction_characters"] == 0
    assert _REFERENCE not in serialized
    assert "Xin chào" not in serialized


def test_request_summary_counts_characters_only() -> None:
    request, _, _, _ = _build_request(
        _namespace(text="Xin chào từ VoxCPM2.", voice_instruction=_INSTRUCTION)
    )
    assert request is not None
    summary = _request_summary(request, "Xin chào từ VoxCPM2.", _INSTRUCTION)
    assert summary["instruction_characters"] == len(_INSTRUCTION)
    assert summary["input_text_characters"] == len("Xin chào từ VoxCPM2.")
    assert summary["reference_audio_used"] is False
    assert summary["reference_transcript_characters"] == 0


def test_continuation_summary_counts_transcript_not_content() -> None:
    request, _, _, _ = _build_request(
        _namespace(
            text="Và đây là phần tiếp theo.",
            prompt_audio=_REFERENCE,
            prompt_text=_TRANSCRIPT,
        )
    )
    assert request is not None
    summary = _request_summary(request, "Và đây là phần tiếp theo.", None)
    assert summary["reference_transcript_characters"] == len(_TRANSCRIPT)
    assert summary["reference_audio_used"] is True
    serialized = json.dumps(summary)
    assert _TRANSCRIPT not in serialized
    assert _REFERENCE not in serialized


# --- dispatch routing ---------------------------------------------------


def test_dispatch_routes_each_request_to_its_operation() -> None:
    calls: list[str] = []

    class _RecordingBackend:
        def synthesize(self, request: SpeechRequest) -> str:
            calls.append("synthesize")
            return request.text

        def design(self, request: VoiceDesignRequest) -> str:
            calls.append("design")
            return request.instruction

        def clone(self, request: CloneRequest) -> str:
            calls.append("clone")
            return request.reference_audio.local_path

        def continue_audio(self, request: ContinuationRequest) -> str:
            calls.append("continue_audio")
            return request.reference_transcript

    backend = _RecordingBackend()
    assert _dispatch_one_shot(backend, SpeechRequest(text="a")) == "a"  # type: ignore[arg-type]
    assert _dispatch_one_shot(backend, VoiceDesignRequest(text="a", instruction="b")) == "b"  # type: ignore[arg-type]
    clone = CloneRequest(text="a", reference_audio=AudioReference(local_path=_REFERENCE))
    assert _dispatch_one_shot(backend, clone) == _REFERENCE  # type: ignore[arg-type]
    continuation = ContinuationRequest(
        text="a",
        reference_audio=AudioReference(local_path=_REFERENCE),
        reference_transcript=_TRANSCRIPT,
    )
    assert _dispatch_one_shot(backend, continuation) == _TRANSCRIPT  # type: ignore[arg-type]
    assert calls == ["synthesize", "design", "clone", "continue_audio"]


def test_dispatch_rejects_unknown_request() -> None:
    from voxcpm_runtime.errors import BackendRequestError

    with pytest.raises(BackendRequestError):
        _dispatch_one_shot(object(), object())  # type: ignore[arg-type]


# --- CLI surface --------------------------------------------------------


def test_help_lists_every_m6_flag() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.generate", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    for flag in (
        "--text",
        "--output",
        "--report",
        "--load-only",
        "--voice-instruction",
        "--reference-audio",
        "--prompt-audio",
        "--prompt-text",
    ):
        assert flag in completed.stdout, flag
    for absent in ("--stream", "--chunk", "--queue"):
        assert absent not in completed.stdout, absent
