import json

import pytest

from voxcpm_runtime.config import ConfigurationError, RuntimeConfig, parse_gpu_devices, parse_workers


def test_defaults_preserve_unspecified_numeric_values():
    config = RuntimeConfig.from_mapping({})
    assert config.backend == "pytorch-voxcpm"
    assert config.offline is True
    assert config.load_denoiser is False
    assert config.profile is None
    assert config.device == "auto"
    assert config.gpu_devices is None
    assert config.workers is None
    assert config.host == "127.0.0.1"
    assert config.port == 8090
    assert config.require_auth is True
    assert config.allow_unauthenticated_external is False
    assert config.max_text_chars is None
    assert config.max_reference_bytes is None
    assert config.max_reference_seconds is None
    assert config.max_prompt_audio_seconds is None
    assert config.max_output_seconds is None
    assert config.max_queue_size is None
    assert config.queue_timeout_seconds is None
    assert config.request_timeout_seconds is None
    assert config.max_inference_seconds is None
    assert config.max_concurrent_requests is None
    assert config.stream_ipc_max_chunks is None
    assert config.stream_ipc_max_bytes is None
    assert config.stream_backpressure_timeout_seconds is None
    assert config.min_tmp_free_bytes is None
    assert config.readiness_mode == "degraded"
    assert config.qualification_strict is False


def test_boolean_forms_are_case_insensitive():
    values = {
        "VOXCPM_OFFLINE": "OFF",
        "VOXCPM_LOAD_DENOISER": "YeS",
        "VOXCPM_REQUIRE_AUTH": "0",
        "VOXCPM_ALLOW_UNAUTHENTICATED_EXTERNAL": "true",
        "VOXCPM_QUALIFICATION_STRICT": "1",
    }
    config = RuntimeConfig.from_mapping(values)
    assert config.offline is False
    assert config.load_denoiser is True
    assert config.require_auth is False
    assert config.allow_unauthenticated_external is True
    assert config.qualification_strict is True


@pytest.mark.parametrize("value", ["maybe", "2", "truthy"])
def test_invalid_boolean_fails(value):
    with pytest.raises(ConfigurationError, match="VOXCPM_OFFLINE"):
        RuntimeConfig.from_mapping({"VOXCPM_OFFLINE": value})


def test_optional_numbers_are_parsed_without_inventing_defaults():
    config = RuntimeConfig.from_mapping(
        {
            "VOXCPM_MAX_TEXT_CHARS": "1000",
            "VOXCPM_MAX_REFERENCE_SECONDS": "2.5",
            "VOXCPM_STREAM_SAMPLE_RATE": "44100",
            "VOXCPM_STREAM_CHANNELS": "2",
        }
    )
    assert config.max_text_chars == 1000
    assert config.max_reference_seconds == 2.5
    assert config.stream_sample_rate == 44100
    assert config.stream_channels == 2


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("VOXCPM_MAX_TEXT_CHARS", "-1"),
        ("VOXCPM_MAX_REFERENCE_SECONDS", "nan"),
        ("VOXCPM_STREAM_SAMPLE_RATE", "0"),
        ("VOXCPM_PORT", "65536"),
        ("VOXCPM_READINESS_MODE", "ready"),
        ("VOXCPM_STREAM_FORMAT", "wav"),
    ],
)
def test_invalid_scalar_values_fail(key, value):
    with pytest.raises(ConfigurationError, match=key):
        RuntimeConfig.from_mapping({key: value})


@pytest.mark.parametrize("value", ["gpu", "CUDA0", "banana", "CUDA"])
def test_invalid_device_fails(value):
    with pytest.raises(ConfigurationError, match="VOXCPM_DEVICE"):
        RuntimeConfig.from_mapping({"VOXCPM_DEVICE": value})


@pytest.mark.parametrize("value", ["-1", "0,,1", "a", "0,a", "0,0", "auto,0"])
def test_invalid_gpu_list_fails(value):
    with pytest.raises(ConfigurationError, match="VOXCPM_GPU_DEVICES"):
        RuntimeConfig.from_mapping({"VOXCPM_GPU_DEVICES": value})


def test_gpu_list_preserves_explicit_order():
    assert parse_gpu_devices("1,0") == (1, 0)
    assert RuntimeConfig.from_mapping({"VOXCPM_GPU_DEVICES": "1,0"}).gpu_devices == (1, 0)


@pytest.mark.parametrize("value", ["0", "-1", "1.0", "two", "auto,1"])
def test_invalid_workers_fail(value):
    with pytest.raises(ConfigurationError, match="VOXCPM_WORKERS"):
        RuntimeConfig.from_mapping({"VOXCPM_WORKERS": value})


def test_workers_accept_auto_and_positive_integer():
    assert parse_workers("auto") is None
    assert parse_workers("3") == 3


def test_secret_is_absent_from_repr_and_redacted_dict():
    secret = "M1_DUMMY_SECRET_DO_NOT_USE"
    config = RuntimeConfig.from_mapping({"VOXCPM_API_TOKEN": secret})
    assert secret not in repr(config)
    assert secret not in json.dumps(config.to_redacted_dict())
    assert config.api_token_configured is True
    assert "api_token" not in config.to_redacted_dict()


def test_from_mapping_does_not_mutate_input():
    values = {"VOXCPM_DEVICE": "cpu", "VOXCPM_API_TOKEN": "value"}
    snapshot = dict(values)
    RuntimeConfig.from_mapping(values)
    assert values == snapshot


def test_empty_optional_values_are_unset_and_empty_defaults_are_safe():
    config = RuntimeConfig.from_mapping(
        {
            "VOXCPM_MODEL_PATH": "",
            "VOXCPM_UPSTREAM_REVISION": "",
            "VOXCPM_MAX_TEXT_CHARS": "",
            "VOXCPM_DEVICE": "",
            "VOXCPM_PORT": "",
        }
    )
    assert config.model_path is None
    assert config.upstream_revision is None
    assert config.max_text_chars is None
    assert config.device == "auto"
    assert config.port == 8090


def test_direct_construction_rejects_invalid_worker_types_clearly():
    with pytest.raises(ConfigurationError, match="VOXCPM_WORKERS"):
        RuntimeConfig(workers=1.5)
    with pytest.raises(ConfigurationError, match="VOXCPM_WORKERS"):
        RuntimeConfig(workers=0)
