from pathlib import Path

from voxcpm_runtime.config import RuntimeConfig


def test_dockerfile_freezes_portable_cpu_contract() -> None:
    source = Path("Dockerfile").read_text(encoding="utf-8")
    assert "VOXCPM_MODEL_PATH=/models/VoxCPM2" in source
    assert "VOXCPM_PROFILE=cpu" in source
    assert "VOXCPM_DEVICE=cpu" in source
    assert "VOXCPM_OFFLINE=1" in source
    assert "VOXCPM_REQUIRE_AUTH=1" in source
    assert "VOXCPM_ALLOW_UNAUTHENTICATED_EXTERNAL=0" in source
    assert "HEALTHCHECK" in source
    assert 'CMD ["voxcpm-serve"]' in source
    assert "USER voxcpm" in source
    assert "f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69" in source


def test_dockerignore_excludes_model_weights_and_runtime_artifacts() -> None:
    lines = {
        line.strip()
        for line in Path(".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    for pattern in ("*.safetensors", "*.bin", "*.pth", "*.pt", "*.ckpt", "*.gguf", "*.log"):
        assert pattern in lines


def test_container_environment_maps_to_existing_runtime_contract() -> None:
    config = RuntimeConfig.from_mapping(
        {
            "VOXCPM_MODEL_PATH": "/models/VoxCPM2",
            "VOXCPM_PROFILE": "cpu",
            "VOXCPM_DEVICE": "cpu",
            "VOXCPM_HOST": "0.0.0.0",
            "VOXCPM_PORT": "8090",
            "VOXCPM_REQUIRE_AUTH": "1",
            "VOXCPM_API_TOKEN": "docker-test-token",
            "VOXCPM_ALLOW_UNAUTHENTICATED_EXTERNAL": "0",
            "VOXCPM_OFFLINE": "1",
        }
    )
    assert config.model_path == "/models/VoxCPM2"
    assert config.profile is not None and config.profile.value == "cpu"
    assert config.device.value == "cpu"
    assert config.host == "0.0.0.0"
    assert config.port == 8090
    assert config.require_auth is True
    assert config.api_token_configured is True
    assert config.allow_unauthenticated_external is False
    assert config.offline is True


def test_contract_smoke_is_test_only_and_uses_fake_worker() -> None:
    source = Path("deploy/docker/smoke_app.py").read_text(encoding="utf-8")
    assert 'backend_kind="fake"' in source
    assert "VOXCPM_API_TOKEN is required" in source
    assert "VOXCPM_MODEL_PATH must be a mounted directory" in source
    assert "smoke_app" not in Path("Dockerfile").read_text(encoding="utf-8").split('FROM base AS runtime', 1)[1]
