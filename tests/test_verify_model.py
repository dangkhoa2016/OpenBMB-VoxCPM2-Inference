import builtins
import json
import os
import subprocess
import sys
from pathlib import Path

import deploy.kaggle.model_mounts as model_mounts
import scripts.verify_model as verify_model


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VERIFY_SCRIPT = REPOSITORY_ROOT / "scripts" / "verify_model.py"


def _environment(**values: str) -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "VOXCPM_MODEL_PATH",
        "VOXCPM_CACHE_DIR",
        "VOXCPM_OFFLINE",
        "VOXCPM_API_TOKEN",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    ):
        environment.pop(name, None)
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
    environment.update(values)
    return environment


def _run(environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
        env=environment,
    )


def _clear_runtime_environment(monkeypatch) -> None:
    for name in (
        "VOXCPM_MODEL_PATH",
        "VOXCPM_CACHE_DIR",
        "VOXCPM_OFFLINE",
        "VOXCPM_API_TOKEN",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_verify_model_emits_structured_success(tmp_path, model_factory):
    model = model_factory(tmp_path / "model")
    secret = "verify-model-secret-placeholder"
    completed = _run(
        _environment(
            VOXCPM_MODEL_PATH=str(model),
            VOXCPM_OFFLINE="1",
            VOXCPM_API_TOKEN=secret,
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
        )
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert payload["status"] == "ok"
    assert payload["operation"] == "metadata-only"
    assert payload["path"] == str(model.resolve())
    assert payload["source_kind"] == "explicit-path"
    assert payload["model_id"] == "openbmb/VoxCPM2"
    assert payload["config_identity"]["architecture"] == "voxcpm2"
    assert payload["remote_acquisition_count"] == 0
    assert payload["inference_performed"] is False
    assert payload["file_count"] == 5
    assert payload["required_artifacts"] == {
        "audiovae_weight": "audiovae.safetensors",
        "config": "config.json",
        "main_weight": "model.safetensors",
        "tokenizer": ["tokenizer.json", "tokenizer_config.json"],
    }
    assert secret not in completed.stdout
    assert secret not in completed.stderr


def test_verify_model_returns_nonzero_for_invalid_explicit_model(
    tmp_path,
    model_factory,
):
    secret = "verify-model-invalid-architecture-placeholder"
    model = model_factory(
        tmp_path / "model",
        config={"architecture": secret},
    )
    completed = _run(
        _environment(
            VOXCPM_MODEL_PATH=str(model),
            VOXCPM_OFFLINE="1",
        )
    )

    assert completed.returncode == 2
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert payload["status"] == "error"
    assert payload["error"]["type"] == "UnexpectedModelArchitectureError"
    assert payload["error"]["code"] == "unexpected_model_architecture"
    assert secret not in completed.stdout
    assert secret not in completed.stderr


def test_verify_model_missing_explicit_path_does_not_fall_back(tmp_path):
    completed = _run(
        _environment(
            VOXCPM_MODEL_PATH=str(tmp_path / "missing"),
            VOXCPM_OFFLINE="1",
        )
    )

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["error"]["type"] == "ModelNotFoundError"
    assert payload["error"]["code"] == "model_not_found"


def test_verify_model_rejects_invalid_configuration(tmp_path, model_factory):
    model = model_factory(tmp_path / "model")
    completed = _run(
        _environment(
            VOXCPM_MODEL_PATH=str(model),
            VOXCPM_OFFLINE="maybe",
        )
    )

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["error"]["type"] == "ConfigurationError"
    assert payload["error"]["code"] == "configuration_error"


def test_explicit_path_skips_kaggle_discovery(
    monkeypatch,
    capsys,
    tmp_path,
    model_factory,
):
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv("VOXCPM_MODEL_PATH", str(model_factory(tmp_path / "model")))
    monkeypatch.setenv("VOXCPM_OFFLINE", "1")

    def fail_discovery():
        raise AssertionError("Kaggle discovery ran for an explicit model path")

    monkeypatch.setattr(
        model_mounts,
        "discover_kaggle_model_candidates",
        fail_discovery,
    )

    assert verify_model.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_kind"] == "explicit-path"


def test_kaggle_discovery_failure_is_structured(monkeypatch, capsys):
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv("VOXCPM_OFFLINE", "1")

    def fail_discovery():
        raise model_mounts.KaggleDiscoveryError("bounded discovery failed")

    monkeypatch.setattr(
        model_mounts,
        "discover_kaggle_model_candidates",
        fail_discovery,
    )

    assert verify_model.main() == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "kaggle_discovery_error"
    assert payload["error"]["type"] == "KaggleDiscoveryError"


def test_import_failure_is_structured(monkeypatch, capsys):
    real_import = builtins.__import__

    def fail_adapter_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "deploy.kaggle.model_mounts":
            raise ImportError("adapter import failed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_adapter_import)

    assert verify_model.main() == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "verification_import_error"
    assert payload["error"]["type"] == "ImportError"
