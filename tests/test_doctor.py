import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_doctor(extra_environment=None):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    if extra_environment:
        environment.update(extra_environment)
    return subprocess.run(
        [sys.executable, "-m", "voxcpm_runtime.doctor"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_doctor_default_outputs_valid_json_and_succeeds():
    result = run_doctor()
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["project"]["version"] == "1.0.0"
    assert payload["execution"]["worker_count"] >= 1
    assert payload["hardware"]["cpu_logical_count"] is None or payload["hardware"]["cpu_logical_count"] >= 1


def test_doctor_explicit_cpu_succeeds_on_a_cuda_capable_host():
    result = run_doctor({"VOXCPM_DEVICE": "cpu"})
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["execution"]["effective_device"] == "cpu"
    assert payload["execution"]["selected_gpu_indices"] == []
    assert payload["execution"]["worker_count"] == 1


def test_doctor_invalid_config_returns_nonzero_without_traceback():
    result = run_doctor({"VOXCPM_DEVICE": "banana"})
    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert payload["errors"]


def test_doctor_invalid_gpu_list_returns_nonzero():
    result = run_doctor({"VOXCPM_GPU_DEVICES": "0,,1"})
    assert result.returncode != 0
    assert json.loads(result.stdout)["status"] == "error"


def test_doctor_does_not_reflect_invalid_configuration_values():
    secret = "M1_DUMMY_SECRET_DO_NOT_USE"
    result = run_doctor({"VOXCPM_API_TOKEN": secret, "VOXCPM_DEVICE": "banana"})
    assert result.returncode != 0
    assert secret not in result.stdout
    assert secret not in result.stderr
    assert json.loads(result.stdout)["status"] == "error"


def test_doctor_redacts_dummy_token():
    secret = "M1_DUMMY_SECRET_DO_NOT_USE"
    result = run_doctor({"VOXCPM_API_TOKEN": secret})
    assert result.returncode == 0, result.stderr
    assert secret not in result.stdout
    assert secret not in result.stderr
    payload = json.loads(result.stdout)
    assert payload["config"]["api_token_configured"] is True


def test_explicit_cuda_fails_on_cpu_only_host():
    probe = run_doctor()
    if probe.returncode != 0:
        return
    if json.loads(probe.stdout)["hardware"]["cuda_available"]:
        return
    result = run_doctor({"VOXCPM_DEVICE": "cuda"})
    assert result.returncode != 0
    assert "CUDA" in result.stderr
    assert "Traceback" not in result.stderr
    assert json.loads(result.stdout)["execution"] is None
