import json
import os
import subprocess
import sys
from pathlib import Path

from voxcpm_runtime.config import RuntimeConfig
from voxcpm_runtime.device import HardwareInventory
from voxcpm_runtime.diagnostics import (
    build_doctor_report,
    collect_environment,
    read_source_lock,
)


ROOT = Path(__file__).resolve().parents[1]
LOCKED_SHA = "f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69"


def test_source_lock_is_reported_without_network_access():
    commit, status = read_source_lock()
    assert status == "loaded"
    assert commit == LOCKED_SHA


def test_environment_report_is_serializable_and_contains_optional_versions():
    inventory = HardwareInventory(torch_version="test-torch", cuda_runtime_version="test-cuda")
    report = collect_environment(inventory).to_dict()
    json.dumps(report)
    assert report["python"]
    assert report["platform"]
    assert report["torch"] == "test-torch"
    for package in ("transformers", "numpy", "soundfile", "fastapi", "pydantic"):
        assert package in report["packages"]


def test_doctor_report_never_contains_api_token_value():
    secret = "M1_DUMMY_SECRET_DO_NOT_USE"
    config = RuntimeConfig.from_mapping({"VOXCPM_API_TOKEN": secret})
    inventory = HardwareInventory(cpu_logical_count=2, cuda_available=False)
    report = build_doctor_report(config, inventory).to_dict()
    encoded = json.dumps(report, sort_keys=True)
    assert secret not in encoded
    assert report["config"]["api_token_configured"] is True
    assert "api_token" not in report["config"]
    assert report["provenance"]["locked_upstream_commit"] == LOCKED_SHA


def test_doctor_report_redacts_configured_paths():
    values = {
        "VOXCPM_MODEL_PATH": "MODEL_PATH_SECRET",
        "VOXCPM_OUTPUT_DIR": "OUTPUT_PATH_SECRET",
        "VOXCPM_TMP_DIR": "TMP_PATH_SECRET",
        "VOXCPM_CACHE_DIR": "CACHE_PATH_SECRET",
        "VOXCPM_LOG_FILE": "LOG_PATH_SECRET",
    }
    config = RuntimeConfig.from_mapping(values)
    report = build_doctor_report(config, HardwareInventory(cuda_available=False)).to_dict()
    encoded = json.dumps(report, sort_keys=True)
    for secret in values.values():
        assert secret not in encoded
    for name in values:
        assert report["config"][f"{name.removeprefix('VOXCPM_').lower()}_configured"] is True


def test_configured_revision_mismatch_is_explicit():
    config = RuntimeConfig.from_mapping({"VOXCPM_UPSTREAM_REVISION": "not-the-lock"})
    inventory = HardwareInventory(cuda_available=False)
    report = build_doctor_report(config, inventory)
    assert report.provenance["upstream_revision_matches"] is False
    assert any("does not match" in warning for warning in report.warnings)


def test_importing_core_modules_does_not_import_torch():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    code = (
        "import sys\n"
        "import voxcpm_runtime.config\n"
        "import voxcpm_runtime.device\n"
        "import voxcpm_runtime.diagnostics\n"
        "import voxcpm_runtime.doctor\n"
        "blocked = {'torch', 'transformers', 'huggingface_hub', 'voxcpm'}\n"
        "assert blocked.isdisjoint(sys.modules)\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=environment, check=True)
