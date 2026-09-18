import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "VERSION"


def test_version_file_matches_release_target():
    assert VERSION_FILE.is_file()
    assert VERSION_FILE.read_bytes() == b"1.0.0\n"


def test_package_import_is_model_free_and_cpu_only(tmp_path):
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "PYTHONPATH": str(ROOT),
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "import voxcpm_runtime\n"
                "assert voxcpm_runtime.__version__ == '1.0.0'\n"
                "blocked = {'torch', 'transformers', 'huggingface_hub', 'voxcpm', 'soundfile'}\n"
                "assert blocked.isdisjoint(sys.modules)\n"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
