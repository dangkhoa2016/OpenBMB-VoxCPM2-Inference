import importlib.util
import json
import sys
from pathlib import Path


def _load_demo_module():
    path = Path("scripts/kaggle_production_demo.py")
    spec = importlib.util.spec_from_file_location("kaggle_production_demo", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_runner_import_does_not_import_torch_or_voxcpm():
    before_torch = "torch" in sys.modules
    before_voxcpm = "voxcpm" in sys.modules
    _load_demo_module()
    assert ("torch" in sys.modules) is before_torch
    assert ("voxcpm" in sys.modules) is before_voxcpm


def test_demo_runner_uses_kaggle_adapter_without_hardcoded_model_leaf():
    source = Path("scripts/kaggle_production_demo.py").read_text(encoding="utf-8")
    assert "discover_kaggle_model_candidates" in source
    assert "/kaggle/input/models/dangkhoa2016" not in source
    assert "VOXCPM_API_TOKEN" in source
    assert "HF_ENDPOINT" in source
    assert "127.0.0.1:9" in source


def test_thin_notebook_has_fresh_workspace_and_selected_profile_contract():
    notebook = json.loads(
        Path("notebooks/kaggle-production-demo.ipynb").read_text(encoding="utf-8")
    )
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 6
    source = "\n".join(
        "".join(cell.get("source", ())) for cell in notebook["cells"]
    )
    for profile in ("cpu", "cuda-single", "cuda-replica", "auto"):
        assert profile in source
    assert "VOXCPM_DEMO_PROFILE" in source
    assert "VOXCPM_DEMO_REF" in source
    assert "shutil.rmtree(WORKSPACE)" in source
    assert "git" in source and "clone" in source
    assert "scripts/kaggle_production_demo.py" in source
    assert ".venv-internal" not in source
    assert "internal_" not in source
    assert "profile_" not in source
