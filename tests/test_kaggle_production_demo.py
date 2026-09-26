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


def test_demo_runner_supports_parallel_dual_audio_showcase():
    source = Path("scripts/kaggle_production_demo.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--parallel-text", default=None)' in source
    assert "ThreadPoolExecutor(max_workers=2)" in source
    assert '"VOXCPM_MAX_CONCURRENT_REQUESTS": str(max(1, resolved.workers))' in source
    assert '"peak_busy_workers"' in source
    assert 'raise RuntimeError("parallel showcase did not observe two busy workers")' in source
    assert "output_path.write_bytes(response.content)" in source
    assert 'print("KAGGLE_T4X2_PARALLEL_SHOWCASE=PASS")' in source


def test_public_notebook_has_bilingual_showcase_and_runtime_contract():
    notebook = json.loads(
        Path("notebooks/kaggle-production-demo.ipynb").read_text(encoding="utf-8")
    )
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 15

    cells = notebook["cells"]
    for index, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            assert index > 0
            assert cells[index - 1]["cell_type"] == "markdown"

    source = "\n".join(
        "".join(cell.get("source", ())) for cell in cells
    )
    for profile in ("cpu", "cuda-single", "cuda-replica", "auto"):
        assert profile in source

    assert "VOXCPM_DEMO_PROFILE" in source
    assert "VOXCPM_DEMO_REF" in source
    assert "v1.0.0" in source
    assert "shutil.rmtree(WORKSPACE)" in source
    assert "git" in source and "clone" in source
    assert "scripts/kaggle_production_demo.py" in source

    assert "English" in source
    assert "Tiếng Việt" in source
    assert "Add Input" in source
    assert "dangkhoa2016/openbmb-voxcpm2" in source
    assert "/kaggle/input/models/" in source
    assert "SHOWCASE_TEXT_EN" in source
    assert "SHOWCASE_TEXT_VI" in source
    assert "--parallel-text" in source
    assert "peak_busy_workers" in source
    assert "Audio(filename=str(EN_WAV)" in source
    assert "Audio(filename=str(VI_WAV)" in source
    assert "Listen again / Nghe lại" in source
    assert "<details" in source
    assert "KAGGLE_T4X2_PARALLEL_SHOWCASE=PASS" in source

    assert ".venv-internal" not in source
    assert "internal_" not in source
    assert "profile_" not in source
