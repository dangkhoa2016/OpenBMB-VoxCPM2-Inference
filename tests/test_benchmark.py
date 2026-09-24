import importlib.util
import json
import sys
from pathlib import Path


def _load_runner():
    path = Path("scripts/benchmark.py")
    spec = importlib.util.spec_from_file_location("benchmark_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_v1_is_frozen_and_complete():
    payload = json.loads(Path("benchmarks/protocol-v1.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["protocol_id"] == "voxcpm2-benchmark-v1"
    assert payload["warmup_policy"] == {
        "cold_start": "Measure process launch through readiness separately.",
        "warmup_requests": 1,
        "measured_warm_repetitions": 3,
        "mix_cold_and_warm": False,
    }
    assert set(payload["inputs"]) == {f"B{i}" for i in range(1, 9)}
    assert payload["inputs"]["B7"]["kind"] == "streaming"
    assert payload["inputs"]["B8"]["kind"] == "concurrency"
    assert len(payload["inputs"]["B8"]["requests"]) == 2
    assert payload["formulas"]["rtf"] == (
        "request_elapsed_seconds / generated_audio_duration_seconds"
    )


def test_runner_import_does_not_import_torch_or_voxcpm():
    before_torch = "torch" in sys.modules
    before_voxcpm = "voxcpm" in sys.modules
    _load_runner()
    assert ("torch" in sys.modules) is before_torch
    assert ("voxcpm" in sys.modules) is before_voxcpm


def test_runner_freezes_cold_warm_stream_and_concurrency_boundaries():
    source = Path("scripts/benchmark.py").read_text(encoding="utf-8")
    for profile in ("cpu", "cuda-single", "cuda-replica"):
        assert profile in source
    assert "warmup_requests" in source
    assert "measured_warm_repetitions" in source
    assert "first_body_seconds" in source
    assert "requests_per_second" in source
    assert "audio_seconds_per_second" in source
    assert "gpu_processes_match_baseline" in source
    assert "HF_ENDPOINT" in source
    assert "127.0.0.1:9" in source
    assert "/kaggle/input/models/dangkhoa2016" not in source
