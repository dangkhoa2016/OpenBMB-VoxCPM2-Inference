# OpenBMB-VoxCPM2-Inference

Portable inference engineering for the OpenBMB VoxCPM2 model family.

## Status

This project is in early development toward its first `v1.0.0` release. M1 provides stable environment configuration and CPU/CUDA execution planning; M2 adds portable local model resolution; M3 freezes the project-owned backend contract; M4 qualifies real CPU standard TTS; and M5 qualifies real standard TTS on one explicitly selected NVIDIA T4. The measured M5 canonical run used `cuda:0`, `bfloat16`, and `optimize=False`, with 5.194 GiB peak allocated VRAM and a measured real-time factor of about 2.97. T4x2/multi-GPU execution, other VoxCPM2 features, streaming, an API, scheduling, multiple workers, and production deployment remain unqualified. It is not production-ready.

## Scope

The intended architecture is portable across CPU, a single GPU, and independent multi-GPU replicas. CPU and one explicitly selected CUDA GPU now have evidence-backed runtime paths through M5. Independent multi-GPU replicas remain a future target and will be enabled only after separate measurement and qualification. Kaggle is one deployment and qualification target, not the core architecture.

## Model and attribution

Model weights are not stored in this repository. The reference model is [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2), and the reference implementation is [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM).

This is an independent engineering project. It is not an official OpenBMB release, and no affiliation is claimed. See `provenance/SOURCE-PROVENANCE.md` for the locked source revision and license findings.

## Development

M0 through M5 support Python 3.10 through 3.12.

```bash
python -m pip install -e .
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

After installation, inspect the current environment without loading a model:

```bash
voxcpm-doctor
VOXCPM_DEVICE=cpu voxcpm-doctor
voxcpm-verify-model
```

The doctor emits deterministic JSON. An explicit `VOXCPM_DEVICE=cuda` request fails when no usable CUDA device is visible; it never silently falls back to CPU. See [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) for the environment contract and device policy.

Synthesizing with a real local model requires the pinned upstream stack in a separate environment, because this project intentionally declares no runtime dependencies:

```bash
CUDA_VISIBLE_DEVICES="" VOXCPM_DEVICE=cpu VOXCPM_OFFLINE=1 \
  VOXCPM_MODEL_PATH=/path/to/local/voxcpm2 \
  voxcpm-generate --text "Xin chào từ VoxCPM2." --output out.wav --report report.json
```

M0 through M5 tests do not download or execute the VoxCPM2 model and do not require a GPU; the real backend is covered with a stubbed upstream import. M0 rejects tracked `.bin` files along with model-weight formats. See [`docs/MODEL-RESOLUTION.md`](docs/MODEL-RESOLUTION.md) for M2, [`docs/BACKEND-CONTRACT.md`](docs/BACKEND-CONTRACT.md) for M3, [`docs/REAL-CPU-RUNTIME.md`](docs/REAL-CPU-RUNTIME.md) for M4 CPU evidence, and [`docs/REAL-GPU-RUNTIME.md`](docs/REAL-GPU-RUNTIME.md) for the measured M5 single-T4 path and its limits.

## License

Apache-2.0. See `LICENSE`.
