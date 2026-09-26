# OpenBMB-VoxCPM2-Inference

Portable inference engineering for the OpenBMB VoxCPM2 model family.

## Status

This project is in early development toward its first `v1.0.0` release. M1 provides stable environment configuration, host and optional CUDA inventory, deterministic CPU/CUDA execution planning, and the `voxcpm-doctor` diagnostic CLI. M2 adds portable, metadata-only local model resolution, bounded Kaggle mount discovery, and the `voxcpm-verify-model` CLI. M3 freezes a project-owned backend contract with normalized errors and a deterministic, hermetic fake backend. M4 adds the first real execution path: a CPU-only backend that loads a pinned local VoxCPM2 model, synthesizes standard TTS, writes a project-owned PCM16 WAV, and reports redacted runtime telemetry through the `voxcpm-generate` CLI. Real CPU load and synthesis were measured successfully on one Kaggle host, at roughly `48x` slower than real time; memory sufficiency, GPU, other VoxCPM2 features, streaming, an API, scheduling, workers, and deployment qualification remain unqualified. It is not production-ready.

## Scope

The intended architecture is portable across CPU, a single GPU, and independent multi-GPU replicas. Those execution modes are future targets and will be enabled only after later measurement and qualification. Kaggle is one deployment and qualification target, not the core architecture.

## Model and attribution

Model weights are not stored in this repository. The reference model is [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2), and the reference implementation is [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM).

This is an independent engineering project. It is not an official OpenBMB release, and no affiliation is claimed. See `provenance/SOURCE-PROVENANCE.md` for the locked source revision and license findings.

## Development

M0 through M4 support Python 3.10 through 3.12.

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

M0 through M4 tests do not download or execute the VoxCPM2 model and do not require a GPU; the real backend is covered with a stubbed upstream import. M0 rejects tracked `.bin` files along with model-weight formats. See [`docs/MODEL-RESOLUTION.md`](docs/MODEL-RESOLUTION.md) for the M2 resolver contract, [`docs/BACKEND-CONTRACT.md`](docs/BACKEND-CONTRACT.md) for the M3 boundary and its limits, and [`docs/REAL-CPU-RUNTIME.md`](docs/REAL-CPU-RUNTIME.md) for what M4 actually measured on CPU, including why it is not a 16 GB or real-time claim. This conservative policy may be refined in a later evidence-backed milestone.

## License

Apache-2.0. See `LICENSE`.
