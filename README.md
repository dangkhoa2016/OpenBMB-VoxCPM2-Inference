# OpenBMB-VoxCPM2-Inference

Portable inference engineering for the OpenBMB VoxCPM2 model family.

## Status

This project is in early development toward its first `v1.0.0` release. M1 provides stable environment configuration and CPU/CUDA execution planning; M2 adds portable local model resolution; M3 freezes the project-owned backend contract; M4 qualifies real CPU standard TTS; M5 qualifies real standard TTS on one explicitly selected NVIDIA T4; M6 qualifies the remaining one-shot voice features; and M7 qualifies the pinned upstream native streaming path for standard TTS, voice design, voice clone, and audio continuation on one Tesla T4. Measured M7 streams used `cuda:0`, `bfloat16`, and `optimize=False`, with peak allocated VRAM between 5.35 GiB and 5.53 GiB. T4x2/multi-GPU execution, HTTP/API streaming, scheduling, multiple workers, and production deployment remain unqualified. It is not production-ready.

## Scope

The intended architecture is portable across CPU, a single GPU, and independent multi-GPU replicas. CPU, one explicitly selected CUDA GPU, one-shot voice features, and native backend streaming now have evidence-backed runtime paths through M7. Independent multi-GPU replicas and network/API streaming remain future targets and will be enabled only after separate measurement and qualification. Kaggle is one deployment and qualification target, not the core architecture.

## Model and attribution

Model weights are not stored in this repository. The reference model is [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2), and the reference implementation is [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM).

This is an independent engineering project. It is not an official OpenBMB release, and no affiliation is claimed. See `provenance/SOURCE-PROVENANCE.md` for the locked source revision and license findings.

## Development

M0 through M7 support Python 3.10 through 3.12.

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

The same entrypoint covers the qualified one-shot voice features through mutually exclusive flags:

```bash
# voice design
voxcpm-generate --text "Xin chào từ VoxCPM2." \
  --voice-instruction "giọng nữ ấm áp, bình tĩnh" --output design.wav

# voice clone from a local reference recording
voxcpm-generate --text "Xin chào, đây là phép thử clone giọng." \
  --reference-audio /path/to/reference.wav --output clone.wav

# audio continuation from a local prefix recording
voxcpm-generate --text "Và đây là phần tiếp theo." \
  --prompt-audio /path/to/prefix.wav \
  --prompt-text "Xin chào, đây là giọng nói tham chiếu." --output continuation.wav
```

`--voice-instruction` cannot be combined with reference or prompt audio, `--reference-audio` cannot be combined with prompt audio or prompt text, and `--prompt-audio` and `--prompt-text` must appear together. Reference audio is read only from the local filesystem, is never fetched remotely, and never appears in a public error or report. M7 adds backend-native streaming through the same CLI; `--stream` consumes project-owned `AudioChunk` values and writes one validation WAV only after the stream completes:

```bash
voxcpm-generate --stream --text "Xin chào từ VoxCPM2." \
  --output streamed.wav --report streamed-report.json
```

This is model/backend streaming, not HTTP, SSE, WebSocket, or browser media streaming.

M0 through M7 tests do not download or execute the VoxCPM2 model and do not require a GPU; the real backend is covered with a stubbed upstream import. M0 rejects tracked `.bin` files along with model-weight formats. See [`docs/MODEL-RESOLUTION.md`](docs/MODEL-RESOLUTION.md) for M2, [`docs/BACKEND-CONTRACT.md`](docs/BACKEND-CONTRACT.md) for M3, [`docs/REAL-CPU-RUNTIME.md`](docs/REAL-CPU-RUNTIME.md) for M4, [`docs/REAL-GPU-RUNTIME.md`](docs/REAL-GPU-RUNTIME.md) for M5, [`docs/REAL-VOICE-FEATURES.md`](docs/REAL-VOICE-FEATURES.md) for M6, and [`docs/REAL-STREAMING-RUNTIME.md`](docs/REAL-STREAMING-RUNTIME.md) for the measured M7 native-streaming path and its limits.

## License

Apache-2.0. See `LICENSE`.
