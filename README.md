# OpenBMB-VoxCPM2-Inference

Portable inference engineering for the OpenBMB VoxCPM2 model family.

## Status

This project is in early development toward its first `v1.0.0` release. M1-M7 qualify configuration, local model resolution, the backend contract, real CPU/single-T4 one-shot inference, voice features, and native backend streaming. M8 adds a spawn-based worker process boundary plus deterministic bounded FIFO scheduling semantics, and qualifies one real worker owning one Tesla T4 through IPC for standard TTS and native streaming. Active cancellation is a process-boundary termination, followed by explicit replacement-worker recovery. T4x2/multi-worker real runtime, FastAPI/HTTP streaming, autoscaling, and production deployment remain unqualified. It is not production-ready.

## Scope

The intended architecture is portable across CPU, a single GPU, and independent process-isolated GPU replicas. Through M8, one worker process can own one qualified backend/device while the parent remains model-free; bounded FIFO admission and failure/cancellation semantics are tested with spawned fake workers. Independent real multi-worker/multi-GPU replicas and network/API streaming remain future targets and require separate measurement. Kaggle is one deployment and qualification target, not the core architecture.

## Model and attribution

Model weights are not stored in this repository. The reference model is [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2), and the reference implementation is [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM).

This is an independent engineering project. It is not an official OpenBMB release, and no affiliation is claimed. See `provenance/SOURCE-PROVENANCE.md` for the locked source revision and license findings.

## Development

M0 through M8 support Python 3.10 through 3.12.

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

This is model/backend streaming, not HTTP, SSE, WebSocket, or browser media streaming. M8 places the qualified backend behind a project-owned spawned worker process and bounded scheduler; this is an internal Python process/IPC boundary, not a network API.

M0 through M8 tests do not download or execute the VoxCPM2 model and do not require a GPU; real runtime paths are covered with stubbed/fake boundaries in normal CI. M0 rejects tracked `.bin` files along with model-weight formats. See [`docs/MODEL-RESOLUTION.md`](docs/MODEL-RESOLUTION.md) for M2, [`docs/BACKEND-CONTRACT.md`](docs/BACKEND-CONTRACT.md) for M3, [`docs/REAL-CPU-RUNTIME.md`](docs/REAL-CPU-RUNTIME.md) for M4, [`docs/REAL-GPU-RUNTIME.md`](docs/REAL-GPU-RUNTIME.md) for M5, [`docs/REAL-VOICE-FEATURES.md`](docs/REAL-VOICE-FEATURES.md) for M6, [`docs/REAL-STREAMING-RUNTIME.md`](docs/REAL-STREAMING-RUNTIME.md) for M7, and [`docs/WORKER-PROCESS-RUNTIME.md`](docs/WORKER-PROCESS-RUNTIME.md) for M8 worker/scheduler semantics and measured single-T4 process isolation.

## License

Apache-2.0. See `LICENSE`.
