# OpenBMB-VoxCPM2-Inference

Portable inference engineering for the OpenBMB VoxCPM2 model family.

## Status

This project is in early development toward its first `v1.0.0` release. M1-M8 qualify configuration, local model resolution, the backend contract, real CPU/single-T4 inference, voice features, native backend streaming, and a spawn-based worker/scheduler boundary. M9 adds a FastAPI/REST surface over that qualified Scheduler + WorkerClient path, including bearer authentication, health/readiness, one-shot WAV responses, bounded admission, incremental PCM16 HTTP streaming, and client-disconnect cancellation on one Tesla T4. T4x2/multi-worker real runtime, SSE/WebSocket, autoscaling, and production deployment remain unqualified. It is not production-ready.

## Scope

The intended architecture is portable across CPU, a single GPU, and independent process-isolated GPU replicas. Through M9, the API parent remains model-free and routes all inference through the M8 Scheduler/WorkerClient boundary; one real worker owns one explicitly selected T4. Independent real multi-worker/multi-GPU replicas remain a future target and require separate measurement. Kaggle is one deployment and qualification target, not the core architecture.

## Model and attribution

Model weights are not stored in this repository. The reference model is [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2), and the reference implementation is [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM).

This is an independent engineering project. It is not an official OpenBMB release, and no affiliation is claimed. See `provenance/SOURCE-PROVENANCE.md` for the locked source revision and license findings.

## Development

M0 through M9 support Python 3.10 through 3.12.

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

M7 covers model/backend streaming; M8 adds the internal process/IPC boundary; M9 adds the first qualified HTTP surface. Install the optional API stack with `python -m pip install -e '.[api]'`, configure one explicit GPU, authentication, and a bounded queue, then run `voxcpm-serve`. The qualified endpoints are `GET /healthz`, `GET /readyz`, `POST /v1/tts`, and `POST /v1/tts/stream`. The stream response is raw `pcm_s16le` over `application/octet-stream`, not SSE or WebSocket.

M0 through M9 normal CI remains GPU-free and model-free. M0 rejects tracked `.bin` files along with model-weight formats. See [`docs/API-RUNTIME.md`](docs/API-RUNTIME.md) for the M9 HTTP contract and measured real single-T4 evidence boundaries.

## License

Apache-2.0. See `LICENSE`.
