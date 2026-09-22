# OpenBMB-VoxCPM2-Inference

Portable CPU/GPU inference engineering for OpenBMB VoxCPM2.

This is an **independent engineering project** around OpenBMB VoxCPM2. It is not an official OpenBMB release and does not claim affiliation with OpenBMB.

Model weights are **not stored in this Git repository**. You must supply or mount a valid VoxCPM2 model copy. The reference model is `openbmb/VoxCPM2`; the Kaggle mirror used by the qualified notebook flow is `dangkhoa2016/openbmb-voxcpm2`.

## Supported scope

The same runtime architecture has measured qualification for Linux CPU, one NVIDIA T4 16 GB, and two NVIDIA T4 GPUs as independent process-isolated model replicas. The project also provides authenticated one-shot REST TTS, native PCM streaming, voice design/clone/continuation through local CLI/backend paths, a fresh-workspace Kaggle demo, and a CPU Docker container contract on generic GitHub-hosted Linux.

Hardware support means **measured/qualified environments only**. It is not a universal guarantee for every CPU, GPU, RAM size, Linux image, or container host.

A "production-style demo" here means a stable API contract, authentication, readiness, bounded queue/concurrency, timeouts, structured errors, offline/local model loading, telemetry, and clean lifecycle. It does **not** mean HA, SLA, autoscaling, permanent public hosting, or enterprise multi-tenancy.

## Requirements

- Python 3.10, 3.11, or 3.12.
- A valid local/mounted VoxCPM2 model copy for real inference.
- A compatible CUDA/PyTorch environment for GPU profiles.
- The upstream VoxCPM runtime stack for real model inference.

## Install

```bash
git clone https://github.com/dangkhoa2016/OpenBMB-VoxCPM2-Inference.git
cd OpenBMB-VoxCPM2-Inference
python -m pip install -e .
```

For the REST API:

```bash
python -m pip install -e '.[api]'
```

Development/test dependencies:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Inspect the environment

```bash
voxcpm-doctor
VOXCPM_DEVICE=cpu voxcpm-doctor
voxcpm-verify-model
```

An explicit CUDA request fails when no usable CUDA device is available; it does not silently fall back to CPU.

## Supply the model

```bash
export VOXCPM_MODEL_PATH=/path/to/VoxCPM2
export VOXCPM_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

For Kaggle, attach model `dangkhoa2016/openbmb-voxcpm2`; the demo runner discovers the mounted model through the Kaggle deployment adapter rather than a hard-coded leaf path.

## One-shot generation

CPU example:

```bash
CUDA_VISIBLE_DEVICES="" \
VOXCPM_DEVICE=cpu \
VOXCPM_MODEL_PATH=/path/to/VoxCPM2 \
VOXCPM_OFFLINE=1 \
voxcpm-generate --text "Xin chào từ VoxCPM2." --output out.wav --report report.json
```

Single-GPU example:

```bash
VOXCPM_DEVICE=cuda \
VOXCPM_GPU_DEVICES=0 \
VOXCPM_WORKERS=1 \
VOXCPM_MODEL_PATH=/path/to/VoxCPM2 \
VOXCPM_OFFLINE=1 \
voxcpm-generate --text "Hello from VoxCPM2." --output out.wav
```

Voice design, clone, continuation and streaming use the same `voxcpm-generate` entrypoint. See [Backends](docs/BACKENDS.md) and [Streaming](docs/STREAMING.md).

## Execution profiles

Canonical profiles are `cpu`, `cuda-single`, `cuda-replica`, and `auto`; aliases `gpu` and `multi-gpu` are also accepted.

`cuda-replica` uses one independent worker/model replica per selected GPU. It is **not** tensor parallelism and does not shard one model across GPUs.

See [CPU](docs/CPU.md), [CUDA](docs/CUDA.md), [Multi-GPU](docs/MULTI-GPU.md), and [Execution profiles](docs/EXECUTION-PROFILES.md).

## REST API

```bash
export VOXCPM_MODEL_PATH=/path/to/VoxCPM2
export VOXCPM_PROFILE=cuda-single
export VOXCPM_MAX_QUEUE_SIZE=1
export VOXCPM_API_TOKEN='replace-with-a-strong-token'
export VOXCPM_REQUIRE_AUTH=1
voxcpm-serve
```

Default bind: `127.0.0.1:8090`.

```bash
curl -sS \
  -H "Authorization: Bearer $VOXCPM_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text":"Xin chào từ VoxCPM2."}' \
  http://127.0.0.1:8090/v1/tts \
  --output out.wav
```

Qualified endpoints are `GET /healthz`, `GET /readyz`, `POST /v1/tts`, and `POST /v1/tts/stream`. Streaming returns raw `pcm_s16le` over `application/octet-stream`.

See [API](docs/API.md) and [Security](docs/SECURITY.md).

## Kaggle

For the fresh-workspace notebook flow: create a Kaggle Notebook, choose **GPU T4 x2** when exercising the full GPU profile matrix, attach `dangkhoa2016/openbmb-voxcpm2`, import `notebooks/kaggle-production-demo.ipynb`, optionally choose `VOXCPM_DEMO_PROFILE`, then Run All.

See [Kaggle](docs/KAGGLE.md).

## Docker

The CPU Docker contract uses a non-root runtime user, mounted model path `/models/VoxCPM2`, healthcheck, authentication-preserving defaults, and offline/local model defaults. Real VoxCPM2 inference **inside Docker** and CUDA Docker are not currently claimed as qualified.

See [Docker](docs/DOCKER.md).

## Documentation

[Architecture](docs/ARCHITECTURE.md) · [Configuration](docs/CONFIGURATION.md) · [Backends](docs/BACKENDS.md) · [IPC](docs/IPC.md) · [Streaming](docs/STREAMING.md) · [Observability](docs/OBSERVABILITY.md) · [CPU](docs/CPU.md) · [CUDA](docs/CUDA.md) · [Multi-GPU](docs/MULTI-GPU.md) · [Kaggle](docs/KAGGLE.md) · [API](docs/API.md) · [Security](docs/SECURITY.md) · [Benchmarks](docs/BENCHMARKS.md) · [Troubleshooting](docs/TROUBLESHOOTING.md) · [Release checklist](docs/RELEASE-CHECKLIST.md)

Public qualification evidence is summarized in `provenance/QUALIFICATION.md`.

## License

Apache-2.0. See [LICENSE](LICENSE).
