# Docker deployment

## Scope

Docker qualification adds a portable CPU Docker image and proves the API/container contract on a generic GitHub-hosted Ubuntu runner.

The image does **not** embed VoxCPM2 model weights. Mount a valid model copy at:

```text
/models/VoxCPM2
```

The default runtime image is CPU-only and starts:

```text
voxcpm-serve
```

CUDA Docker remains unqualified in Docker qualification.

## Build

Build the CPU runtime image from the repository root:

```bash
docker build --target runtime -t openbmb-voxcpm2-inference:cpu .
```

The image installs:

- the current project source;
- the API extra;
- CPU PyTorch from the official CPU wheel index;
- the pinned upstream VoxCPM source commit
  `f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69`.

The qualified Docker build resolved:

```text
Python       3.12
PyTorch      2.14.0+cpu
VoxCPM       2.0.3.post33+gf772e498a
CUDA         unavailable
```

These are observations from the qualification build. The Dockerfile does not pin PyTorch to an exact release, so future rebuilds can resolve a newer compatible CPU wheel unless the image is built from a frozen external dependency snapshot.

## Run

The container is intentionally fail-closed for authentication. Supply a token explicitly:

```bash
docker run --rm   -p 8090:8090   -v /absolute/path/to/VoxCPM2:/models/VoxCPM2:ro   -e VOXCPM_API_TOKEN='replace-with-a-strong-token'   openbmb-voxcpm2-inference:cpu
```

The default environment contract is:

```text
VOXCPM_MODEL_PATH=/models/VoxCPM2
VOXCPM_PROFILE=cpu
VOXCPM_DEVICE=cpu
VOXCPM_HOST=0.0.0.0
VOXCPM_PORT=8090
VOXCPM_REQUIRE_AUTH=1
VOXCPM_ALLOW_UNAUTHENTICATED_EXTERNAL=0
VOXCPM_OFFLINE=1
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

The model path is a volume contract, not a path baked into model-resolution core logic.

## Healthcheck

The image contains a Docker healthcheck against:

```text
GET http://127.0.0.1:8090/healthz
```

The check uses Python's standard library rather than requiring an external monitoring client.

Application readiness remains available at:

```text
GET /readyz
```

Health and readiness retain the same semantics as non-container deployments.

## API contract

The qualified public HTTP surface is unchanged:

```text
GET  /healthz
GET  /readyz
POST /v1/tts
POST /v1/tts/stream
```

Authentication is still bearer-token based for inference requests. Non-streaming output is WAV. Streaming output is raw `pcm_s16le` over `application/octet-stream`.

## Non-root execution

Both Docker targets run as:

```text
user: voxcpm
uid: 10001
```

The model directory is expected to be mounted read-only for serving.

## Build-context hygiene

`.dockerignore` excludes common model-weight and runtime-artifact formats, including:

```text
*.safetensors
*.bin
*.pth
*.pt
*.ckpt
*.gguf
*.log
```

This complements the repository's existing no-model-weights policy.

## Qualification boundary

Docker qualification proves the following on a generic, non-Kaggle GitHub-hosted Ubuntu Docker environment:

- Docker build of the contract-smoke target;
- mounted `/models/VoxCPM2` directory plumbing;
- image healthcheck;
- `/healthz` and `/readyz`;
- bearer-auth enforcement;
- one-shot WAV API semantics;
- streaming PCM16 API semantics;
- non-root runtime;
- build of the full CPU runtime image;
- CPU-only PyTorch identity;
- pinned upstream VoxCPM source identity;
- default command `voxcpm-serve`.

The contract-smoke target uses the project's fake backend only for model-free CI verification. It is not the default runtime target and is not a production inference mode.

Docker qualification does **not** claim:

- real VoxCPM2 model inference was executed inside Docker;
- a minimum Docker RAM requirement;
- CUDA Docker support;
- GPU passthrough qualification;
- image reproducibility across future unpinned transitive dependency releases;
- SLA, HA, autoscaling, or permanent hosting.

Real CPU inference is separately qualified on the measured CPU-runtime qualification Linux environment. Docker qualification's portability claim is therefore limited to the container/runtime/API contract demonstrated here.

## Qualification authority

The public qualification summary records the measured Docker portability scope and its claim boundaries.

See `provenance/QUALIFICATION.md`.
