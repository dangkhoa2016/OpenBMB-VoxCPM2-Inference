# Triển khai Docker

## Phạm vi

Docker qualification bổ sung CPU Docker image portable và chứng minh API/container contract trên generic GitHub-hosted Ubuntu runner, không phải Kaggle.

Image **không** chứa VoxCPM2 model weights. Cần mount một model copy hợp lệ tại:

```text
/models/VoxCPM2
```

Default runtime image là CPU-only và chạy:

```text
voxcpm-serve
```

CUDA Docker chưa được qualification trong Docker qualification.

## Build

Từ repository root:

```bash
docker build --target runtime -t openbmb-voxcpm2-inference:cpu .
```

Image cài:

- current project source;
- API extra;
- CPU PyTorch từ official CPU wheel index;
- pinned upstream VoxCPM commit
  `f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69`.

Docker build đã qualification và resolve:

```text
Python       3.12
PyTorch      2.14.0+cpu
VoxCPM       2.0.3.post33+gf772e498a
CUDA         unavailable
```

Đây là measured build observation. Dockerfile chưa pin chính xác PyTorch release, vì vậy future rebuild có thể resolve CPU wheel tương thích mới hơn nếu không dùng một external dependency snapshot đã freeze.

## Run

Container fail-closed cho authentication. Cần truyền token rõ ràng:

```bash
docker run --rm   -p 8090:8090   -v /absolute/path/to/VoxCPM2:/models/VoxCPM2:ro   -e VOXCPM_API_TOKEN='replace-with-a-strong-token'   openbmb-voxcpm2-inference:cpu
```

Default environment contract:

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

Model path là volume contract, không phải Kaggle/Docker path hard-code trong core model resolver.

## Healthcheck

Image có Docker healthcheck:

```text
GET http://127.0.0.1:8090/healthz
```

Healthcheck dùng Python standard library.

Application readiness vẫn là:

```text
GET /readyz
```

Health/readiness semantics giống các deployment ngoài container.

## API contract

Qualified HTTP surface không đổi:

```text
GET  /healthz
GET  /readyz
POST /v1/tts
POST /v1/tts/stream
```

Inference request vẫn dùng bearer authentication. Non-streaming output là WAV. Streaming output là raw `pcm_s16le` qua `application/octet-stream`.

## Non-root

Cả hai Docker target chạy bằng:

```text
user: voxcpm
uid: 10001
```

Khi serve, model directory nên được mount read-only.

## Build-context hygiene

`.dockerignore` loại trừ các model-weight/runtime-artifact format phổ biến:

```text
*.safetensors
*.bin
*.pth
*.pt
*.ckpt
*.gguf
*.log
```

Điều này bổ sung cho policy không lưu model weights trong Git.

## Qualification boundary

Docker qualification chứng minh trên generic non-Kaggle GitHub-hosted Ubuntu Docker environment:

- build contract-smoke image;
- mount plumbing cho `/models/VoxCPM2`;
- image healthcheck;
- `/healthz` và `/readyz`;
- bearer-auth enforcement;
- one-shot WAV API semantics;
- streaming PCM16 API semantics;
- non-root runtime;
- build full CPU runtime image;
- CPU-only PyTorch identity;
- pinned upstream VoxCPM source identity;
- default command `voxcpm-serve`.

Target `contract-smoke` chỉ dùng fake backend của project cho model-free CI verification. Nó không phải default runtime target và không phải production inference mode.

Docker qualification **không** claim:

- real VoxCPM2 inference đã chạy trong Docker;
- minimum Docker RAM requirement;
- CUDA Docker support;
- GPU passthrough qualification;
- image reproducibility qua các future unpinned transitive dependencies;
- SLA, HA, autoscaling hoặc permanent hosting.

Real CPU inference đã được qualification riêng trên measured CPU-runtime qualification Linux environment. Vì vậy portability claim của Docker qualification chỉ giới hạn ở container/runtime/API contract đã chứng minh.

## Qualification authority

The public qualification summary records the measured Docker portability scope and its claim boundaries.

Xem `provenance/QUALIFICATION.md`.
