# OpenBMB-VoxCPM2-Inference

Kỹ thuật inference portable trên CPU/GPU cho OpenBMB VoxCPM2.

Đây là một **dự án kỹ thuật độc lập** xây dựng quanh OpenBMB VoxCPM2. Đây không phải bản phát hành chính thức của OpenBMB và không tuyên bố có liên kết với OpenBMB.

Trọng số model **không được lưu trong Git repository này**. Người dùng phải tự cung cấp hoặc mount một bản VoxCPM2 hợp lệ. Model tham chiếu là `openbmb/VoxCPM2`; Kaggle mirror dùng trong notebook đã qualification là `dangkhoa2016/openbmb-voxcpm2`.

## Phạm vi hỗ trợ

Cùng một kiến trúc runtime đã được đo/qualification trên Linux CPU, một NVIDIA T4 16 GB và hai NVIDIA T4 dưới dạng hai model replica độc lập, cách ly theo process. Project cũng có authenticated one-shot REST TTS, native PCM streaming, voice design/clone/continuation qua local CLI/backend, fresh-workspace Kaggle demo và CPU Docker container contract trên generic GitHub-hosted Linux.

Hỗ trợ phần cứng chỉ có nghĩa là **môi trường đã được đo/qualification**. Không nên hiểu thành đảm bảo cho mọi CPU, GPU, dung lượng RAM, Linux image hay container host.

"Production-style demo" ở đây nghĩa là API contract ổn định, authentication, readiness, bounded queue/concurrency, timeout, structured error, offline/local model loading, telemetry và clean lifecycle. Nó **không** có nghĩa HA, SLA, autoscaling, public hosting vĩnh viễn hay enterprise multi-tenancy.

## Yêu cầu

- Python 3.10, 3.11 hoặc 3.12.
- Một bản VoxCPM2 local/mounted hợp lệ để chạy inference thật.
- Môi trường CUDA/PyTorch phù hợp cho GPU profile.
- Upstream VoxCPM runtime stack để chạy model thật.

## Cài đặt

```bash
git clone https://github.com/dangkhoa2016/OpenBMB-VoxCPM2-Inference.git
cd OpenBMB-VoxCPM2-Inference
python -m pip install -e .
```

Cài REST API:

```bash
python -m pip install -e '.[api]'
```

Dependency cho development/test:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Kiểm tra môi trường

```bash
voxcpm-doctor
VOXCPM_DEVICE=cpu voxcpm-doctor
voxcpm-verify-model
```

Nếu yêu cầu CUDA tường minh nhưng không có CUDA device dùng được, runtime sẽ fail thay vì âm thầm fallback sang CPU.

## Cung cấp model

```bash
export VOXCPM_MODEL_PATH=/path/to/VoxCPM2
export VOXCPM_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Trên Kaggle, attach model `dangkhoa2016/openbmb-voxcpm2`; demo runner tìm model qua Kaggle deployment adapter thay vì hard-code leaf path.

## Generate one-shot

Ví dụ CPU:

```bash
CUDA_VISIBLE_DEVICES="" \
VOXCPM_DEVICE=cpu \
VOXCPM_MODEL_PATH=/path/to/VoxCPM2 \
VOXCPM_OFFLINE=1 \
voxcpm-generate --text "Xin chào từ VoxCPM2." --output out.wav --report report.json
```

Ví dụ single GPU:

```bash
VOXCPM_DEVICE=cuda \
VOXCPM_GPU_DEVICES=0 \
VOXCPM_WORKERS=1 \
VOXCPM_MODEL_PATH=/path/to/VoxCPM2 \
VOXCPM_OFFLINE=1 \
voxcpm-generate --text "Hello from VoxCPM2." --output out.wav
```

Voice design, clone, continuation và streaming dùng cùng entrypoint `voxcpm-generate`. Xem [Backends](docs/BACKENDS.vi.md) và [Streaming](docs/STREAMING.vi.md).

## Execution profile

Các profile canonical là `cpu`, `cuda-single`, `cuda-replica` và `auto`; alias `gpu` và `multi-gpu` cũng được chấp nhận.

`cuda-replica` dùng một worker/model replica độc lập cho mỗi GPU được chọn. Đây **không** phải tensor parallelism và không shard một model qua nhiều GPU.

Xem [CPU](docs/CPU.vi.md), [CUDA](docs/CUDA.vi.md), [Multi-GPU](docs/MULTI-GPU.vi.md) và [Execution profiles](docs/EXECUTION-PROFILES.vi.md).

## REST API

```bash
export VOXCPM_MODEL_PATH=/path/to/VoxCPM2
export VOXCPM_PROFILE=cuda-single
export VOXCPM_MAX_QUEUE_SIZE=1
export VOXCPM_API_TOKEN='replace-with-a-strong-token'
export VOXCPM_REQUIRE_AUTH=1
voxcpm-serve
```

Bind mặc định: `127.0.0.1:8090`.

```bash
curl -sS \
  -H "Authorization: Bearer $VOXCPM_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text":"Xin chào từ VoxCPM2."}' \
  http://127.0.0.1:8090/v1/tts \
  --output out.wav
```

Các endpoint đã qualification là `GET /healthz`, `GET /readyz`, `POST /v1/tts` và `POST /v1/tts/stream`. Streaming trả raw `pcm_s16le` qua `application/octet-stream`.

Xem [API](docs/API.vi.md) và [Security](docs/SECURITY.vi.md).

## Kaggle

Fresh-workspace notebook flow: tạo Kaggle Notebook, chọn **GPU T4 x2** khi muốn chạy đầy đủ GPU profile matrix, attach `dangkhoa2016/openbmb-voxcpm2`, import `notebooks/kaggle-production-demo.ipynb`, tùy chọn `VOXCPM_DEMO_PROFILE`, rồi Run All.

Xem [Kaggle](docs/KAGGLE.vi.md).

## Docker

CPU Docker contract dùng non-root runtime user, model mount tại `/models/VoxCPM2`, healthcheck, authentication-preserving defaults và offline/local model defaults. Real VoxCPM2 inference **bên trong Docker** và CUDA Docker hiện chưa được claim là đã qualification.

Xem [Docker](docs/DOCKER.vi.md).

## Tài liệu

[Architecture](docs/ARCHITECTURE.vi.md) · [Configuration](docs/CONFIGURATION.md) · [Backends](docs/BACKENDS.vi.md) · [IPC](docs/IPC.vi.md) · [Streaming](docs/STREAMING.vi.md) · [Observability](docs/OBSERVABILITY.vi.md) · [CPU](docs/CPU.vi.md) · [CUDA](docs/CUDA.vi.md) · [Multi-GPU](docs/MULTI-GPU.vi.md) · [Kaggle](docs/KAGGLE.vi.md) · [API](docs/API.vi.md) · [Security](docs/SECURITY.vi.md) · [Benchmarks](docs/BENCHMARKS.vi.md) · [Troubleshooting](docs/TROUBLESHOOTING.vi.md) · [Release checklist](docs/RELEASE-CHECKLIST.vi.md)

Evidence qualification public được tóm tắt tại `provenance/QUALIFICATION.md`.

## License

MIT cho code và tài liệu nguyên bản của repository này. Xem [LICENSE](LICENSE).
