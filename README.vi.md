# OpenBMB-VoxCPM2-Inference

Kỹ thuật inference portable cho họ mô hình OpenBMB VoxCPM2.

## Trạng thái

Dự án đang ở giai đoạn phát triển ban đầu hướng tới bản phát hành `v1.0.0` đầu tiên. M1-M9 qualification cấu hình, model local, inference CPU/single-T4, các tính năng giọng, native streaming, process isolation, bounded scheduling và FastAPI/REST. M10 qualification một FastAPI parent với hai real worker độc lập trên hai Tesla T4, gồm GPU ownership, HTTP TTS/streaming song song, bounded admission, degraded readiness, failure isolation, offline local-model operation và resource cleanup đã đo. SSE/WebSocket, autoscaling, tensor parallelism, model sharding và production deployment vẫn chưa được qualification. Dự án chưa sẵn sàng cho production.

## Phạm vi

Kiến trúc mục tiêu là portable trên CPU, một GPU và các replica GPU độc lập theo process. Qua M10, API parent vẫn model-free và mọi inference đi qua Scheduler/WorkerClient. Topology T4x2 đã qualification dùng hai model replica độc lập, mỗi process sở hữu một physical GPU; hệ thống không chia một model qua hai GPU. Kaggle là một mục tiêu triển khai và qualification, không phải kiến trúc lõi.

## Mô hình và ghi công

Trọng số mô hình không được lưu trong repository này. Mô hình tham chiếu là [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) và implementation tham chiếu là [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM).

Đây là một dự án kỹ thuật độc lập, không phải bản phát hành chính thức của OpenBMB và không có tuyên bố liên kết nào. Xem `provenance/SOURCE-PROVENANCE.md` để biết revision nguồn đã khóa và kết quả xác minh license.

## Phát triển

M0 đến M10 hỗ trợ Python 3.10 đến 3.12.

```bash
python -m pip install -e .
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Sau khi cài đặt, có thể kiểm tra môi trường hiện tại mà không tải mô hình:

```bash
voxcpm-doctor
VOXCPM_DEVICE=cpu voxcpm-doctor
voxcpm-verify-model
```

Doctor xuất JSON xác định. Yêu cầu tường minh `VOXCPM_DEVICE=cuda` sẽ thất bại khi không có thiết bị CUDA dùng được; hệ thống không âm thầm chuyển sang CPU. Xem [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) để biết hợp đồng biến môi trường và chính sách thiết bị.

Tổng hợp bằng mô hình local thật cần có ngăn xếp upstream đã ghim trong một môi trường riêng, vì dự án này cố ý không khai báo phụ thuộc runtime nào:

```bash
CUDA_VISIBLE_DEVICES="" VOXCPM_DEVICE=cpu VOXCPM_OFFLINE=1 \
  VOXCPM_MODEL_PATH=/path/to/local/voxcpm2 \
  voxcpm-generate --text "Xin chào từ VoxCPM2." --output out.wav --report report.json
```

Cùng entrypoint đó phục vụ các tính năng giọng one-shot đã qualification thông qua các cờ loại trừ lẫn nhau:

```bash
# thiết kế giọng
voxcpm-generate --text "Xin chào từ VoxCPM2." \
  --voice-instruction "giọng nữ ấm áp, bình tĩnh" --output design.wav

# nhân bản giọng từ một bản ghi tham chiếu local
voxcpm-generate --text "Xin chào, đây là phép thử clone giọng." \
  --reference-audio /path/to/reference.wav --output clone.wav

# nối tiếp âm thanh từ một bản ghi tiền tố local
voxcpm-generate --text "Và đây là phần tiếp theo." \
  --prompt-audio /path/to/prefix.wav \
  --prompt-text "Xin chào, đây là giọng nói tham chiếu." --output continuation.wav
```

`--voice-instruction` không được kết hợp với reference hay prompt audio, `--reference-audio` không được kết hợp với prompt audio hay prompt text, và `--prompt-audio` cùng `--prompt-text` phải xuất hiện cùng nhau. Reference audio chỉ được đọc từ filesystem local, không bao giờ được tải từ xa và không xuất hiện trong error/report public. M7 bổ sung backend-native streaming qua cùng CLI; `--stream` tiêu thụ các `AudioChunk` thuộc dự án và chỉ ghi một validation WAV sau khi stream hoàn tất:

```bash
voxcpm-generate --stream --text "Xin chào từ VoxCPM2." \
  --output streamed.wav --report streamed-report.json
```

M7 qualification model/backend streaming; M8 bổ sung process/IPC boundary nội bộ; M9 bổ sung HTTP surface; M10 qualification hai real GPU worker độc lập dưới cùng API parent. Cài API extra bằng `python -m pip install -e '.[api]'`, cấu hình GPU tường minh, auth và bounded queue, rồi chạy `voxcpm-serve`. Với T4x2 đã qualification, dùng `VOXCPM_DEVICE=cuda`, `VOXCPM_GPU_DEVICES=0,1` và `VOXCPM_WORKERS=2`. Các endpoint đã qualification là `GET /healthz`, `GET /readyz`, `POST /v1/tts` và `POST /v1/tts/stream`. Stream trả raw `pcm_s16le` qua `application/octet-stream`, không phải SSE hay WebSocket.

CI thông thường M0 đến M10 vẫn GPU-free và model-free. M0 từ chối tracked `.bin` cùng các định dạng model-weight. Xem [`docs/API-RUNTIME.vi.md`](docs/API-RUNTIME.vi.md) cho HTTP contract và [`docs/T4X2-TWO-WORKER-RUNTIME.vi.md`](docs/T4X2-TWO-WORKER-RUNTIME.vi.md) cho ranh giới M10 T4x2 đã đo.

## License

Apache-2.0. Xem `LICENSE`.
