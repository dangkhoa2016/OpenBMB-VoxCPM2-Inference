# OpenBMB-VoxCPM2-Inference

Kỹ thuật inference portable cho họ mô hình OpenBMB VoxCPM2.

## Trạng thái

Dự án đang ở giai đoạn phát triển ban đầu hướng tới bản phát hành `v1.0.0` đầu tiên. M1 cung cấp cấu hình và kế hoạch thực thi CPU/CUDA; M2 bổ sung phân giải model local portable; M3 đóng băng hợp đồng backend thuộc dự án; M4 qualification TTS chuẩn thật trên CPU; và M5 qualification TTS chuẩn thật trên đúng một NVIDIA T4 được chọn tường minh. Canonical run M5 đo trên `cuda:0`, `bfloat16`, `optimize=False`, với peak allocated VRAM 5.194 GiB và real-time factor khoảng 2.97. Thực thi T4x2/multi-GPU, các tính năng VoxCPM2 khác, streaming, API, scheduler, nhiều worker và production deployment vẫn chưa được qualification. Dự án chưa sẵn sàng cho production.

## Phạm vi

Kiến trúc mục tiêu là portable trên CPU, một GPU và các replica multi-GPU độc lập. CPU và đúng một GPU CUDA được chọn tường minh hiện đã có runtime path dựa trên bằng chứng qua M5. Các replica multi-GPU độc lập vẫn là mục tiêu tương lai và chỉ được bật sau qualification riêng. Kaggle là một mục tiêu triển khai và qualification, không phải kiến trúc lõi.

## Mô hình và ghi công

Trọng số mô hình không được lưu trong repository này. Mô hình tham chiếu là [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) và implementation tham chiếu là [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM).

Đây là một dự án kỹ thuật độc lập, không phải bản phát hành chính thức của OpenBMB và không có tuyên bố liên kết nào. Xem `provenance/SOURCE-PROVENANCE.md` để biết revision nguồn đã khóa và kết quả xác minh license.

## Phát triển

M0 đến M5 hỗ trợ Python 3.10 đến 3.12.

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

Test M0 đến M5 không tải hoặc chạy mô hình VoxCPM2 và không yêu cầu GPU; backend thật được kiểm thử với import upstream được giả lập. M0 từ chối mọi file `.bin` được Git theo dõi cùng với các định dạng trọng số mô hình. Xem [`docs/MODEL-RESOLUTION.md`](docs/MODEL-RESOLUTION.md) cho M2, [`docs/BACKEND-CONTRACT.vi.md`](docs/BACKEND-CONTRACT.vi.md) cho M3, [`docs/REAL-CPU-RUNTIME.vi.md`](docs/REAL-CPU-RUNTIME.vi.md) cho bằng chứng CPU M4 và [`docs/REAL-GPU-RUNTIME.vi.md`](docs/REAL-GPU-RUNTIME.vi.md) cho đường single-T4 M5 đã đo cùng các giới hạn của nó.

## License

Apache-2.0. Xem `LICENSE`.
