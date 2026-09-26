# OpenBMB-VoxCPM2-Inference

Kỹ thuật inference portable cho họ mô hình OpenBMB VoxCPM2.

## Trạng thái

Dự án đang ở giai đoạn phát triển ban đầu hướng tới bản phát hành `v1.0.0` đầu tiên. M1 cung cấp cấu hình môi trường ổn định, inventory host/CUDA tùy chọn, kế hoạch thực thi CPU/CUDA xác định và CLI chẩn đoán `voxcpm-doctor`. M2 bổ sung phân giải mô hình local chỉ đọc metadata, discovery giới hạn trên Kaggle mount và CLI `voxcpm-verify-model`. M3 đóng băng hợp đồng backend thuộc dự án với lỗi chuẩn hóa và fake backend xác định, hermetic. M4 bổ sung đường thực thi thật đầu tiên: một backend chỉ-CPU nạp mô hình VoxCPM2 local đã ghim, tổng hợp TTS chuẩn, ghi WAV PCM16 thuộc dự án, và báo cáo telemetry runtime đã che thông tin nhạy cảm qua CLI `voxcpm-generate`. Việc nạp và tổng hợp CPU thật đã được đo thành công trên một host Kaggle, ở mức chậm hơn thời gian thực khoảng `48 lần`; tính đủ bộ nhớ, GPU, các tính năng VoxCPM2 khác, phát trực tuyến, API, scheduler, worker và qualification triển khai vẫn chưa được đánh giá. Dự án chưa sẵn sàng cho production.

## Phạm vi

Kiến trúc mục tiêu là portable trên CPU, một GPU và các replica multi-GPU độc lập. Các chế độ thực thi này là mục tiêu trong tương lai và chỉ được bật sau khi có đo đạc cùng đánh giá chất lượng ở các giai đoạn sau. Kaggle là một mục tiêu triển khai và qualification, không phải kiến trúc lõi.

## Mô hình và ghi công

Trọng số mô hình không được lưu trong repository này. Mô hình tham chiếu là [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) và implementation tham chiếu là [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM).

Đây là một dự án kỹ thuật độc lập, không phải bản phát hành chính thức của OpenBMB và không có tuyên bố liên kết nào. Xem `provenance/SOURCE-PROVENANCE.md` để biết revision nguồn đã khóa và kết quả xác minh license.

## Phát triển

M0 đến M4 hỗ trợ Python 3.10 đến 3.12.

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

Test M0 đến M4 không tải hoặc chạy mô hình VoxCPM2 và không yêu cầu GPU; backend thật được kiểm thử với import upstream được giả lập. M0 từ chối mọi file `.bin` được Git theo dõi cùng với các định dạng trọng số mô hình. Xem [`docs/MODEL-RESOLUTION.md`](docs/MODEL-RESOLUTION.md) để biết hợp đồng resolver M2, [`docs/BACKEND-CONTRACT.vi.md`](docs/BACKEND-CONTRACT.vi.md) để biết ranh giới M3 cùng giới hạn của nó, và [`docs/REAL-CPU-RUNTIME.vi.md`](docs/REAL-CPU-RUNTIME.vi.md) để biết M4 thực sự đo được gì trên CPU, bao gồm lý do đó không phải là tuyên bố 16 GB hay thời gian thực. Chính sách thận trọng này có thể được tinh chỉnh sau khi có bằng chứng ở một giai đoạn sau.

## License

Apache-2.0. Xem `LICENSE`.
