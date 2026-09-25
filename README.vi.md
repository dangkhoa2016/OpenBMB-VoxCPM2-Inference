# OpenBMB-VoxCPM2-Inference

Kỹ thuật inference portable cho họ mô hình OpenBMB VoxCPM2.

## Trạng thái

Dự án đang ở giai đoạn phát triển ban đầu hướng tới bản phát hành `v1.0.0` đầu tiên. M1 cung cấp cấu hình môi trường ổn định, inventory host/CUDA tùy chọn, kế hoạch thực thi CPU/CUDA xác định và CLI chẩn đoán `voxcpm-doctor`. M2 bổ sung phân giải mô hình local chỉ đọc metadata, discovery giới hạn trên Kaggle mount và CLI `voxcpm-verify-model`. Cả hai milestone đều không tải trọng số mô hình và chưa cung cấp inference, API, streaming, scheduler hay hỗ trợ triển khai. Dự án chưa sẵn sàng cho production.

## Phạm vi

Kiến trúc mục tiêu là portable trên CPU, một GPU và các replica multi-GPU độc lập. Các chế độ thực thi này là mục tiêu trong tương lai và chỉ được bật sau khi có đo đạc cùng đánh giá chất lượng ở các giai đoạn sau. Kaggle là một mục tiêu triển khai và qualification, không phải kiến trúc lõi.

## Mô hình và ghi công

Trọng số mô hình không được lưu trong repository này. Mô hình tham chiếu là [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) và implementation tham chiếu là [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM).

Đây là một dự án kỹ thuật độc lập, không phải bản phát hành chính thức của OpenBMB và không có tuyên bố liên kết nào. Xem `provenance/SOURCE-PROVENANCE.md` để biết revision nguồn đã khóa và kết quả xác minh license.

## Phát triển

M0 đến M2 hỗ trợ Python 3.10 đến 3.12.

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

Test M0 đến M2 không tải hoặc chạy mô hình VoxCPM2 và không yêu cầu GPU. M0 từ chối mọi file `.bin` được Git theo dõi cùng với các định dạng trọng số mô hình. Xem [`docs/MODEL-RESOLUTION.md`](docs/MODEL-RESOLUTION.md) để biết hợp đồng resolver M2. Chính sách thận trọng này có thể được tinh chỉnh sau khi có bằng chứng ở một giai đoạn sau.

## License

Apache-2.0. Xem `LICENSE`.
