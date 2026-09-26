# Xử lý sự cố

## CUDA tường minh bị fail

Chạy `voxcpm-doctor` và `VOXCPM_DEVICE=cuda voxcpm-doctor`. Nếu không có CUDA device dùng được thì failure là chủ ý; runtime không âm thầm chuyển sang CPU.

## Không resolve được model path

Đặt `VOXCPM_MODEL_PATH` tới thư mục VoxCPM2 local/mounted tuyệt đối rồi chạy `voxcpm-verify-model`.

## API trả 401

Kiểm tra `VOXCPM_API_TOKEN`, `VOXCPM_REQUIRE_AUTH=1` và chỉ một header `Authorization: Bearer ...`.

## API trả 429

Bounded pending queue đã đầy. Giảm client concurrency hoặc chỉ thay đổi queue size trong measured resource envelope.

## `/healthz` là 200 nhưng `/readyz` là 503

Process còn sống nhưng không có worker usable ở trạng thái ready. Kiểm tra worker/runtime state; cancellation hoặc timeout có thể chủ ý stop worker.

## Streaming bị stall hoặc kết thúc

Kiểm tra tốc độ đọc của client và `VOXCPM_STREAM_BACKPRESSURE_TIMEOUT_SECONDS`.

## Startup fail vì temporary storage

Kiểm tra `VOXCPM_TMP_DIR` và `VOXCPM_MIN_TMP_FREE_BYTES`.

## Kaggle không tìm thấy model

Xác nhận `dangkhoa2016/openbmb-voxcpm2` đã được attach vào notebook. Xem [Kaggle](KAGGLE.vi.md).

## Docker không load được model

Mount model tại `/models/VoxCPM2` hoặc override `VOXCPM_MODEL_PATH`. Image không embed model weights.
