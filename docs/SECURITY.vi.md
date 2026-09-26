# Gia cố bảo mật

Tài liệu này mô tả trạng thái bảo mật security qualification của HTTP runtime đã qualification. Đây là hồ sơ qualification kỹ thuật, không phải chứng nhận bảo mật.

## HTTP surface đã qualification

Public API surface được security qualification bao phủ:

- `GET /healthz`
- `GET /readyz`
- `POST /v1/tts`
- `POST /v1/tts/stream`

Các endpoint POST nhận JSON text input. HTTP surface hiện tại không nhận upload reference audio hoặc prompt audio.

## Xác thực và external bind

Bearer authentication được yêu cầu mặc định. security qualification giữ constant-time token comparison và từ chối thiếu credential, sai scheme, bearer rỗng hoặc sai, whitespace variant không khớp chính xác contract, Authorization value quá lớn và nhiều header Authorization trùng nhau.

Non-loopback bind khi tắt auth vẫn fail-closed trừ khi `VOXCPM_ALLOW_UNAUTHENTICATED_EXTERNAL=1` được bật tường minh.

Lỗi validation dùng public error envelope chuẩn hóa và không phản chiếu raw request input.

## Bounded admission và time limit

security qualification enforce contract cấu hình hiện có:

- `VOXCPM_MAX_QUEUE_SIZE`: pending queue có giới hạn.
- `VOXCPM_QUEUE_TIMEOUT_SECONDS`: hết hạn request đang chờ trong queue.
- `VOXCPM_REQUEST_TIMEOUT_SECONDS`: giới hạn thời gian chờ HTTP one-shot và cancel work bị timeout.
- `VOXCPM_MAX_INFERENCE_SECONDS`: terminate worker nếu active inference vượt ceiling cấu hình.
- `VOXCPM_MAX_CONCURRENT_REQUESTS`: giới hạn số request active đồng thời kể cả khi còn worker READY.
- `VOXCPM_STREAM_BACKPRESSURE_TIMEOUT_SECONDS`: ngăn stream consumer không đọc giữ dispatcher bị block vô hạn.

Request vượt pending queue limit nhận queue-full rejection hiện có thay vì làm tăng tài nguyên không giới hạn.

## Guard cho temporary storage

Khi cấu hình `VOXCPM_MIN_TMP_FREE_BYTES`, ứng dụng kiểm tra free space của `VOXCPM_TMP_DIR`, hoặc system temporary directory nếu không cấu hình directory riêng.

Runtime fail-closed khi free-space floor không đạt yêu cầu.

Các HTTP endpoint hiện tại không tạo temporary file cho reference audio hay prompt audio. Vì vậy security qualification không tuyên bố temp-upload cleanup cho một API flow chưa tồn tại.

## Redaction và public error

Central redaction bao phủ URL, absolute filesystem path, home-directory path, các token shape phổ biến và bearer credential value.

API validation error được chuẩn hóa để framework validation detail không phản chiếu raw input về client.

Diagnostics chỉ cho biết API token đã được cấu hình hay chưa, không bao giờ xuất token.

## Secret scanning

security qualification quét tracked file cho các dạng token và private-key phổ biến. Các token-shaped value tìm thấy trong repository chỉ nằm trong explicit test fixture dùng để kiểm tra redaction. Cùng phép quét đó cho kết quả không có secret-shape match nào ngoài `tests/`.

Không được commit credential thật. Dùng environment variable hoặc external secret store.

## Dependency review

API extra của project pin:

```text
fastapi==0.136.1
uvicorn[standard]==0.46.0
httpx==0.28.1
```

Kaggle qualification host có một số package cài sẵn không liên quan với project tạo `pip check` conflict. security qualification không coi các host-image conflict đó là lỗi dependency của project.

Standard-library `venv` trên qualification host không bootstrap được `ensurepip`, đồng thời `pip-audit` và `safety` không được cài. Vì vậy security qualification không tuyên bố dependency graph không có vulnerability.

Docker qualification đã ghi nhận Docker runtime pin upstream VoxCPM source commit nhưng chưa freeze toàn bộ transitive runtime dependency. Việc freeze chính xác transitive dependency vẫn là quyết định reproducibility cho release.

## Ranh giới audio validation

Reference và prompt audio validation tồn tại ở backend và CLI path từ các qualification trước, nhưng security qualification HTTP surface chỉ nhận text. security qualification không claim HTTP multipart audio content sniffing, từ chối uploaded WAV malformed, uploaded reference byte hoặc duration limit, uploaded prompt-audio duration limit hay temporary upload cleanup.

Các control này cần qualification khi một HTTP audio-upload endpoint được bổ sung.

## Ranh giới output duration

`VOXCPM_MAX_OUTPUT_SECONDS` vẫn thuộc parsed configuration contract, nhưng security qualification không claim HTTP output-duration enforcement đã qualification. Streaming response bắt đầu trước khi biết final duration, nên policy tương lai phải định nghĩa deterministic truncation và error semantics trước khi quảng bá knob này như một HTTP security guarantee.

## Giới hạn vận hành

security qualification không thiết lập security certification, đảm bảo hoàn toàn không có vulnerability, khả năng chống DDoS, multi-node rate limiting, autoscaling, HA/SLA guarantee, real-model Docker security qualification hay CUDA Docker security qualification.

Xem `provenance/QUALIFICATION.md` để biết evidence cụ thể và trạng thái từng gate.
