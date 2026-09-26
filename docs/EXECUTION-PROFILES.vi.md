# Execution profile

Runtime công bố bốn tên execution profile canonical:

```text
cpu
cuda-single
cuda-replica
auto
```

Các alias thân thiện được chấp nhận ở đầu vào cấu hình:

```text
gpu       -> cuda-single
multi-gpu -> cuda-replica
```

Alias luôn được chuẩn hóa về tên canonical; diagnostics và tài liệu dùng tên canonical.

## Cấu hình

Chọn profile bằng:

```bash
VOXCPM_PROFILE=cpu
VOXCPM_PROFILE=cuda-single
VOXCPM_PROFILE=cuda-replica
VOXCPM_PROFILE=auto
```

Profile layer materialize topology hiện có của `VOXCPM_DEVICE`, `VOXCPM_GPU_DEVICES` và `VOXCPM_WORKERS`. Nó không tạo scheduler hay backend path thứ hai.

### `cpu`

- bắt buộc CPU execution;
- đúng một worker;
- từ chối GPU selection và worker oversubscription.

### `cuda-single`

- yêu cầu CUDA usable;
- chọn đúng một GPU;
- dùng đúng một worker;
- nếu không khai báo GPU list thì chọn GPU runtime-visible đầu tiên;
- không bao giờ âm thầm fallback sang CPU.

### `cuda-replica`

- yêu cầu ít nhất hai GPU usable;
- tạo một worker/model replica độc lập trên mỗi GPU được chọn;
- giữ nguyên worker-per-GPU isolation contract;
- từ chối worker count khác số GPU đã chọn.

Đây là replica parallelism, không phải tensor parallelism hay model sharding.

### `auto`

- probe CUDA bên ngoài API parent process;
- không có GPU usable -> CPU với một worker;
- một GPU usable -> CUDA với một worker;
- nhiều GPU usable -> một CUDA worker trên mỗi GPU visible;
- tôn trọng các constraint device/GPU/worker tương thích đã khai báo;
- từ chối oversubscription hoặc explicit CUDA khi CUDA không usable.

CUDA probe chạy trong một subprocess ngắn. API parent không import Torch hoặc VoxCPM trong lúc profile discovery.

## Tương thích ngược

`VOXCPM_PROFILE` là optional. Khi không đặt biến này, contract low-level hiện có của `VOXCPM_DEVICE`, `VOXCPM_GPU_DEVICES` và `VOXCPM_WORKERS` không thay đổi.

Profile và low-level setting xung đột sẽ fail rõ ràng thay vì bị override âm thầm.

## API parity

HTTP client contract không phụ thuộc profile. Cùng request body, authentication header, request ID, endpoint và response format được dùng cho mọi profile.

Real qualification đã dùng cùng request:

```json
{"text":"Xin chào."}
```

với `POST /v1/tts` trên cả bốn canonical profile. Cả bốn đều trả HTTP 200 với WAV mono 48 kHz. Timing đo được chỉ là evidence về functional parity, không phải benchmark guarantee.

## Ranh giới phạm vi

Execution profile chỉ freeze các tên topology được hỗ trợ. Chúng không qualification:

- tensor parallelism;
- model sharding;
- automatic worker respawn;
- mọi loại GPU;
- performance/scaling claim chung;
- production deployment.

Xem `provenance/M11-EXECUTION-PROFILE-EVIDENCE.md`.
