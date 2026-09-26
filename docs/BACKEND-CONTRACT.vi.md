# Hợp đồng backend M3

M3 đóng băng ranh giới thuộc dự án cho các implementation inference trong tương lai. `InferenceBackend` là abstraction thực thi duy nhất mà worker ở các giai đoạn sau được phép dùng. M3 không nối ranh giới này với `ModelResolver`, `DeviceManager`, package VoxCPM thật, trọng số mô hình, worker, scheduler hay HTTP API.

Hợp đồng được định nghĩa mà không cần PyTorch, NumPy, Transformers, thư viện audio, FastAPI hoặc type VoxCPM upstream.

## Thao tác

| Thao tác | Hợp đồng | Kết quả |
| --- | --- | --- |
| `load()` | Chuyển backend từ trạng thái created sang loaded | `BackendInfo` |
| `synthesize(request)` | Tổng hợp giọng nói từ văn bản | `AudioResult` |
| `design(request)` | Tổng hợp giọng nói với chỉ dẫn thiết kế giọng thuộc dự án | `AudioResult` |
| `clone(request)` | Tổng hợp giọng nói có điều kiện từ descriptor audio tham chiếu | `AudioResult` |
| `continue_audio(request)` | Tiếp tục từ audio tham chiếu và transcript của audio đó | `AudioResult` |
| `stream(request)` | Tạo các chunk audio có thứ tự mà không có framing HTTP | `Iterator[AudioChunk]` |
| `close()` | Chuyển backend sang trạng thái closed | `None` |

Nguồn upstream đã pin tại commit `f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69` chỉ được đọc để xác định ngữ nghĩa. Tạo giọng nói thông thường từ văn bản ánh xạ thành `SpeechRequest`; thiết kế giọng bằng văn bản ánh xạ thành `VoiceDesignRequest`; clone từ audio tham chiếu ánh xạ thành `CloneRequest`; tiếp tục từ audio tiền tố kèm transcript ánh xạ thành `ContinuationRequest`. M3 không công khai tên tham số, class, prompt cache, tensor hay cờ implementation của upstream.

## Kiểu request

Mọi request type là dataclass bất biến thuộc dự án và chỉ chứa giá trị của standard library.

- `SpeechRequest(text)` chứa văn bản đích không chỉ toàn khoảng trắng.
- `VoiceDesignRequest(text, instruction)` chứa văn bản đích và chỉ dẫn thiết kế giọng, cả hai đều không chỉ toàn khoảng trắng.
- `CloneRequest(text, reference_audio)` chứa văn bản đích và một `AudioReference`.
- `ContinuationRequest(text, reference_audio, reference_transcript)` chứa văn bản đích, `AudioReference` và transcript không chỉ toàn khoảng trắng của audio tiền tố.
- `StreamRequest(request)` bọc đúng một request one-shot để streaming giữ nguyên ngữ nghĩa thao tác.

`AudioReference(local_path)` chỉ là descriptor. M3 không mở, giải mã, resample, tải hoặc kiểm tra sự tồn tại của file được tham chiếu. Ranh giới worker ở giai đoạn sau có thể truyền audio lớn qua file, nhưng việc thu hình không thuộc hợp đồng này.

## Kiểu audio

`AudioResult` chứa:

```text
samples: tuple[float, ...]
sample_rate_hz: int dương
channels: int dương
metadata: cặp scalar bất biến đã sắp xếp
```

`AudioChunk` bổ sung `sequence` bắt đầu từ 0 và cờ `is_final`. Sample được xen kẽ theo channel theo thứ tự biểu diễn bởi `channels`; fake backend tạo audio mono. Sample container là tuple bất biến gồm các Python float hữu hạn, không phải NumPy, Torch, WAV hoặc media type HTTP.

Output chỉ dùng cho fake là 48 kHz mono với 480 sample. Tần số 48 kHz là lựa chọn `test/fake contract only` và không khẳng định gì về sample rate native của VoxCPM2. Sample của fake là Python binary64 float. Contract test serialize theo thứ tự IEEE-754 binary64 little-endian (`struct.pack("<d", sample)`) trước khi tính SHA-256.

## Vòng đời

Vòng đời là rõ ràng:

```text
created -> load() -> loaded -> close() -> closed
```

`load()` và `close()` idempotent trong khi trạng thái cho phép. Gọi generation trước `load()` ném `BackendStateError` với code `backend_not_loaded`. Sử dụng sau `close()` hoặc gọi `load()` sau close ném `BackendStateError` với code `backend_closed`. Backend đã đóng không tự mở lại.

## Chuẩn hóa lỗi

Hierarchy lỗi của dự án là:

```text
BackendError
├── BackendStateError
├── BackendRequestError
├── BackendUnsupportedError
└ BackendExecutionError
```

Mỗi lỗi có machine-readable code ổn định, public message an toàn, cờ retry và các chi tiết scalar bất biến. `to_dict()` trả về envelope an toàn xác định chỉ gồm `code`, `details`, `message` và `retryable`.

`normalize_backend_error()` giữ nguyên một `BackendError` hiện có. Exception lạ được chuyển thành `BackendExecutionError`, chỉ đưa thao tác an toàn và tên exception đã lọc vào details. Nó không sao chép message thô, traceback, filesystem path, URL, token hoặc object exception upstream.

## Tính xác định và isolation của fake

`FakeVoxCPMBackend` triển khai mọi thao tác bằng audio xác định trong bộ nhớ. Nó không tải model, chạm `ModelResolver` hoặc Kaggle adapter, dùng CUDA, đọc biến môi trường, mở file, truy cập network, sleep, dùng wall-clock time, random không seed hoặc gọi Python `hash()`.

Fake canonicalize tên thao tác và field request thuộc dự án thành JSON UTF-8 có key sắp xếp, hash byte bằng SHA-256 rồi sinh sample binary64 có biên độ giới hạn. Request giống nhau tạo kết quả giống nhau qua các lần gọi lặp lại và các instance mới. Tên thao tác, văn bản, chỉ dẫn, descriptor tham chiếu hoặc transcript khác nhau làm output xác định thay đổi.

Với fake backend xác định, streaming chia cùng waveform đầy đủ thành các chunk giới hạn và không rỗng: sequence liên tục, có đúng một chunk final, và nối sample của các chunk bằng chính xác kết quả fake one-shot tương đương. Tính bằng nhau tuyệt đối với one-shot này là bảo đảm của fake backend, không phải yêu cầu phổ quát cho implementation native streaming thật. Real backend phải giữ đúng ngữ nghĩa request, thứ tự chunk, sample không rỗng, đúng một chunk final và kiểu dữ liệu thuộc dự án, đồng thời phải tài liệu hóa riêng mức tương đương số học đo được giữa native stream và one-shot.

`FakeFailurePlan` bị tắt theo mặc định. Khi được cấu hình tường minh, lỗi runtime private của fake được chuẩn hóa tại public boundary và message private không xuất hiện trong lỗi public.

## M3 chứng minh điều gì

Fake backend pass chỉ chứng minh hợp đồng dự án ổn định, hành vi xác định, lỗi được chuẩn hóa, vòng đời đúng và ranh giới thực thi hermetic.

Nó không chứng minh rằng:

- VoxCPM2 thật load được;
- ràng buộc CPU, CUDA, T4, multi-GPU, RAM hoặc VRAM được đáp ứng;
- sample rate native đã được qualification;
- thiết kế giọng, clone, continuation hoặc streaming thật hoạt động;
- chất lượng audio hoặc hiệu năng thời gian thực đạt yêu cầu.

Model loading và runtime qualification thật nằm ngoài M3. Milestone kế tiếp được phép sau khi M3 đóng hoàn tất là M4 real CPU runtime investigation; hợp đồng này không bắt đầu M4.
