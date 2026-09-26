# Runtime worker process

## Trạng thái

M8 qualification worker process boundary dùng `spawn` thuộc dự án và bounded FIFO scheduler semantics.

CI thông thường qualification process lifecycle, admission, cancellation, worker-crash containment, error transport đã sanitize, bounded stream IPC và shutdown không để orphan bằng fake backend.

Real runtime qualification cố ý hẹp hơn: đúng một spawned worker sở hữu đúng một Tesla T4 và một real VoxCPM2 backend.

## Authority

```text
M8 base SHA             665af656234f355eace5c8a254c4f74ef394f1fd
worker boundary SHA     27e139b98ade2c754302900337b6a12762e12f61
worker boundary CI      36230495838 PASS
scheduler SHA           1252f23d4b14a4c57a8d059c11b2f12351bdee4e
scheduler CI            36230777274 PASS
VERSION                 1.0.0
```

## Kiến trúc

```text
caller
  -> Scheduler / bounded FIFO admission
  -> WorkerClient
  -> spawn IPC
  -> WorkerProcess
  -> InferenceBackend
  -> PytorchVoxCPMBackend
```

Parent sở hữu topology và không load model. Child sở hữu một backend instance. Real CUDA worker nhận một physical GPU selector trước khi import Torch/upstream; sau visibility isolation, child runtime dùng CUDA index 0 và worker_count=1.

Guardrail single-worker của real backend không bị nới lỏng.

## Scheduler semantics

M8 dùng:

```text
FIFO pending queue
bounded max_pending_requests
một active request mỗi worker
duplicate request ID bị từ chối
queued cancellation loại request trước dispatch
active cancellation terminate worker process
worker chết bất ngờ -> worker_exited
không còn healthy worker -> no_healthy_worker
shutdown join/terminate child tường minh
```

Các multi-request semantics này được qualification bằng spawned fake worker trong CI. M8 không claim real multi-worker throughput.

## Real single-T4 worker

Startup worker đo được:

```text
spawn -> READY             29.785 s
worker child RSS           4,072,284,160 B (3.793 GiB)
worker GPU memory          5,496,635,392 B (5.119 GiB)
physical GPU               Tesla T4
physical GPU memory        16,106,127,360 B
parent RSS trước worker    23,957,504 B
parent RSS sau READY       24,510,464 B
parent import torch        false
parent import voxcpm       false
```

GPU memory được đo bằng NVML theo PID của worker khi process còn sống.

## Real one-shot IPC

Canonical request:

```text
Xin chào từ VoxCPM2.
```

Kết quả:

```text
IPC wall time              4.578 s
request pickle bytes       170
result pickle bytes        691,676
sample rate                48,000 Hz
channels                   1
frames                     76,800
audio duration             1.60 s
WAV SHA-256                70f6623a5b27da9d0eb5c718914a5aeb89958008d450ed09d5d8dc2da219cde2
```

WAV được ghi và đọc độc lập trong parent process.

## Real native stream IPC

Canonical native stream đi qua process boundary theo từng chunk:

```text
first parent chunk         0.774 s
stream completion          3.321 s
chunk count                9
samples mỗi chunk          7,680
sequence contiguous        true
final chunk count          1
max serialized chunk       69,513 B
sample rate                48,000 Hz
channels                   1
frames                     69,120
audio duration             1.44 s
WAV SHA-256                a88e7d0cd6544bc50984325f3650f9a5392ccba6324c5e55f079dc866e5afc8a
```

Worker result queue là bounded nên parent chậm sẽ tạo backpressure thay vì tích lũy audio không giới hạn.

## Active cancellation

Một native stream dài được bắt đầu. Sau khi parent nhận chunk đầu tiên 7,680 samples, request được cancel.

```text
cancellation mechanism     process termination + join
cancel duration            0.360 s
worker PID gone            true
later chunks               none
```

M8 không claim cooperative cancellation của GPU kernel.

## Manual recovery và offline proof

Sau cancellation, một replacement worker được khởi động tường minh.

```text
replacement startup        29.900 s
replacement child RSS      4,091,260,928 B
offline recovery TTS       6.693 s
HF_HOME files sau run      0
replacement PID gone       true sau shutdown
```

Recovery WAV:

```text
sample rate                48,000 Hz
channels                   1
frames                     122,880
duration                   2.56 s
SHA-256                    0184f960a6461fca07a9eb1387812c4708d073b1a3c5e074d36adb120eb7a059
```

Replacement worker dùng offline flags, HF cache mới rỗng và model endpoint/proxy blackhole. Điều này chứng minh run recovery đã đo không cần remote model acquisition.

## Resource và process evidence

Sau real sequence:

```text
cgroup low                 0
cgroup high                0
cgroup max                 0
cgroup oom                 0
cgroup oom_kill            0
cgroup oom_group_kill      0
mọi PID worker đã ghi      absent
```

Không còn orphan worker process.

## Ranh giới phạm vi

M8 không qualification:

- hai real worker process;
- T4x2 throughput;
- multi-GPU scheduling;
- auto respawn worker;
- HTTP, REST, SSE, WebSocket hay browser streaming;
- external queue;
- shared-memory audio transport;
- batching hay dynamic batching;
- autoscaling;
- production deployment;
- release/tag.

Xem `provenance/M8-REAL-WORKER-PROCESS-EVIDENCE.md` cho gate record đầy đủ.
