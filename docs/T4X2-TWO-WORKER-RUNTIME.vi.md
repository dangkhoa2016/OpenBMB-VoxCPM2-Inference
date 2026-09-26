# Runtime T4x2 với hai worker

## Trạng thái

M10 qualification hai process worker VoxCPM2 thật, độc lập, chạy dưới một FastAPI parent trên máy Kaggle có hai GPU Tesla T4.

```text
FastAPI parent
  -> Scheduler
     -> worker0 -> physical GPU 0 -> child-local cuda:0
     -> worker1 -> physical GPU 1 -> child-local cuda:0
```

Mỗi worker sở hữu một model replica hoàn chỉnh. M10 không dùng tensor parallelism, model sharding, shared CUDA context hay shared model object.

## Authority

```text
M10 implementation base   2426aeb21a0ad2b41bc3ff463fd76af21d692897
VERSION                   1.0.0
physical GPUs             2 x Tesla T4
GPU memory mỗi GPU        16,106,127,360 B
model config SHA-256      405f0dcd92f7feba6011ed4eac5c8d4f74cba9712f07fd5cfa3063bbdd95402c
```

Model được resolve từ local mirror đã attach trong offline mode với zero remote acquisition.

## GPU ownership và parent isolation

Fresh-machine startup probe hoàn tất trong 81.852 s.

```text
worker0 PID / GPU / VRAM    7733 / 0 / 5,496,635,392 B
worker1 PID / GPU / VRAM    7758 / 1 / 5,496,635,392 B
worker0 RSS                  4,062,543,872 B
worker1 RSS                  4,103,450,624 B
API parent RSS                  21,827,584 B
parent Torch imported        false
parent VoxCPM imported       false
```

Child-visibility probe chứng minh mỗi child chỉ thấy đúng một CUDA device runtime và dùng `cuda:0`, đồng thời giữ đúng mapping physical GPU từ parent. Sau shutdown, cả hai GPU không còn compute process.

## Parallel one-shot HTTP TTS

```text
sequential A / B            9.311 s / 8.820 s
sequential total            18.133 s
parallel A / B              9.758 s / 10.122 s
parallel makespan           10.157 s
observed ratio               1.785 x
```

Tất cả request đều HTTP 200 với WAV mono 48 kHz hợp lệ. Tỷ lệ trên chỉ là characterization của một lần chạy.

## Bounded admission

Với hai worker và một pending slot:

```text
A -> active -> HTTP 200 trong 29.631 s
B -> active -> HTTP 200 trong 34.718 s
C -> pending -> HTTP 200 trong 61.811 s
D -> HTTP 429 queue_full trong 0.018 s
mid-run readyz -> 200 với ready=0, busy=2, failed=0
```

## Parallel native HTTP streaming

| Stream | First body | Completion | Client chunks | PCM bytes |
| --- | ---: | ---: | ---: | ---: |
| A | 1.039 s | 12.760 s | 33 | 506,880 |
| B | 1.034 s | 12.960 s | 34 | 522,240 |

Cả hai stream trả byte đầu tiên trước completion và giữ request ID độc lập.

## Failure isolation và degraded readiness

Một active stream bị disconnect sau body chunk đầu tiên 15,360 byte tại 1.132 s. Worker xử lý stream đó biến mất khỏi GPU compute-process list. Worker thứ hai vẫn sống, hoàn tất stream trong 25.765 s và sau đó xử lý request C đang pending trong 37.673 s. `/readyz` vẫn HTTP 200 với một survivor.

Một phép thử deliberate active-worker failure riêng qualification zero-healthy semantics mà không giả định disconnect luôn được phát hiện trong một deadline cố định:

```text
2 healthy workers             readyz 200
worker0 bị terminate          request 503 worker_exited
1 healthy worker              readyz 200, ready=1, failed=1
worker1 bị terminate          request 503 worker_exited
0 healthy workers             readyz 503, ready=0, failed=2
healthz                       200
GPU compute processes         0
```

M10 không auto-respawn worker.

## Fresh offline/no-download proof

Fresh two-worker server dùng offline flags, `HF_HOME` mới và rỗng, cùng remote endpoint/proxy bị blackhole về loopback.

```text
HF_HOME files trước/sau       0 / 0
offline request A             HTTP 200, 29.291 s
offline request B             HTTP 200, 12.677 s
ready workers sau request     2
```

Resource đo sau request:

```text
GPU0 worker memory          5,996 MiB
GPU1 worker memory          5,624 MiB
API parent RSS             83,912 KiB
worker0 RSS             3,467,772 KiB
worker1 RSS             3,945,756 KiB
cgroup low/high/max              0/0/0
cgroup oom/oom_kill              0/0
cgroup oom_group_kill                0
```

Sau final shutdown: API/worker process count = 0, GPU compute-process count = 0, port 8090 đóng và OOM counters vẫn bằng 0.

## Ranh giới phạm vi

M10 qualification topology T4x2 hai worker đã đo, bounded scheduling, one-shot và native streaming song song, degraded readiness, isolation khi một worker/request bị disconnect hoặc fail, zero-healthy readiness, local/offline model use và resource safety đã đo.

M10 không qualification tensor parallelism, model sharding, shared model state, auto-respawn, batching, autoscaling, mọi loại GPU, production deployment hay performance guarantee chung.

Xem `provenance/M10-REAL-T4X2-TWO-WORKER-EVIDENCE.md`.
