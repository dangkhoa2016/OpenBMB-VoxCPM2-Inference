# T4x2 two-worker runtime

## Status

M10 qualifies two independent real VoxCPM2 worker processes under one FastAPI parent on a Kaggle host exposing two Tesla T4 GPUs.

```text
FastAPI parent
  -> Scheduler
     -> worker0 -> physical GPU 0 -> child-local cuda:0
     -> worker1 -> physical GPU 1 -> child-local cuda:0
```

Each worker owns one complete model replica. M10 does not use tensor parallelism, model sharding, a shared CUDA context, or a shared model object.

## Authority

```text
M10 implementation base   2426aeb21a0ad2b41bc3ff463fd76af21d692897
VERSION                   1.0.0
physical GPUs             2 x Tesla T4
GPU memory each           16,106,127,360 B
model config SHA-256      405f0dcd92f7feba6011ed4eac5c8d4f74cba9712f07fd5cfa3063bbdd95402c
```

The model was resolved from the attached local mirror in offline mode with zero remote acquisition.

## GPU ownership and parent isolation

The fresh-machine direct startup probe completed in 81.852 s.

```text
worker0 PID / GPU / VRAM    7733 / 0 / 5,496,635,392 B
worker1 PID / GPU / VRAM    7758 / 1 / 5,496,635,392 B
worker0 RSS                  4,062,543,872 B
worker1 RSS                  4,103,450,624 B
API parent RSS                  21,827,584 B
parent Torch imported        false
parent VoxCPM imported       false
```

A child-visibility probe proved each child sees exactly one runtime CUDA device and uses `cuda:0`, while preserving the parent physical GPU mapping. After shutdown both GPU compute-process lists were empty.

## Parallel one-shot HTTP TTS

```text
sequential A / B            9.311 s / 8.820 s
sequential total            18.133 s
parallel A / B              9.758 s / 10.122 s
parallel makespan           10.157 s
observed ratio               1.785 x
```

All requests returned HTTP 200 with valid 48 kHz mono WAV output. The ratio is one-run characterization, not a universal speedup claim.

## Bounded admission

With two workers and one pending slot:

```text
A -> active -> HTTP 200 in 29.631 s
B -> active -> HTTP 200 in 34.718 s
C -> pending -> HTTP 200 in 61.811 s
D -> HTTP 429 queue_full in 0.018 s
mid-run readyz -> 200 with ready=0, busy=2, failed=0
```

## Parallel native HTTP streaming

| Stream | First body | Completion | Client chunks | PCM bytes |
| --- | ---: | ---: | ---: | ---: |
| A | 1.039 s | 12.760 s | 33 | 506,880 |
| B | 1.034 s | 12.960 s | 34 | 522,240 |

Both streams delivered first bytes before completion and kept distinct request IDs.

## Failure isolation and degraded readiness

One active stream was disconnected after the first 15,360-byte body chunk at 1.132 s. Its worker left the GPU compute-process list. The second worker remained alive, completed its stream in 25.765 s, and then served pending request C in 37.673 s. `/readyz` remained HTTP 200 with one survivor.

A separate deliberate active-worker failure sequence qualified zero-healthy semantics without relying on a fixed disconnect-detection deadline:

```text
2 healthy workers             readyz 200
worker0 terminated            request 503 worker_exited
1 healthy worker              readyz 200, ready=1, failed=1
worker1 terminated            request 503 worker_exited
0 healthy workers             readyz 503, ready=0, failed=2
healthz                       200
GPU compute processes         0
```

No automatic worker respawn is implemented or claimed.

## Fresh offline/no-download proof

A fresh two-worker server used offline flags, a fresh empty `HF_HOME`, and loopback-blackholed remote endpoints/proxies.

```text
HF_HOME files before/after    0 / 0
offline request A             HTTP 200, 29.291 s
offline request B             HTTP 200, 12.677 s
ready workers after requests  2
```

Measured post-request resources:

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

After final shutdown, API/worker process count was zero, GPU compute-process count was zero, port 8090 was closed, and OOM counters remained zero.

## Scope boundary

M10 qualifies this measured T4x2 two-worker topology, bounded scheduling, parallel one-shot and native streaming requests, degraded readiness, one-worker disconnect/failure isolation, zero-healthy readiness, local/offline model use, and measured resource safety.

M10 does not qualify tensor parallelism, model sharding, shared model state, automatic worker respawn, batching, autoscaling, all GPU types, production deployment, or a universal performance guarantee.

See `provenance/M10-REAL-T4X2-TWO-WORKER-EVIDENCE.md` for the gate record.
