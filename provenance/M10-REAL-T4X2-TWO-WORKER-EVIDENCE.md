# M10 Real T4x2 Two-Worker Evidence

## Status

- Repository: OpenBMB-VoxCPM2-Inference
- Branch: main
- M10 implementation base: `2426aeb21a0ad2b41bc3ff463fd76af21d692897`
- VERSION: `1.0.0`
- No release tag was created during M10 qualification.
- Real hardware: two physical Tesla T4 GPUs.
- Topology: two independent spawned workers, one model replica per GPU, one FastAPI parent.

## Model and runtime authority

```text
upstream VoxCPM commit     f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69
installed voxcpm           2.0.3.post33+gf772e498a
PyTorch                    2.10.0+cu128
CUDA runtime               12.8
model id                   openbmb/VoxCPM2
model config SHA-256       405f0dcd92f7feba6011ed4eac5c8d4f74cba9712f07fd5cfa3063bbdd95402c
model metadata status      ok
offline                    true
remote acquisition count   0
```

Absolute local model paths, API tokens, and temporary qualification paths are intentionally omitted.

## Model-free baseline and GPU remapping

```text
454 passed
1 skipped
```

The skip is the existing root filesystem-permission case.

Spawn visibility proof:

```text
physical GPU 0 -> CUDA_VISIBLE_DEVICES=0 -> device_count=1 -> current_device=0 -> Tesla T4
physical GPU 1 -> CUDA_VISIBLE_DEVICES=1 -> device_count=1 -> current_device=0 -> Tesla T4
exit codes [0, 0]
```

## Real two-worker startup

```text
startup                       81.852 s
worker0 PID / GPU / VRAM      7733 / 0 / 5,496,635,392 B
worker1 PID / GPU / VRAM      7758 / 1 / 5,496,635,392 B
worker0 RSS                   4,062,543,872 B
worker1 RSS                   4,103,450,624 B
API parent RSS                   21,827,584 B
parent Torch imported         false
parent VoxCPM imported        false
```

After shutdown both worker processes and both GPU compute-process entries were gone.

## Parallel HTTP and admission

```text
sequential total             18.133 s
parallel makespan            10.157 s
observed ratio                1.785 x
parallel A / B               9.758 s / 10.122 s
```

Every one-shot response was HTTP 200 with valid 48 kHz mono WAV data. The ratio is characterization only.

With two workers and one pending slot:

```text
mid-run ready/busy/failed     0/2/0
A                             200, 29.631 s
B                             200, 34.718 s
C                             200, 61.811 s
D                             429 queue_full, 0.018 s
D retryable                   true
```

## Parallel native streaming

```text
A first/completion/chunks     1.039 s / 12.760 s / 33
A PCM bytes                   506,880
B first/completion/chunks     1.034 s / 12.960 s / 34
B PCM bytes                   522,240
parallel stream makespan      12.965 s
```

Both streams produced first bytes before completion and used distinct request IDs.

## Disconnect isolation and zero-healthy readiness

Stream A was closed after the first 15,360-byte body chunk at 1.132 s. Its worker left the GPU compute-process list. Worker B remained alive, completed its stream in 25.765 s, and pending request C returned HTTP 200 in 37.673 s. Final survivor state was `ready=1, busy=0, failed=0`.

A separate exploratory final-worker stream-close did not transition readiness within the observation window, so M10 does not claim a universal disconnect-detection deadline. The required zero-healthy state was qualified independently by deliberate active worker loss:

```text
initial                       ready=2, busy=0, failed=0
terminate worker0             active request -> 503 worker_exited
after worker0 loss            readyz 200, ready=1, failed=1
terminate worker1             active request -> 503 worker_exited
after worker1 loss            readyz 503, ready=0, failed=2
healthz                       200
GPU compute process count     0
```

No automatic respawn occurred.

## Fresh offline/no-download proof

```text
HF_HOME files before/after    0 / 0
request A                     200, 29.291 s, 48 kHz mono
request B                     200, 12.677 s, 48 kHz mono
ready state after             ready=2, busy=0, failed=0
```

Post-request resources:

```text
GPU0 worker memory            5,996 MiB
GPU1 worker memory            5,624 MiB
API parent RSS               83,912 KiB
worker0 RSS               3,467,772 KiB
worker1 RSS               3,945,756 KiB
cgroup low/high/max                0/0/0
cgroup oom/oom_kill                0/0
cgroup oom_group_kill                  0
```

Final cleanup:

```text
API/worker process count      0
GPU compute process count     0
port 8090                     closed
cgroup OOM counters           0
```

## Scope exclusions

M10 does not qualify tensor parallelism, model sharding, shared model state, automatic worker respawn, batching, autoscaling, universal GPU compatibility, production deployment, or universal latency/speedup claims.

## Gates

```text
M10_PRE_IMPLEMENTATION_AUDIT=PASS
M10_PARENT_TOPOLOGY_PLAN_GATE=PASS
M10_TWO_WORKER_FACTORY_GATE=PASS
M10_GPU_ORDER_PRESERVATION_GATE=PASS
M10_CHILD_GPU_REMAPPING_GATE=PASS
M10_PARTIAL_STARTUP_CLEANUP_GATE=PASS
M10_TWO_WORKER_SCHEDULER_GATE=PASS
M10_TWO_ACTIVE_REQUESTS_GATE=PASS
M10_PENDING_THIRD_REQUEST_GATE=PASS
M10_SINGLE_WORKER_FAILURE_ISOLATION_GATE=PASS
M10_ONE_OF_TWO_CANCEL_READINESS_GATE=PASS
M10_ZERO_HEALTHY_WORKER_READINESS_GATE=PASS
M10_NO_ORPHAN_WORKER_GATE=PASS
M10_M0_M9_REGRESSION_GATE=PASS
M10_IMPLEMENTATION_EXACT_SHA_CI=PASS
M10_REAL_T4X2_STARTUP_GATE=PASS
M10_REAL_GPU_OWNERSHIP_GATE=PASS
M10_REAL_API_PARENT_MODEL_ISOLATION_GATE=PASS
M10_REAL_PARALLEL_TTS_GATE=PASS
M10_REAL_TWO_ACTIVE_ONE_PENDING_GATE=PASS
M10_REAL_QUEUE_FULL_GATE=PASS
M10_REAL_FAILURE_ISOLATION_GATE=PASS
M10_REAL_SURVIVOR_WORKER_GATE=PASS
M10_REAL_PARALLEL_STREAM_GATE=PASS
M10_REAL_T4X2_OFFLINE_GATE=PASS
M10_REAL_MEMORY_MEASUREMENT_GATE=PASS
M10_REAL_RESOURCE_SAFETY_GATE=PASS
M10_TENSOR_PARALLELISM_REMAINS_OUT_OF_SCOPE_GATE=PASS
M10_MODEL_SHARDING_REMAINS_OUT_OF_SCOPE_GATE=PASS
M10_INVESTIGATION_COMPLETENESS=PASS
M10_T4X2_TWO_WORKER_RUNTIME=PASS
```

The final evidence-commit exact-SHA CI gate is recorded only after this document is committed, pushed, and the corresponding CI run is green.
