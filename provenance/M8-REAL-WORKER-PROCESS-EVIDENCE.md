# M8 Real Worker Process Evidence

## Status

- Repository: OpenBMB-VoxCPM2-Inference
- Branch: main
- M8 base SHA: 665af656234f355eace5c8a254c4f74ef394f1fd
- Worker boundary SHA: 27e139b98ade2c754302900337b6a12762e12f61
- Worker boundary CI: 36230495838, PASS
- Scheduler SHA: 1252f23d4b14a4c57a8d059c11b2f12351bdee4e
- Scheduler CI: 36230777274, PASS
- VERSION: 1.0.0
- No tag or release was created

## Implementation summary

M8 adds a spawn-based `WorkerClient` / `WorkerProcess`, safe project-owned IPC envelopes, one backend instance per worker, child-local CUDA visibility remapping, bounded stream-result buffering, bounded FIFO scheduler admission, duplicate request-ID rejection, pending/active cancellation, crash containment, deterministic `no_healthy_worker` failure, and explicit no-orphan shutdown.

The real backend single-worker/single-device guardrail remains unchanged.

## Model-free CI authority

```text
431 passed
1 skipped
ruff          PASS
compileall    PASS
CI YAML parse PASS
targeted worker/scheduler suite 20 passed
```

CI also asserts the M8 parent-import, child-remapping, spawn-context, one-backend-per-worker, single-active-request, bounded-admission, FIFO, stream-backpressure, error-sanitization, no-orphan and out-of-scope gates.

## Real hardware and isolation

The real qualification used one physical Tesla T4 on a T4x2 host.

```text
Torch imported in parent       false
VoxCPM imported in parent      false
parent RSS before worker       23,957,504 B
parent RSS after READY         24,510,464 B
spawn -> READY                 29.785037 s
child RSS at READY             4,072,284,160 B (3.793 GiB)
backend                        pytorch-voxcpm
device                         cuda
capabilities                   clone,continue_audio,design,stream,synthesize
model id                       openbmb/VoxCPM2
```

Independent NVML measurement while a real worker was alive:

```text
physical GPU                   Tesla T4
physical GPU total memory      16,106,127,360 B
worker PID GPU memory          5,496,635,392 B (5.119 GiB)
parent Torch imported          false
parent VoxCPM imported         false
```

## Real one-shot IPC

Canonical text: `Xin chào từ VoxCPM2.`

```text
wall time                      4.577619 s
serialized request             170 B
serialized AudioResult         691,676 B
sample rate                    48,000 Hz
channels                       1
frames                         76,800
duration                       1.60 s
WAV bytes                      153,644
WAV SHA-256                    70f6623a5b27da9d0eb5c718914a5aeb89958008d450ed09d5d8dc2da219cde2
```

## Real native stream IPC

```text
first chunk observed by parent 0.773944 s
stream completion              3.321499 s
chunk count                    9
chunk samples                  7,680 each
sequence contiguous            true
final chunk count              1
max serialized chunk           69,513 B
sample rate                    48,000 Hz
channels                       1
frames                         69,120
duration                       1.44 s
WAV bytes                      138,284
WAV SHA-256                    a88e7d0cd6544bc50984325f3650f9a5392ccba6324c5e55f079dc866e5afc8a
```

## Active cancellation

After the first 7,680-sample chunk arrived, the active stream was cancelled by worker-process termination plus join.

```text
first chunk sequence           0
cancel duration                0.359682 s
cancelled worker PID gone      true
```

M8 does not claim cooperative cancellation of an in-flight GPU kernel.

## Manual recovery and offline proof

```text
replacement startup            29.900268 s
replacement child RSS          4,091,260,928 B
recovery TTS time              6.692752 s
HF_HOME files after run        0
replacement PID gone           true after shutdown
```

Recovery WAV:

```text
sample rate                    48,000 Hz
channels                       1
frames                         122,880
duration                       2.56 s
bytes                          245,804
SHA-256                        0184f960a6461fca07a9eb1387812c4708d073b1a3c5e074d36adb120eb7a059
```

The replacement worker used offline flags, a fresh empty HF cache, and loopback blackhole model endpoints/proxies. No remote model acquisition was required.

## Resource and process safety

```text
cgroup low                     0
cgroup high                    0
cgroup max                     0
cgroup oom                     0
cgroup oom_kill                0
cgroup oom_group_kill          0
```

Every recorded worker PID was independently checked absent from `/proc` after cancellation/shutdown. No orphan worker remained.

## Public evidence hygiene

The public qualification JSON was scanned for absolute Kaggle paths, temporary qualification workspace paths, and GitHub token prefixes; the scan returned zero hits. Raw child stderr includes local model paths and remains private only. Generated WAV files are not committed.

## M8 gates

```text
M8_PRE_IMPLEMENTATION_AUDIT=PASS
M8_PROCESS_CONTRACT_GATE=PASS
M8_SPAWN_PROCESS_BOUNDARY_GATE=PASS
M8_CHILD_PLAN_REMAPPING_GATE=PASS
M8_PARENT_MODEL_ISOLATION_GATE=PASS
M8_WORKER_STARTUP_HANDSHAKE_GATE=PASS
M8_SINGLE_ACTIVE_REQUEST_PER_WORKER_GATE=PASS
M8_BOUNDED_ADMISSION_GATE=PASS
M8_FIFO_SCHEDULER_GATE=PASS
M8_DUPLICATE_REQUEST_ID_GATE=PASS
M8_PENDING_CANCEL_GATE=PASS
M8_ACTIVE_CANCEL_GATE=PASS
M8_WORKER_CRASH_CONTAINMENT_GATE=PASS
M8_WORKER_ERROR_SANITIZATION_GATE=PASS
M8_STREAM_IPC_SEQUENCE_GATE=PASS
M8_STREAM_IPC_BACKPRESSURE_GATE=PASS
M8_NO_ORPHAN_WORKER_GATE=PASS
M8_GRACEFUL_SHUTDOWN_GATE=PASS
M8_M0_M7_REGRESSION_GATE=PASS
M8_IMPLEMENTATION_EXACT_SHA_CI=PASS
M8_REAL_WORKER_STARTUP_GATE=PASS
M8_REAL_WORKER_TTS_GATE=PASS
M8_REAL_WORKER_STREAM_GATE=PASS
M8_REAL_ACTIVE_CANCEL_GATE=PASS
M8_REAL_MANUAL_RECOVERY_GATE=PASS
M8_REAL_WORKER_OFFLINE_GATE=PASS
M8_REAL_SINGLE_T4_DEVICE_PLACEMENT_GATE=PASS
M8_REAL_WORKER_MEMORY_MEASUREMENT=PASS
M8_HTTP_REMAINS_OUT_OF_SCOPE_GATE=PASS
M8_MULTI_GPU_RUNTIME_REMAINS_UNQUALIFIED_GATE=PASS
M8_INVESTIGATION_COMPLETENESS=PASS
M8_WORKER_PROCESS_RUNTIME=PASS
```

## Scope boundary

M8 qualifies spawned worker lifecycle, bounded FIFO scheduler semantics with fake workers, one real worker process owning one T4, real one-shot IPC, real native-stream IPC, process-boundary cancellation, explicit replacement-worker recovery, and offline/no-download worker startup/inference.

M8 does not qualify two real workers, T4x2 throughput, real multi-GPU scheduling, automatic respawn, FastAPI/REST, HTTP/SSE/WebSocket streaming, external queues, batching, autoscaling, production deployment, or any public release.

The evidence commit and its final exact-SHA CI run are the final M8 authority.
