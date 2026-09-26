# Worker process runtime

## Status

M8 qualifies a project-owned spawned worker process boundary and bounded FIFO scheduling semantics.

Normal CI qualifies process lifecycle, admission, cancellation, worker-crash containment, sanitized error transport, bounded stream IPC, and no-orphan shutdown with fake backends.

Real runtime qualification remains deliberately narrower: exactly one spawned worker owns exactly one Tesla T4 and one real VoxCPM2 backend.

## Authority

```text
M8 base SHA             665af656234f355eace5c8a254c4f74ef394f1fd
worker boundary SHA     27e139b98ade2c754302900337b6a12762e12f61
worker boundary CI      36230495838 PASS
scheduler SHA           1252f23d4b14a4c57a8d059c11b2f12351bdee4e
scheduler CI            36230777274 PASS
VERSION                 1.0.0
```

## Architecture

```text
caller
  -> Scheduler / bounded FIFO admission
  -> WorkerClient
  -> spawn IPC
  -> WorkerProcess
  -> InferenceBackend
  -> PytorchVoxCPMBackend
```

The parent owns topology and never loads the model. The child owns one backend instance. Real CUDA workers receive one physical GPU selector before Torch/upstream import; after visibility isolation, the child runtime uses CUDA index 0 and worker_count=1.

The real backend single-worker guardrail is not weakened.

## Scheduler semantics

M8 uses:

```text
FIFO pending queue
bounded max_pending_requests
one active request per worker
duplicate request IDs rejected
queued cancellation removes work before dispatch
active cancellation terminates the worker process
unexpected worker death becomes worker_exited
no healthy worker becomes no_healthy_worker
shutdown explicitly joins/terminates children
```

These multi-request semantics are qualified with spawned fake workers in normal CI. M8 does not claim real multi-worker throughput.

## Real single-T4 worker

Measured worker startup:

```text
spawn -> READY             29.785 s
worker child RSS           4,072,284,160 B (3.793 GiB)
worker GPU memory          5,496,635,392 B (5.119 GiB)
physical GPU               Tesla T4
physical GPU memory        16,106,127,360 B
parent RSS before worker   23,957,504 B
parent RSS after READY     24,510,464 B
parent torch imported      false
parent voxcpm imported     false
```

The GPU memory value was measured by NVML for the worker PID while the worker was alive.

## Real one-shot IPC

Canonical request:

```text
Xin chào từ VoxCPM2.
```

Result:

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

The WAV was written and independently read in the parent process.

## Real native stream IPC

Canonical native stream crossed the process boundary incrementally:

```text
first parent chunk         0.774 s
stream completion          3.321 s
chunk count                9
samples per chunk          7,680
sequence contiguous        true
final chunk count          1
max serialized chunk       69,513 B
sample rate                48,000 Hz
channels                   1
frames                     69,120
audio duration             1.44 s
WAV SHA-256                a88e7d0cd6544bc50984325f3650f9a5392ccba6324c5e55f079dc866e5afc8a
```

The worker result queue is bounded, so a slow parent applies backpressure instead of allowing unlimited audio accumulation.

## Active cancellation

A longer native stream was started. After the parent received the first 7,680-sample chunk, it cancelled the active request.

```text
cancellation mechanism     process termination + join
cancel duration            0.360 s
worker PID gone            true
later chunks               none
```

M8 does not claim cooperative GPU-kernel cancellation.

## Manual recovery and offline proof

After cancellation, a replacement worker was explicitly started.

```text
replacement startup        29.900 s
replacement child RSS      4,091,260,928 B
offline recovery TTS       6.693 s
HF_HOME files after run    0
replacement PID gone       true after shutdown
```

Recovery WAV:

```text
sample rate                48,000 Hz
channels                   1
frames                     122,880
duration                   2.56 s
SHA-256                    0184f960a6461fca07a9eb1387812c4708d073b1a3c5e074d36adb120eb7a059
```

The replacement worker used offline flags, a fresh HF cache, and blackholed model endpoints/proxies. This demonstrates no remote model acquisition in the measured recovery run.

## Resource and process evidence

After the real sequence:

```text
cgroup low                 0
cgroup high                0
cgroup max                 0
cgroup oom                 0
cgroup oom_kill            0
cgroup oom_group_kill      0
all recorded worker PIDs   absent
```

No worker process was left orphaned.

## Scope boundary

M8 does not qualify:

- two real worker processes;
- T4x2 throughput;
- multi-GPU scheduling;
- automatic worker respawn;
- HTTP, REST, SSE, WebSocket or browser streaming;
- external queues;
- shared-memory audio transport;
- batching or dynamic batching;
- autoscaling;
- production deployment;
- release/tag work.

See `provenance/M8-REAL-WORKER-PROCESS-EVIDENCE.md` for the complete gate record.
