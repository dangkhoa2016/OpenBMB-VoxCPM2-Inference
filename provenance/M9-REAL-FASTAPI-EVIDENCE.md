# M9 Real FastAPI Evidence

## Status

- Repository: OpenBMB-VoxCPM2-Inference
- Branch: main
- M9 base SHA: c99cd3022440f05d5990ffbb63c1bb540827a135
- FastAPI surface SHA: c2e0263d453d724c952721d8a257eddef1481f85
- FastAPI surface CI: 36233240601, PASS
- HTTP streaming SHA: 12383d0e7894bcb1384cd707f13ca4fb5ab6f9b7
- HTTP streaming CI: 36233550988, PASS
- VERSION: 1.0.0
- No tag or release was created

## Implementation authority

M9 adds a FastAPI parent process over the M8 Scheduler + WorkerClient boundary.

The qualified topology is:

```text
HTTP client
  -> FastAPI parent
  -> Scheduler
  -> WorkerClient
  -> spawn IPC
  -> one WorkerProcess
  -> one real backend
  -> one Tesla T4
```

The HTTP layer does not call the real backend directly.

## Dependency isolation

The core package still declares no mandatory runtime dependency. M9 adds an explicit API extra pinned to the versions exercised during qualification:

```text
fastapi 0.136.1
uvicorn 0.46.0
httpx 0.28.1
```

An isolated import-only virtual environment without FastAPI successfully imported:

```text
voxcpm_runtime
backend
backend_types
scheduler
worker_client
worker_process
```

FastAPI and Uvicorn were not imported by those core modules.

## Model-free CI

After the HTTP streaming implementation:

```text
443 passed
1 skipped
ruff          PASS
compileall    PASS
CI YAML parse PASS
```

The existing local skip is the unreadable-file case under root.

Exact-SHA CI:

```text
FastAPI one-shot surface
SHA c2e0263d453d724c952721d8a257eddef1481f85
run 36233240601
Python 3.10 PASS
Python 3.11 PASS
Python 3.12 PASS

HTTP native streaming
SHA 12383d0e7894bcb1384cd707f13ca4fb5ab6f9b7
run 36233550988
Python 3.10 PASS
Python 3.11 PASS
Python 3.12 PASS
```

## Real API topology and model isolation

The first real server was bound to loopback only:

```text
host                         127.0.0.1
port                         8090
API authentication           required
queue capacity               1 pending request
worker count                 1
selected physical GPU        one Tesla T4
```

Before inference:

```text
API parent RSS               55,608 KiB
API parent Torch/model maps  absent
worker GPU memory            5,496,635,392 B
GPU                          Tesla T4
```

Only the worker process appeared in the NVIDIA compute-process list. The API parent never appeared as a GPU compute process.

## Health and readiness

With the worker healthy:

```text
GET /healthz   200
GET /readyz    200
ready workers  1
busy workers   0
failed workers 0
```

After client-disconnect cancellation terminated the only worker:

```text
GET /healthz   200
GET /readyz    503
GPU processes  0
```

This distinguishes API-process liveness from inference readiness.

## Authentication

For `POST /v1/tts`:

```text
missing bearer token    401
incorrect bearer token  401
correct bearer token    200
```

The qualification token was stored only in a temporary local file, was not emitted in server logs or public artifacts, and was deleted after qualification.

## Real one-shot HTTP TTS

Canonical request text:

```text
Xin chào từ VoxCPM2.
```

Measured response:

```text
HTTP status              200
Content-Type             audio/wav
request ID               m9-real-tts
wall time                10.388755 s
response bytes           399,404
channels                 1
sample width             2 bytes
sample rate              48,000 Hz
frames                   199,680
audio duration           4.16 s
WAV SHA-256              a91af6d394d40960acb0e37f243dabe1fd562eee4436588032bb1d3063dea3d9
```

The client independently opened the returned WAV successfully.

## Real HTTP native streaming

The qualified streaming endpoint returned raw PCM16 little-endian over `application/octet-stream`.

Client-observed result:

```text
HTTP status              200
request ID               m9-real-stream
audio format             pcm_s16le
sample rate header       48000
channels header          1
client chunk count       12
bytes per chunk          15,360
time to first body       1.076264 s
completion               4.732486 s
total PCM bytes          184,320
frames                   92,160
audio duration           1.92 s
```

Client-side WAV reconstruction:

```text
SHA-256                  bdb7a647b87cd1d932f5addff5e0d7690bbf00b112686e4375eb524cc85f8d2f
```

The first HTTP body bytes arrived well before stream completion. The server therefore did not wait for the complete waveform before sending audio.

The API stream bridge is bounded and the underlying M8 worker-result queue is also bounded.

## Real bounded admission

The real server used one worker and one pending slot.

Three concurrent requests produced:

```text
request A   active, HTTP 200, 63.063 s
request B   pending then dispatched, HTTP 200, 66.506 s
request C   HTTP 429 queue_full, rejected in 0.055939 s
```

The third request was rejected immediately rather than silently queued or routed around the M8 Scheduler.

## Real client disconnect

A deliberately longer HTTP native stream was opened.

The client consumed the first HTTP body chunk and then closed the response before stream completion.

Measured:

```text
first body bytes         15,360
first body arrival       1.530725 s
client close             1.530866 s
readyz became 503        1.965011 s from request start
worker GPU process gone  true
healthz remained 200     true
```

The disconnect propagated through the API generator cleanup to Scheduler cancellation. M8 active cancellation terminated and joined the worker.

M9 does not auto-respawn the worker.

## Offline/no-download HTTP proof

A fresh server was started after the disconnect test with:

```text
VOXCPM_OFFLINE=1
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
fresh empty HF_HOME
model endpoint -> loopback blackhole
HTTP/HTTPS proxy -> loopback blackhole
```

Measured startup and request:

```text
server ready             31.375833 s
authenticated HTTP TTS   200
HTTP TTS time            8.341589 s
response bytes           307,244
HF_HOME files after run  0
worker GPU memory        5,897,191,424 B
```

Offline response WAV:

```text
channels                 1
sample width             2 bytes
sample rate              48,000 Hz
frames                   153,600
duration                 3.20 s
SHA-256                  d63bfc47661567410c8c459d94de081a2a65a7fb1e1a939e3fed7b6a136f0088
```

No remote model acquisition was required.

## Process and resource safety

After both server lifecycles and the disconnect cancellation:

```text
recorded API PIDs        absent
recorded worker PIDs     absent
GPU compute processes    0
cgroup low               0
cgroup high              0
cgroup max               0
cgroup oom               0
cgroup oom_kill          0
cgroup oom_group_kill    0
```

No API or worker process remained orphaned.

## Security and public-evidence hygiene

- bearer token comparison uses constant-time comparison;
- non-loopback unauthenticated bind is rejected unless explicitly allowed;
- no permissive CORS middleware is installed by default;
- authorization tokens are not accepted in query parameters or paths;
- qualification token was absent from both server logs;
- temporary qualification token was deleted;
- generated WAV and PCM files remain outside Git;
- absolute model paths and private server stderr remain outside public evidence.

## M9 gates

```text
M9_PRE_IMPLEMENTATION_AUDIT=PASS
M9_API_DEPENDENCY_ISOLATION_GATE=PASS
M9_CORE_IMPORT_WITHOUT_API_EXTRA_GATE=PASS
M9_FASTAPI_LIFESPAN_GATE=PASS
M9_SINGLE_API_PROCESS_GATE=PASS
M9_EXTERNAL_BIND_AUTH_POLICY_GATE=PASS
M9_BEARER_AUTH_GATE=PASS
M9_TOKEN_REDACTION_GATE=PASS
M9_HEALTH_LIVENESS_GATE=PASS
M9_READINESS_GATE=PASS
M9_REQUEST_VALIDATION_GATE=PASS
M9_REQUEST_ID_GATE=PASS
M9_SCHEDULER_ONLY_INFERENCE_GATE=PASS
M9_BOUNDED_ADMISSION_HTTP_GATE=PASS
M9_QUEUE_FULL_HTTP_GATE=PASS
M9_SAFE_ERROR_MAPPING_GATE=PASS
M9_EVENT_LOOP_NONBLOCKING_GATE=PASS
M9_HTTP_ONESHOT_WAV_GATE=PASS
M9_HTTP_NATIVE_STREAM_NOT_BUFFERED_GATE=PASS
M9_HTTP_STREAM_SEQUENCE_GATE=PASS
M9_HTTP_STREAM_BACKPRESSURE_GATE=PASS
M9_CLIENT_DISCONNECT_CANCEL_GATE=PASS
M9_NO_PERMISSIVE_CORS_DEFAULT_GATE=PASS
M9_M0_M8_REGRESSION_GATE=PASS
M9_IMPLEMENTATION_EXACT_SHA_CI=PASS
M9_REAL_API_STARTUP_GATE=PASS
M9_REAL_AUTH_GATE=PASS
M9_REAL_HEALTH_READINESS_GATE=PASS
M9_REAL_HTTP_TTS_GATE=PASS
M9_REAL_HTTP_STREAM_GATE=PASS
M9_REAL_QUEUE_FULL_GATE=PASS
M9_REAL_DISCONNECT_CANCEL_GATE=PASS
M9_REAL_API_PARENT_MODEL_ISOLATION_GATE=PASS
M9_REAL_WORKER_MEMORY_MEASUREMENT=PASS
M9_REAL_OFFLINE_HTTP_GATE=PASS
M9_NO_ORPHAN_PROCESS_GATE=PASS
M9_T4X2_RUNTIME_REMAINS_UNQUALIFIED_GATE=PASS
M9_WEBSOCKET_SSE_REMAIN_OUT_OF_SCOPE_GATE=PASS
M9_INVESTIGATION_COMPLETENESS=PASS
M9_FASTAPI_RUNTIME=PASS
```

## Scope boundary

M9 qualifies only the measured single-API-process, single-worker, single-T4 HTTP surface.

It does not qualify:

- T4x2 or two real workers;
- real multi-GPU scheduling or throughput;
- automatic worker respawn;
- SSE or WebSocket;
- multipart/reference-audio upload endpoints;
- browser playback behavior;
- permissive CORS;
- external message brokers;
- batching or autoscaling;
- production deployment;
- a public release.

The M9 evidence commit and its final exact-SHA CI run are the final M9 authority.
