# FastAPI / REST runtime

## Status

M9 qualifies the first network API surface over the M8 Scheduler + WorkerClient boundary.

Qualified topology:

```text
HTTP client
  -> FastAPI parent
  -> Scheduler
  -> WorkerClient
  -> spawn IPC
  -> one WorkerProcess
  -> one PytorchVoxCPMBackend
  -> one Tesla T4
```

The API parent remains model-free. M9 does not qualify T4x2, two real workers, SSE, WebSocket, autoscaling, or production deployment.

## Authority

```text
M9 base SHA             c99cd3022440f05d5990ffbb63c1bb540827a135
FastAPI surface SHA     c2e0263d453d724c952721d8a257eddef1481f85
FastAPI surface CI      36233240601 PASS
HTTP streaming SHA      12383d0e7894bcb1384cd707f13ca4fb5ab6f9b7
HTTP streaming CI       36233550988 PASS
VERSION                 1.0.0
```

## Installation

The core package still has no required runtime dependencies. Install the API extra explicitly:

```bash
python -m pip install -e '.[api]'
```

The API extra exercised by M9 is pinned to:

```text
fastapi 0.136.1
uvicorn 0.46.0
httpx 0.28.1
```

An isolated import-only environment without FastAPI successfully imported the core backend, scheduler, and worker modules.

## Required API configuration

M9 uses the existing M1 configuration contract.

For the qualified real single-T4 path, configure at least:

```text
VOXCPM_DEVICE=cuda
VOXCPM_GPU_DEVICES=0
VOXCPM_WORKERS=1
VOXCPM_MAX_QUEUE_SIZE=1
VOXCPM_API_TOKEN=<secret>
VOXCPM_REQUIRE_AUTH=1
```

The real M9 server also ran offline with a local model path.

M9 requires exactly one explicitly selected GPU for the real API runtime.

## Bind and authentication policy

Default bind remains:

```text
127.0.0.1:8090
```

If authentication is required, an API token must be configured.

If the server is bound to a non-loopback address while authentication is disabled, startup is rejected unless `VOXCPM_ALLOW_UNAUTHENTICATED_EXTERNAL=1` was set deliberately.

Authenticated endpoints use:

```http
Authorization: Bearer <token>
```

Token comparison uses constant-time comparison. Tokens are never accepted from query parameters or URL paths.

## Endpoints

Qualified M9 endpoints:

```text
GET  /healthz
GET  /readyz
POST /v1/tts
POST /v1/tts/stream
```

`/healthz` is liveness only.

`/readyz` reflects Scheduler/worker availability:

```text
200 -> worker ready/busy and service can continue
503 -> no usable worker
```

Health and readiness do not expose model paths, tokens, PIDs, or GPU details.

## One-shot TTS

Request:

```json
{
  "text": "Xin chào từ VoxCPM2."
}
```

Response:

```text
Content-Type: audio/wav
X-Request-ID: <safe-id>
Cache-Control: no-store
```

The API parent receives an `AudioResult` through M8 IPC and writes the WAV response in memory. It does not call the real backend directly.

Example:

```bash
curl -sS \
  -H "Authorization: Bearer $VOXCPM_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text":"Xin chào từ VoxCPM2."}' \
  http://127.0.0.1:8090/v1/tts \
  --output out.wav
```

## HTTP streaming

`POST /v1/tts/stream` exposes M7/M8 native chunks incrementally.

Response:

```text
Content-Type: application/octet-stream
X-Audio-Format: pcm_s16le
X-Audio-Sample-Rate: 48000
X-Audio-Channels: 1
X-Request-ID: <safe-id>
Cache-Control: no-store
```

The body is raw PCM16 little-endian, not a WAV container, SSE, WebSocket, JSON, or base64.

The API bridge uses a bounded queue. M8 worker IPC is also bounded, so M9 does not add an unbounded buffering layer between model streaming and the HTTP response.

## Admission and cancellation

The M8 Scheduler remains the only admission authority.

With one real worker and `VOXCPM_MAX_QUEUE_SIZE=1`:

```text
request A -> active
request B -> pending
request C -> 429 queue_full
```

Client disconnect from an active stream propagates cancellation through the API bridge to M8. M8 active cancellation terminates and joins the worker process.

After the only worker has been terminated:

```text
/healthz -> 200
/readyz  -> 503
```

M9 does not auto-respawn the worker.

## Real measurements

### API parent and worker ownership

Before inference:

```text
API parent RSS                 55,608 KiB
API parent Torch/model maps    absent
worker GPU memory              5,496,635,392 B
physical GPU                   Tesla T4
```

Only the worker PID appeared in the GPU compute-process list.

### Real one-shot HTTP

```text
HTTP status                    200
wall time                      10.389 s
WAV bytes                      399,404
sample rate                    48,000 Hz
channels                       1
frames                         199,680
audio duration                 4.16 s
WAV SHA-256                    a91af6d394d40960acb0e37f243dabe1fd562eee4436588032bb1d3063dea3d9
```

Missing and incorrect bearer tokens both returned 401.

### Real native HTTP stream

```text
HTTP status                    200
client-observed chunks         12
bytes per observed chunk       15,360
time to first body bytes       1.076 s
completion                     4.732 s
total PCM bytes                184,320
frames                         92,160
audio duration                 1.92 s
reconstructed WAV SHA-256      bdb7a647b87cd1d932f5addff5e0d7690bbf00b112686e4375eb524cc85f8d2f
```

First bytes arrived before completion, proving the real HTTP path did not wait for the full waveform.

### Real queue-full

With one worker and one pending slot:

```text
request A                      200, active
request B                      200, pending then dispatched
request C                      429 queue_full
request C rejection time       0.056 s
```

### Real client disconnect

A long HTTP stream was closed immediately after the first body chunk:

```text
first body bytes               15,360
first body arrival             1.531 s
client closed                  1.531 s
readyz became 503              1.965 s from request start
worker GPU process gone        true
healthz remained 200           true
```

This is the measured M9 disconnect-cancellation contract.

### Offline restart

A fresh server + worker was started with a fresh empty HF cache, offline flags, and loopback blackhole model endpoints/proxies.

```text
server ready                   31.376 s
authenticated HTTP TTS         200
HTTP TTS wall time             8.342 s
HF_HOME files after run        0
worker GPU memory              5,897,191,424 B
WAV frames                     153,600
WAV duration                   3.20 s
WAV SHA-256                    d63bfc47661567410c8c459d94de081a2a65a7fb1e1a939e3fed7b6a136f0088
```

## M10 two-worker extension

M10 extends the same API surface to two independent real workers on two physical Tesla T4 GPUs. The API parent still does not own a model or CUDA compute context. The qualified configuration is `VOXCPM_DEVICE=cuda`, `VOXCPM_GPU_DEVICES=0,1`, `VOXCPM_WORKERS=2`, and `VOXCPM_MAX_QUEUE_SIZE=1`.

With two workers, readiness is degraded-capacity aware: two healthy workers -> 200, one healthy worker -> 200, and zero healthy workers -> 503 while `/healthz` remains 200. Admission with one pending slot is A/B active, C pending, D -> 429 `queue_full`. M10 also qualifies two concurrent native PCM streams and one-worker disconnect/failure isolation. No automatic worker respawn is added. See `docs/T4X2-TWO-WORKER-RUNTIME.md` and `provenance/M10-REAL-T4X2-TWO-WORKER-EVIDENCE.md`.

## Configuration fields not newly enforced by M9

M9 enforces the fields needed for the qualified surface, including host/port, auth, explicit GPU selection, text length, queue capacity, and stream IPC chunk capacity.

Other previously parsed fields such as universal request deadlines, max inference time, output directories, and autoscaling behavior remain outside the M9 runtime contract unless explicitly documented otherwise.

## Scope boundary

M9 does not qualify:

- T4x2 or two real worker processes;
- real multi-GPU scheduling;
- automatic worker respawn;
- SSE or WebSocket;
- multipart/reference-audio HTTP upload;
- browser-specific playback;
- permissive CORS;
- external message brokers;
- batching or autoscaling;
- production deployment;
- public release/tag work.

See `provenance/M9-REAL-FASTAPI-EVIDENCE.md` for the complete gate record.
