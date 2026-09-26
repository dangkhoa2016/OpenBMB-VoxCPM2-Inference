# Runtime FastAPI / REST

## Trạng thái

M9 qualification network API surface đầu tiên trên M8 Scheduler + WorkerClient boundary.

Topology đã qualification:

```text
HTTP client
  -> FastAPI parent
  -> Scheduler
  -> WorkerClient
  -> spawn IPC
  -> một WorkerProcess
  -> một PytorchVoxCPMBackend
  -> một Tesla T4
```

API parent vẫn model-free. M9 không qualification T4x2, hai real worker, SSE, WebSocket, autoscaling hay production deployment.

## Authority

```text
M9 base SHA             c99cd3022440f05d5990ffbb63c1bb540827a135
FastAPI surface SHA     c2e0263d453d724c952721d8a257eddef1481f85
FastAPI surface CI      36233240601 PASS
HTTP streaming SHA      12383d0e7894bcb1384cd707f13ca4fb5ab6f9b7
HTTP streaming CI       36233550988 PASS
VERSION                 1.0.0
```

## Cài đặt

Core package vẫn không có required runtime dependency. API extra phải được cài tường minh:

```bash
python -m pip install -e '.[api]'
```

M9 đã chạy với:

```text
fastapi 0.136.1
uvicorn 0.46.0
httpx 0.28.1
```

Một môi trường import-only không có FastAPI vẫn import thành công backend, scheduler và worker core.

## Cấu hình API

M9 dùng lại M1 configuration contract.

Real single-T4 path cần tối thiểu:

```text
VOXCPM_DEVICE=cuda
VOXCPM_GPU_DEVICES=0
VOXCPM_WORKERS=1
VOXCPM_MAX_QUEUE_SIZE=1
VOXCPM_API_TOKEN=<secret>
VOXCPM_REQUIRE_AUTH=1
```

M9 yêu cầu đúng một GPU được chọn tường minh cho real API runtime.

## Bind và authentication

Default bind:

```text
127.0.0.1:8090
```

Nếu auth được yêu cầu thì token phải tồn tại.

Non-loopback bind + auth disabled sẽ bị từ chối khi startup, trừ khi `VOXCPM_ALLOW_UNAUTHENTICATED_EXTERNAL=1` được bật tường minh.

Authenticated endpoint dùng Bearer token và so sánh constant-time. Token không được nhận qua query hay URL path.

## Endpoint

```text
GET  /healthz
GET  /readyz
POST /v1/tts
POST /v1/tts/stream
```

`/healthz` chỉ phản ánh API process còn sống.

`/readyz` phản ánh Scheduler/worker availability:

```text
200 -> worker usable
503 -> không còn usable worker
```

## One-shot TTS

`POST /v1/tts` nhận JSON chứa `text` và trả:

```text
Content-Type: audio/wav
X-Request-ID: <safe-id>
Cache-Control: no-store
```

API parent nhận `AudioResult` qua M8 IPC rồi encode WAV trong memory; nó không gọi real backend trực tiếp.

## HTTP streaming

`POST /v1/tts/stream` trả raw PCM16 little-endian:

```text
Content-Type: application/octet-stream
X-Audio-Format: pcm_s16le
X-Audio-Sample-Rate: 48000
X-Audio-Channels: 1
```

Đây không phải WAV container, SSE, WebSocket, JSON hay base64.

API bridge và M8 worker IPC đều bounded, vì vậy M9 không chèn unbounded buffering layer.

## Admission và cancellation

M8 Scheduler vẫn là admission authority duy nhất.

Với một worker và một pending slot:

```text
A -> active
B -> pending
C -> 429 queue_full
```

Client disconnect từ active HTTP stream sẽ propagate thành M8 active cancellation, tức terminate + join worker.

Sau khi sole worker bị terminate:

```text
/healthz -> 200
/readyz  -> 503
```

M9 không auto-respawn.

## Real measurements

### API parent / worker ownership

```text
API parent RSS                 55,608 KiB
API parent Torch/model maps    absent
worker GPU memory              5,496,635,392 B
GPU                            Tesla T4
```

Chỉ worker xuất hiện trong GPU compute-process list.

### One-shot HTTP thật

```text
HTTP                           200
wall time                      10.389 s
WAV bytes                      399,404
48 kHz mono
frames                         199,680
duration                       4.16 s
SHA-256                        a91af6d394d40960acb0e37f243dabe1fd562eee4436588032bb1d3063dea3d9
```

Thiếu token và token sai đều trả 401.

### Native HTTP stream thật

```text
HTTP                           200
client chunks                  12
bytes/chunk                    15,360
first body                     1.076 s
completion                     4.732 s
PCM bytes                      184,320
frames                         92,160
duration                       1.92 s
reconstructed WAV SHA-256      bdb7a647b87cd1d932f5addff5e0d7690bbf00b112686e4375eb524cc85f8d2f
```

First bytes đến trước completion, chứng minh HTTP path không chờ full waveform.

### Queue-full thật

```text
A                              active, 200
B                              pending rồi 200
C                              429 queue_full
C rejection                    0.056 s
```

### Client disconnect thật

```text
first body bytes               15,360
first body                     1.531 s
client close                   1.531 s
readyz -> 503                  1.965 s từ request start
worker GPU process gone        true
healthz vẫn 200                true
```

### Offline restart

```text
server ready                   31.376 s
HTTP TTS                       200
HTTP TTS time                  8.342 s
HF_HOME files                  0
worker GPU memory              5,897,191,424 B
frames                         153,600
duration                       3.20 s
WAV SHA-256                    d63bfc47661567410c8c459d94de081a2a65a7fb1e1a939e3fed7b6a136f0088
```

## Các trường cấu hình chưa được M9 enforce mới

M9 enforce các trường cần cho surface đã qualification, gồm host/port, auth, single-GPU selection tường minh, giới hạn độ dài text, bounded queue capacity và stream IPC chunk capacity.

Các trường đã được parse từ trước như universal request deadline, max inference time, output directory và autoscaling vẫn nằm ngoài M9 runtime contract nếu không được tài liệu này nêu riêng.

## Ranh giới phạm vi

M9 không qualification T4x2, hai real worker, real multi-GPU scheduling, auto-respawn, SSE, WebSocket, multipart/reference-audio upload, browser playback, permissive CORS, external broker, batching, autoscaling, production deployment hay release/tag.

Xem `provenance/M9-REAL-FASTAPI-EVIDENCE.md`.
