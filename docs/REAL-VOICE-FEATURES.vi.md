# Qualification tính năng giọng one-shot thật

## Trạng thái

M6 qualification ba operation one-shot còn lại đã được M3 backend contract đóng băng: thiết kế giọng, nhân bản giọng, và nối tiếp âm thanh.

Cả ba đều chạy qua entrypoint non-streaming của upstream trên `PytorchVoxCPMBackend` sẵn có, trên đúng một GPU CUDA được chọn tường minh. Không thêm abstraction backend public mới.

~~~text
VOXCPM_DEVICE=cuda
VOXCPM_GPU_DEVICES=0
VOXCPM_WORKERS=1
CUDA_VISIBLE_DEVICES=0
effective_device=cuda
selected_gpu_indices=[0]
worker_count=1
upstream_device=cuda:0
~~~

Đây là qualification one-shot. Không phải qualification streaming, multi-GPU, worker, scheduling, hay API.

## Authority

~~~text
M6 starting SHA         2ea0e8d52e21b6dcc97764cc86eb69a04f3e8294
M6 implementation SHA   805ed80a057d60bfab2af2691d03f4d8f62ba2d7
implementation CI       36223105464 (PASS trên đúng SHA implementation)
VERSION                 1.0.0
upstream commit         f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69
installed voxcpm        2.0.3.post33+gf772e498a
source-lock SHA-256     10097c137461308dfb2693b1bf53f2fdff3deb8508ff1add2f1d3047ecae367a
~~~

Model là openbmb/VoxCPM2 tại revision 32279effe8c19989596f05d353d1447f51d9e915, config SHA-256 405f0dcd92f7feba6011ed4eac5c8d4f74cba9712f07fd5cfa3063bbdd95402c. Đường dẫn model tuyệt đối và đường dẫn reference audio tuyệt đối được chủ động loại khỏi bằng chứng public. M2 bind run vào mirror local đã attach với zero remote acquisition.

## Ánh xạ upstream

Entrypoint `VoxCPM.generate` của upstream đã pin nhận `text`, `reference_wav_path`, `prompt_wav_path`, và `prompt_text`. M6 ánh xạ mỗi request của dự án sang đúng một trong các mode đó.

| Request của dự án | Lệnh upstream |
| --- | --- |
| `SpeechRequest(text=T)` | `generate(text=T)` |
| `VoiceDesignRequest(text=T, instruction=I)` | `generate(text=f"({I.strip()}){T}")` |
| `CloneRequest(text=T, reference_audio=P)` | `generate(text=T, reference_wav_path=P)` |
| `ContinuationRequest(text=T, reference_audio=P, reference_transcript=R)` | `generate(text=T, prompt_wav_path=P, prompt_text=R)` |

Dạng instruction trong ngoặc khớp với quy tắc `build_final_text` của CLI upstream đã pin. Quy tắc định dạng đó được giữ riêng tư ở ranh giới backend; `VoiceDesignRequest` public vẫn mang instruction thô, và report public chỉ ghi số ký tự của instruction.

Mode kết hợp `reference_wav_path + prompt_wav_path` cố ý không qualification trong M6. Clone không truyền prompt pair, và continuation không truyền reference path.

## Capabilities

~~~text
BackendInfo capabilities
  clone
  continue_audio
  design
  synthesize

stream                  vắng mặt
generate_streaming      vắng mặt
~~~

Hằng số capability chỉ dành cho M4 đã được thay bằng `REAL_BACKEND_CAPABILITIES` trung tính milestone. Streaming vẫn ném `stream_not_qualified` với `milestone: M7`.

## Chính sách reference audio

M6 là milestone đầu tiên mà `AudioReference.local_path` chạm tới code upstream thật. Backend validate descriptor tại thời điểm thực thi và không thay đổi dataclass contract của M3.

~~~text
path không tồn tại    reference_audio_not_found
path là directory     reference_audio_not_file
path không đọc được   reference_audio_unreadable
descriptor sai kiểu   invalid_backend_request
~~~

Mọi message public đều trung tính về path. Không absolute path, nội dung instruction, hay transcript tham chiếu nào tới được error envelope đã normalize, tuple metadata của kết quả, hay report runtime public. Reference audio chỉ được đọc từ filesystem local và không bao giờ được tải từ xa.

## Fixture tham chiếu

Fixture tham chiếu canonical M6 được sinh local bằng chính đường TTS chuẩn đã qualification ở M5, sau đó đóng băng và dùng lại cho cả clone lẫn continuation. Không tải mẫu giọng của bên thứ ba, nên không phát sinh mơ hồ về licensing hay privacy.

~~~text
reference text      Xin chào, đây là giọng nói tham chiếu.
exit code           0
load seconds        27.751947
synthesis seconds   6.359500
channels            1
sample width        2 bytes
sample rate         48,000 Hz
frames              115,200
duration            2.40 s
byte size           230,444
SHA-256             419fc4ca5bba2d2d74eb0918e27108b85ad88f61c9a6e5b9c99075c2c5cc85fa
peak abs PCM16      29,445
RMS PCM16           5,144.13
nonzero samples     114,808 / 115,200
WAV validation      valid
~~~

Reference tự sinh chứng minh đường reference-audio thật chạy được. Nó không chứng minh độ trung thành khi clone so với một người nói thật.

## Các run thật

| Run | Exit | Load | Generate | Duration | Peak allocated VRAM | Peak reserved VRAM | Host peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reference fixture | 0 | 27.752 s | 6.360 s | 2.40 s | 5.194 GiB | 5.363 GiB | 11.033 GiB |
| Voice design | 0 | 27.457 s | 5.290 s | 1.92 s | 5.180 GiB | 5.363 GiB | 11.025 GiB |
| Voice clone | 0 | 27.423 s | 10.055 s | 3.20 s | 5.397 GiB | 5.463 GiB | 11.009 GiB |
| Audio continuation | 0 | 27.507 s | 7.002 s | 1.76 s | 5.370 GiB | 5.463 GiB | 11.041 GiB |
| Offline clone proof | 0 | 27.529 s | 8.982 s | 2.72 s | 5.383 GiB | 5.463 GiB | 11.054 GiB |

### Thiết kế giọng

~~~text
target text          Xin chào từ VoxCPM2.
instruction          giọng nữ ấm áp, bình tĩnh
operation            voice-design
conditioning         instruction
upstream effective   (giọng nữ ấm áp, bình tĩnh)Xin chào từ VoxCPM2.
channels             1
sample rate          48,000 Hz
frames               92,160
duration             1.92 s
byte size            184,364
SHA-256              9b4d9d321e1f783e23379c8b701d1dfe75363ccdae77c2f1e791b6feef71d9f4
peak abs PCM16       30,749
RMS PCM16            3,214.95
nonzero samples      89,336 / 92,160
~~~

Gate chứng minh đường design có điều kiện thật chạy thành công và trả về audio hợp lệ về cấu trúc. Nó không khẳng định giọng sinh ra thực sự nghe nữ, ấm, hay bình tĩnh. Điều đó cần đánh giá của con người hoặc đánh giá ngữ nghĩa.

### Nhân bản giọng

~~~text
target text          Xin chào, đây là phép thử clone giọng.
reference            canonical M6 reference fixture, consumed locally
operation            voice-clone
conditioning         reference
upstream mode        reference_wav_path=<local fixture>
                     prompt_wav_path=None
                     prompt_text=None
channels             1
sample rate          48,000 Hz
frames               153,600
duration             3.20 s
byte size            307,244
SHA-256              43b841d1bb5c6ca78c84f5e5bd562383a51b030b2054ee7f04a1c5aac4c0b629
peak abs PCM16       25,866
RMS PCM16            4,600.49
nonzero samples      138,814 / 153,600
CPU fallback         false
~~~

Gate chứng minh đường có reference thật chạy trên GPU được chọn và trả về `AudioResult` hợp lệ cùng WAV hợp lệ. Nó không khẳng định độ giống giọng nói.

### Nối tiếp âm thanh

~~~text
target text          Và đây là phần tiếp theo.
prompt audio         canonical M6 reference fixture
prompt transcript    Xin chào, đây là giọng nói tham chiếu.
operation            audio-continuation
conditioning         continuation
upstream mode        prompt_wav_path=<local fixture>
                     prompt_text=<reference transcript>
                     reference_wav_path=None
channels             1
sample rate          48,000 Hz
frames               84,480
duration             1.76 s
byte size            169,004
SHA-256              d79cffea396d7879e75862028eaf412bceb8de71c6e23097a8d9257318483854
peak abs PCM16       27,933
RMS PCM16            4,500.63
nonzero samples      76,285 / 84,480
~~~

Gate chứng minh đường continuation thật chạy thành công và trả về audio hợp lệ về cấu trúc. Nó không khẳng định chất lượng nối tiếp về ngữ nghĩa.

## Vị trí thiết bị và bộ nhớ

Mọi process canonical dùng cùng cách isolate một T4 như M5.

~~~text
CUDA available         true
CUDA device count      1
device name            Tesla T4
upstream device        cuda:0
upstream optimize      false
denoiser               false
parameter count        888
parameter device types ["cuda"]
buffer count           10
buffer device types    ["cuda"]
all on expected device true
CPU fallback           false
~~~

Với các run reference fixture, design, clone, continuation, và offline clone, PyTorch báo maximum allocated trong khoảng 5.18 GiB đến 5.40 GiB và maximum reserved trong khoảng 5.36 GiB đến 5.46 GiB trên thiết bị 15,636,037,632 B. Host peak RSS giữ trong khoảng 11.01 GiB đến 11.05 GiB.

## Kiểm tra WAV

Mọi output canonical được kiểm tra hai lần: một lần bằng validator của dự án, một lần bằng Python standard-library `wave` độc lập. Cả hai cùng khớp về channels, sample width, sample rate, frame count, duration và byte size cho cả bốn file. Tín hiệu không rỗng và không suy biến trong mọi trường hợp.

~~~text
reference      1 ch / 2 bytes / 48,000 Hz / 115,200 frames / 2.40 s / 230,444 B
design         1 ch / 2 bytes / 48,000 Hz / 92,160 frames / 1.92 s / 184,364 B
clone          1 ch / 2 bytes / 48,000 Hz / 153,600 frames / 3.20 s / 307,244 B
continuation   1 ch / 2 bytes / 48,000 Hz / 84,480 frames / 1.76 s / 169,004 B
~~~

Image không có binary `nvidia-smi`; vì vậy bằng chứng GPU đến từ allocator PyTorch, bao phủ trực tiếp process inference. Sampling upstream là stochastic, nên không khẳng định tái lập byte-for-byte.

## Chứng minh offline / không tải về

Operation clone được lặp lại dưới `HF_HOME` mới rỗng, cờ offline, và endpoint cùng proxy loopback chặn.

~~~text
VOXCPM_OFFLINE=1
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_HOME=<fresh empty directory>
HF_ENDPOINT=http://127.0.0.1:1
HTTP_PROXY=http://127.0.0.1:1
HTTPS_PROXY=http://127.0.0.1:1
~~~

~~~text
exit code                      0
local_files_only               true
HF_HOME files before run       0
HF_HOME files after run        0
valid WAV                      true
channels                       1
sample width                   2 bytes
sample rate                    48,000 Hz
frames                         130,560
duration                       2.72 s
byte size                      261,164
SHA-256                        7e428f351c03d9ee5869ac0bfeeda88809f56b8c66029bc9362c7a4ec5d6c0fb
peak abs PCM16                 26,235
RMS PCM16                      4,620.45
nonzero samples                123,934 / 130,560
~~~

Vì reference audio là local và model là local, operation có reference chạy thành công mà không cần bất kỳ acquisition nào.

## Bộ nhớ và bằng chứng OOM

Mọi snapshot cgroup `memory.events` đã ghi đều bằng zero cho `low`, `high`, `max` và `oom` qua các run reference fixture, design, clone, continuation và offline clone. Bộ đếm host `oom`, `oom_kill`, và `oom_group_kill` vẫn bằng zero trước và sau toàn bộ phiên M6. Mọi process đều exit zero. Không quan sát thấy GPU OOM, host OOM, signal termination, hay allocation failure.

## Bề mặt CLI

M6 mở rộng `voxcpm-generate` bằng các mode operation loại trừ lẫn nhau thay vì thêm các script rời rạc.

~~~bash
voxcpm-generate --text "..." --output out.wav
voxcpm-generate --text "..." --voice-instruction "..." --output out.wav
voxcpm-generate --text "..." --reference-audio /path/ref.wav --output out.wav
voxcpm-generate --text "..." --prompt-audio /path/prefix.wav --prompt-text "..." --output out.wav
~~~

~~~text
--voice-instruction với reference hoặc prompt audio   bị từ chối
--reference-audio với prompt audio hoặc prompt text   bị từ chối
--prompt-audio mà không có --prompt-text              bị từ chối
--prompt-text mà không có --prompt-audio              bị từ chối
cờ streaming                                        vắng mặt
~~~

## Report runtime

Schema report giữ tương thích ngược. M6 thêm trường `operation` và phần tóm tắt `request` an toàn.

~~~text
operation   standard-tts | voice-design | voice-clone | audio-continuation | load-only
request     input_text_characters
            instruction_characters
            reference_audio_used
            reference_input_kind
            reference_transcript_characters
~~~

Phần tóm tắt chỉ ghi số đếm và giá trị boolean. Toàn bộ text request, nội dung instruction thô, nội dung transcript thô, và absolute reference path không bao giờ được đưa vào.

## Kiểm chứng implementation

~~~text
baseline before M6        342 passed
after corrective M6       397 passed, 1 skipped
additional passing tests  55
additional local skips    1
ruff                      PASS
compileall                PASS
git diff --check          PASS
CI YAML parse             PASS
implementation CI         36223105464 PASS
~~~

CI thường vẫn GPU-free, model-free, và network-free. Các gate CI của M6 dùng stubbed upstream import và không bao giờ tải trọng số model.

## M6 gates

Danh sách dưới đây gồm 22 closeout gate bắt buộc của M6 cộng thêm một hygiene gate về không acquisition audio từ xa.

~~~text
M6_PRE_IMPLEMENTATION_AUDIT=PASS
M6_VOICE_DESIGN_MAPPING_GATE=PASS
M6_VOICE_CLONE_MAPPING_GATE=PASS
M6_AUDIO_CONTINUATION_MAPPING_GATE=PASS
M6_REFERENCE_PATH_REDACTION_GATE=PASS
M6_REAL_BACKEND_CAPABILITY_GATE=PASS
M6_STREAMING_REMAINS_UNQUALIFIED_GATE=PASS
M6_NO_REMOTE_AUDIO_ACQUISITION_GATE=PASS
M6_IMPLEMENTATION_EXACT_SHA_CI=PASS
M6_REFERENCE_FIXTURE_GATE=PASS
M6_REAL_VOICE_DESIGN_GATE=PASS
M6_REAL_VOICE_CLONE_GATE=PASS
M6_REAL_AUDIO_CONTINUATION_GATE=PASS
M6_GPU_DEVICE_PLACEMENT_GATE=PASS
M6_GPU_MEMORY_MEASUREMENT=PASS
M6_WAV_VALIDATION_GATE=PASS
M6_OFFLINE_REFERENCE_FEATURE_GATE=PASS
M6_NO_CPU_FALLBACK_GATE=PASS
M6_RESOURCE_SAFETY_GATE=PASS
M6_M0_M5_REGRESSION_GATE=PASS
M6_INVESTIGATION_COMPLETENESS=PASS
M6_REAL_ONE_SHOT_VOICE_FEATURES=PASS
M6_SINGLE_T4_RUNTIME=PASS
~~~

## Những gì M6 không qualification

M6 chỉ qualification ba đường design, clone, và continuation one-shot đã đo trên một Tesla T4.

Nó không qualification streaming, `generate_streaming`, thực thi `StreamRequest` thật, sequencing chunk, ghép lại chunk, hành vi ngắt, FastAPI hay REST, HTTP framing, scheduling, worker process, WorkerPool, admission control, T4x2 hay multi-GPU, nhiều worker, batching, `torch.compile`, `optimize=True` của upstream, override FP16, quantization, denoiser, mode kết hợp reference cộng prompt, prompt cache như public API, chấm điểm độ giống giọng chủ quan, ngưỡng xác thực người nói, tuyên bố chất lượng ngữ nghĩa qua ASR, hỗ trợ phổ quát cho GPU 16 GB, toàn bộ NVIDIA GPU, cam kết độ trễ real-time, production readiness, hay bất kỳ public release nào.

Streaming vẫn là M7. Public release đầu tiên vẫn là v1.0.0 và vẫn là release gate rõ ràng ở sau.
