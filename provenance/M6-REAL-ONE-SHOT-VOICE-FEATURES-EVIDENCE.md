# M6 Real One-Shot Voice Features Evidence

## Status

- UTC evidence snapshot: 2026-09-26T06:24:24Z
- Repository: OpenBMB-VoxCPM2-Inference
- Branch: main
- M6 starting SHA: 2ea0e8d52e21b6dcc97764cc86eb69a04f3e8294
- M6 implementation SHA: 805ed80a057d60bfab2af2691d03f4d8f62ba2d7
- Implementation CI run: 36223105464, PASS on the exact implementation SHA
- Version: 1.0.0, unchanged
- Source-lock SHA-256: 10097c137461308dfb2693b1bf53f2fdff3deb8508ff1add2f1d3047ecae367a
- No tag or release was created

## Upstream and model authority

~~~text
upstream repository    OpenBMB/VoxCPM
pinned commit          f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69
installed version      2.0.3.post33+gf772e498a
upstream license       Apache-2.0
model id               openbmb/VoxCPM2
model revision         32279effe8c19989596f05d353d1447f51d9e915
architecture           voxcpm2
config SHA-256         405f0dcd92f7feba6011ed4eac5c8d4f74cba9712f07fd5cfa3063bbdd95402c
file count             10
total model bytes      4,960,734,297
remote acquisition     0
~~~

The pinned upstream `VoxCPM.generate` signature was inspected directly from the pinned source tree and accepts `text`, `prompt_wav_path`, `prompt_text`, `reference_wav_path`, `cfg_value`, `inference_timesteps`, `normalize`, `denoise`, and `seed`. The parenthesised instruction form used by M6 matches the pinned upstream CLI rule `build_final_text(text, control) = f"({control}){text}"`.

The resolver selected the attached local deployment mount. The absolute model path is intentionally omitted from public evidence.

## Hardware and software

Before isolation, PyTorch observed two Tesla T4 devices. The canonical M6 processes used CUDA_VISIBLE_DEVICES=0, after which runtime discovery reported exactly one CUDA device.

~~~text
Python                  3.12.13
PyTorch                 2.10.0+cu128
torchaudio              2.10.0+cu128
transformers            5.0.0
huggingface-hub         1.33.0
numpy                   2.0.2
safetensors             0.7.0
librosa                 0.11.0
einops                  0.8.2
CUDA runtime            12.8
CUDA driver API         13.0
GPU                     Tesla T4
runtime visible GPUs    1
GPU total memory        15,636,037,632 B (14.562 GiB)
compute capability      7.5
configured dtype        bfloat16
upstream optimize       false
denoiser                false
~~~

The image did not expose nvidia-smi; PyTorch supplied device identity and allocator telemetry.

## Implementation verification

M6 adds real one-shot voice design, clone, and continuation mappings, execution-time local reference-audio validation, a milestone-neutral capability set, mutually exclusive CLI operation modes, and a safe request summary in the runtime report. Lazy upstream imports and the existing backend boundary are preserved. During corrective closeout, waveform-conversion failures were also bound to the actual operation instead of a hard-coded synthesize label; the real M6 qualification was repeated after the corrected implementation passed exact-SHA CI.

~~~text
baseline before M6       342 passed
after corrective M6      397 passed, 1 skipped
additional passing tests 55
additional local skips   1
ruff                     PASS
compileall               PASS
git diff --check          PASS
CI YAML parse             PASS
implementation CI         36223105464 PASS
~~~

The single skipped test asserts an unreadable-file rejection that the root user cannot observe, because root bypasses filesystem read permissions. The remaining 397 tests are green. Relative to the 342-test M5 baseline, M6 adds 55 passing cases plus that one environment-dependent local skip.

### M6 CI gates

All M6 gates run on stubbed upstream imports and never load model weights.

~~~text
M6_VOICE_DESIGN_MAPPING_GATE=PASS
M6_VOICE_CLONE_MAPPING_GATE=PASS
M6_AUDIO_CONTINUATION_MAPPING_GATE=PASS
M6_REFERENCE_PATH_REDACTION_GATE=PASS
M6_REAL_BACKEND_CAPABILITY_GATE=PASS
M6_STREAMING_REMAINS_UNQUALIFIED_GATE=PASS
M6_NO_REMOTE_AUDIO_ACQUISITION_GATE=PASS
~~~

## Mapping semantics verified

| Project request | Verified upstream call |
| --- | --- |
| `SpeechRequest` | `generate(text=T)` |
| `VoiceDesignRequest` | `generate(text=f"({I.strip()}){T}")` |
| `CloneRequest` | `generate(text=T, reference_wav_path=P)` |
| `ContinuationRequest` | `generate(text=T, prompt_wav_path=P, prompt_text=R)` |

The combined `reference_wav_path + prompt_wav_path` mode is not qualified. Clone passes no prompt pair; continuation passes no reference path.

## Reference audio validation

Validated at execution time at the backend boundary, with path-neutral public messages.

~~~text
missing path      reference_audio_not_found
directory path    reference_audio_not_file
unreadable path   reference_audio_unreadable
~~~

No absolute reference path, instruction text, or reference transcript appeared in any normalized error envelope, result metadata tuple, or public runtime report. A corrective scan of all five JSON runtime reports for absolute paths and request content returned zero hits.

## Reference fixture

Generated with the qualified M5 standard-TTS path, then frozen and reused for both clone and continuation. The WAV is not committed to Git.

~~~text
reference text      Xin chào, đây là giọng nói tham chiếu.
exit code           0
load duration       27.751947 s
synthesis duration  6.359500 s
channels            1
sample width        2 bytes
sample rate         48,000 Hz
frames              115,200
duration            2.40 s
file size           230,444 B
SHA-256             419fc4ca5bba2d2d74eb0918e27108b85ad88f61c9a6e5b9c99075c2c5cc85fa
peak abs PCM16      29,445
RMS PCM16           5,144.13
nonzero samples     114,808 / 115,200
project validation  valid
~~~

## Real M6 runs

| Run | Exit | Load | Generate | Duration | Peak allocated | Peak reserved | Host peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reference fixture | 0 | 27.752 s | 6.360 s | 2.40 s | 5,577,359,872 B | 5,758,779,392 B | 11,846,856,704 B |
| Voice design | 0 | 27.457 s | 5.290 s | 1.92 s | 5,561,635,328 B | 5,758,779,392 B | 11,837,554,688 B |
| Voice clone | 0 | 27.423 s | 10.055 s | 3.20 s | 5,795,059,712 B | 5,865,734,144 B | 11,820,601,344 B |
| Audio continuation | 0 | 27.507 s | 7.002 s | 1.76 s | 5,765,563,392 B | 5,865,734,144 B | 11,855,065,088 B |
| Offline clone proof | 0 | 27.529 s | 8.982 s | 2.72 s | 5,780,305,408 B | 5,865,734,144 B | 11,869,257,728 B |

No CPU fallback and no OOM occurred in any run.

### Voice design

~~~text
target text          Xin chào từ VoxCPM2.
instruction          giọng nữ ấm áp, bình tĩnh
upstream effective   (giọng nữ ấm áp, bình tĩnh)Xin chào từ VoxCPM2.
operation            voice-design
conditioning         instruction
channels             1
sample rate          48,000 Hz
frames               92,160
duration             1.92 s
file size            184,364 B
SHA-256              9b4d9d321e1f783e23379c8b701d1dfe75363ccdae77c2f1e791b6feef71d9f4
peak abs PCM16       30,749
RMS PCM16            3,214.95
nonzero samples      89,336 / 92,160
~~~

This gate proves the real conditioned design path executes and returns structurally valid audio. It does not claim the voice sounds female, warm, or calm.

### Voice clone

~~~text
target text          Xin chào, đây là phép thử clone giọng.
reference            frozen M6 reference fixture, consumed locally
upstream mode        reference_wav_path=<local fixture>, no prompt pair
operation            voice-clone
conditioning         reference
channels             1
sample rate          48,000 Hz
frames               153,600
duration             3.20 s
file size            307,244 B
SHA-256              43b841d1bb5c6ca78c84f5e5bd562383a51b030b2054ee7f04a1c5aac4c0b629
peak abs PCM16       25,866
RMS PCM16            4,600.49
nonzero samples      138,814 / 153,600
~~~

This gate proves the real reference-conditioned path executes on the selected GPU. It does not claim speaker similarity.

### Audio continuation

~~~text
target text          Và đây là phần tiếp theo.
prompt audio         frozen M6 reference fixture
prompt transcript    Xin chào, đây là giọng nói tham chiếu.
upstream mode        prompt_wav_path=<local fixture>, prompt_text=<transcript>, no reference path
operation            audio-continuation
conditioning         continuation
channels             1
sample rate          48,000 Hz
frames               84,480
duration             1.76 s
file size            169,004 B
SHA-256              d79cffea396d7879e75862028eaf412bceb8de71c6e23097a8d9257318483854
peak abs PCM16       27,933
RMS PCM16            4,500.63
nonzero samples      76,285 / 84,480
~~~

This gate proves the real continuation path executes and returns structurally valid audio. It does not claim semantic continuity quality.

## Device placement evidence

Identical across all five runs.

~~~text
upstream device              cuda:0
parameter count              888
parameter device types       ["cuda"]
buffer count                 10
buffer device types          ["cuda"]
all on expected device       true
CPU fallback                 false
~~~

## Independent WAV validation

Every canonical output was validated with the project validator and independently with the Python standard-library `wave` module. Both agreed on all structural fields.

~~~text
reference      1 ch / 2 bytes / 48,000 Hz / 115,200 frames / 2.40 s / 230,444 B
design         1 ch / 2 bytes / 48,000 Hz / 92,160 frames / 1.92 s / 184,364 B
clone          1 ch / 2 bytes / 48,000 Hz / 153,600 frames / 3.20 s / 307,244 B
continuation   1 ch / 2 bytes / 48,000 Hz / 84,480 frames / 1.76 s / 169,004 B
~~~

Upstream sampling is stochastic, so byte-for-byte reproducibility is not claimed. The offline clone digest differs from the canonical clone digest for that reason.

## Offline/no-download proof

A clone run was repeated with a fresh empty HF_HOME, offline flags, and loopback blackhole endpoint and proxy settings.

~~~text
VOXCPM_OFFLINE=1
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_HOME                    fresh empty directory
HF_ENDPOINT                http://127.0.0.1:1
HTTP_PROXY                 http://127.0.0.1:1
HTTPS_PROXY                http://127.0.0.1:1
exit code                  0
local_files_only           true
HF_HOME files before run   0
HF_HOME files after run    0
valid WAV                  true
channels                   1
sample rate                48,000 Hz
frames                     130,560
duration                   2.72 s
file size                  261,164 B
SHA-256                    7e428f351c03d9ee5869ac0bfeeda88809f56b8c66029bc9362c7a4ec5d6c0fb
peak abs PCM16             26,235
RMS PCM16                  4,620.45
nonzero samples            123,934 / 130,560
~~~

Because the reference audio is local and the model is local, the reference-conditioned operation succeeds without acquisition.

## OOM evidence

Every corrective runtime report recorded zero for cgroup `low`, `high`, `max`, and `oom`. The host `oom`, `oom_kill`, and `oom_group_kill` counters were zero before and after the corrective real-runtime sequence. Every run exited zero. No GPU OOM, host OOM, signal termination, or allocation failure was observed.

## Environment limitations

Standard venv creation failed at ensurepip, so the qualification environment used --without-pip --system-site-packages to preserve the working CUDA stack.

A global pip check reports unrelated pre-existing Kaggle package conflicts. The exact pinned VoxCPM import passed, all declared non-torch upstream dependencies were present, and PyTorch remained 2.10.0+cu128.

The image lacks nvidia-smi; M6 GPU-memory evidence therefore uses the PyTorch allocator, which directly covers the inference process.

## Security and hygiene

- The GitHub PAT value was never printed.
- The local credential file mode remained 0600.
- Generated WAVs and raw stderr were not added to Git.
- Model weights were not added to Git.
- Absolute model mount paths and absolute reference-audio paths are omitted from public evidence.
- Raw upstream stderr contains absolute model paths and is preserved privately only.

## Gate summary

The summary contains the 22 required M6 closeout gates plus one additional no-remote-audio-acquisition hygiene gate.

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

## Scope boundary

M6 qualifies only the measured one-shot design, clone, and continuation paths on one Tesla T4.

It does not qualify streaming, `generate_streaming`, `StreamRequest` real execution, FastAPI or REST, HTTP framing, scheduling, worker processes, WorkerPool, admission control, T4x2 or multi-GPU, multiple workers, batching, `torch.compile`, upstream `optimize=True`, FP16 overrides, quantization, the denoiser, the combined reference-plus-prompt mode, prompt caches as a public API, subjective speaker-similarity scoring, speaker verification thresholds, ASR semantic quality claims, universal 16 GB GPU support, all NVIDIA GPUs, real-time latency guarantees, production deployment, or any release.

Streaming remains M7. The evidence commit and its final exact-SHA CI run are the final M6 authority. The first public release remains v1.0.0 and stays a later explicit release gate.
