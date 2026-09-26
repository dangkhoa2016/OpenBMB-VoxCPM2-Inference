# Real one-shot voice feature qualification

## Status

M6 qualifies the three remaining one-shot operations already frozen by the M3 backend contract: voice design, voice clone, and audio continuation.

All three run through the existing `PytorchVoxCPMBackend` non-streaming upstream entrypoint on one explicitly selected CUDA GPU. No new public backend abstraction was introduced.

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

This is a one-shot qualification. It is not a streaming, multi-GPU, worker, scheduling, or API qualification.

## Authority

~~~text
M6 starting SHA         2ea0e8d52e21b6dcc97764cc86eb69a04f3e8294
M6 implementation SHA   805ed80a057d60bfab2af2691d03f4d8f62ba2d7
implementation CI       36223105464 (PASS on the exact implementation SHA)
VERSION                 1.0.0
upstream commit         f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69
installed voxcpm        2.0.3.post33+gf772e498a
source-lock SHA-256     10097c137461308dfb2693b1bf53f2fdff3deb8508ff1add2f1d3047ecae367a
~~~

The model is openbmb/VoxCPM2 at revision 32279effe8c19989596f05d353d1447f51d9e915, config SHA-256 405f0dcd92f7feba6011ed4eac5c8d4f74cba9712f07fd5cfa3063bbdd95402c. The absolute model path and the absolute reference-audio path are intentionally omitted from public evidence. M2 bound the run to the attached local mirror with zero remote acquisition.

## Upstream mapping

The pinned upstream `VoxCPM.generate` entrypoint accepts `text`, `reference_wav_path`, `prompt_wav_path`, and `prompt_text`. M6 maps each project request onto exactly one of those modes.

| Project request | Upstream call |
| --- | --- |
| `SpeechRequest(text=T)` | `generate(text=T)` |
| `VoiceDesignRequest(text=T, instruction=I)` | `generate(text=f"({I.strip()}){T}")` |
| `CloneRequest(text=T, reference_audio=P)` | `generate(text=T, reference_wav_path=P)` |
| `ContinuationRequest(text=T, reference_audio=P, reference_transcript=R)` | `generate(text=T, prompt_wav_path=P, prompt_text=R)` |

The parenthesised instruction form matches the pinned upstream CLI rule `build_final_text`. That formatting rule stays private to the backend boundary; the public `VoiceDesignRequest` still carries the raw instruction, and the public report records only the instruction character count.

The combined `reference_wav_path + prompt_wav_path` mode is deliberately not qualified in M6. Clone passes no prompt pair, and continuation passes no reference path.

## Capabilities

~~~text
BackendInfo capabilities
  clone
  continue_audio
  design
  synthesize

stream                  absent
generate_streaming      absent
~~~

The M4-only capability constant was replaced by a milestone-neutral `REAL_BACKEND_CAPABILITIES`. Streaming still raises `stream_not_qualified` with `milestone: M7`.

## Reference audio policy

M6 is the first milestone where `AudioReference.local_path` reaches real upstream code. The backend validates the descriptor at execution time and never changes the M3 dataclass contract.

~~~text
missing path      reference_audio_not_found
directory path    reference_audio_not_file
unreadable path   reference_audio_unreadable
non-descriptor    invalid_backend_request
~~~

Every public message is path-neutral. No absolute path, instruction text, or reference transcript reaches a normalized error envelope, a result metadata tuple, or the public runtime report. Reference audio is read only from the local filesystem and is never fetched remotely.

## Reference fixture

The canonical M6 reference fixture was generated locally with the already-qualified M5 standard-TTS path, then frozen and reused for both clone and continuation. No third-party voice sample was downloaded, so no licensing or privacy ambiguity is introduced.

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

A self-generated reference proves the real reference-audio path executes. It does not prove cloning fidelity against a human speaker.

## Real runs

| Run | Exit | Load | Generate | Duration | Peak allocated VRAM | Peak reserved VRAM | Host peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reference fixture | 0 | 27.752 s | 6.360 s | 2.40 s | 5.194 GiB | 5.363 GiB | 11.033 GiB |
| Voice design | 0 | 27.457 s | 5.290 s | 1.92 s | 5.180 GiB | 5.363 GiB | 11.025 GiB |
| Voice clone | 0 | 27.423 s | 10.055 s | 3.20 s | 5.397 GiB | 5.463 GiB | 11.009 GiB |
| Audio continuation | 0 | 27.507 s | 7.002 s | 1.76 s | 5.370 GiB | 5.463 GiB | 11.041 GiB |
| Offline clone proof | 0 | 27.529 s | 8.982 s | 2.72 s | 5.383 GiB | 5.463 GiB | 11.054 GiB |

### Voice design

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

The gate proves the real conditioned design path executes and returns structurally valid audio. It does not claim the generated voice actually sounds female, warm, or calm. That would require human or semantic evaluation.

### Voice clone

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

The gate proves the real reference-conditioned path executes on the selected GPU and returns a valid `AudioResult` and valid WAV. It does not claim speaker similarity.

### Audio continuation

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

The gate proves the real continuation path executes and returns structurally valid audio. It does not claim semantic continuity quality.

## Device placement and memory

Every canonical process used the same single-T4 isolation as M5.

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

For the reference fixture, design, clone, continuation, and offline clone runs, PyTorch reported a maximum allocated between 5.18 GiB and 5.40 GiB and a maximum reserved between 5.36 GiB and 5.46 GiB on a 15,636,037,632 B device. Host peak RSS stayed between 11.01 GiB and 11.05 GiB.

## WAV validation

Every canonical output was validated twice: once by the project validator and once by an independent Python standard-library `wave` readback. Both agreed on channels, sample width, sample rate, frame count, duration, and byte size for all four files. The signal is non-empty and non-degenerate in every case.

~~~text
reference      1 ch / 2 bytes / 48,000 Hz / 115,200 frames / 2.40 s / 230,444 B
design         1 ch / 2 bytes / 48,000 Hz / 92,160 frames / 1.92 s / 184,364 B
clone          1 ch / 2 bytes / 48,000 Hz / 153,600 frames / 3.20 s / 307,244 B
continuation   1 ch / 2 bytes / 48,000 Hz / 84,480 frames / 1.76 s / 169,004 B
~~~

The image does not expose `nvidia-smi`; GPU evidence therefore comes from the PyTorch allocator, which directly covers the inference process. Upstream sampling is stochastic, so byte-for-byte reproducibility is not claimed.

## Offline / no-download proof

The clone operation was repeated under a fresh empty `HF_HOME`, offline flags, and loopback blackhole endpoint and proxy settings.

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

Because the reference audio is local and the model is local, the reference-conditioned operation succeeds without any acquisition.

## Memory and OOM evidence

Every recorded cgroup `memory.events` snapshot was zero for `low`, `high`, `max`, and `oom` across the reference fixture, design, clone, continuation, and offline clone runs. The host `oom`, `oom_kill`, and `oom_group_kill` counters remained zero before and after the whole M6 session. Every process exited zero. No GPU OOM, host OOM, signal termination, or allocation failure was observed.

## CLI surface

M6 extends `voxcpm-generate` with mutually exclusive operation modes rather than adding unrelated scripts.

~~~bash
voxcpm-generate --text "..." --output out.wav
voxcpm-generate --text "..." --voice-instruction "..." --output out.wav
voxcpm-generate --text "..." --reference-audio /path/ref.wav --output out.wav
voxcpm-generate --text "..." --prompt-audio /path/prefix.wav --prompt-text "..." --output out.wav
~~~

~~~text
--voice-instruction with reference or prompt audio   rejected
--reference-audio with prompt audio or prompt text   rejected
--prompt-audio without --prompt-text                 rejected
--prompt-text without --prompt-audio                 rejected
streaming flags                                     absent
~~~

## Runtime report

The report schema stays backward compatible. M6 adds an `operation` field and a safe `request` summary.

~~~text
operation   standard-tts | voice-design | voice-clone | audio-continuation | load-only
request     input_text_characters
            instruction_characters
            reference_audio_used
            reference_input_kind
            reference_transcript_characters
~~~

The summary records counts and booleans only. Full request text, raw instruction content, raw transcript content, and absolute reference paths are never included.

## Implementation verification

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

Normal CI remains GPU-free, model-free, and network-free. The M6 CI gates use a stubbed upstream import and never load model weights.

## M6 gates

The list below contains the 22 required M6 closeout gates plus one additional no-remote-audio-acquisition hygiene gate.

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

## What M6 does not qualify

M6 qualifies only the measured one-shot design, clone, and continuation paths on one Tesla T4.

It does not qualify streaming, `generate_streaming`, `StreamRequest` real execution, chunk sequencing, reassembly, interruption behaviour, FastAPI or REST, HTTP framing, scheduling, worker processes, WorkerPool, admission control, T4x2 or multi-GPU, multiple workers, batching, `torch.compile`, upstream `optimize=True`, FP16 overrides, quantization, the denoiser, the combined reference-plus-prompt mode, prompt caches as a public API, subjective speaker-similarity scoring, speaker verification thresholds, ASR semantic quality claims, universal 16 GB GPU support, all NVIDIA GPUs, real-time latency guarantees, production readiness, or any public release.

Streaming remains M7. The first public release remains v1.0.0 and stays a later explicit release gate.
