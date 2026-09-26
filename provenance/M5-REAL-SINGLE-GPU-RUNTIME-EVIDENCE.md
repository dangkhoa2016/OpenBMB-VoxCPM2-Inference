# M5 Real Single-GPU Runtime Evidence

## Status

- UTC evidence snapshot: 2026-09-26T03:30:59Z
- Repository: OpenBMB-VoxCPM2-Inference
- Branch: main
- M5 starting SHA: ec28285ca5dcb50e7f45ccb1984e19529fb21f88
- M5 implementation SHA: d1e91b99c0457401c2cbf0c46eeb576aa08350ee
- Implementation CI run: 36214646774, PASS on exact implementation SHA
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

The resolver selected the attached local deployment mount. The absolute model path is intentionally omitted from public evidence.
## Hardware and software

Before isolation, PyTorch observed two Tesla T4 devices. The canonical process used CUDA_VISIBLE_DEVICES=0, after which runtime discovery reported exactly one CUDA device.

~~~text
Python                  3.12.13
PyTorch                 2.10.0+cu128
torchaudio              2.10.0+cu128
transformers            5.0.0
huggingface-hub         1.33.0
numpy                   2.0.2
safetensors             0.7.0
librosa                 0.11.0
einops                   0.8.2
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

The image did not expose nvidia-smi; PyTorch supplied device identity and allocator telemetry, and libcuda.so.1 supplied the driver API version.

## Implementation verification

M5 adds indexed single-GPU mapping, explicit multi-GPU refusal, GPU placement reporting, and process-local CUDA allocator telemetry while preserving lazy upstream imports and the existing backend boundary.

~~~text
baseline before M5      334 passed
after M5                342 passed
new M5 tests            8
ruff                     PASS
compileall               PASS
git diff --check         PASS
CI YAML parse            PASS
implementation CI        36214646774 PASS
~~~
## Real load-only run

~~~text
exit code                    0
load duration                45.282575 s
wall time                    48.35 s
host peak RSS                11,827,089,408 B (11.015 GiB)
peak GPU allocated           5,310,254,592 B (4.946 GiB)
peak GPU reserved            5,372,903,424 B (5.004 GiB)
effective device             cuda
selected GPU indices         [0]
worker count                 1
worker GPU indices           [0]
upstream device              cuda:0
CUDA visible count           1
parameter count              888
parameter device types       ["cuda"]
buffer count                 10
buffer device types          ["cuda"]
all on expected device       true
CPU fallback                 false
~~~

## Canonical real TTS

Canonical text: "Xin chào từ VoxCPM2."

~~~text
exit code                    0
load duration                27.079539 s
synthesis duration           7.137908 s
wall time                    37.63 s
audio duration               2.40 s
real-time factor             2.9741
host peak RSS                11,814,817,792 B (11.003 GiB)
peak GPU allocated           5,577,354,752 B (5.194 GiB)
peak GPU reserved            5,758,779,392 B (5.363 GiB)
~~~
Canonical WAV:

~~~text
channels                     1
sample width                 2 bytes
sample rate                  48,000 Hz
frames                       115,200
duration                     2.40 s
file size                    230,444 B
SHA-256                      8a5b4641f6ddc9b08bc733baa5071a1fc851ab71f13e3e8ab132bd6b649515a2
peak abs PCM16               15,446
RMS PCM16                    2,034.97
nonzero samples              114,130 / 115,200
~~~

Project WAV validation and independent Python wave readback both passed. No subjective listening, ASR intelligibility, or speaker-fidelity claim is made.

## Offline/no-download proof

A full TTS run was repeated with a fresh empty HF_HOME, offline flags, and loopback blackhole endpoint/proxy settings.

~~~text
exit code                     0
load duration                 27.168456 s
synthesis duration            5.611368 s
wall time                     35.86 s
HF_HOME files after run       0
network/proxy/download errors 0
local_files_only              true
valid WAV                     true
peak GPU allocated            5,566,538,752 B
peak GPU reserved             5,758,779,392 B
host peak RSS                 11,846,963,200 B (11.033 GiB)
~~~

Offline WAV SHA-256: 94dd93dcd6b083574b714ece1bebb224df678ddbe4b2f57d3c3cf2af62cdba66.

The WAV differs from the canonical digest because upstream sampling is stochastic. Byte-for-byte reproducibility is not claimed.
## OOM evidence

For load-only, canonical TTS, and offline-proof TTS, cgroup memory-event deltas were zero for high, low, max, oom, oom_kill, and oom_group_kill. Every run exited zero. No GPU OOM, host OOM, signal termination, or allocation failure was observed.

## Environment limitations

Standard venv creation failed at ensurepip, so the qualification environment used --without-pip --system-site-packages to preserve the working CUDA stack.

A global pip check reports unrelated pre-existing Kaggle package conflicts. The exact pinned VoxCPM import passed, all declared non-torch upstream dependencies were present, and PyTorch remained 2.10.0+cu128.

The image lacks nvidia-smi; M5 GPU-memory evidence therefore uses the PyTorch allocator, which directly covers the inference process.

## Security and hygiene

- The GitHub PAT value was never printed.
- The local credential file mode was tightened to 0600.
- The temporary GIT_ASKPASS helper was removed after push.
- Generated WAVs and raw stderr were not added to Git.
- Model weights were not added to Git.
- Absolute model mount paths are omitted from public evidence.

## Gate summary

~~~text
M5_PRE_IMPLEMENTATION_AUDIT=PASS
M5_SINGLE_GPU_PLAN_GATE=PASS
M5_CUDA_DEVICE_MAPPING_GATE=PASS
M5_REAL_GPU_LOAD_GATE=PASS
M5_REAL_GPU_TTS_GATE=PASS
M5_GPU_DEVICE_PLACEMENT_GATE=PASS
M5_GPU_MEMORY_MEASUREMENT=PASS
M5_WAV_VALIDATION_GATE=PASS
M5_OFFLINE_LOCAL_MODEL_GATE=PASS
M5_NO_CPU_FALLBACK_GATE=PASS
M5_M0_M4_REGRESSION_GATE=PASS
M5_REAL_SINGLE_GPU_RUNTIME=PASS
M5_INVESTIGATION_COMPLETENESS=PASS
M5_SINGLE_T4_RUNTIME=PASS
~~~
## Scope boundary

M5 qualifies only the measured single-T4 standard-TTS path.

It does not qualify T4x2 or any multi-GPU execution, multiple workers, scheduling, torch.compile, FP16 overrides, quantization, voice design, cloning, continuation, streaming, API behavior, universal 16 GB GPU support, all NVIDIA GPUs, real-time latency, subjective audio quality, or production deployment.

The evidence commit and its final exact-SHA CI run are the final M5 authority.
