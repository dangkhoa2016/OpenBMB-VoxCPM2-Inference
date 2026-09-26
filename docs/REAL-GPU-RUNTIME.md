# Real single-GPU runtime qualification

## Status

M5 qualifies the real VoxCPM2 backend for standard TTS on one explicitly selected CUDA GPU.

The host exposed two Tesla T4 devices before isolation. The canonical M5 process used CUDA_VISIBLE_DEVICES=0, so the project runtime saw exactly one CUDA device.

~~~text
VOXCPM_DEVICE=cuda
VOXCPM_GPU_DEVICES=0
VOXCPM_WORKERS=1
effective_device=cuda
selected_gpu_indices=[0]
worker_count=1
upstream_device=cuda:0
~~~

This is a single-GPU qualification. It is not a T4x2 or multi-GPU qualification.

## Authority

~~~text
M5 starting SHA        ec28285ca5dcb50e7f45ccb1984e19529fb21f88
M5 implementation SHA  d1e91b99c0457401c2cbf0c46eeb576aa08350ee
implementation CI      36214646774 (PASS)
VERSION                 1.0.0
upstream commit         f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69
installed voxcpm        2.0.3.post33+gf772e498a
~~~
The model is openbmb/VoxCPM2 at revision 32279effe8c19989596f05d353d1447f51d9e915. Its config SHA-256 is 405f0dcd92f7feba6011ed4eac5c8d4f74cba9712f07fd5cfa3063bbdd95402c. The absolute model path is intentionally omitted from public reports; M2 bound the run to the attached local mirror with zero remote acquisition.

## Runtime environment

~~~text
Python                  3.12.13
PyTorch                 2.10.0+cu128
torchaudio              2.10.0+cu128
CUDA runtime            12.8
CUDA driver API         13.0
GPU                     Tesla T4
runtime visible GPUs    1
GPU memory              15,636,037,632 B (14.562 GiB)
compute capability      7.5
configured dtype        bfloat16
upstream optimize       false
denoiser                false
~~~

The checkpoint dtype was not modified. M5 does not add an FP16 override and does not qualify torch.compile.

The Kaggle image did not provide the nvidia-smi binary. GPU identity and allocator memory were recorded through PyTorch, while the driver API was queried through libcuda.so.1.

## Real runs

| Run | Exit | Load | Synthesize | Wall | Peak allocated VRAM | Peak reserved VRAM | Host peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Load-only | 0 | 45.283 s | n/a | 48.35 s | 4.946 GiB | 5.004 GiB | 11.015 GiB |
| Canonical TTS | 0 | 27.080 s | 7.138 s | 37.63 s | 5.194 GiB | 5.363 GiB | 11.003 GiB |
| Offline-proof TTS | 0 | 27.168 s | 5.611 s | 35.86 s | 5.184 GiB | 5.363 GiB | 11.033 GiB |
Canonical text: "Xin chào từ VoxCPM2."

The canonical synthesis produced 2.40 seconds of audio in 7.138 seconds, a measured real-time factor of about 2.97. This is one measured T4 run, not a latency guarantee and not a real-time claim.

## Device placement

~~~text
parameter_count           888
parameter_device_types    ["cuda"]
buffer_count              10
buffer_device_types       ["cuda"]
expected_device           "cuda"
all_on_expected_device    true
cpu_only                  false
~~~

The backend maps one selected runtime GPU to cuda:0. A real backend plan selecting more than one GPU is rejected as multi_gpu_not_qualified; M5 never silently falls back to CPU.

## WAV result

~~~text
channels              1
sample width          2 bytes (PCM16)
sample rate           48,000 Hz
frames                115,200
duration              2.40 s
file size             230,444 B
SHA-256               8a5b4641f6ddc9b08bc733baa5071a1fc851ab71f13e3e8ab132bd6b649515a2
peak abs PCM16        15,446
RMS PCM16             2,034.97
nonzero samples       114,130 / 115,200
~~~

The project validator and an independent Python wave readback agreed on the WAV structure. The signal is non-empty and non-degenerate. M5 did not perform listening evaluation, ASR transcription, speaker-fidelity scoring, or subjective audio-quality qualification.
The upstream sampler is stochastic. The offline-proof run produced a different valid WAV digest and duration; byte-for-byte reproducibility is not claimed.

## Offline proof

A second full TTS run used a fresh empty HF_HOME, offline flags, and loopback blackhole endpoints/proxies.

~~~text
exit code                       0
local_files_only                true
HF_HOME files after run         0
network/proxy/download errors   0
valid WAV                       yes
~~~

This proves the measured run used the attached local model without remote model acquisition. The method uses blackholed endpoints/proxies rather than a separate network namespace.

## Resource evidence

For load-only, canonical TTS, and offline-proof TTS, cgroup deltas were zero for high, low, max, oom, oom_kill, and oom_group_kill. Every process exited zero. No GPU OOM, host OOM, signal kill, or allocation failure was observed.

## Qualification-environment caveat

The qualification venv inherited the Kaggle CUDA stack with --system-site-packages. Standard venv creation failed in ensurepip, so M5 used --without-pip --system-site-packages and reused the system pip module through the venv interpreter.

A global pip check reports unrelated pre-existing Kaggle package conflicts. The exact pinned VoxCPM import passed, all declared non-torch upstream dependencies were present, and the CUDA PyTorch identity remained unchanged. These conflicts are an environment limitation, not a VoxCPM runtime failure.
## M5 gates

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

## What M5 does not qualify

M5 does not qualify T4x2/multi-GPU execution, multiple workers, scheduling, torch.compile, FP16 overrides, quantization, streaming, voice design, cloning, continuation, API behavior, real-time guarantees, universal 16 GB GPU support, all NVIDIA GPUs, production readiness, or subjective audio quality.
