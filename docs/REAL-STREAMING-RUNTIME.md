# Real native streaming runtime

## Status

M7 qualifies the pinned upstream native streaming path for standard TTS, voice design, voice clone, and audio continuation on one explicitly selected Tesla T4.

This is backend/model streaming only. It does not qualify HTTP, SSE, WebSocket, browser playback, workers, scheduling, IPC, T4x2, or multi-GPU execution.

## Authority

```text
M7 base SHA              c5127c07c88d3077adb693bfa071a489c413151b
contract clarification   5adf5866e3056294bc14cddde14395269c5520c2
clarification CI         36225098683 PASS
implementation SHA       f97e671d0b08e838375ad70c45e61960182e4756
implementation CI        36225562822 PASS
VERSION                  1.0.0
upstream commit          f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69
model revision           32279effe8c19989596f05d353d1447f51d9e915
```

## Characterization

Pinned upstream native streaming is real incremental execution:

```text
generate_streaming
→ incremental inference
→ AudioVAE.streaming_decode
→ decode_chunk
→ yielded CPU waveform chunks
```

Fixed-seed characterization showed native streams are exactly repeatable for the same seed and chunk boundaries, while one-shot versus streaming differs slightly at floating-point level because the decode paths differ.

Across measured seeds, one-shot and native streaming kept the same total sample count. The largest measured float difference was on the order of 1e-6, and PCM16 differences were at most one least-significant bit. Exact one-shot equality remains a fake-backend guarantee; real native streaming is measured rather than assumed bit-identical.

See `provenance/M7-NATIVE-STREAMING-CHARACTERIZATION.md`.

## Real single-T4 runs

All measured stream runs used:

```text
CUDA_VISIBLE_DEVICES=0
effective_device=cuda
selected_gpu_indices=[0]
worker_count=1
upstream_device=cuda:0
dtype=bfloat16
optimize=false
visible CUDA devices=1
GPU=Tesla T4
```

| Operation | Chunks | First chunk | Completion | Audio | Peak allocated VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard TTS | 13 | 1.721 s | 5.702 s | 2.08 s | 5.346 GiB |
| Voice design | 12 | 1.751 s | 5.388 s | 1.92 s | 5.348 GiB |
| Voice clone | 15 | 3.637 s | 8.372 s | 2.40 s | 5.528 GiB |
| Audio continuation | 13 | 3.620 s | 7.624 s | 2.08 s | 5.529 GiB |
| Offline clone stream | 15 | 3.605 s | 8.337 s | 2.40 s | 5.528 GiB |

Every real stream had:

```text
native=true
sequence_contiguous=true
all_chunks_non_empty=true
final_chunk_count=1
first_chunk_samples=7680
min_chunk_samples=7680
max_chunk_samples=7680
sample_rate=48000 Hz
channels=1
CPU fallback=false
OOM counters=0
```

## Early cancellation

A long native stream was started, one project `AudioChunk` was consumed, and the iterator was closed immediately.

```text
first chunk sequence          0
first chunk final             false
first chunk samples           7680
first chunk latency           1.883 s
iterator close time           0.000552 s
allocated before close        5,527,718,912 B
allocated after close         5,502,608,384 B
```

After cancellation, the same backend successfully completed a normal one-shot TTS request:

```text
post-cancel generation        3.946 s
samples                       84,480
sample rate                   48,000 Hz
channels                      1
```

This proves early iterator close propagates without leaving the backend unusable in the measured single-threaded case.

## Offline proof

The clone stream was repeated with a fresh empty `HF_HOME`, offline flags, and loopback blackhole endpoint/proxy settings.

```text
HF_HOME files after run       0
local_files_only              true
stream completed              true
valid WAV                     true
```

This demonstrates the measured native stream used the attached local model and local reference audio without model acquisition.

## Independent WAV validation

| Output | Frames | Duration | SHA-256 |
| --- | ---: | ---: | --- |
| Standard stream | 99,840 | 2.08 s | `b85ed975135aaac99db73cda5f98f4a59d64a2d54953d12328058eb889fe379d` |
| Design stream | 92,160 | 1.92 s | `d9f93864b11dcca55c177930cfcf2dfdc7e36fafd5e1ff7903291a12529c70c4` |
| Clone stream | 115,200 | 2.40 s | `0bd58807b3e6bbbf20f72a884c42a5851fcf39661a0ec9be2d9f1a6c5f0d3858` |
| Continuation stream | 99,840 | 2.08 s | `feda434243e7a6d9e4cbd0448a8e32811f7ada68e8fea3cbcdbd162e6b112fdd` |
| Offline clone stream | 115,200 | 2.40 s | `d79ff31e561038d89cb631858406dc3adb6a19e7a75c8953968363e4a85a90ab` |

All were mono PCM16 at 48 kHz and were independently read successfully with Python `wave`.

## M7 gates

```text
M7_PRE_IMPLEMENTATION_AUDIT=PASS
M7_UPSTREAM_STREAMING_EXISTS_GATE=PASS
M7_UPSTREAM_STREAMING_REPEATABILITY_GATE=PASS
M7_STREAM_VS_ONESHOT_CHARACTERIZATION=PASS
M7_PUBLIC_STREAM_CONTRACT_COMPATIBILITY=PASS_WITH_CLARIFICATION
M7_NATIVE_STREAM_MAPPING_GATE=PASS
M7_NATIVE_NOT_BUFFERED_GATE=PASS
M7_STREAM_SEQUENCE_GATE=PASS
M7_FINAL_CHUNK_SEMANTICS_GATE=PASS
M7_LAZY_STREAM_ITERATION_GATE=PASS
M7_STREAM_ERROR_NORMALIZATION_GATE=PASS
M7_STREAM_REFERENCE_REDACTION_GATE=PASS
M7_STREAM_LIFECYCLE_GATE=PASS
M7_IMPLEMENTATION_EXACT_SHA_CI=PASS
M7_REAL_STANDARD_STREAM_GATE=PASS
M7_REAL_DESIGN_STREAM_GATE=PASS
M7_REAL_CLONE_STREAM_GATE=PASS
M7_REAL_CONTINUATION_STREAM_GATE=PASS
M7_REAL_EARLY_CANCEL_GATE=PASS
M7_GPU_DEVICE_PLACEMENT_GATE=PASS
M7_GPU_MEMORY_MEASUREMENT=PASS
M7_STREAM_WAV_VALIDATION_GATE=PASS
M7_OFFLINE_NATIVE_STREAM_GATE=PASS
M7_NO_CPU_FALLBACK_GATE=PASS
M7_STREAM_RESOURCE_SAFETY_GATE=PASS
M7_M0_M6_REGRESSION_GATE=PASS
M7_INVESTIGATION_COMPLETENESS=PASS
M7_REAL_NATIVE_STREAMING=PASS
M7_SINGLE_T4_RUNTIME=PASS
```

## Scope boundary

M7 does not qualify network/media streaming, FastAPI, REST, HTTP framing, SSE, WebSocket, workers, WorkerPool, schedulers, admission control, IPC, batching, T4x2, multi-GPU, `torch.compile`, `optimize=True`, quantization, denoiser, production deployment, or any public release.

The validation WAV is written only after stream completion and is not evidence of live network playback.
