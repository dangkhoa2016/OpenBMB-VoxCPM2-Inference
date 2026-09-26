# M7 Real Native Streaming Evidence

## Status

- Repository: OpenBMB-VoxCPM2-Inference
- Branch: main
- M7 base SHA: c5127c07c88d3077adb693bfa071a489c413151b
- Contract clarification SHA: 5adf5866e3056294bc14cddde14395269c5520c2
- Contract clarification CI: 36225098683, PASS
- Native streaming implementation SHA: f97e671d0b08e838375ad70c45e61960182e4756
- Implementation CI: 36225562822, PASS
- VERSION: 1.0.0
- No tag or release was created

## Upstream and model authority

```text
upstream repository      OpenBMB/VoxCPM
upstream commit          f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69
installed voxcpm         2.0.3.post33+gf772e498a
model id                 openbmb/VoxCPM2
model revision           32279effe8c19989596f05d353d1447f51d9e915
config SHA-256           405f0dcd92f7feba6011ed4eac5c8d4f74cba9712f07fd5cfa3063bbdd95402c
file count               10
total model bytes        4,960,734,297
remote acquisition       0
```

Absolute model paths are omitted from public evidence.

## Runtime environment

```text
Python                    3.12.13
PyTorch                   2.10.0+cu128
CUDA runtime              12.8
GPU                       Tesla T4
visible CUDA devices      1
GPU memory                15,636,037,632 B
compute capability        7.5
configured dtype          bfloat16
upstream device           cuda:0
upstream optimize         false
denoiser                  false
```

The host physically exposed two T4 devices before isolation. M7 processes used `CUDA_VISIBLE_DEVICES=0`, so the project runtime saw exactly one GPU.

## Pre-implementation characterization

Pinned upstream native streaming was characterized before backend implementation.

Canonical fixed-seed run:

```text
text                         Xin chào từ VoxCPM2.
seed                         42
one-shot samples             99,840
stream chunks                13
stream samples               99,840
stream A == stream B         exact float32 equality
one-shot == stream           false at float32 level
max abs difference           4.54e-7
RMS difference               4.48e-8
```

Additional characterization confirmed the same pattern across multiple seeds: same sample count, exactly repeatable native stream for the same seed, and tiny numerical differences from one-shot caused by the distinct incremental decoder path.

Exact one-shot equality is therefore kept as a deterministic fake-backend guarantee, not a universal real-native-stream requirement.

See `provenance/M7-NATIVE-STREAMING-CHARACTERIZATION.md`.

## Implementation verification

The real backend implementation:

- calls `model.generate_streaming()` rather than `model.generate()`;
- adapts upstream native chunks into project-owned immutable `AudioChunk` values;
- uses one-chunk lookahead so exactly the last chunk has `is_final=True`;
- keeps sequence values contiguous from zero;
- rejects an empty upstream stream;
- normalizes stream errors through the project error hierarchy;
- closes the upstream iterator when the consumer closes early;
- preserves M6 local-reference validation for clone and continuation streams;
- advertises `stream` only in the qualified real backend capability set.

Local regression on the implementation SHA:

```text
411 passed
1 skipped
ruff          PASS
compileall    PASS
implementation CI 36225562822 PASS
Python 3.10   PASS
Python 3.11   PASS
Python 3.12   PASS
```

The single local skip is the pre-existing unreadable-file case under root.

## Real native stream measurements

| Operation | Chunks | First chunk | Completion | Audio | Peak allocated | Peak reserved | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard TTS | 13 | 1.721 s | 5.702 s | 2.08 s | 5.346 GiB | 5.537 GiB | 11.024 GiB |
| Voice design | 12 | 1.751 s | 5.388 s | 1.92 s | 5.348 GiB | 5.539 GiB | 11.012 GiB |
| Voice clone | 15 | 3.637 s | 8.372 s | 2.40 s | 5.528 GiB | 5.740 GiB | 11.016 GiB |
| Audio continuation | 13 | 3.620 s | 7.624 s | 2.08 s | 5.529 GiB | 5.740 GiB | 11.025 GiB |
| Offline clone stream | 15 | 3.605 s | 8.337 s | 2.40 s | 5.528 GiB | 5.740 GiB | 11.033 GiB |

For every completed stream:

```text
native                       true
all_chunks_non_empty         true
sequence_contiguous          true
final_chunk_count            1
chunk size                   7,680 samples
sample rate                  48,000 Hz
channels                     1
model all on expected device true
CPU fallback                 false
cgroup low/high/max/oom      0
```

## Early cancellation

The project backend was loaded once and a deliberately longer native stream was started.

Only the first chunk was consumed:

```text
sequence                     0
is_final                     false
samples                      7,680
first chunk latency          1.883153 s
```

The iterator was then closed:

```text
close duration               0.000552 s
allocated before close       5,527,718,912 B
allocated after close        5,502,608,384 B
```

The same backend immediately completed a normal one-shot TTS request afterward:

```text
generation                   3.945931 s
samples                      84,480
sample rate                  48,000 Hz
channels                     1
```

This is evidence that consumer cancellation propagates to the upstream generator and leaves the measured backend reusable.

## Offline/no-download proof

A real voice-clone stream was repeated with:

```text
VOXCPM_OFFLINE=1
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
fresh empty HF_HOME
HF endpoint -> loopback blackhole
HTTP/HTTPS proxy -> loopback blackhole
```

Result:

```text
HF_HOME files after run      0
local_files_only             true
native stream completed      true
valid WAV                    true
OOM counters                 0
```

This proves the measured run did not require remote model acquisition. It is not a network-namespace isolation claim.

## Independent WAV validation

Every completed stream was concatenated into one project `AudioResult` and written as a validation WAV only after stream completion.

| Output | Frames | Duration | SHA-256 |
| --- | ---: | ---: | --- |
| Standard stream | 99,840 | 2.08 s | `b85ed975135aaac99db73cda5f98f4a59d64a2d54953d12328058eb889fe379d` |
| Design stream | 92,160 | 1.92 s | `d9f93864b11dcca55c177930cfcf2dfdc7e36fafd5e1ff7903291a12529c70c4` |
| Clone stream | 115,200 | 2.40 s | `0bd58807b3e6bbbf20f72a884c42a5851fcf39661a0ec9be2d9f1a6c5f0d3858` |
| Continuation stream | 99,840 | 2.08 s | `feda434243e7a6d9e4cbd0448a8e32811f7ada68e8fea3cbcdbd162e6b112fdd` |
| Offline clone stream | 115,200 | 2.40 s | `d79ff31e561038d89cb631858406dc3adb6a19e7a75c8953968363e4a85a90ab` |

Independent Python `wave` readback confirmed mono, PCM16, 48 kHz structure and non-degenerate signals for all outputs.

Generated WAV files are not committed.

## Public report hygiene

All six public JSON reports were scanned for:

- absolute Kaggle paths;
- qualification workspace paths;
- raw voice-design instruction;
- raw reference transcript;
- raw continuation target text.

The scan returned zero hits.

## M7 gates

```text
M7_PRE_IMPLEMENTATION_AUDIT=PASS
M7_UPSTREAM_STREAMING_EXISTS_GATE=PASS
M7_UPSTREAM_STREAMING_REPEATABILITY_GATE=PASS
M7_STREAM_VS_ONESHOT_CHARACTERIZATION=PASS
M7_NATIVE_REASSEMBLY_EQUIVALENCE=PASS_TOLERANCE
M7_PUBLIC_STREAM_CONTRACT_COMPATIBILITY=PASS_WITH_CLARIFICATION
M7_NATIVE_STREAM_MAPPING_GATE=PASS
M7_NATIVE_NOT_BUFFERED_GATE=PASS
M7_STREAM_SEQUENCE_GATE=PASS
M7_FINAL_CHUNK_SEMANTICS_GATE=PASS
M7_LAZY_STREAM_ITERATION_GATE=PASS
M7_STREAM_ERROR_NORMALIZATION_GATE=PASS
M7_STREAM_REFERENCE_REDACTION_GATE=PASS
M7_STREAM_LIFECYCLE_GATE=PASS
M7_REAL_BACKEND_STREAM_CAPABILITY_GATE=PASS
M7_HTTP_REMAINS_OUT_OF_SCOPE_GATE=PASS
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

## Security and repository hygiene

- No generated WAV is added to Git.
- No model weight is added to Git.
- Public reports omit absolute model/reference paths.
- No tag or release is created.
- VERSION remains 1.0.0.

## Scope boundary

M7 qualifies only the measured backend-native streaming paths on one Tesla T4.

It does not qualify HTTP, REST, SSE, WebSocket, browser playback, network media framing, workers, WorkerPool, scheduler, admission control, IPC, batching, T4x2, multi-GPU, `torch.compile`, `optimize=True`, quantization, denoiser, production deployment, or a public release.

The evidence commit and its final exact-SHA CI run are the final M7 authority.
