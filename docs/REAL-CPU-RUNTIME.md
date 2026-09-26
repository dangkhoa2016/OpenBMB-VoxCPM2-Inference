# Real CPU Runtime

This document records what M4 actually measured when the pinned upstream VoxCPM2 implementation was
loaded and executed on CPU against a real local model. It is an investigation report, not a support
commitment.

Read this first:

- A real local-model CPU load and a real CPU standard-TTS synthesis both succeeded on this host.
- The measured peak resident set was `10.819 GiB`. That number is **not** a 16 GB support claim. See
  [16 GB status](#16-gb-status).
- CPU synthesis here was roughly `48x` slower than real time. This is **not** a real-time CPU claim.
- The report path is environment-specific evidence. It does not qualify GPU, T4, voice design,
  cloning, continuation, streaming, batching, or worker isolation.

## Scope

M4 adds the first real execution path behind the M3 backend boundary:

| Area | State |
| --- | --- |
| Real local model load on CPU | Qualified on this host |
| Real standard TTS to WAV | Qualified on this host |
| WAV write and validation | Qualified |
| No-CUDA proof for the canonical run | Qualified |
| Voice design / cloning / continuation | Not implemented (M5+) |
| Streaming | Not implemented (M6+) |
| GPU / T4 / single-GPU runtime | Not qualified (M5) |
| API, scheduler, worker isolation, multi-GPU | Not started (M7-M10) |

## Architecture

`voxcpm_runtime` keeps a hard split between the dependency-free contract core and the real
implementation:

```text
voxcpm_runtime/backend.py        contract only, no upstream imports
voxcpm_runtime/backend_types.py  project-owned types only
voxcpm_runtime/errors.py         normalized error hierarchy
voxcpm_runtime/fake_backend.py   deterministic, hermetic, no model
voxcpm_runtime/pytorch_backend.py  the ONLY module that reaches upstream VoxCPM
voxcpm_runtime/wav_io.py         project-owned PCM16 WAV writer and validator
voxcpm_runtime/runtime_metrics.py  RSS, device, CUDA, and version telemetry
```

`pytorch_backend.py` imports the upstream package exactly once, through
`importlib.import_module("voxcpm")`, inside its load path. There is no module-level import of
`torch`, `numpy`, `transformers`, or `voxcpm` anywhere in `voxcpm_runtime`. This is enforced in CI
by a dedicated gate rather than left to convention.

The project package declares no runtime dependencies. Torch, torchaudio, transformers, and the
upstream package live in a separate qualification environment, so importing this project never
drags in the ML stack.

## CPU optimization policy

M4 forces upstream `optimize=False` on CPU. The upstream `optimize=True` path requires the
`torch.compile` stack and a real compilation investigation that belongs to a later milestone, so
this milestone never enables it.

A non-CPU optimization request is rejected deliberately rather than silently downgraded:

```text
BackendUnsupportedError
code = optimization_semantics_not_qualified
```

Silently running a different optimization regime would make timing and memory numbers
unfalsifiable, so the backend refuses instead.

## Using it

```bash
export VOXCPM_MODEL_PATH=/path/to/local/voxcpm2
export VOXCPM_OFFLINE=1
export CUDA_VISIBLE_DEVICES=""
export VOXCPM_DEVICE=cpu

# Load the real model, report device and memory facts, generate nothing.
voxcpm-generate --load-only --report load.json

# Real standard TTS to a mono PCM16 WAV plus a JSON runtime report.
voxcpm-generate \
  --text "Xin chào từ VoxCPM2." \
  --output out.wav \
  --report report.json
```

The canonical CPU qualification environment is always launched with both `CUDA_VISIBLE_DEVICES=""`
and `VOXCPM_DEVICE=cpu`.

## What the report contains

The JSON report is built for sharing. It records the effective device and execution plan, CUDA
availability and initialization state, upstream parameter and buffer device types, host and cgroup
memory facts, resident set size before load, after load, and after generation, peak RSS, phase
timings, the model identity and config digest, resolved dependency versions, WAV metadata and
SHA-256, and an independent WAV validation block.

Absolute filesystem paths, home directories, model mount locations, and URLs are redacted. The
report states that paths were withheld rather than silently omitting them, and it records that model
weights are never committed to the repository.

## Measured results

Host and environment for every number below:

```text
platform            Linux-6.12.90+-x86_64-with-glibc2.35
python              3.12.13 (CPython)
cpu                 4 logical, 2 physical
host ram            33,659,379,712 B (31.348 GiB)
cgroup limit        32,212,254,720 B (30.000 GiB)
torch               2.14.0+cpu
torchaudio          2.11.0+cpu
upstream package    2.0.3.post33+gf772e498a
upstream commit     f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69
model               openbmb/VoxCPM2
model revision      32279effe8c19989596f05d353d1447f51d9e915  (bound by M2 authority)
model config sha256 405f0dcd92f7feba6011ed4eac5c8d4f74cba9712f07fd5cfa3063bbdd95402c
model files         10 files, 4,960,734,297 B
```

Three real runs were performed against the model. Peak RSS is reported twice on purpose: once from
in-process telemetry, and once from an external `/usr/bin/time -v` harness, because internal
telemetry cannot report after a hard OOM kill.

| Run | Exit | Load | Synthesize | Wall | Peak RSS (external) |
| --- | --- | --- | --- | --- | --- |
| Load-only probe | `0` | 78.57 s | n/a | 82.61 s | 11,595,091,968 B (10.799 GiB) |
| Canonical CPU TTS | `0` | 43.15 s | 100.66 s | 148.08 s | 11,616,759,808 B (10.819 GiB) |
| Offline-proof CPU TTS | `0` | 34.32 s | 101.38 s | 139.96 s | 11,717,046,272 B (10.912 GiB) |

Memory detail for the canonical run:

```text
rss before load        232,341,504 B (221.6 MiB)
rss after load       6,710,083,584 B (6.249 GiB)
rss after generation 6,895,919,104 B (6.422 GiB)
peak rss            11,616,759,808 B (10.819 GiB)
```

Load dominates the resident set. The `4.62 GiB` model plus the AudioVAE accounts for most of the
step from `221.6 MiB` to `6.249 GiB`; the gap between steady-state RSS and peak RSS is transient
decode-time working memory, not additional stored weights.

Generated audio, canonical run:

```text
channels                1 (mono)
sample rate             48,000 Hz
sample width            2 bytes (PCM16)
frames                  99,840
duration                2.08 s
file size               199,724 B
wav sha256              5a7608fe4af915c33dacc84f2193b3fade8dcf5054d3fe70e356a296a4e3df53
peak abs pcm16          32,342
rms pcm16               5,918.34
```

The `48,000 Hz` rate is read from the loaded upstream model at runtime. It is not hard-coded in the
backend, so a different model revision reporting a different rate would be reported rather than
overwritten.

WAV validation is performed twice and independently: once by the project's validator during the
run, and once afterwards by the Python standard library `wave` module reading the file from disk.
Both agreed on channels, width, rate, frame count, and SHA-256.

### Synthesis is far from real time

Canonical `100.66 s` of synthesis produced `2.08 s` of audio, a real-time factor of about `48.4`.
This host has 2 physical CPU cores and the model runs in `bfloat16` on CPU, which is not a
throughput-optimized path.

This measurement is recorded to prevent an overclaim. It is not a performance target, and no
latency or real-time guarantee is offered.

## Proving no CUDA use

CUDA is not merely absent from the report; the canonical run was launched with every relevant
knob set to exclude it, and the model was then inspected after load:

```text
CUDA_VISIBLE_DEVICES            ""            (empty)
VOXCPM_DEVICE                   cpu
execution_plan.effective_device cpu
execution_plan.selected_gpu_indices   []
torch.cuda.is_available()       false
torch.cuda.is_initialized()     false
torch.cuda.device_count()       0
model parameter device types    ["cpu"]
model buffer device types       ["cpu"]
model cpu_only                  true
```

CUDA memory-allocating APIs were deliberately never called, because merely probing some of them can
initialize a CUDA context and would invalidate the proof.

The model path is `explicit-path` in the canonical run, so the report's own `revision` field is
`null` by construction. Model identity and revision are bound to the M2 authority record instead,
which is the documented source of truth for the mirror revision and config digest.

## Proving no download

The offline-proof run repeated full synthesis with every network route blackholed:

```text
VOXCPM_OFFLINE=1, HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1
HF_HOME                = a fresh empty directory
HF_ENDPOINT            = http://127.0.0.1:1   (closed port)
http_proxy/https_proxy = http://127.0.0.1:1   (closed port)
HTTP_PROXY/HTTPS_PROXY = http://127.0.0.1:1
```

The run exited `0`, produced a valid WAV, and logged no connection, proxy, or download error. After
the run, `HF_HOME` contained `0` files. Combined with `local_files_only=true` in the report, this
demonstrates that neither loading nor synthesis needs the network.

The canonical run and the offline-proof run produced different WAV digests with identical frame
counts. Upstream sampling is stochastic, which is expected; M4 asserts nothing about byte-for-byte
reproducibility of real synthesis.

## 16 GB status

```text
M4_CPU_16GB_SUFFICIENCY=NOT_QUALIFIED
```

The measured peak RSS was `10.819 GiB`, which is below `16 GB`. But this host had a `30 GiB` cgroup
limit and `31.348 GiB` of RAM, and **no 16 GB constraint was imposed on the run**. A process that
fits in `31 GiB` was never shown to fit in `16 GB`.

To qualify that gate honestly, the same load and synthesis would have to be re-run under an enforced
`16 GB` limit with an OOM harness watching the cgroup. Until then the number is a measurement, not a
support claim.

## Memory and OOM handling

Peak RSS is captured by two independent mechanisms:

1. In-process telemetry samples resident set size before load, after load, after generation, and
   tracks a high-water mark.
2. An external `/usr/bin/time -v` harness captures the child's own maximum RSS, wall clock, and exit
   status, and survives a hard kill.

The cgroup `memory.events` counters were sampled before and after every run. Across all three real
runs, `oom`, `oom_kill`, `max`, `high`, and `low` were all `0`, and every run exited `0`. No run was
killed, and no run raised an allocation failure.

A non-zero exit is not reported as an OOM on its own. Classification as an OOM requires evidence
such as an increased `oom` or `oom_kill` counter, a genuine allocation failure raised by the
runtime, or explicit OOM termination evidence from the environment. Otherwise the outcome is
recorded as an unexplained signal or resource termination.

## Testing

The real model is never loaded in normal CI. The test suite stubs the upstream import, so the whole
suite stays fast and hermetic.

```text
baseline before M4      226 passed
after M4                334 passed  (60 backend, 23 WAV, 25 metrics)
M3 target suites         77 passed  (contract, errors, fake backend)
ruff check              PASS
compileall              PASS
```

M4-specific tests cover the lazy-import boundary, the `optimize=False` CPU mapping, upstream
positional-argument handling, sample-rate passthrough from the model, non-finite and empty waveform
rejection, lifecycle and state errors, every unsupported operation, WAV atomicity, clipping,
SHA-256, RIFF validation, RSS peak tracking, and report redaction.

## What M4 does not qualify

- 16 GB or any other memory floor as a support claim
- Real-time, streaming, or interactive-latency CPU inference
- GPU, T4, single-GPU, or multi-GPU execution
- Voice design, voice cloning, audio continuation, or any other VoxCPM2 feature
- Audio quality, speaker fidelity, or intelligibility
- Multi-worker, multi-process, or scheduler behavior
- Any API surface

Those remain M5 and later, and each needs its own measurement and evidence.

## Related documents

- [`docs/BACKEND-CONTRACT.md`](BACKEND-CONTRACT.md) for the M3 contract boundary
- [`docs/CONFIGURATION.md`](CONFIGURATION.md) for the environment contract and device policy
- [`docs/MODEL-RESOLUTION.md`](MODEL-RESOLUTION.md) for the M2 resolver contract
- [`provenance/M4-REAL-CPU-RUNTIME-EVIDENCE.md`](../provenance/M4-REAL-CPU-RUNTIME-EVIDENCE.md) for the
  raw gate-by-gate evidence record
- `docs/REAL-CPU-RUNTIME.vi.md` for the Vietnamese edition of this document
