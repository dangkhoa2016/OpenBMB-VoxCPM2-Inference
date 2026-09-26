# M1 configuration and diagnostics

M1 provides a side-effect-free configuration loader, a host inventory, deterministic CPU/CUDA execution planning, and the `voxcpm-doctor` command. It does not download, resolve, load, or execute VoxCPM2.

## Configuration

`RuntimeConfig.from_env()` reads the following environment contract once. Blank optional values remain unset. No model path is opened and no directory is created while loading configuration.

### Model and backend

| Variable | Default | M1 behavior |
| --- | --- | --- |
| `VOXCPM_MODEL_PATH` | unset | Stored only |
| `VOXCPM_BACKEND` | `pytorch-voxcpm` | Validated enum |
| `VOXCPM_OFFLINE` | `1` | Parsed boolean |
| `VOXCPM_LOAD_DENOISER` | `0` | Parsed boolean |
| `VOXCPM_OPTIMIZE` | `auto` | Validated enum |
| `VOXCPM_UPSTREAM_REVISION` | unset | Informational provenance value |

### Device and workers

| Variable | Default | M1 behavior |
| --- | --- | --- |
| `VOXCPM_DEVICE` | `auto` | `auto`, `cpu`, or `cuda` |
| `VOXCPM_GPU_DEVICES` | `auto` | `auto` or comma-separated non-negative indices |
| `VOXCPM_WORKERS` | `auto` | `auto` or a positive integer |

GPU indices use PyTorch's runtime-visible namespace. Explicit order is preserved, for example `1,0`. Duplicate, negative, malformed, and out-of-range indices are rejected.

### API and request contract

M1 parses and stores the API and request-limit variables without starting a server or implementing authentication:

```text
VOXCPM_HOST=127.0.0.1
VOXCPM_PORT=8090
VOXCPM_API_TOKEN=
VOXCPM_REQUIRE_AUTH=1
VOXCPM_ALLOW_UNAUTHENTICATED_EXTERNAL=0
VOXCPM_MAX_TEXT_CHARS=
VOXCPM_MAX_REFERENCE_BYTES=
VOXCPM_MAX_REFERENCE_SECONDS=
VOXCPM_MAX_PROMPT_AUDIO_SECONDS=
VOXCPM_MAX_OUTPUT_SECONDS=
VOXCPM_MAX_QUEUE_SIZE=
VOXCPM_QUEUE_TIMEOUT_SECONDS=
VOXCPM_REQUEST_TIMEOUT_SECONDS=
VOXCPM_MAX_INFERENCE_SECONDS=
VOXCPM_MAX_CONCURRENT_REQUESTS=
```

The blank request-limit fields have no M1 numeric defaults.

M9 now implements the first FastAPI surface using this existing contract. The qualified M9 runtime enforces host/port, bearer-auth settings, explicit single-GPU selection, text-length limits, bounded queue capacity, and stream IPC chunk capacity. Universal request deadlines, max-inference enforcement, autoscaling, and multi-worker real runtime remain outside the M9 qualification unless documented separately in `docs/API-RUNTIME.md`.

### Streaming, storage, logging, and readiness

The loader also stores the declared streaming, output/storage, logging, and readiness values:

```text
VOXCPM_STREAM_FORMAT=pcm_s16le
VOXCPM_STREAM_SAMPLE_RATE=48000
VOXCPM_STREAM_CHANNELS=1
VOXCPM_STREAM_IPC_MAX_CHUNKS=
VOXCPM_STREAM_IPC_MAX_BYTES=
VOXCPM_STREAM_BACKPRESSURE_TIMEOUT_SECONDS=
VOXCPM_OUTPUT_DIR=
VOXCPM_TMP_DIR=
VOXCPM_CACHE_DIR=
VOXCPM_MIN_TMP_FREE_BYTES=
VOXCPM_LOG_LEVEL=INFO
VOXCPM_LOG_FORMAT=json
VOXCPM_LOG_FILE=
VOXCPM_READINESS_MODE=degraded
VOXCPM_QUALIFICATION_STRICT=0
```

These values do not enable streaming, storage, readiness endpoints, or inference in M1.

## Boolean and numeric parsing

Boolean values are case-insensitive. Accepted true values are `1`, `true`, `yes`, and `on`; accepted false values are `0`, `false`, `no`, and `off`. Other values fail with a configuration error.

Optional integer and float values are unset when missing or empty. Invalid values, negative quantities where a non-negative value is required, and non-finite floats fail clearly. No unspecified production limit is guessed.

## Execution policy

The policy is deterministic:

- `cpu` always resolves to CPU, even when CUDA hardware is visible. It selects no GPUs and defaults to one worker.
- `cuda` fails when no usable CUDA device is visible. It never falls back to CPU.
- `auto` resolves to CUDA when the runtime-visible CUDA inventory is usable, otherwise to CPU.
- For CUDA, the default worker count equals the number of selected GPUs. An explicit worker count above that number is rejected rather than silently capped.
- A selected GPU list must refer to visible runtime GPU indices.

The resolver operates on an injectable `HardwareInventory`, so CPU-only CI can test synthetic CUDA inventories without a GPU.

## `voxcpm-doctor`

Install the package and run:

```bash
voxcpm-doctor
VOXCPM_DEVICE=cpu voxcpm-doctor
```

The default output is deterministic JSON. It includes the redacted configuration, Python and optional package versions, host CPU/RAM facts, CUDA/GPU inventory, source-lock provenance, and the execution plan. A missing optional dependency is represented as `null` and does not by itself make the doctor fail.

On a CPU-only host:

```bash
VOXCPM_DEVICE=cuda voxcpm-doctor
```

returns a non-zero status with a clear error and no CPU fallback. Invalid device and GPU-list values also return non-zero without a traceback.

## Secret handling

`VOXCPM_API_TOKEN` is never emitted. Diagnostics expose only `api_token_configured: true` or `false`. Authorization headers, bearer tokens, and raw environment dumps are not produced.

## Scope boundary

M1 does not implement model resolution, model acquisition, model loading, inference, APIs, streaming, scheduling, worker processes, or deployment qualification.
