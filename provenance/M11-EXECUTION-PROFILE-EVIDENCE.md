# M11 Execution Profile Evidence

## Status

M11 freezes the supported execution-profile names over the already-qualified runtime topology.

Implementation authority:

```text
base SHA                    aa35bd1bad44bb9970e37797eb77812b3518f76a
implementation SHA          47193a8fb4bcacbed8b94f166bf288d67461cc74
implementation CI run       36255061615
implementation CI result    success
VERSION                      1.0.0
```

No tag or release was created during M11.

## Frozen names

Canonical profiles:

```text
cpu
cuda-single
cuda-replica
auto
```

Accepted aliases:

```text
gpu       -> cuda-single
multi-gpu -> cuda-replica
```

Aliases normalize to canonical values before topology materialization.

## Topology semantics

`cpu` resolves to one CPU worker.

`cuda-single` requires usable CUDA and resolves to exactly one selected GPU and one worker. It does not silently fall back to CPU.

`cuda-replica` requires at least two usable GPUs and resolves to one independent worker/model replica per selected GPU. Worker count must match selected GPU count.

`auto` resolves deterministically from runtime-visible hardware:

```text
0 usable GPUs -> cpu / 1 worker
1 usable GPU  -> cuda / 1 worker
N usable GPUs -> cuda / N independent workers
```

Conflicting low-level settings are rejected rather than silently overridden. With `VOXCPM_PROFILE` unset, the previous `VOXCPM_DEVICE`, `VOXCPM_GPU_DEVICES`, and `VOXCPM_WORKERS` behavior remains available.

## Parent model isolation

The hardware probe used by API profile discovery executes in a short subprocess. The API parent process itself did not import Torch:

```text
torch before probe           false
runtime GPU probe            (0, 1)
torch after probe            false
torch after materialization  false
```

On the qualification T4x2 host:

```text
cpu           -> cpu / no GPU selection / 1 worker
cuda-single   -> cuda / GPU 0 / 1 worker
cuda-replica  -> cuda / GPUs 0,1 / 2 workers
auto           -> cuda / GPUs 0,1 / 2 workers
```

## Same-client real API matrix

The same HTTP endpoint, request body, authentication shape, and request ID were used without client-side changes for every canonical profile.

Request body:

```json
{"text":"Xin chào."}
```

Request ID:

```text
m11-profile-request
```

Measured functional results from this qualification run:

| Profile | HTTP | Workers observed | GPU ownership | Wall time | WAV |
| --- | ---: | ---: | --- | ---: | --- |
| `cuda-single` | 200 | 1 | GPU0 | 3.619 s | 48 kHz mono, 53,760 frames |
| `cuda-replica` | 200 | 2 | GPU0 + GPU1 | 3.632 s | 48 kHz mono, 53,760 frames |
| `auto` | 200 | 2 | GPU0 + GPU1 | 3.644 s | 48 kHz mono, 53,760 frames |
| `cpu` | 200 | 1 | none | 28.289 s | 48 kHz mono, 53,760 frames |

Each response contained 107,564 WAV bytes and 1.12 s of audio.

These timings are functional qualification observations only. M11 does not freeze benchmark numbers or publish a speedup claim; formal benchmark/resource qualification remains a later milestone.

## Lifecycle cleanup

After each profile run:

```text
port 8090 listeners          0
GPU compute processes        0
```

The qualification run finished with no GPU compute-process leak.

## Regression and CI

Local closeout before the implementation commit:

```text
pytest                        481 passed, 1 skipped
ruff                          PASS
compileall                    PASS
workflow YAML parse           PASS
git diff --check              PASS
```

Exact-SHA CI for `47193a8fb4bcacbed8b94f166bf288d67461cc74`:

```text
Bootstrap on Python 3.10     success
Bootstrap on Python 3.11     success
Bootstrap on Python 3.12     success
Verify M11 execution profiles success on all three jobs
```

## Scope exclusions

M11 freezes profile names and their mapping onto the existing runtime. It does not qualify:

- tensor parallelism;
- model sharding;
- automatic worker respawn;
- universal GPU compatibility;
- benchmark/resource envelopes;
- universal performance or scaling claims;
- production deployment;
- Kaggle `Run All` reproducibility.

## Gates

```text
M11_PROFILE_NAMES_GATE=PASS
M11_ALIAS_NORMALIZATION_GATE=PASS
M11_LEGACY_CONFIG_COMPATIBILITY_GATE=PASS
M11_CPU_PROFILE_GATE=PASS
M11_CUDA_SINGLE_PROFILE_GATE=PASS
M11_CUDA_REPLICA_PROFILE_GATE=PASS
M11_AUTO_PROFILE_GATE=PASS
M11_NO_SILENT_FALLBACK_GATE=PASS
M11_PARENT_MODEL_ISOLATION_GATE=PASS
M11_SAME_API_REQUEST_GATE=PASS
M11_REAL_PROFILE_MATRIX_GATE=PASS
M11_PROFILE_CLEANUP_GATE=PASS
M11_FULL_REGRESSION_GATE=PASS
M11_IMPLEMENTATION_EXACT_SHA_CI=PASS
```

The final evidence-commit exact-SHA CI is verified after this file is committed and pushed.
