# Execution profiles

The runtime exposes four canonical execution profile names:

```text
cpu
cuda-single
cuda-replica
auto
```

Friendly aliases are accepted at configuration input:

```text
gpu       -> cuda-single
multi-gpu -> cuda-replica
```

Aliases normalize to the canonical name; diagnostics and documentation use the canonical form.

## Configuration

Set a profile with:

```bash
VOXCPM_PROFILE=cpu
VOXCPM_PROFILE=cuda-single
VOXCPM_PROFILE=cuda-replica
VOXCPM_PROFILE=auto
```

The profile layer materializes the existing `VOXCPM_DEVICE`, `VOXCPM_GPU_DEVICES`, and `VOXCPM_WORKERS` topology. It does not introduce a second scheduler or backend path.

### `cpu`

- forces CPU execution;
- uses exactly one worker;
- rejects GPU selection and worker oversubscription.

### `cuda-single`

- requires usable CUDA;
- selects exactly one GPU;
- uses exactly one worker;
- when no GPU list is supplied, selects the first runtime-visible GPU;
- never silently falls back to CPU.

### `cuda-replica`

- requires at least two usable GPUs;
- creates one independent worker/model replica per selected GPU;
- preserves the existing worker-per-GPU isolation contract;
- rejects a worker count that differs from the selected GPU count.

This is replica parallelism, not tensor parallelism or model sharding.

### `auto`

- probes CUDA availability outside the API parent process;
- no usable GPU -> CPU with one worker;
- one usable GPU -> CUDA with one worker;
- multiple usable GPUs -> one CUDA worker per visible GPU;
- respects explicit compatible device/GPU/worker constraints;
- rejects oversubscription or an explicit CUDA constraint when CUDA is unavailable.

The CUDA probe runs in a short subprocess. The API parent itself does not import Torch or VoxCPM during profile discovery.

## Backward compatibility

`VOXCPM_PROFILE` is optional. When it is unset, the existing low-level `VOXCPM_DEVICE`, `VOXCPM_GPU_DEVICES`, and `VOXCPM_WORKERS` contract remains unchanged.

Conflicting profile and low-level settings fail clearly rather than being silently overridden.

## API parity

The HTTP client contract is profile-independent. The same request body, authentication header, request ID, endpoint, and response format are used for every profile.

A real qualification run used the same request:

```json
{"text":"Xin chào."}
```

against `POST /v1/tts` under all four canonical profiles. All four returned HTTP 200 with a 48 kHz mono WAV. The measured timings from that run are evidence of functional parity only and are not published as benchmark guarantees.

## Scope

Execution profiles freeze supported topology names. They do not qualify:

- tensor parallelism;
- model sharding;
- automatic worker respawn;
- universal GPU compatibility;
- universal performance or scaling claims;
- production deployment.

See `provenance/M11-EXECUTION-PROFILE-EVIDENCE.md` for the qualification record.
