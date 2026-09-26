# Security hardening

This document describes the security qualification security posture for the qualified HTTP runtime. It is an engineering qualification record, not a security certification.

## Qualified HTTP surface

The security qualification public API surface is:

- `GET /healthz`
- `GET /readyz`
- `POST /v1/tts`
- `POST /v1/tts/stream`

The POST endpoints accept JSON text input. The current HTTP surface does not accept reference-audio or prompt-audio uploads.

## Authentication and external binding

Bearer authentication is required by default. security qualification retains constant-time token comparison and rejects missing credentials, wrong schemes, empty or incorrect bearer values, whitespace variants that do not exactly match the contract, oversized authorization values, and duplicate Authorization headers.

A non-loopback bind with authentication disabled remains fail-closed unless `VOXCPM_ALLOW_UNAUTHENTICATED_EXTERNAL=1` is explicitly set.

Validation failures use a normalized public error envelope and do not echo raw request inputs.

## Bounded admission and time limits

security qualification enforces the existing configuration contract:

- `VOXCPM_MAX_QUEUE_SIZE`: bounded pending queue.
- `VOXCPM_QUEUE_TIMEOUT_SECONDS`: expires requests waiting in the queue.
- `VOXCPM_REQUEST_TIMEOUT_SECONDS`: bounds one-shot HTTP waiting time and cancels timed-out work.
- `VOXCPM_MAX_INFERENCE_SECONDS`: terminates a worker whose active inference exceeds the configured ceiling.
- `VOXCPM_MAX_CONCURRENT_REQUESTS`: limits concurrently active requests even when more workers are ready.
- `VOXCPM_STREAM_BACKPRESSURE_TIMEOUT_SECONDS`: prevents a non-reading stream consumer from blocking the dispatcher indefinitely.

Requests beyond the pending queue limit receive the existing queue-full rejection instead of causing unbounded growth.

## Temporary storage guard

When `VOXCPM_MIN_TMP_FREE_BYTES` is configured, application creation checks free space on `VOXCPM_TMP_DIR`, or on the system temporary directory when no explicit directory is configured.

The runtime fails closed when the configured free-space floor is not met.

The current HTTP endpoints do not create reference-audio or prompt-audio temporary files. security qualification therefore does not claim upload-temp cleanup behavior that the present API does not exercise.

## Redaction and public errors

Central redaction covers URL-like values, absolute filesystem paths, home-directory paths, common token shapes, and bearer credential values.

API validation errors are normalized so framework validation details do not reflect raw input values back to the caller.

Diagnostics expose whether an API token is configured, never the token itself.

## Secret scanning

security qualification scanned tracked files for common token and private-key shapes. Token-shaped values found in the repository were confined to explicit test fixtures used to validate redaction. The same scan reported zero matching secret shapes outside `tests/`.

Real credentials must never be committed. Use environment variables or an external secret store.

## Dependency review

The project API extra pins:

```text
fastapi==0.136.1
uvicorn[standard]==0.46.0
httpx==0.28.1
```

The Kaggle qualification host contains unrelated preinstalled packages with `pip check` conflicts. Those host-image conflicts are not presented as project dependency failures.

The standard-library `venv` path on the qualification host could not bootstrap `ensurepip`, and neither `pip-audit` nor `safety` was installed. security qualification therefore does not claim that the dependency graph is vulnerability-free.

Docker qualification already records that the Docker runtime pins the upstream VoxCPM source commit but does not freeze every transitive runtime dependency. Exact transitive freezing remains a release-reproducibility decision.

## Audio validation boundary

Reference and prompt audio validation exists in backend and CLI paths from earlier backend and CLI work, but the security qualification HTTP surface is text-only. security qualification does not claim HTTP multipart audio content sniffing, malformed uploaded WAV rejection, uploaded reference byte or duration limits, uploaded prompt-audio duration limits, or temporary upload cleanup.

Those controls should be qualified when an HTTP audio-upload endpoint is introduced.

## Output-duration boundary

`VOXCPM_MAX_OUTPUT_SECONDS` remains part of the parsed configuration contract, but security qualification does not claim a qualified HTTP output-duration enforcement policy. Streaming responses begin before final duration is known, so a future policy must define deterministic truncation and error semantics before this knob can be advertised as an HTTP security guarantee.

## Operational limits

security qualification does not establish a security certification, complete vulnerability absence, DDoS resistance, multi-node rate limiting, autoscaling, HA/SLA guarantees, real-model Docker security qualification, or CUDA Docker security qualification.

See `provenance/QUALIFICATION.md` for concrete evidence and gate status.
