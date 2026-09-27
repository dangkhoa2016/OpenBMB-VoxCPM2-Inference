# Qualification summary

This document is the public qualification record for OpenBMB-VoxCPM2-Inference. It intentionally summarizes externally meaningful capabilities and measured environments instead of exposing the project's internal development ledger.

## Source and model identity

- Upstream source: OpenBMB VoxCPM at commit `f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69`.
- Model: `openbmb/VoxCPM2`.
- Model revision: `32279effe8c19989596f05d353d1447f51d9e915`.
- Model config SHA-256: `405f0dcd92f7feba6011ed4eac5c8d4f74cba9712f07fd5cfa3063bbdd95402c`.
- Kaggle mirror used by the demo flow: `dangkhoa2016/openbmb-voxcpm2`.
- Runtime model acquisition is blocked after bootstrap; measured inference uses the mounted/local model copy.

See `SOURCE-PROVENANCE.md` and `voxcpm-source-lock.json` for the pinned upstream source contract.

## Measured runtime matrix

The project has been exercised on the following measured environments.

| Surface | Measured environment | Result |
|---|---|---|
| CPU TTS | Kaggle/Linux host with roughly 32 GiB RAM | PASS |
| Single-GPU TTS | NVIDIA Tesla T4 16 GiB | PASS |
| Two-worker GPU runtime | NVIDIA Tesla T4 x2, one isolated worker/model replica per GPU | PASS |
| Voice design | Tesla T4 | PASS |
| Voice cloning | Tesla T4 | PASS |
| Audio continuation | Tesla T4 | PASS |
| Native streaming | Tesla T4 | PASS |
| Authenticated REST API | CPU, single-T4 and T4x2 qualified paths | PASS |
| Docker portability contract | GitHub-hosted Linux CPU image | PASS |

These are measured environments, not universal hardware guarantees.

## Representative measurements

Fresh release qualification recorded:

- CPU profile: 1 worker, HTTP 200, valid 48 kHz mono PCM16 WAV, clean shutdown, no OOM.
- Single T4 profile: 1 worker, HTTP 200, valid 48 kHz mono PCM16 WAV, worker memory in the measured request lifecycle around 5.2-5.6 GiB, clean shutdown.
- T4x2 replica profile: 2 workers, one worker per physical GPU, HTTP 200, valid 48 kHz mono PCM16 WAV, clean shutdown.
- Auto profile on T4x2: 2 workers; a cold start measured 139.940 seconds, which is why worker startup timeout is configurable and defaults to 300 seconds.
- Native streaming: 22 non-empty contiguous chunks in the retained qualification run, with a single final chunk and a valid 48 kHz mono validation WAV.
- Voice design, clone and continuation each produced valid 48 kHz mono WAV output in the measured single-T4 qualification.

Performance values are observations from specific measured runs, not service-level guarantees.

## API and scheduling behavior

The qualified runtime includes:

- bearer-authenticated FastAPI endpoints;
- health and readiness separation;
- bounded request admission;
- bounded active concurrency;
- queue timeout;
- request timeout;
- maximum inference duration;
- bounded streaming backpressure;
- cancellation propagation;
- worker crash containment;
- degraded readiness while at least one worker remains healthy;
- 503 readiness when no workers remain healthy;
- no silent CPU fallback when explicit CUDA execution fails.

The API parent remains model-free. GPU model ownership stays inside isolated worker processes.

## Security and resource boundaries

Automated tests cover:

- malformed and duplicate Authorization handling;
- secret redaction;
- non-reflective validation errors;
- request and queue timeout behavior;
- stream backpressure timeout;
- temporary-space admission;
- concurrent request ceilings;
- worker cleanup and orphan prevention.

The current HTTP surface is text-first. Reference/prompt audio operations are available through backend/CLI paths; this qualification summary does not claim multipart upload hardening for audio files.

## Kaggle showcase

The public notebook is designed for a fresh Kaggle workspace and documents:

- GPU T4 x2 selection;
- Add Input -> Models -> `dangkhoa2016/openbmb-voxcpm2`;
- automatic mounted-model discovery;
- English and Vietnamese prompts;
- two playable WAV outputs;
- a production API path using the selected execution profile.

The latest notebook code contains an explicit two-request T4x2 showcase path and automated contract tests. A fresh hosted Kaggle T4x2 Run All should be performed after the public-history rewrite before treating that latest notebook revision itself as new measured notebook evidence.

## Automated verification

The cleaned public tree passes:

```text
516 passed
1 skipped
```

The single skip is the root/filesystem-permission case where root bypasses ordinary read-permission semantics.

Ruff and `git diff --check` also pass on the cleaned public tree.

## Claim boundaries

This project does not claim:

- universal 16 GiB CPU compatibility;
- universal 16 GiB VRAM compatibility;
- support for every CUDA GPU/driver combination;
- tensor parallelism or model sharding;
- automatic failed-worker respawn;
- HA, SLA or autoscaling;
- real-model inference inside Docker;
- CUDA Docker qualification;
- semantic speaker-similarity guarantees;
- universal real-time latency guarantees.

The repository is an independent engineering project around OpenBMB VoxCPM2 and is not an official OpenBMB release.
