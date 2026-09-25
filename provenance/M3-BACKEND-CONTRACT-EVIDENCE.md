# M3 Backend Contract Evidence

## Status

- UTC evidence snapshot: `2026-09-25T15:31:16Z`
- Repository: `OpenBMB-VoxCPM2-Inference`
- Branch: `main`
- M3 starting SHA: `ed265570041ed67b5f1f04bf57b66835fc0fb64a`
- Implementation SHA: `1be5e340e29a76e23fe10a07521bdd532f5b50c8`
- Implementation CI run: `36154845384` (`https://github.com/dangkhoa2016/OpenBMB-VoxCPM2-Inference/actions/runs/36154845384`), `PASS` on exact SHA `1be5e340e29a76e23fe10a07521bdd532f5b50c8`
- Implementation CI matrix: Python `3.10` `PASS`, Python `3.11` `PASS`, Python `3.12` `PASS`
- Evidence follow-up SHA: `PENDING_EVIDENCE_COMMIT`
- Python (host): `3.12.13`
- Python (isolated verification): `3.10.12`, `3.12.13`
- Python `3.11` is not installed in this session environment; it is covered only by the CI matrix.
- Version: `1.0.0` (unchanged)
- Source lock SHA-256: `10097c137461308dfb2693b1bf53f2fdff3deb8508ff1add2f1d3047ecae367a`
- Source-lock upstream commit: `f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69`
- M2 gate state at M3 start: `M2_PORTABLE_MODEL_RESOLVER=PASS`

## M2 baseline before M3 implementation

- `python -m pytest -q`: `PASS`, `149 passed`
- `ruff check voxcpm_runtime deploy tests scripts`: `PASS`
- `python -m compileall -q voxcpm_runtime deploy tests scripts`: `PASS`
- Isolated `uv` environment (`/tmp/opencode/m3-baseline-uv`): `PASS`, `149 passed`, `uv pip check` `PASS`

## New M3 files

```text
voxcpm_runtime/backend.py
voxcpm_runtime/backend_types.py
voxcpm_runtime/errors.py
voxcpm_runtime/fake_backend.py
tests/test_backend_contract.py
tests/test_backend_errors.py
tests/test_fake_backend.py
docs/BACKEND-CONTRACT.md
docs/BACKEND-CONTRACT.vi.md
provenance/M3-BACKEND-CONTRACT-EVIDENCE.md
```

Modified:

```text
README.md
README.vi.md
.github/workflows/ci.yml
```

## Contract surface

- Protocol `InferenceBackend`: `PASS`
- Operations: `load`, `synthesize`, `design`, `clone`, `continue_audio`, `stream`, `close`
- Project-owned types: `BackendInfo`, `AudioResult`, `AudioChunk`
- Project-owned requests: `SpeechRequest`, `VoiceDesignRequest`, `CloneRequest`, `ContinuationRequest`, `StreamRequest`
- Upstream VoxCPM types or parameters in the contract: `NONE`
- Runtime dependency count: `0` (`project.dependencies = []`)

## Fake sample format used for tests

```text
sample_rate_hz = 48000
channels = 1
sample_count = 480
sample_type = Python binary64 float
digest_input = sha256 over b"".join(struct.pack("<d", sample) for sample in samples)
golden_sha256 = 75b074a436b54ec3947272220ab2bb35223c712ea43985ce3610cf512387cd7e
```

- The waveform is a deterministic function of the canonical JSON request, so no RNG is used.
- No binary audio fixture is committed; the golden digest is compared against an in-memory synthesis.
- `48 kHz` and `fake/cpu-test` device strings are contract-scope placeholders, recorded in metadata as `sample_rate_scope=test/fake contract only`.

## Determinism and streaming evidence

- Same request, same instance: `PASS`
- Same request, fresh instance: `PASS`
- Different text/operation/reference input produces deterministically different output: `PASS`
- Samples finite and non-empty: `PASS`
- Streaming chunk boundaries deterministic: `PASS`
- Streaming reassembly equals the one-shot result for every request shape: `PASS`
- Chunk-size matrix (including 1, non-divisor, and larger-than-signal sizes): `PASS`
- Golden digest reproduced on Python `3.10.12` and `3.12.13`: `PASS`

## Isolation evidence

- Model required: `NO`
- GPU required: `NO`
- Network attempt count under the socket-deny harness (`socket.create_connection`, `socket.getaddrinfo`, `socket.socket.connect`, `socket.socket.connect_ex`, `socket.socket.sendto`): `0`
- Upstream `voxcpm` imports in M3 runtime: `0`
- `torch` / `transformers` / `huggingface_hub` / `numpy` imports in `voxcpm_runtime`: `0`
- `ModelResolver` or `/kaggle/input` coupling in M3 core modules: `0`
- Network URL, socket call, or private absolute path in M3 core modules: `0`
- `PytorchVoxCPMBackend`, FastAPI/server, worker, scheduler, GPU logic, or benchmark code: `NONE`
- M4 work: `NONE`

## Normalized error evidence

- Hierarchy: `BackendError`, `BackendStateError`, `BackendRequestError`, `BackendUnsupportedError`, `BackendExecutionError`
- `normalize_backend_error()` converts unknown backend failures into `BackendExecutionError` with a safe `code`, public `message`, `retryable=False`, and immutable scalar-only `details`
- Safe identifiers only: exception type and operation name are token-validated with a fallback; raw messages, tracebacks, paths, URLs, and tokens are not carried over
- Error details are sorted and exposed through a read-only mapping; non-scalar or non-finite detail values are rejected at construction
- Failure injection covers every operation: `load`, `synthesize`, `design`, `clone`, `continue_audio`, `stream`, `close`: `PASS`
- Raw private injected detail (`m3-private-backend-detail-placeholder`) absent from normalized output: `PASS`
- Exception chaining suppressed (`__context__ is None`) for normalized failures: `PASS`
- Pre-load generation fails with `backend_not_loaded`: `PASS`
- Post-close generation fails with `backend_closed`: `PASS`
- Reopen after close rejected: `PASS`
- Unknown request type rejected with `invalid_backend_request`: `PASS`
- Unicode surrogate text rejected at request construction: `PASS`

## Local gates

- `python -m pytest -q`: `PASS`, `226 passed in 14.29s`
- M3 target tests: `PASS`, `77 passed in 0.40s` (16 contract, 38 errors, 23 fake backend)
- M3 target tests on Python `3.10.12`: `PASS`, `77 passed in 0.44s`
- Full suite in an isolated Python `3.12.13` environment: `PASS`, `226 passed in 2.72s`
- Full suite in an isolated Python `3.10.12` environment: `PASS`, `226 passed in 2.09s`
- `ruff check voxcpm_runtime deploy tests scripts`: `PASS`
- `python -m compileall -q voxcpm_runtime deploy tests scripts`: `PASS`
- `git diff --check`: `PASS`
- CI determinism smoke script (executed locally): `PASS`, `M3_FAKE_BACKEND_SMOKE=PASS`
- CI forbidden-import scan (executed locally): `PASS`, `M3_FORBIDDEN_IMPORT_SCAN=PASS`
- CI workflow YAML parse: `PASS`
- Tracked model-weight scan: `PASS`
- Untracked/pending binary audio or weight files: `NONE`
- Token-pattern scan of M3 sources, tests, docs, and CI: `PASS`
- Isolated `uv pip check` (Python `3.10.12` and `3.12.13`): `PASS`
- Host `python -m pip check`: `FAIL` only for pre-existing Kaggle environment packages (`bigframes`, `google-adk`, `google-colab`, `dopamine-rl`, `moviepy`); this project has no runtime dependencies, matching the recorded M2 environment condition
- Implementation exact-SHA CI: `PASS`, run `36154845384` on SHA `1be5e340e29a76e23fe10a07521bdd532f5b50c8`

## Repository integrity

- `VERSION` = `1.0.0`
- Source lock changed: `NO`
- Model weights tracked: `NO`
- Real backend added: `NO`
- M4 work added: `NO`
- Release or tag created: `NO`
- M0/M1/M2 history modified: `NO`

## Scope notes

- This evidence qualifies the project-owned backend contract, the deterministic fake backend, and error normalization.
- It does not qualify real VoxCPM2 loading, CPU inference, real sample rate, voice design, cloning, continuation, real streaming, GPU/T4 behavior, memory sufficiency, real-time factor, or audio quality. Those require a later real-model milestone.
- `M4` is not authorized by the M3 runbook and is not started in this session.

## Gate decision

```text
M3_BACKEND_CONTRACT_GATE=PASS
M3_FAKE_BACKEND_GATE=PASS
M3_ERROR_NORMALIZATION_GATE=PASS
M3_UPSTREAM_ISOLATION_GATE=PASS
M3_BACKEND_ABSTRACTION=PASS
```

The gate is bound to implementation commit `1be5e340e29a76e23fe10a07521bdd532f5b50c8` and its exact-SHA CI run `36154845384`. The follow-up evidence commit carries documentation only and is verified by the final exact-SHA CI run.
