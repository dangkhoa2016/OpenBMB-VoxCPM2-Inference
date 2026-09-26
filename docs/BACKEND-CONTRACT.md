# M3 backend contract

M3 freezes a project-owned boundary for future inference implementations. `InferenceBackend` is the only execution abstraction that later workers may use. M3 does not connect this boundary to `ModelResolver`, `DeviceManager`, a real VoxCPM package, model weights, workers, scheduling, or an HTTP API.

The contract is defined without PyTorch, NumPy, Transformers, audio-library, FastAPI, or upstream VoxCPM types.

## Operations

| Operation | Contract | Result |
| --- | --- | --- |
| `load()` | Move a backend from created to loaded state | `BackendInfo` |
| `synthesize(request)` | Generate speech from text | `AudioResult` |
| `design(request)` | Generate speech with a project-owned voice instruction | `AudioResult` |
| `clone(request)` | Generate speech conditioned on a reference-audio descriptor | `AudioResult` |
| `continue_audio(request)` | Continue from reference audio and its transcript | `AudioResult` |
| `stream(request)` | Generate deterministic audio chunks without HTTP framing | `Iterator[AudioChunk]` |
| `close()` | Move a backend to closed state | `None` |

The pinned upstream source at commit `f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69` was inspected for semantics only. Ordinary text generation maps to `SpeechRequest`; textual voice design maps to `VoiceDesignRequest`; reference-audio cloning maps to `CloneRequest`; and prefix-audio continuation with its transcript maps to `ContinuationRequest`. M3 does not expose upstream argument names, classes, prompt caches, tensors, or implementation flags.

## Request types

All request types are immutable project dataclasses and contain only standard-library values.

- `SpeechRequest(text)` contains non-blank target text.
- `VoiceDesignRequest(text, instruction)` contains non-blank target text and a non-blank voice-design instruction.
- `CloneRequest(text, reference_audio)` contains target text and an `AudioReference`.
- `ContinuationRequest(text, reference_audio, reference_transcript)` contains target text, an `AudioReference`, and a non-blank transcript for the audio prefix.
- `StreamRequest(request)` wraps exactly one of those one-shot request types so streaming preserves the same operation semantics.

`AudioReference(local_path)` is only a descriptor. M3 does not open, decode, resample, fetch, or validate the existence of the referenced file. A later worker boundary may transfer large audio through files, but acquisition is not part of this contract.

## Audio types

`AudioResult` contains:

```text
samples: tuple[float, ...]
sample_rate_hz: positive int
channels: positive int
metadata: immutable sorted scalar pairs
```

`AudioChunk` adds a zero-based `sequence` and `is_final` flag. Samples are interleaved by channel in the order represented by `channels`; the fake backend produces mono audio. The sample container is an immutable tuple of finite Python floats, not NumPy, Torch, WAV, or an HTTP media type.

The fake-only output is 48 kHz mono with 480 samples. Its 48 kHz rate is a `test/fake contract only` choice and does not claim anything about the native VoxCPM2 sample rate. Fake samples are Python binary64 floats. Contract tests serialize them in little-endian IEEE-754 binary64 order (`struct.pack("<d", sample)`) before taking SHA-256.

## Lifecycle

The lifecycle is explicit:

```text
created -> load() -> loaded -> close() -> closed
```

`load()` and `close()` are idempotent while their state permits them. Generation before `load()` raises `BackendStateError` with code `backend_not_loaded`. Use after `close()` or `load()` after close raises `BackendStateError` with code `backend_closed`. A closed backend never silently reopens.

## Error normalization

The project error hierarchy is:

```text
BackendError
├── BackendStateError
├── BackendRequestError
├── BackendUnsupportedError
└ BackendExecutionError
```

Every error has a stable machine-readable code, a safe public message, a retryability flag, and immutable scalar details. `to_dict()` returns a deterministic safe envelope containing only `code`, `details`, `message`, and `retryable`.

`normalize_backend_error()` preserves an existing `BackendError`. An unknown exception becomes `BackendExecutionError` with only the safe operation and sanitized exception type in details. It does not copy the raw exception message, traceback, filesystem path, URL, token, or upstream exception object.

## Fake determinism and isolation

`FakeVoxCPMBackend` implements every operation with in-memory deterministic audio. It does not load a model, touch `ModelResolver` or the Kaggle adapter, use CUDA, read environment variables, open files, access the network, sleep, use wall-clock time, use unseeded randomness, or call Python `hash()`.

The fake canonicalizes the operation and project request fields as sorted UTF-8 JSON, hashes the bytes with SHA-256, and derives bounded binary64 samples. Identical requests produce identical results across repeated calls and fresh instances. Different operation names, text, instructions, reference descriptors, or transcripts affect the deterministic output.

For the deterministic fake backend, streaming splits the same full waveform into bounded, non-empty chunks: sequence values are contiguous, exactly one final chunk is emitted, and concatenating chunk samples equals the equivalent fake one-shot result exactly. This exact one-shot equality is a fake-backend guarantee, not a universal requirement for real native streaming implementations. A real backend must preserve request semantics, chunk ordering, non-empty samples, one final chunk, and safe project-owned types, while separately documenting measured native-stream versus one-shot numerical equivalence.

`FakeFailurePlan` is disabled by default. When explicitly configured, a private fake runtime error is normalized at the public boundary, and its private message does not appear in the public error.

## What M3 proves

A passing fake backend proves only the stable project contract, deterministic behavior, normalized errors, lifecycle behavior, and hermetic execution boundary.

It does not prove that:

- real VoxCPM2 loads;
- CPU, CUDA, T4, multi-GPU, RAM, or VRAM constraints are satisfied;
- the native sample rate is qualified;
- real voice design, cloning, continuation, or streaming works;
- audio quality or real-time performance is acceptable.

Real model loading and runtime qualification are outside M3. The next authorized milestone after a completed M3 closeout is M4 real CPU runtime investigation, which is not started by this contract.
