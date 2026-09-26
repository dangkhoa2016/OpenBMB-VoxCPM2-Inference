# M4 Real CPU Runtime Investigation Evidence

## Status

- UTC evidence snapshot: `2026-09-26T01:39:58Z`
- Repository: `OpenBMB-VoxCPM2-Inference`
- Branch: `main`
- M4 starting SHA: `d2035b7c6af84dc91c6e2f11d976a8f0dffa9b46`
- Implementation SHA: `9959150770b65a030faf1cb32a3b0ed753966ea3`
- Implementation CI run: `36209158464` (`https://github.com/dangkhoa2016/OpenBMB-VoxCPM2-Inference/actions/runs/36209158464`), `PASS` on exact SHA `9959150770b65a030faf1cb32a3b0ed753966ea3`
- Implementation CI matrix: Python `3.10` `PASS`, Python `3.11` `PASS`, Python `3.12` `PASS`
- Evidence follow-up SHA: `PENDING_EVIDENCE_COMMIT`
- Version: `1.0.0` (unchanged)
- Source lock SHA-256: `10097c137461308dfb2693b1bf53f2fdff3deb8508ff1add2f1d3047ecae367a` (unchanged)
- M3 gate state at M4 start: `M3_BACKEND_ABSTRACTION=PASS`

## Upstream authority

- Upstream repository: `https://github.com/OpenBMB/VoxCPM`
- Pinned upstream commit: `f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69`
- Installed upstream distribution: `voxcpm 2.0.3.post33+gf772e498a`
- Upstream license: `Apache-2.0`
- Upstream checkout was detached at the pinned commit: `PASS`
- Upstream import in the isolated qualification environment: `PASS`
- `pip check` in the isolated qualification environment: `PASS`
- Upstream license is `Apache-2.0`, so no dependency was added to this project: `PASS`

## Model identity

- Model id: `openbmb/VoxCPM2`
- Model revision: `32279effe8c19989596f05d353d1447f51d9e915` (bound by the M2 mirror authority record)
- `config.json` SHA-256: `405f0dcd92f7feba6011ed4eac5c8d4f74cba9712f07fd5cfa3063bbdd95402c`
- Architecture: `voxcpm2`
- File count: `10`
- Total model bytes: `4,960,734,297` (`4.62 GiB`)
- M2 mirror verification (`voxcpm-verify-model`, `inference_performed=false`): `PASS`, `M4_MODEL_IDENTITY=PASS`
- Main weight `model.safetensors`: `4,580,080,592 B`
- AudioVAE weight `audiovae.pth`: `376,951,122 B`
- Model weights committed to this repository: `NO`
- Model download during any M4 run: `NONE` (see offline proof)

The canonical M4 run resolves the model through an explicitly configured path, so the M4 report's own
`revision` field is `null` by construction and reports `source_kind=explicit-path`. Model identity and
revision are therefore bound to the M2 authority record, which is the documented source of truth for
the mirror revision and the config digest. Both digests above were independently re-derived from the
model files during this session.

## Host and environment

```text
platform                 Linux-6.12.90+-x86_64-with-glibc2.35
python                   3.12.13 (CPython)
cpu logical              4
cpu physical             2
host total ram           33,659,379,712 B (31.348 GiB)
cgroup memory limit      32,212,254,720 B (30.000 GiB)
gpu count                0
nvidia-smi               absent
torch                    2.14.0+cpu
torchaudio               2.11.0+cpu
```

- Host facts captured before dependency changes through the M1 `DeviceManager`: `PASS`
- CPU-only host: `PASS`, `cuda_available=false`, `cuda_probe_status=cuda_unavailable`

## Dependency versions in the qualification environment

```text
voxcpm           2.0.3.post33+gf772e498a
torch            2.14.0+cpu
torchaudio       2.11.0+cpu
transformers     5.17.0
huggingface-hub  1.33.0
numpy            2.5.3
safetensors      0.8.0
librosa          1.0.0
einops           0.8.2
pydantic         2.13.5
tqdm             4.70.1
```

- The project itself declares `project.dependencies = []`: `PASS`
- The ML stack lives only in a separate qualification environment: `PASS`
- A pre-existing Kaggle `sitecustomize` warning about a missing `wrapt` module appears on every
  interpreter start in this host image. It is unrelated to this project, does not affect `pip` or the
  runtime, and was not treated as a failure.

## Implementation

New files:

```text
voxcpm_runtime/pytorch_backend.py
voxcpm_runtime/wav_io.py
voxcpm_runtime/runtime_metrics.py
scripts/generate.py
tests/test_pytorch_backend.py
tests/test_wav_io.py
tests/test_runtime_metrics.py
docs/REAL-CPU-RUNTIME.md
docs/REAL-CPU-RUNTIME.vi.md
```

Modified:

```text
pyproject.toml            (voxcpm-generate console entry point)
.github/workflows/ci.yml  (two new M4 gates)
README.md
README.vi.md
```

Import boundary:

- `pytorch_backend.py` is the only module that reaches the upstream package: `PASS`
- Upstream package is imported exactly once, via `importlib.import_module("voxcpm")`, inside the load
  path: `PASS`
- Module-level imports of `torch`, `numpy`, `transformers`, or `voxcpm` anywhere in
  `voxcpm_runtime`: `0`
- Unexpected runtime imports in any other `voxcpm_runtime` module: `0`
- M3 core modules still import no upstream execution code: `PASS`
- The M3 forbidden-import CI gate was kept strict for the contract core and a new gate,
  `M4_LAZY_UPSTREAM_IMPORT_SCAN`, was added to enforce the M4 boundary explicitly.

CPU policy:

- Upstream `optimize` is forced to `False` on CPU: `PASS`
- A non-CPU optimization request is rejected with
  `BackendUnsupportedError(code=optimization_semantics_not_qualified)`: `PASS`
- Sample rate is read from the loaded upstream model at runtime, not hard-coded: `PASS`
- Real sample rate observed: `48,000 Hz`

## Real CPU runs

Three real runs were executed against the real local model, each launched with
`CUDA_VISIBLE_DEVICES=""` and `VOXCPM_DEVICE=cpu`.

| Run | Operation | Exit | Load | Synthesize | Wall | External peak RSS |
| --- | --- | --- | --- | --- | --- | --- |
| Load-only probe | `load-only` | `0` | 78.57 s | n/a | 82.61 s | 11,595,091,968 B (10.799 GiB) |
| Canonical CPU TTS | `standard-tts` | `0` | 43.15 s | 100.66 s | 148.08 s | 11,616,759,808 B (10.819 GiB) |
| Offline-proof CPU TTS | `standard-tts` | `0` | 34.32 s | 101.38 s | 139.96 s | 11,717,046,272 B (10.912 GiB) |

Canonical text: `Xin chào từ VoxCPM2.` (`20` characters).

Memory detail, canonical run:

```text
rss before load          232,341,504 B (221.6 MiB)
rss after load         6,710,083,584 B (6.249 GiB)
rss after generation   6,895,919,104 B (6.422 GiB)
peak rss              11,616,759,808 B (10.819 GiB)
```

- Internal in-process telemetry and the external `/usr/bin/time -v` harness were both used, because
  internal telemetry cannot report after a hard OOM kill: `PASS`
- Generation is roughly `48.4x` slower than real time on 2 physical CPU cores in `bfloat16`. This is a
  recorded measurement, not a performance target, and no real-time claim is made.

Generated audio, canonical run:

```text
channels                1 (mono)
sample rate             48,000 Hz
sample width            2 bytes (PCM16)
frames                  99,840
duration                2.08 s
file size               199,724 B
sha256                  5a7608fe4af915c33dacc84f2193b3fade8dcf5054d3fe70e356a296a4e3df53
peak abs pcm16          32,342
rms pcm16               5,918.34
```

- Audio samples finite and non-empty: `PASS`
- Audio is non-silent speech, not a degenerate constant or clipped square wave: `PASS`
- Generated WAV committed to this repository: `NO`
- M4 does not require a committed audio fixture: `PASS`

WAV validation was performed twice and independently: once by the project validator during the run,
and once afterwards by the Python standard library `wave` module reading the file back from disk. Both
agreed on channels, sample width, sample rate, frame count, and SHA-256: `PASS`.

## No-CUDA evidence

Recorded after `import torch`, without calling any CUDA memory-allocating API:

```text
execution_plan.effective_device        cpu
execution_plan.selected_gpu_indices    []
execution_plan.worker_count            1
CUDA_VISIBLE_DEVICES                   ""   (empty, set by the canonical launch)
VOXCPM_DEVICE                          cpu
torch.cuda.is_available()              false
torch.cuda.is_initialized()            false
torch.cuda.device_count()              0
model parameter device types           ["cpu"]
model buffer device types              ["cpu"]
model cpu_only                         true
```

- CUDA memory-allocating APIs were deliberately never called, because probing some of them can
  initialize a CUDA context and invalidate the proof: `PASS`
- No CUDA device appears in any model parameter or buffer after load: `PASS`

## No-download evidence

The offline-proof run repeated full synthesis with every network route blackholed:

```text
VOXCPM_OFFLINE=1, HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1
HF_HOME                = a fresh empty directory
HF_ENDPOINT            = http://127.0.0.1:1   (closed port)
http_proxy/https_proxy = http://127.0.0.1:1   (closed port)
HTTP_PROXY/HTTPS_PROXY = http://127.0.0.1:1
```

- Exit code: `0`
- Valid WAV produced: `PASS`
- Connection, proxy, or download errors logged: `0`
- Files present in `HF_HOME` after the run: `0`
- `local_files_only` in the report: `true`
- `unshare -n` was attempted for a stronger network-namespace proof but is not permitted in this
  container, so egress blackholing plus an empty `HF_HOME` was used instead. This is recorded as a
  limitation of the method, not as a stronger claim.

The canonical and offline-proof runs produced different WAV digests with identical frame counts.
Upstream sampling is stochastic. M4 asserts nothing about byte-for-byte reproducibility of real
synthesis.

## Memory and OOM evidence

- `memory.events` sampled before and after every run: `PASS`
- `oom`, `oom_kill`, `max`, `high`, `low` after all three real runs: all `0`
- Every run exit code: `0`
- Runs killed by a signal: `0`
- Genuine allocation failures raised: `0`
- OOM classification rule applied: a non-zero exit alone is never reported as an OOM; classification
  requires an increased `oom` or `oom_kill` counter, a genuine runtime allocation failure, or explicit
  environment OOM evidence
- Exit classification for this session: `no resource failure; all runs completed`

## 16 GB status

```text
M4_CPU_16GB_SUFFICIENCY=NOT_QUALIFIED
```

The measured peak RSS was `10.819 GiB`, which is below `16 GB`. However this host had a `30 GiB`
cgroup limit and `31.348 GiB` of RAM, and **no 16 GB constraint was imposed on the run**. A process
observed to fit in `31 GiB` has not been shown to fit in `16 GB`. Qualifying this gate requires
re-running the same load and synthesis under an enforced `16 GB` limit with a cgroup OOM harness.

This is a measurement, not a support claim.

## Local gates

- `python -m pytest -q`: `PASS`, `334 passed in 21.54s`
- Baseline before M4: `PASS`, `226 passed`
- New M4 tests: `108` (60 real backend, 23 WAV, 25 runtime metrics)
- M3 target suites: `PASS`, `77 passed` (contract, errors, fake backend)
- `ruff check voxcpm_runtime deploy tests scripts`: `PASS`
- `python -m compileall -q voxcpm_runtime deploy tests scripts`: `PASS`
- `git diff --check`: `PASS`
- `git diff --cached --check`: `PASS`
- CI workflow YAML parse: `PASS`
- CI forbidden-import scan (executed locally): `PASS`, `M3_FORBIDDEN_IMPORT_SCAN=PASS`
- CI M4 lazy-upstream scan (executed locally): `PASS`, `M4_LAZY_UPSTREAM_IMPORT_SCAN=PASS`
- CI generate-CLI smoke (executed locally): `PASS`, `M4_GENERATE_CLI_SMOKE=PASS`
- Report redaction audit over the real report: `PASS`, no absolute path, home directory, mount path,
  site-packages path, or URL
- Tracked model-weight and audio scan (`*.safetensors`, `*.bin`, `*.pth`, `*.pt`, `*.ckpt`, `*.wav`,
  `*.flac`, `*.mp3`, `*.ogg`, `*.npy`, `*.npz`): `PASS`, none tracked
- Untracked or pending binary audio or weight files: `NONE`
- Secret and token pattern scan over all changed files: `PASS`
- Hard-coded Kaggle mount leaf in reusable core (`voxcpm_runtime`): `NONE`
- Implementation exact-SHA CI: `PASS`, run `36209158464` on SHA
  `9959150770b65a030faf1cb32a3b0ed753966ea3`
- Ordinary CI loaded or downloaded the real model: `NO`

M4 test coverage includes the lazy-import boundary, the `optimize=False` CPU mapping, upstream
positional-argument handling, sample-rate passthrough from the model, rejection of empty and
non-finite waveforms, lifecycle and state errors, every unsupported operation, WAV atomicity,
clipping, SHA-256, RIFF validation, RSS peak tracking, and report redaction.

## Documentation

- `docs/REAL-CPU-RUNTIME.md` and `docs/REAL-CPU-RUNTIME.vi.md` created: `PASS`
- Heading parity between the English and Vietnamese documents: `PASS`, `14` headings each
- Measured-value parity across all shared code blocks: `PASS`, `9` blocks
- Measured-value parity across all shared table rows: `PASS`, `15` rows
- `README.md` and `README.vi.md` updated to state the M4 result conservatively: `PASS`

README and documentation deliberately do not claim 16 GB support, real-time CPU inference, T4 or GPU
support, streaming, or full VoxCPM2 feature support.

## Repository integrity

- `VERSION` = `1.0.0`, unchanged: `PASS`
- Source lock changed: `NO`
- `provenance/voxcpm-source-lock.json` SHA-256 unchanged: `PASS`
- M0-M3 history modified: `NO`
- Model weights tracked: `NO`
- Generated real WAV tracked: `NO`
- Release or tag created: `NO`
- Version bump performed: `NO`
- Project runtime dependency count: `0`
- M5 work started: `NO`

## Scope notes

- This evidence qualifies one real local-model CPU load and one real CPU standard-TTS synthesis path
  on one specific Kaggle host.
- It does not qualify 16 GB or any other memory floor, real-time or streaming inference, GPU or T4
  execution, voice design, voice cloning, audio continuation, any other VoxCPM2 feature, audio
  quality, speaker fidelity, multi-worker or scheduler behavior, or any API surface.
- Real-model evidence is environment-specific and is not reproduced by ordinary GitHub CI.
- `M5` and later milestones are not started by this evidence.

## Gate decision

```text
M4_REAL_BACKEND_CODE_GATE=PASS
M4_REAL_CPU_LOAD_GATE=PASS
M4_REAL_CPU_TTS_GATE=PASS
M4_WAV_VALIDATION_GATE=PASS
M4_CPU_NO_CUDA_GATE=PASS
M4_CPU_MEMORY_MEASUREMENT=PASS
M4_REAL_CPU_RUNTIME=PASS
M4_CPU_16GB_SUFFICIENCY=NOT_QUALIFIED
```

The gate is bound to implementation commit `9959150770b65a030faf1cb32a3b0ed753966ea3` and its
exact-SHA CI run `36209158464`. The follow-up evidence commit carries documentation only and is
verified by the final exact-SHA CI run.
