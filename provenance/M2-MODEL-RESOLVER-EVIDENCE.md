# M2 Model Resolver Evidence

## Status

- UTC evidence snapshot: `2026-09-25T12:24:44Z`
- Repository: `OpenBMB-VoxCPM2-Inference`
- Branch: `main`
- Starting SHA: `e02b22f6252a20da507ff981501c5cc5e92f3a9e`
- Implementation SHA: `e50ee55a2f9b9004b6719a66b21d9428c3fa17e8`
- Evidence follow-up SHA: `this evidence commit`
- Python: `3.12.13`
- Version: `1.0.0`
- Source lock SHA-256: `10097c137461308dfb2693b1bf53f2fdff3deb8508ff1add2f1d3047ecae367a`
- Source-lock upstream commit: `f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69`

## Local gates

- `python -m pytest -q`: `PASS`, `149 passed in 13.39s`
- `ruff check voxcpm_runtime deploy tests scripts`: `PASS`
- `python -m compileall -q voxcpm_runtime deploy tests scripts`: `PASS`
- `git diff --check`: `PASS`
- Source lock working-tree diff: `PASS`, unchanged
- Tracked model-weight scan: `PASS`
- Wheel and sdist build: `PASS` using setuptools `84.0.0`
- Wheel content/entry-point smoke: `PASS` in an offline `uv` environment
- Isolated package `pip check`: `PASS`
- Host `python -m pip check`: `FAIL` only for pre-existing environment packages (`bigframes`, `google-adk`, `google-colab`, `dopamine-rl`, `moviepy`); this project has no runtime dependencies
- Full implementation CI: `PASS`, run `36134661011` (`https://github.com/dangkhoa2016/OpenBMB-VoxCPM2-Inference/actions/runs/36134661011`), exact SHA `e50ee55a2f9b9004b6719a66b21d9428c3fa17e8`

## Portable resolver checks

- `RuntimeConfig` reuse: `PASS`
- Explicit-path precedence: `PASS`
- Invalid explicit path fail-closed: `PASS`
- Deployment precedence: `PASS`
- Local-cache fallback after invalid cache: `PASS`
- Remote source remains last: `PASS`
- Offline remote-call count: `PASS`, `0`
- Network-negative tests: `PASS`
- Wrong architecture and required artifact groups: `PASS`
- Tokenizer metadata and bounded JSON validation: `PASS`
- Bounded inventory and malformed-path handling: `PASS`
- Kaggle fake-root fixture discovery: `PASS`
- Malformed manifest does not hide a later valid candidate: `PASS`
- Core Kaggle-coupling scan: `PASS`
- No tensor/model-load implementation scan: `PASS`
- No M3/backend implementation scan: `PASS`
- No model weights committed: `PASS`

## Architecture and required-file evidence

The pinned real mirror is:

```text
/kaggle/input/models/dangkhoa2016/openbmb-voxcpm2/pytorch/default/1
```

The resolver verified:

- `config.json` top-level `architecture`: `voxcpm2`
- `config.json` SHA-256: `405f0dcd92f7feba6011ed4eac5c8d4f74cba9712f07fd5cfa3063bbdd95402c`
- Main weight group: `model.safetensors`
- AudioVAE weight group: `audiovae.pth`
- Tokenizer files: `tokenizer.json`, `tokenizer_config.json`
- Mirror manifest revision: `32279effe8c19989596f05d353d1447f51d9e915`
- Mirror handle: `dangkhoa2016/openbmb-voxcpm2/pyTorch/default`

The required-file rule accepts exactly one non-empty artifact from each main-weight and AudioVAE group, and does not hash or deserialize weights. The tokenizer check is evidence-based on the pinned `tokenizer.json` BPE version `1.0` format and the tokenizer-class metadata; the real inventory also contains `special_tokens_map.json` and `tokenization_voxcpm2.py`.

## Canonical Kaggle offline gate

The canonical verifier was run against the real mirror with:

```text
VOXCPM_OFFLINE=1
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

Observed result:

- Adapter discovery: `PASS`
- Resolved source kind: `deployment-mount`
- Architecture validation: `PASS`
- Required artifacts: `PASS`
- Inventory: `10` top-level files, `4,960,734,297` bytes total
- `remote_acquisition_count`: `0`
- `inference_performed`: `false`
- Exit code: `0`
- Internet disabled: `UNVERIFIED`

The application-level offline flags are not proof that the operating system network is disabled. The attempted namespace isolation command failed with `unshare: unshare failed: Operation not permitted`. Therefore this evidence does not claim a real Internet-OFF Kaggle run.

## Gate decision

```text
M2_PORTABLE_CODE_GATE=PASS
M2_KAGGLE_OFFLINE_GATE=HOLD
M2_PORTABLE_MODEL_RESOLVER=HOLD
```

The portable code gate is complete: local checks and exact-SHA CI passed. The canonical Kaggle gate remains `HOLD` because the operating system Internet-OFF prerequisite could not be verified; this evidence-only follow-up must also receive green CI. M3 is not authorized.
