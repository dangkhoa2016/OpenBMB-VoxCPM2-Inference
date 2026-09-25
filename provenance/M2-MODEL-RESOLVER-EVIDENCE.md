# M2 Model Resolver Evidence

## Status

- UTC evidence snapshot: `2026-09-25T14:30:26Z`
- Repository: `OpenBMB-VoxCPM2-Inference`
- Branch: `main`
- Original M2 implementation starting SHA: `e02b22f6252a20da507ff981501c5cc5e92f3a9e`
- Implementation SHA: `e50ee55a2f9b9004b6719a66b21d9428c3fa17e8`
- Evidence follow-up SHA: `4aba91978182d15beab468bcb7ee8ca67b6acd23`
- Closeout starting SHA: `4aba91978182d15beab468bcb7ee8ca67b6acd23`
- Python: `3.12.13`
- Version: `1.0.0`
- Source lock SHA-256: `10097c137461308dfb2693b1bf53f2fdff3deb8508ff1add2f1d3047ecae367a`
- Source-lock upstream commit: `f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69`

## Local gates

- `python -m pytest -q`: `PASS`, `149 passed in 13.84s`
- Offline remote-acquirer negative tests: `PASS`, `2 passed in 0.04s`
- `ruff check voxcpm_runtime deploy tests scripts`: `PASS`
- `python -m compileall -q voxcpm_runtime deploy tests scripts`: `PASS`
- `git diff --check`: `PASS`
- Source lock working-tree diff: `PASS`, unchanged
- Tracked model-weight scan: `PASS`
- Process-level network-deny harness: `PASS`, `0` attempted calls
- Wheel and sdist build: `PASS` using setuptools `84.0.0`
- Wheel content/entry-point smoke: `PASS` in an offline `uv` environment
- Isolated package `pip check`: `PASS`
- Host `python -m pip check`: `FAIL` only for pre-existing environment packages (`bigframes`, `google-adk`, `google-colab`, `dopamine-rl`, `moviepy`); this project has no runtime dependencies
- Full implementation CI: `PASS`, run `36134661011` (`https://github.com/dangkhoa2016/OpenBMB-VoxCPM2-Inference/actions/runs/36134661011`), exact SHA `e50ee55a2f9b9004b6719a66b21d9428c3fa17e8`
- Evidence follow-up CI: `PASS`, run `36134943817` (`https://github.com/dangkhoa2016/OpenBMB-VoxCPM2-Inference/actions/runs/36134943817`), exact SHA `4aba91978182d15beab468bcb7ee8ca67b6acd23`

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
- Static local-resolution scan: `PASS`; no `requests`, `httpx`, `aiohttp`, `huggingface_hub`, Kaggle downloader, `curl`/`wget`, or `from_pretrained` path
- No tensor deserialization, model construction, or inference path: `PASS`
- Runtime code unchanged: `PASS`
- Source lock unchanged: `PASS`

## Architecture and required-file evidence

The pinned real mirror was discovered by the deployment adapter at the deployment-specific path:

```text
/kaggle/input/models/dangkhoa2016/openbmb-voxcpm2/pytorch/default/1
```

The path is metadata only and is not hard-coded in the generic resolver. The canonical run did not set `VOXCPM_MODEL_PATH`. The resolver verified:

- `config.json` top-level `architecture`: `voxcpm2`
- `config.json` SHA-256: `405f0dcd92f7feba6011ed4eac5c8d4f74cba9712f07fd5cfa3063bbdd95402c`
- Main weight group: `model.safetensors`
- AudioVAE weight group: `audiovae.pth`
- Tokenizer files: `tokenizer.json`, `tokenizer_config.json`
- Mirror manifest revision: `32279effe8c19989596f05d353d1447f51d9e915`
- Mirror handle: `dangkhoa2016/openbmb-voxcpm2/pyTorch/default`

The required-file rule accepts exactly one non-empty artifact from each main-weight and AudioVAE group, and does not hash or deserialize weights. The tokenizer check is evidence-based on the pinned `tokenizer.json` BPE version `1.0` format and the tokenizer-class metadata; the real inventory also contains `special_tokens_map.json` and `tokenization_voxcpm2.py`.

## Canonical Kaggle application-level offline gate

The M2 offline invariant is model-resolution network independence, not host-wide network isolation.

Historical host-level observation:

- raw TCP connectivity to external addresses was observed;
- the prior corrective run observed successful connections to `1.1.1.1:443` and `8.8.8.8:53`;
- this is retained as context;
- it is not an M2 failure condition;
- the host network may remain reachable, and this evidence does not claim host Internet isolation.

Host Internet isolation: `NOT REQUIRED FOR M2`

Canonical application-level qualification:

- `VOXCPM_OFFLINE=1`
- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- real Kaggle mirror discovered by the deployment adapter
- no `VOXCPM_MODEL_PATH` shortcut
- `source_kind=deployment-mount`
- `model_id=openbmb/VoxCPM2`
- `architecture=voxcpm2`
- config SHA-256: `405f0dcd92f7feba6011ed4eac5c8d4f74cba9712f07fd5cfa3063bbdd95402c`
- revision: `32279effe8c19989596f05d353d1447f51d9e915`
- required artifacts: `PASS`
- `file_count=10`
- `total_bytes=4960734297`
- process network-deny harness: `PASS`
- patched network entry points: `socket.create_connection`, `socket.getaddrinfo`, `urllib.request.urlopen`
- process network attempts: `0`
- offline remote-acquirer negative tests: `PASS`, `2 passed`
- `operation=metadata-only`
- `remote_acquisition_count=0`
- `inference_performed=false`
- exit code: `0`

Application-level model network access: `BLOCKED / NOT USED`

The process-scoped proof is the M2 invariant. The application-level offline flags and the network-deny harness demonstrate that this resolver run did not initiate remote model acquisition or network activity; they do not claim that the host has no Internet connection.

## Gate decision

```text
M2_PORTABLE_CODE_GATE=PASS
M2_APPLICATION_OFFLINE_GATE=PASS
M2_KAGGLE_OFFLINE_GATE=PASS
M2_PORTABLE_MODEL_RESOLVER=PASS
```

`M2_KAGGLE_OFFLINE_GATE=PASS` means real Kaggle application-level offline model resolution. It does not mean host-wide Internet isolation. Runtime code and the source lock are unchanged. M3 is not authorized.
