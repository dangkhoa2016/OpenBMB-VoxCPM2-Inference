# M1 configuration, device, and doctor evidence

- Evidence timestamp: `2026-09-25T10:57:39Z`
- Repository: `https://github.com/dangkhoa2016/OpenBMB-VoxCPM2-Inference.git`
- Branch: `main`
- Starting SHA: `8b5aca0b4f281fbbc50a021a264d4e193c476ed4`
- Candidate implementation commit: `pending`
- Remote CI for the candidate: `pending`
- Python: `3.12.13`
- Optional system PyTorch: `2.10.0+cpu`; isolated `.venv` PyTorch: `null`
- Real host inventory: CPU logical `4`; CPU physical `2`; host RAM `33659379712` bytes; CUDA available `false`; GPU count `0`
- Local install: `PASS`, editable install and `pip check`
- Local syntax/metadata checks: `PASS`, Python compilation, TOML parsing, and JSON parsing
- Lint: `PASS`, `ruff check voxcpm_runtime tests`
- Tests: `PASS`, `.venv/bin/python -m pytest -q` (`77 passed`)
- M0 regression: `PASS`, included in the full test suite
- Doctor default: `PASS`, exit `0`, effective device `cpu`, worker count `1`
- Doctor explicit CPU: `PASS`, exit `0`, effective device `cpu`, worker count `1`
- Doctor explicit CUDA: `PASS` negative case, exit `2`, no CPU fallback
- Invalid device: `PASS` negative case, exit `2`
- Malformed GPU list: `PASS` negative case, exit `2`
- Doctor JSON validation: `PASS`, all smoke outputs parsed as JSON
- API-token redaction: `PASS`, dummy token absent from stdout and stderr
- Synthetic CUDA policy tests: `PASS`, explicit CPU isolation, explicit CUDA failure without CUDA, deterministic auto selection, ordered GPU selection, worker limits, direct plan invariants, and GPU memory alias validation
- Redaction hardening: `PASS`, nested visible-device mapping values, unknown mapping strings, configured paths, API tokens, and invalid-config diagnostics do not expose secret values
- Source lock: `PASS`, `f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69`
- Version authorities: `PASS`, `VERSION`, `pyproject.toml`, and `voxcpm_runtime.__version__` are all `1.0.0`
- Tracked-weight scan: `PASS`, no model-weight or blocked binary files
- Large-file scan: `PASS`, no accidental file over 20 MiB outside ignored environments
- Hard-coded Kaggle core-path scan: `PASS`, no hard-coded Kaggle input/working path in reusable runtime, tests, CI, or package configuration
- High-confidence secret scan: `PASS`, no private key, cloud token, Hugging Face token, or GitHub token pattern
- M2/model-loading scan: `PASS`, no ModelResolver, inference backend, model import, or model-loading implementation
- Package artifact check: `PASS`, wheel/sdist build completed; target wheel install exposes the `voxcpm-doctor` entry point and source-lock report
- Working-tree review: `PASS`, only intended M1 files are modified or untracked; generated caches and build artifacts were removed before staging
- Candidate status: `HOLD` only for remote CI verification; local M1 validation is complete
- Next authorized action: stage and review the complete M1 diff, commit, push, and verify CI for the exact pushed SHA

The isolated environment emitted a non-fatal pre-existing `sitecustomize` warning about a missing `wrapt` module; commands exited successfully. No model weights were downloaded, staged, or committed.
