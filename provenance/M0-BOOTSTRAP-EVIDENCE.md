# M0 bootstrap evidence

- UTC timestamp: `2026-09-25T09:34:44Z`
- Repository remote: `https://github.com/dangkhoa2016/OpenBMB-VoxCPM2-Inference.git`
- Branch: `main`
- Pre-commit candidate state: unborn branch with the complete M0 candidate staged; no commit existed before this bootstrap.
- Python: `3.12.13`
- Test command: `.venv/bin/python -m pytest -q`
- Test result: `PASS`, 2 tests passed.
- Syntax and metadata checks: Python compilation, JSON parsing, TOML parsing, and `pip check` passed.
- VERSION result: `PASS`, exact bytes are `1.0.0\n`.
- Tracked-weight scan: `PASS`, no tracked model-weight or blocked binary extension.
- Large-file scan: `PASS`, no working-tree file exceeded 20 MiB outside `.git` and `.venv`.
- Inherited-behavior scan: `PASS`, no unrelated runtime/product behavior was found in tracked files.
- Hard-coded Kaggle path scan: `PASS`, no `/kaggle/input` or `/kaggle/working` path in reusable code, tests, CI, or package configuration.
- High-confidence secret scan: `PASS`, no token or private-key pattern was found in tracked files.
- Upstream repository: `https://github.com/OpenBMB/VoxCPM`
- Exact upstream source SHA: `f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69`
- License verification: `PASS`, Apache-2.0 in upstream package metadata and license text; no upstream NOTICE file was present.
- Reference model: `openbmb/VoxCPM2`, model API revision `32279effe8c19989596f05d353d1447f51d9e915`, model-card license `apache-2.0`.
- Source lock: created with the exact verified upstream code SHA.
- Local M0 gate: `PASS`.
- Push and baseline CI: `PENDING` until the M0 commit is created and pushed.
- Final M0 gate at report time: `HOLD`, pending push and baseline CI for the exact commit.
- Next authorized action: stop after M0 unless a new instruction authorizes another milestone.

The Kaggle shell emitted a non-fatal pre-existing `sitecustomize` warning about a missing `wrapt` module while invoking Python. All test and validation commands exited successfully; remote CI provides an independent clean-run verification.
