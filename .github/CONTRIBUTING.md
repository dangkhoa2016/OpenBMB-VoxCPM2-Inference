# Contributing

> Language / Ngôn ngữ: **English** | [Tiếng Việt](CONTRIBUTING.vi.md)

Thank you for contributing to OpenBMB-VoxCPM2-Inference.

## Principles

Changes should remain:

- focused and reviewable;
- reproducible;
- explicit about CPU, single-GPU, or T4x2 claim impact;
- fail-closed when hardware, model, authentication, or runtime contracts do not match.

Do not weaken a qualification gate only to make a demo pass.

## Development validation

Run at least:

    python -m compileall -q voxcpm_runtime deploy tests scripts
    python -m pytest -q
    python -m ruff check .
    git diff --check

For Kaggle-demo changes, also run:

    python -m pytest -q tests/test_kaggle_production_demo.py

CPU/static PASS is not evidence of successful live T4x2 inference.

## Documentation policy

When a public Markdown document has an English/Vietnamese pair, update both files in the same change. Keep a language switcher near the top of paired community documents.

## Runtime and provenance boundaries

State clearly whether a change affects:

- upstream source/model identity;
- CPU/CUDA execution profiles;
- worker scheduling or API behavior;
- authentication, timeout, or resource limits;
- Kaggle notebook behavior;
- Docker portability;
- public qualification claims.

Do not commit model weights, secrets, tokens, runtime caches, generated evidence archives, or personal credentials.

## Pull requests

Describe what changed and why, validation commands actually run, whether the change is CPU/static only or includes new T4x2 evidence, compatibility/security/provenance impact, and documentation or release impact.
