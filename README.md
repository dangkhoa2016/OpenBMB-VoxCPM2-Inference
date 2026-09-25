# OpenBMB-VoxCPM2-Inference

Portable inference engineering for the OpenBMB VoxCPM2 model family.

## Status

This project is in early development toward its first `v1.0.0` release. The current repository baseline provides verified source provenance, an installable Python package shell, a minimal import test, and CPU-only CI. It does not yet load model weights or provide inference, an API, streaming, scheduling, or deployment support. It is not production-ready.

## Scope

The intended architecture is portable across CPU, a single GPU, and independent multi-GPU replicas. Those execution modes are future targets and will be enabled only after later measurement and qualification. Kaggle is one deployment and qualification target, not the core architecture.

## Model and attribution

Model weights are not stored in this repository. The reference model is [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2), and the reference implementation is [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM).

This is an independent engineering project. It is not an official OpenBMB release, and no affiliation is claimed. See `provenance/SOURCE-PROVENANCE.md` for the locked source revision and license findings.

## Development

The bootstrap supports Python 3.10 through 3.12.

```bash
python -m pip install -e .
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The baseline tests do not download or execute the VoxCPM2 model and do not require a GPU.

M0 rejects tracked `.bin` files along with model-weight formats. This conservative policy may be refined in a later evidence-backed milestone.

## License

Apache-2.0. See `LICENSE`.
