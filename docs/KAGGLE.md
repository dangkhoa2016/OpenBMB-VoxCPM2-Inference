# Kaggle production demo

## Scope

Kaggle demo qualification provides a thin Kaggle notebook that reproduces the qualified VoxCPM2 API runtime from a fresh workspace.

Notebook:

```text
notebooks/kaggle-production-demo.ipynb
```

The notebook is intentionally orchestration-only. Runtime behavior remains in the project package and `scripts/kaggle_production_demo.py`; the notebook does not duplicate the scheduler, worker, backend, or API implementation.

## Kaggle setup

For the full Kaggle demo qualification matrix:

1. Create a new Kaggle Notebook.
2. Set **Accelerator** to **GPU T4 x2**.
3. Attach the Kaggle model **dangkhoa2016/openbmb-voxcpm2**.
4. Enable Internet for source/dependency bootstrap.
5. Upload or import `notebooks/kaggle-production-demo.ipynb`.
6. Choose an execution profile before **Run All** by setting `VOXCPM_DEMO_PROFILE` in the environment. If it is not set, the notebook uses `auto`.

Canonical profiles:

```text
cpu
cuda-single
cuda-replica
auto
```

A T4x2 notebook can exercise all four profiles. The `cpu` profile explicitly uses CPU even while the Kaggle accelerator remains enabled.

## Fresh-workspace contract

Every Run All execution removes and recreates its selected demo workspace. The notebook then:

1. clones this project into the new workspace;
2. checks out `VOXCPM_DEMO_REF`;
3. reads the pinned upstream source commit from `provenance/voxcpm-source-lock.json`;
4. clones and checks out that exact upstream commit;
5. creates a new virtual environment using Kaggle's existing CUDA/PyTorch stack;
6. installs project/API and upstream dependencies;
7. verifies the attached Kaggle model in offline mode;
8. starts the selected execution profile;
9. waits for `/readyz`;
10. sends a real authenticated `POST /v1/tts` request;
11. validates the returned 48 kHz mono WAV;
12. shuts down the server and checks that the API port and GPU process state return to baseline.

The notebook does not read or reuse earlier T4x2 qualification/execution-profile qualification virtual environments, logs, generated audio, or project checkouts.

## Source and model network boundary

Internet is used during bootstrap to clone public source and install Python dependencies.

After bootstrap, model runtime is local/offline:

```text
VOXCPM_OFFLINE=1
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_ENDPOINT=http://127.0.0.1:9
HTTP_PROXY=http://127.0.0.1:9
HTTPS_PROXY=http://127.0.0.1:9
ALL_PROXY=http://127.0.0.1:9
```

Loopback traffic remains allowed so the notebook can call the local FastAPI server.

The model is discovered through the Kaggle deployment adapter rather than a hard-coded model leaf in reusable runtime code.

## Reproducible ref

For qualification or review, use an immutable project SHA:

```bash
export VOXCPM_DEMO_REF=<project-commit-sha>
```

The reproducible public demo is bound to the immutable `v1.0.0` release tag rather than a moving branch.

Using `main` is convenient for development but does not provide the same immutable evidence binding.

## Real Kaggle qualification matrix

All four profiles were executed through the notebook with fresh, distinct workspaces on one Kaggle T4x2 session.

| Profile | Resolved topology | Ready workers | Startup | TTS request | Result |
| --- | --- | ---: | ---: | ---: | --- |
| `auto` | CUDA GPUs 0,1 / 2 workers | 2 | 81.053 s | 5.994 s | HTTP 200 |
| `cuda-single` | CUDA GPU 0 / 1 worker | 1 | 33.041 s | 3.933 s | HTTP 200 |
| `cuda-replica` | CUDA GPUs 0,1 / 2 workers | 2 | 62.572 s | 5.360 s | HTTP 200 |
| `cpu` | CPU / 1 worker | 1 | 28.586 s | 28.886 s | HTTP 200 |

Every response was a valid 48 kHz mono WAV. These timings are observations from the qualification runs, not benchmark guarantees. Formal benchmark characterization belongs to a later qualification work.

For every run:

- model identity was `openbmb/VoxCPM2`;
- model revision was `32279effe8c19989596f05d353d1447f51d9e915`;
- mirror config SHA-256 was `405f0dcd92f7feba6011ed4eac5c8d4f74cba9712f07fd5cfa3063bbdd95402c`;
- runtime model acquisition was offline/blackholed;
- cleanup returned GPU compute-process state to baseline;
- port 8090 was closed;
- cgroup OOM and OOM-kill counters stayed zero.

## Evidence files

The runner writes a structured JSON evidence file under the selected fresh workspace:

```text
<workspace>/evidence/kaggle-<profile>-evidence.json
```

Generated evidence/log files remain runtime artifacts and are not committed. The public qualification summary is stored in:

```text
`provenance/QUALIFICATION.md`
```

## Boundaries

Kaggle demo qualification qualifies a fresh Kaggle production-style demo and Run All reproducibility for the four execution-profile qualification profiles on the measured environment.

Kaggle demo qualification does not qualify:

- a permanent production service;
- autoscaling;
- automatic worker respawn;
- tensor parallelism or model sharding;
- all Kaggle accelerator types;
- benchmark guarantees;
- model training or fine-tuning.
