# Source provenance

## Project identity

- Project: `dangkhoa2016/OpenBMB-VoxCPM2-Inference`
- Verification time: `2026-09-25T09:20:42Z`
- This repository is an independent engineering project around the upstream OpenBMB model. It is not an official OpenBMB release, and no affiliation is claimed.

## Upstream code authority

- Repository: `https://github.com/OpenBMB/VoxCPM`
- Default branch discovered during verification: `main`
- Exact selected commit: `f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69`
- Commit timestamp: `2026-09-02T20:12:35+08:00`
- Package metadata: `pyproject.toml` declares `license = "Apache-2.0"` and `requires-python = ">=3.10"`.
- License file: the repository `LICENSE` is the Apache License, Version 2.0. Its appendix carries `Copyright OpenBMB`.
- NOTICE: no `NOTICE`, `NOTICE.txt`, or equivalent notice file was present in the resolved tree.
- Source lock: `provenance/voxcpm-source-lock.json` pins the exact code commit and verified license identifier.
- Version authority: the exact source commit is authoritative. The upstream package derives its distribution version with `setuptools_scm`; shallow checkouts can therefore report a different generated package version when tag/history metadata is absent, even though the checked-out source commit is identical.

The following files were inspected at the locked commit:

- `pyproject.toml`
- `README.md`
- `LICENSE`
- `src/voxcpm/__init__.py`
- `src/voxcpm/core.py`
- `src/voxcpm/cli.py`
- `src/voxcpm/model/voxcpm2.py`

The upstream API exposes `VoxCPM` from the `voxcpm` package, `VoxCPM.from_pretrained`, `VoxCPM2Model`, and a VoxCPM2-first CLI. This inspection establishes the later integration authority only. bootstrap validation does not import, vendor, or execute upstream runtime code.

## Upstream model identity

- Reference model: `openbmb/VoxCPM2`
- Authoritative source: Hugging Face model repository `https://huggingface.co/openbmb/VoxCPM2`
- Model repository revision observed by the API: `32279effe8c19989596f05d353d1447f51d9e915`
- Model-card metadata and text identify the license as `apache-2.0`.
- No model weights were downloaded, copied, or committed during bootstrap validation.

The code authority, model identity, and this engineering repository are separate provenance records.

## Kaggle qualification mirror

- Intended later qualification mirror: `dangkhoa2016/openbmb-voxcpm2`
- The unauthenticated Hugging Face API probe returned HTTP 401 during verification. Its availability, contents, revision, and license were not independently established and must be revalidated before any later use.
- The mirror is not treated as upstream authorship or as the source-code authority.

## License handling for this repository

This repository's original code and documentation are licensed under the MIT License, Copyright (c) 2026 Đăng Khoa <i.am@dangkhoa.dev>. The pinned upstream VoxCPM source and the reference VoxCPM2 model remain separate third-party works under their own verified license terms (Apache-2.0 at the revisions recorded above). This repository does not relicense upstream code, model weights, or other third-party material. Any reuse or redistribution of upstream material must preserve its applicable copyright, license, attribution, and NOTICE obligations.
