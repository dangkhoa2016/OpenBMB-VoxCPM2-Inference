# Release checklist

This checklist is for the first stable `v1.0.0` publication. Completing documentation alone does not authorize a release.

## Source and repository

- [ ] `VERSION` remains `1.0.0`.
- [ ] `main` is clean and synchronized with `origin/main`.
- [ ] no model weights or generated audio are tracked.
- [ ] no real credentials are tracked.
- [ ] all release-blocking CI is green on the exact release-candidate SHA.

## Documentation

- [ ] README contains no internal development codename language.
- [ ] README states the project is independent and not an official OpenBMB release.
- [ ] README states model weights are not stored in Git.
- [ ] model acquisition/mount instructions are reproducible.
- [ ] hardware claims match measured evidence.
- [ ] API/security limits match the implementation.
- [ ] no dead internal links.
- [ ] paired English/Vietnamese public docs remain aligned.

## Qualification

- [ ] CPU claim is backed by measured CPU evidence.
- [ ] single-T4 claim is backed by measured T4 evidence.
- [ ] T4x2 replica claim is backed by measured two-worker evidence.
- [ ] streaming claim is backed by measured evidence.
- [ ] clone/continuation claims match qualified paths.
- [ ] Docker claims stay within the measured container boundary.
- [ ] fresh Kaggle notebook reproduction is verified.
- [ ] the final acceptance matrix is frozen and passed.

## Publication

- [ ] prepare release notes for `v1.0.0`.
- [ ] create the annotated `v1.0.0` tag only after final qualification.
- [ ] create the GitHub Release from the exact accepted SHA.
- [ ] link the canonical Kaggle notebook.
- [ ] publish only measured hardware/performance claims.
- [ ] perform post-publication link verification.

Do not bump beyond `v1.0.0` merely because development required corrective commits.
