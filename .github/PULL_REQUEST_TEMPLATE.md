# Pull Request

> Language / Ngôn ngữ: **English** | [Tiếng Việt](PULL_REQUEST_TEMPLATE.vi.md)

## Summary

<!-- What changed and why? Keep the PR focused on one reviewable purpose. -->

## Scope

- [ ] Runtime / backend
- [ ] Worker scheduling / execution profiles
- [ ] API / security
- [ ] Kaggle notebook / public demo
- [ ] Docker / deployment
- [ ] Tests / CI
- [ ] Documentation / repository metadata
- [ ] Other

## Validation performed

- [ ] python -m compileall -q voxcpm_runtime deploy tests scripts
- [ ] relevant pytest tests
- [ ] python -m ruff check .
- [ ] git diff --check
- [ ] English/Vietnamese pair checked when paired Markdown changed
- [ ] no credentials, model weights, runtime caches, or generated evidence archives added

## Runtime and qualification impact

<!-- State whether this is CPU/static only or includes new single-GPU/T4x2 evidence. Explain any requalification impact. -->

## Compatibility / security / provenance

<!-- Note source/model identity, dependency, API compatibility, security, or release implications. -->
