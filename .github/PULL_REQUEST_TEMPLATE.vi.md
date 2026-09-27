# Pull Request

> Language / Ngôn ngữ: [English](PULL_REQUEST_TEMPLATE.md) | **Tiếng Việt**

## Tóm tắt

<!-- Thay đổi gì và vì sao? Giữ PR tập trung vào một mục tiêu có thể review rõ ràng. -->

## Phạm vi

- [ ] Runtime / backend
- [ ] Worker scheduling / execution profile
- [ ] API / security
- [ ] Kaggle notebook / public demo
- [ ] Docker / deployment
- [ ] Test / CI
- [ ] Tài liệu / repository metadata
- [ ] Khác

## Validation đã chạy

- [ ] python -m compileall -q voxcpm_runtime deploy tests scripts
- [ ] các pytest test liên quan
- [ ] python -m ruff check .
- [ ] git diff --check
- [ ] kiểm tra cặp English/Vietnamese khi paired Markdown thay đổi
- [ ] không thêm credential, model weights, runtime cache hoặc generated evidence archive

## Tác động runtime và qualification

<!-- Nêu rõ thay đổi chỉ CPU/static hay có single-GPU/T4x2 evidence mới. Giải thích tác động requalification nếu có. -->

## Tương thích / bảo mật / provenance

<!-- Ghi rõ tác động source/model identity, dependency, API compatibility, security hoặc release. -->
