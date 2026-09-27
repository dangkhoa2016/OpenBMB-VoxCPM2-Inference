# Checklist phát hành

Checklist này dành cho lần public stable release `v1.0.0` đầu tiên. Hoàn thành documentation không tự động cho phép release.

## Source và repository

- [ ] `VERSION` vẫn là `1.0.0`.
- [ ] `main` sạch và đồng bộ với `origin/main`.
- [ ] không track model weight hay generated audio.
- [ ] không track credential thật.
- [ ] mọi release-blocking CI green trên exact release-candidate SHA.

## Documentation

- [ ] README không còn internal development codename language.
- [ ] README nêu rõ đây là project độc lập, không phải official OpenBMB release.
- [ ] README nêu rõ model weights không nằm trong Git.
- [ ] hướng dẫn model acquisition/mount có thể tái tạo.
- [ ] hardware claim khớp measured evidence.
- [ ] API/security limit khớp implementation.
- [ ] không có dead internal link.
- [ ] public docs EN/VI có cặp vẫn được giữ đồng bộ.

## Qualification

- [ ] CPU claim có measured CPU evidence.
- [ ] single-T4 claim có measured T4 evidence.
- [ ] T4x2 replica claim có measured two-worker evidence.
- [ ] streaming claim có measured evidence.
- [ ] clone/continuation claim khớp qualified path.
- [ ] Docker claim không vượt measured container boundary.
- [ ] fresh Kaggle notebook reproduction được verify.
- [ ] final acceptance matrix được freeze và pass.

## Publication

- [ ] chuẩn bị release notes cho `v1.0.0`.
- [ ] chỉ tạo annotated tag `v1.0.0` sau final qualification.
- [ ] tạo GitHub Release từ exact accepted SHA.
- [ ] link canonical Kaggle notebook.
- [ ] chỉ publish measured hardware/performance claim.
- [ ] verify link sau publication.

Không bump vượt `v1.0.0` chỉ vì quá trình phát triển có corrective commit.
