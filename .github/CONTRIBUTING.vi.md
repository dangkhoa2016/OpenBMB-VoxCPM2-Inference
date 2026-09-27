# Hướng dẫn đóng góp

> Language / Ngôn ngữ: [English](CONTRIBUTING.md) | **Tiếng Việt**

Cảm ơn bạn đã đóng góp cho OpenBMB-VoxCPM2-Inference.

## Nguyên tắc

Mọi thay đổi nên:

- tập trung và dễ review;
- có khả năng tái lập;
- nói rõ tác động tới claim CPU, single-GPU hoặc T4x2;
- fail-closed khi hardware, model, authentication hoặc runtime contract không khớp.

Không nới lỏng qualification gate chỉ để demo chạy PASS.

## Development validation

Tối thiểu chạy:

    python -m compileall -q voxcpm_runtime deploy tests scripts
    python -m pytest -q
    python -m ruff check .
    git diff --check

Với thay đổi Kaggle demo, chạy thêm:

    python -m pytest -q tests/test_kaggle_production_demo.py

CPU/static PASS không phải evidence rằng live T4x2 inference đã thành công.

## Chính sách tài liệu

Khi một tài liệu Markdown public có cặp English/Vietnamese, hãy cập nhật cả hai file trong cùng thay đổi. Các community document song ngữ nên có language switcher gần đầu file.

## Ranh giới runtime và provenance

Hãy nêu rõ thay đổi có ảnh hưởng tới upstream source/model identity, CPU/CUDA execution profile, worker scheduling hoặc API behavior, authentication/timeouts/resource limits, Kaggle notebook behavior, Docker portability và public qualification claim hay không.

Không commit model weights, secret, token, runtime cache, generated evidence archive hoặc credential cá nhân.

## Pull request

Hãy mô tả thay đổi gì và vì sao, các lệnh validation đã chạy, thay đổi chỉ CPU/static hay có T4x2 evidence mới, tác động compatibility/security/provenance và tác động tài liệu hoặc release.
