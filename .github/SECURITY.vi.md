# Chính sách bảo mật

> Language / Ngôn ngữ: [English](SECURITY.md) | **Tiếng Việt**

## Báo cáo

Không mở public issue cho lỗ hổng nghi ngờ, credential bị lộ, xử lý artifact không an toàn, dependency compromise, command-injection path, authentication bypass hoặc phát hiện nhạy cảm về bảo mật khác.

Dùng GitHub private vulnerability reporting khi repository đã bật. Nếu chưa có, gửi email tới i.am@dangkhoa.dev kèm mô tả ngắn gọn, component và revision bị ảnh hưởng, bước tái hiện hoặc proof of concept, tác động dự kiến và mitigation đề xuất nếu đã biết.

Không gửi credential production thật hoặc dữ liệu riêng tư không liên quan.

## Ranh giới bảo mật riêng của dự án

Các boundary quan trọng gồm:

- bearer authentication cho protected API endpoint;
- bounded request admission và active concurrency;
- queue/request/inference/stream-backpressure timeout;
- worker-process isolation và cleanup;
- explicit local/offline model loading sau bootstrap;
- secret redaction và validation error không phản chiếu input nhạy cảm;
- explicit CUDA profile fail-closed, không silent CPU fallback.

HTTP surface hiện tại là text-first. Reference/prompt audio operation nằm ở local backend/CLI và không được claim là hardened multipart upload endpoint.

## Phiên bản được hỗ trợ

Security fix nhắm tới main và stable release mới nhất còn được hỗ trợ khi phù hợp. Historical commit không được đảm bảo backport.
