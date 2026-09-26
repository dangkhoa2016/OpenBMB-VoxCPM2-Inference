# Observability

Runtime expose operational signal có cấu trúc mà không chuyển model ownership vào API parent.

Các signal gồm:

- `/healthz` liveness;
- `/readyz` readiness/degraded state;
- scheduler và worker state;
- request/queue timing;
- structured runtime metrics;
- process RSS và GPU-memory khi được đo;
- diagnostics đã redact qua `voxcpm-doctor`.

Secret/path redaction được xử lý tập trung. Diagnostics cho biết API token đã được cấu hình hay chưa nhưng không in giá trị token.

Xem [Benchmarks](BENCHMARKS.vi.md) cho performance/resource envelope và [Security](SECURITY.vi.md) cho redaction boundary.
