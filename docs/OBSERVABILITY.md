# Observability

The runtime exposes structured operational signals without moving model ownership into the API parent.

Available signals include:

- `/healthz` liveness;
- `/readyz` readiness/degraded state;
- scheduler and worker state;
- request/queue timing;
- structured runtime metrics;
- process RSS and GPU-memory measurements where collected;
- redacted diagnostics through `voxcpm-doctor`.

Secret/path redaction is centralized. Diagnostics indicate whether an API token is configured but do not print its value.

Measured performance and resource envelopes are documented in [Benchmarks](BENCHMARKS.md). Security/redaction boundaries are documented in [Security](SECURITY.md).
