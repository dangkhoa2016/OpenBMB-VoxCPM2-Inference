# Security Policy

> Language / Ngôn ngữ: **English** | [Tiếng Việt](SECURITY.vi.md)

## Reporting

Do not open a public issue for a suspected vulnerability, exposed credential, unsafe artifact handling, dependency compromise, command-injection path, authentication bypass, or other security-sensitive finding.

Use GitHub private vulnerability reporting when enabled. Otherwise email i.am@dangkhoa.dev with a concise description, affected component and revision, reproduction steps or proof of concept, expected impact, and suggested mitigation if known.

Do not send real production credentials or unrelated private data.

## Project-specific boundaries

Important security boundaries include:

- bearer authentication for protected API endpoints;
- bounded request admission and active concurrency;
- queue, request, inference, and stream-backpressure timeouts;
- worker-process isolation and cleanup;
- explicit local/offline model loading after bootstrap;
- secret redaction and non-reflective validation errors;
- fail-closed explicit CUDA profiles without silent CPU fallback.

The current HTTP surface is text-first. Reference/prompt audio operations are local backend/CLI paths and are not claimed as hardened multipart upload endpoints.

## Supported versions

Security fixes target main and the latest supported stable release when practical. Historical commits are not guaranteed to receive backports.
