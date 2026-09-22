# Architecture

The runtime keeps model ownership out of the HTTP/API parent.

```text
CLI or HTTP client
  -> configuration / profile resolution
  -> API parent or local command
  -> Scheduler
  -> WorkerClient
  -> spawned WorkerProcess
  -> InferenceBackend
  -> VoxCPM2 runtime
  -> CPU or one process-isolated GPU
```

Key properties:

- the API parent is model-free;
- explicit CUDA never silently falls back to CPU;
- workers own model/runtime state;
- multi-GPU mode uses independent replicas, not model sharding;
- queue and concurrency are bounded;
- cancellation and timeout behavior are explicit;
- model resolution is portable and does not hard-code Kaggle paths.

Read next: [Backends](BACKENDS.md), [IPC](IPC.md), [Execution profiles](EXECUTION-PROFILES.md), [API](API.md), and [Observability](OBSERVABILITY.md).

Detailed implementation and qualification material remains in `BACKENDS.md`, `IPC.md`, and the provenance records.
