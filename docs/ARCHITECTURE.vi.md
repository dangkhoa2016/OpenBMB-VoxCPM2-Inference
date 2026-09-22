# Kiến trúc

Runtime giữ model ownership ra khỏi HTTP/API parent.

```text
CLI hoặc HTTP client
  -> configuration / profile resolution
  -> API parent hoặc local command
  -> Scheduler
  -> WorkerClient
  -> spawned WorkerProcess
  -> InferenceBackend
  -> VoxCPM2 runtime
  -> CPU hoặc một GPU được cách ly theo process
```

Các đặc tính chính:

- API parent không sở hữu model;
- CUDA được yêu cầu tường minh sẽ không âm thầm fallback sang CPU;
- worker sở hữu model/runtime state;
- multi-GPU dùng các replica độc lập, không shard model;
- queue và concurrency có giới hạn;
- cancellation và timeout có semantics rõ ràng;
- model resolution portable và không hard-code Kaggle path.

Đọc tiếp: [Backends](BACKENDS.vi.md), [IPC](IPC.vi.md), [Execution profiles](EXECUTION-PROFILES.vi.md), [API](API.vi.md), và [Observability](OBSERVABILITY.vi.md).

Chi tiết implementation/qualification vẫn nằm trong `BACKENDS.vi.md`, `IPC.vi.md` và các provenance record.
