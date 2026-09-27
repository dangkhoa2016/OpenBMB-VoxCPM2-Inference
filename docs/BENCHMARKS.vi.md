# Benchmark và measured resource envelope

## Phạm vi

benchmark qualification freeze benchmark protocol `voxcpm2-benchmark-v1` và chỉ công bố các số đo có frozen evidence.

Protocol:

```text
benchmarks/protocol-v1.json
SHA-256 6feeeb602cd510b09cdc7bb2508e53c5180b5788baacc455fbc86a1d2c6058d2
```

Cold startup và warm request được tách riêng. GPU service benchmark dùng một warm-up request rồi ba measured warm repetitions cho B1-B3.

Project API hiện chỉ truyền `text` cho standard TTS, nên các run này dùng upstream defaults đã pin:

```text
cfg_value            2.0
inference_timesteps  10
seed                 None
denoise              false
upstream optimize    false
```

Đây là các run unseeded. WAV hash chỉ là identity của từng output, không phải yêu cầu bit-identical giữa các run.

## Công thức

```text
RTF = request elapsed seconds / generated audio duration seconds

requests/s =
  completed requests / concurrent makespan seconds

audio-seconds/s =
  tổng generated audio duration / concurrent makespan seconds

measured scaling ratio =
  two-worker requests/s / one-worker requests/s
```

RTF < 1 mới có nghĩa nhanh hơn real-time. Các T4 measurement dưới đây đều có RTF > 1 nên dự án không claim real-time.

## Frozen inputs dùng cho formal GPU timing

| ID | Mục đích | Input |
| --- | --- | --- |
| B1 | short English | `Hello from VoxCPM2.` |
| B2 | short Vietnamese | `Xin chào từ VoxCPM2.` |
| B3 | sustained generation | longer English paragraph đã freeze trong protocol-v1 |
| B7 | streaming TTFA | câu B2 |
| B8 | concurrency | một B1 + một B2 |

B4-B6 cũng được freeze cho voice design, cloning và continuation, nhưng benchmark qualification performance publication chỉ dùng CPU/T4/T4x2 evidence bên dưới.

## CPU envelope

CPU performance/resource envelope được promote từ frozen CPU-runtime qualification real CPU qualification; không biến estimate thành số đo benchmark qualification mới.

Measured environment:

```text
CPU logical / physical      4 / 2
host total RAM              31.348 GiB
cgroup memory limit         30.000 GiB
GPU                         none
PyTorch                     2.14.0+cpu
VoxCPM                      2.0.3.post33+gf772e498a
dtype                       bfloat16
```

Canonical text trùng B2: `Xin chào từ VoxCPM2.`

| Metric | Measured value |
| --- | ---: |
| Model load | 43.15 s |
| Generation | 100.66 s |
| Audio duration | 2.08 s |
| RTF | khoảng 48.4 |
| Peak RSS, canonical | 10.819 GiB |
| Peak RSS, offline proof | 10.912 GiB |
| OOM / OOM kill | 0 / 0 |

Target CPU khoảng 16 GB trong master plan **chưa được qualification** bởi evidence này. Evidence chỉ chứng minh môi trường cgroup 30 GiB đã đo, không chứng minh minimum RAM.

Kaggle demo qualification đã chứng minh current project source vẫn chạy CPU profile functionally bằng fresh Kaggle Run All. benchmark qualification không công bố exploratory repeated CPU service-matrix bị dừng giữa chừng làm benchmark evidence.

## Một NVIDIA T4

Formal benchmark qualification service timing:

```text
profile          cuda-single
GPU              Tesla T4
workers          1
Python           3.12.13
PyTorch          2.10.0+cu128
CUDA runtime     12.8
protocol         voxcpm2-benchmark-v1
```

Cold startup tới `/readyz`: **33.416 s**.

Warm one-shot median, ba repetitions:

| Input | Median request | Median audio | Median RTF |
| --- | ---: | ---: | ---: |
| B1 | 5.444 s | 2.40 s | 2.268 |
| B2 | 4.382 s | 1.92 s | 2.282 |
| B3 | 15.583 s | 6.88 s | 2.265 |

B7 native streaming:

```text
first body / TTFA      0.842 s
completion             5.233 s
audio                  2.24 s
RTF                    2.336
client chunks          14
PCM bytes              215,040
```

B8, một worker với hai simultaneous client requests:

```text
makespan               9.181 s
completed requests     2
requests/s             0.21784
audio-seconds/s        0.43567
```

NVML process memory snapshot là 5,242 MiB lúc READY và 5,842 MiB sau measured workload. Đây là snapshot, không claim peak allocator.

Frozen single-GPU qualification peak allocator envelope:

```text
canonical load                    27.079539 s
canonical generation               7.137908 s
canonical audio                    2.40 s
canonical RTF                      2.9741
peak GPU allocated                 5.194 GiB
peak GPU reserved                  5.363 GiB
host peak RSS                     11.003 GiB
OOM / OOM kill                     0 / 0
```

single-GPU qualification và benchmark qualification là hai formal run khác nhau; không gộp thành một synthetic single-run result.

## T4x2 replica throughput

Formal benchmark qualification service timing:

```text
profile          cuda-replica
GPUs             2 x Tesla T4
workers          2
protocol         voxcpm2-benchmark-v1
```

Cold startup tới two-worker readiness: **61.954 s**.

Warm one-shot median, ba repetitions:

| Input | Median request | Median audio | Median RTF |
| --- | ---: | ---: | ---: |
| B1 | 6.932 s | 3.04 s | 2.280 |
| B2 | 4.805 s | 2.08 s | 2.310 |
| B3 | 17.037 s | 7.36 s | 2.315 |

B8, hai simultaneous requests trên hai replica:

```text
makespan               4.882 s
completed requests     2
requests/s             0.40971
audio-seconds/s        0.75386
```

Measured request-throughput scaling so với one-worker B8:

```text
0.409706 / 0.217837 = 1.8808x
```

Nghĩa chính xác: measured two-worker request rate bằng 1.8808 lần measured one-worker request rate trong frozen B8 protocol. Không phải universal “1.88x faster” và không giả định 2.0x.

Post-workload NVML snapshots:

```text
GPU0 worker memory     5,824 MiB
GPU1 worker memory     5,624 MiB
OOM / OOM kill         0 / 0
```

## Queue và degraded behavior

T4x2 qualification là frozen measured service-behavior evidence được benchmark qualification dùng cho phần queue/failure.

Với hai workers và một pending slot:

```text
A active               HTTP 200, 29.631 s
B active               HTTP 200, 34.718 s
C pending              HTTP 200, 61.811 s
D rejected             HTTP 429 queue_full, 0.018 s
```

Failure/degraded readiness:

```text
2 healthy workers      /readyz 200
1 healthy worker       /readyz 200
0 healthy workers      /readyz 503
/healthz               200
GPU processes after    0
```

Survivor worker hoàn tất active stream và sau đó xử lý pending request sau khi worker kia bị cancel/disconnect. Không claim automatic respawn.

## Evidence integrity

Raw benchmark qualification runtime JSON vẫn là local qualification artifact, không track vào repository.

```text
cuda-single JSON SHA-256
a34636b1016f9e6b346f78f7010340f64c31203fbb3311c79a32f6a1d44bcfdb

cuda-replica JSON SHA-256
e50bf7c69e5c1928b503da0f4308b196a8becab5d8f998146b0c108863f47437
```

Cả hai run đều đưa GPU process state về baseline, đóng port 8090 và giữ OOM counters bằng 0.

## Ranh giới diễn giải

Các số đo chỉ áp dụng cho source/model/runtime/hardware đã nêu.

Chúng không phải:

- SLA hoặc availability guarantee;
- p50/p95 service latency claim;
- guarantee cho mọi T4 hoặc mọi CUDA GPU;
- bằng chứng rằng 16 GB system RAM đủ cho CPU;
- deterministic waveform guarantee;
- tensor-parallel hoặc model-sharding benchmark;
- claim rằng hai GPU luôn scale 2x.

Xem `provenance/QUALIFICATION.md`.
