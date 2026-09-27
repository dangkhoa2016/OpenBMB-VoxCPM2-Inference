# Benchmarks and measured resource envelopes

## Scope

benchmark qualification freezes benchmark protocol `voxcpm2-benchmark-v1` and publishes only measurements backed by retained evidence.

Protocol file:

```text
benchmarks/protocol-v1.json
SHA-256 6feeeb602cd510b09cdc7bb2508e53c5180b5788baacc455fbc86a1d2c6058d2
```

The benchmark protocol separates cold startup from warm requests. GPU service measurements use one warm-up request followed by three measured warm repetitions for B1-B3.

The project API currently forwards only `text` for standard TTS. Therefore the measured runs use pinned upstream defaults:

```text
cfg_value            2.0
inference_timesteps  10
seed                 None
denoise              false
upstream optimize    false
```

These runs are intentionally unseeded. Audio hashes are output identities for individual runs, not cross-run deterministic requirements.

## Formulas

```text
RTF = request elapsed seconds / generated audio duration seconds

requests/s =
  completed requests / concurrent makespan seconds

audio-seconds/s =
  sum generated audio duration / concurrent makespan seconds

measured scaling ratio =
  two-worker requests/s / one-worker requests/s
```

RTF below 1 would mean faster-than-real-time generation. The qualified T4 measurements below have RTF above 1, so no real-time claim is made.

## Frozen inputs used for formal GPU timing

| ID | Purpose | Input |
| --- | --- | --- |
| B1 | short English | `Hello from VoxCPM2.` |
| B2 | short Vietnamese | `Xin chào từ VoxCPM2.` |
| B3 | sustained generation | fixed longer English paragraph from protocol-v1 |
| B7 | streaming TTFA | B2 Vietnamese sentence |
| B8 | concurrency | one B1 request plus one B2 request |

B4-B6 are also frozen in the protocol for voice design, cloning and continuation qualification, but benchmark qualification performance publication is limited to the CPU/T4/T4x2 measurements listed below.

## CPU envelope

The CPU performance/resource envelope is promoted from the frozen CPU-runtime qualification real CPU qualification rather than re-labelling an estimate as a new benchmark qualification measurement.

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

Canonical B2-equivalent text: `Xin chào từ VoxCPM2.`

| Metric | Measured value |
| --- | ---: |
| Model load | 43.15 s |
| Generation | 100.66 s |
| Audio duration | 2.08 s |
| RTF | about 48.4 |
| Peak RSS, canonical run | 10.819 GiB |
| Peak RSS, offline proof | 10.912 GiB |
| OOM / OOM kill | 0 / 0 |

The approximate 16 GB CPU target from the project plan is **not qualified by this evidence**. The measured run proves the listed 30 GiB-cgroup environment; it does not prove a minimum RAM requirement.

Kaggle demo qualification separately proves that the current project source still executes the CPU profile functionally from a fresh Kaggle Run All. benchmark qualification does not publish the interrupted exploratory repeated CPU service-matrix run as benchmark evidence.

## One NVIDIA T4

Formal benchmark qualification service timing source:

```text
profile          cuda-single
GPU              Tesla T4
workers          1
Python           3.12.13
PyTorch          2.10.0+cu128
CUDA runtime     12.8
protocol         voxcpm2-benchmark-v1
```

Cold startup through `/readyz`: **33.416 s**.

Warm one-shot medians, three measured repetitions:

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

B8 with one worker and two simultaneous client requests:

```text
makespan               9.181 s
completed requests     2
requests/s             0.21784
audio-seconds/s        0.43567
```

Current NVML process memory was 5,242 MiB at readiness and 5,842 MiB after the measured workload. These are snapshots, not peak-allocation claims.

For a peak allocator envelope, frozen single-GPU qualification evidence measured:

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

The single-GPU qualification and benchmark qualification values come from different formal runs and are not merged into a synthetic single-run number.

## T4x2 replica throughput

Formal benchmark qualification service timing source:

```text
profile          cuda-replica
GPUs             2 x Tesla T4
workers          2
protocol         voxcpm2-benchmark-v1
```

Cold startup through two-worker readiness: **61.954 s**.

Warm one-shot medians, three measured repetitions:

| Input | Median request | Median audio | Median RTF |
| --- | ---: | ---: | ---: |
| B1 | 6.932 s | 3.04 s | 2.280 |
| B2 | 4.805 s | 2.08 s | 2.310 |
| B3 | 17.037 s | 7.36 s | 2.315 |

B8, two simultaneous requests on two replicas:

```text
makespan               4.882 s
completed requests     2
requests/s             0.40971
audio-seconds/s        0.75386
```

Measured request-throughput scaling against the one-worker B8 run:

```text
0.409706 / 0.217837 = 1.8808x
```

This means the measured two-worker request rate was 1.8808 times the measured one-worker request rate for this frozen B8 protocol. It is not a universal “1.88x faster” claim and no 2.0x expectation is asserted.

Post-workload NVML snapshots:

```text
GPU0 worker memory     5,824 MiB
GPU1 worker memory     5,624 MiB
OOM / OOM kill         0 / 0
```

## Queue and degraded behavior

T4x2 qualification provides the frozen measured service-behavior evidence used by the benchmark qualification resource/throughput publication.

With two workers and one pending slot:

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

The surviving worker completed its active stream and subsequently served a pending request after the other worker was cancelled/disconnected. No automatic worker respawn is claimed.

## Evidence integrity

Raw benchmark qualification runtime JSON remains a local qualification artifact rather than a tracked repository file.

```text
cuda-single JSON SHA-256
a34636b1016f9e6b346f78f7010340f64c31203fbb3311c79a32f6a1d44bcfdb

cuda-replica JSON SHA-256
e50bf7c69e5c1928b503da0f4308b196a8becab5d8f998146b0c108863f47437
```

Both runs returned GPU process state to baseline, closed port 8090, and retained zero cgroup OOM counters.

## Interpretation boundaries

These measurements apply to the explicitly identified source, model, runtime and hardware evidence only.

They are not:

- SLA or availability guarantees;
- p50/p95 service latency claims;
- guarantees for all T4 hosts or all CUDA GPUs;
- proof that 16 GB system RAM is sufficient for CPU;
- deterministic waveform guarantees;
- tensor-parallel or model-sharding benchmarks;
- claims that two GPUs always provide 2x scaling.

See `provenance/QUALIFICATION.md` for source binding and gate status.
