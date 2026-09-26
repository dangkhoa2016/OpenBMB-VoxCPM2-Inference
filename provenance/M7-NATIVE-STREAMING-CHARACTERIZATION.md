# M7 Native Streaming Characterization

## Status

M7 pre-implementation characterization completed on 2026-09-26 from:

~~~text
repository base        c5127c07c88d3077adb693bfa071a489c413151b
VERSION                1.0.0
upstream commit        f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69
installed voxcpm       2.0.3.post33+gf772e498a
model id               openbmb/VoxCPM2
model revision         32279effe8c19989596f05d353d1447f51d9e915
config SHA-256         405f0dcd92f7feba6011ed4eac5c8d4f74cba9712f07fd5cfa3063bbdd95402c
GPU                    Tesla T4
visible CUDA devices   1
dtype                  bfloat16
optimize               false
remote acquisition     0
~~~

No M7 backend implementation existed during this characterization.

## Native upstream path

Pinned upstream exposes real incremental streaming:

~~~text
VoxCPM.generate_streaming(...)
→ _generate(..., streaming=True)
→ model incremental inference
→ AudioVAE.streaming_decode()
→ decode_chunk(...)
→ yield CPU waveform chunks
~~~

This is not project-side slicing of a completed one-shot waveform.

## Canonical fixed-seed characterization

Text:

~~~text
Xin chào từ VoxCPM2.
~~~

Seed:

~~~text
42
~~~

One-shot:

~~~text
samples              99,840
sample rate          48,000 Hz
generation           5.572 s
float32 SHA-256      15782b6ab14f3a5188009637ac05790b27dd609f02d44b9a74e6b1e4e95cc382
~~~

Native stream A:

~~~text
chunks               13
samples per chunk    7,680 for every chunk
total samples        99,840
generation           4.725 s
first chunk          0.417 s
float32 SHA-256      5842f594de1a9a599c07e7e54c642b1aaa51c7a9f31439812e7bf70d4dfeffce
~~~

Native stream B with the same seed:

~~~text
chunks               13
samples per chunk    7,680 for every chunk
total samples        99,840
generation           4.715 s
first chunk          0.402 s
float32 SHA-256      5842f594de1a9a599c07e7e54c642b1aaa51c7a9f31439812e7bf70d4dfeffce
~~~

Stream A and B were exactly equal at float32 sample level and had identical chunk boundaries.

## One-shot versus native stream

For seed 42:

~~~text
same length                  true
exact float equality         false
max absolute difference      4.54e-7
mean absolute difference     2.63e-8
RMS difference               4.48e-8
first differing sample       0
~~~

The same experiment was repeated with seeds 7, 42, and 123.

| Seed | Samples | Chunks | Max abs float diff | RMS diff | PCM16 differing samples | Max PCM16 diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 76,800 | 10 | 2.61e-7 | 2.11e-8 | 28 | 1 LSB |
| 42 | 99,840 | 13 | 3.87e-7 | 4.48e-8 | 82 | 1 LSB |
| 123 | 76,800 | 10 | 1.21e-6 | 8.77e-8 | 113 | 1 LSB |

For all measured seeds:

~~~text
one-shot sample count == native stream sample count
float exact equality == false
PCM16 exact equality == false
PCM16 maximum per-sample difference == 1 LSB
~~~

The observed difference is consistent with using a different incremental decoder path, not a different request or grossly different waveform. M7 does not generalize these measured bounds beyond the tested model, configuration, text, seeds, and hardware.

## Contract decision

Outcome:

~~~text
M7_UPSTREAM_STREAMING_EXISTS_GATE=PASS
M7_UPSTREAM_STREAMING_REPEATABILITY_GATE=PASS
M7_STREAM_VS_ONESHOT_CHARACTERIZATION=PASS
M7_NATIVE_REASSEMBLY_EQUIVALENCE=PASS_TOLERANCE
M7_PUBLIC_STREAM_CONTRACT_COMPATIBILITY=PASS_WITH_CLARIFICATION
~~~

The previous generic wording that every stream reassembles to the equivalent one-shot result exactly was too strong for a real native streaming implementation.

The contract is clarified as follows:

- exact reassembly equality to one-shot remains a deterministic fake-backend guarantee;
- real native streaming must preserve the same project request semantics;
- real chunks must remain ordered, non-empty, project-owned values with exactly one final chunk;
- native-stream versus one-shot equivalence must be measured and documented for the qualified real backend rather than assumed bit-identical.

No buffered fake streaming is authorized.

## Scope

This characterization does not yet qualify the real backend `stream` capability. It only authorizes implementation work against the clarified contract.

HTTP, REST, WebSocket, worker, scheduler, IPC, T4x2, and multi-GPU streaming remain out of scope.
