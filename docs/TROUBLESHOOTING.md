# Troubleshooting

## Explicit CUDA fails

Run `voxcpm-doctor` and `VOXCPM_DEVICE=cuda voxcpm-doctor`. If no usable CUDA device is visible, failure is intentional; the runtime does not silently switch to CPU.

## Model path cannot be resolved

Set `VOXCPM_MODEL_PATH` to an absolute local/mounted VoxCPM2 directory and run `voxcpm-verify-model`.

## API returns 401

Verify `VOXCPM_API_TOKEN`, `VOXCPM_REQUIRE_AUTH=1`, and exactly one `Authorization: Bearer ...` header.

## API returns 429

The bounded pending queue is full. Reduce client concurrency or change the queue size only within a measured resource envelope.

## `/healthz` is 200 but `/readyz` is 503

The process is alive but no usable worker is ready. Inspect worker/runtime state; cancellation or timeout can intentionally stop a worker.

## Streaming stalls or ends

Check the client consumption rate and `VOXCPM_STREAM_BACKPRESSURE_TIMEOUT_SECONDS`.

## Startup fails on temporary storage

Check `VOXCPM_TMP_DIR` and `VOXCPM_MIN_TMP_FREE_BYTES`.

## Kaggle cannot find the model

Confirm `dangkhoa2016/openbmb-voxcpm2` is attached to the notebook. See [Kaggle](KAGGLE.md).

## Docker cannot load the model

Mount the model at `/models/VoxCPM2` or override `VOXCPM_MODEL_PATH`. The image does not embed model weights.
