# M2 portable model resolution

M2 resolves and validates an existing local VoxCPM2 model directory. It does not import the upstream `voxcpm` package, construct `VoxCPM2Model`, load tensors, initialize CUDA, or perform inference.

## Resolution order

`ModelResolver.resolve()` uses the existing `RuntimeConfig` and evaluates sources in this order:

1. `VOXCPM_MODEL_PATH` (`explicit-path`)
2. Ordered deployment candidates (`deployment-mount`)
3. `VOXCPM_CACHE_DIR` itself (`local-cache`)
4. An injected `RemoteAcquirer` (`remote-acquired`)

An explicit path is fail-closed. If it is missing, malformed, incomplete, or has the wrong architecture, resolution fails without checking deployment, cache, or remote sources. A deployment candidate is evaluated in caller-supplied order; the Kaggle adapter sorts its discovered paths deterministically. An invalid cache is recorded and resolution continues to the offline/remote boundary.

The core resolver has no Kaggle path literals or discovery logic. Deployment adapters provide local candidates and metadata.

## Offline behavior

`VOXCPM_OFFLINE=1` prevents the remote acquirer from being called. The resolver does not import or invoke a downloader. An injected acquirer is useful for application integration tests, but its returned path and error details are not trusted and signed URLs or other secret-bearing values are not emitted.

A missing local model in offline mode returns `OfflineResolutionError` with structured, non-sensitive attempt metadata. Network-negative tests replace socket connection, DNS, and UDP primitives with exploding guards and assert zero remote-acquirer calls. Offline mode constrains model resolution and acquisition behavior; it does not require host-wide network isolation.

## Kaggle adapter

`deploy/kaggle/model_mounts.py` searches only the configured local root, defaulting to `/kaggle/input/models`. Traversal is sorted, non-recursive across symlinks, depth-bounded to 8, and entry-bounded to 10,000. A configuration directory is not allowed to hide a valid nested model.

The default adapter qualification tuple is:

```text
model_id: openbmb/VoxCPM2
revision: 32279effe8c19989596f05d353d1447f51d9e915
config_sha256: 405f0dcd92f7feba6011ed4eac5c8d4f74cba9712f07fd5cfa3063bbdd95402c
```

When `_kaggle_mirror_provenance.json` is present, schema version, model identity, revision, target handle, and exactly one config SHA-256 are validated against caller-supplied trusted expectations. The default expectations are the pinned values above. A manifest cannot authenticate itself by changing both its claim and the local config. Malformed candidate manifests are skipped when another candidate is available; if no candidate remains, the adapter returns a structured discovery error.

## Model validation

The resolver checks that the selected path exists, is a directory, and contains:

```text
config.json
one non-empty model.safetensors or pytorch_model.bin
one non-empty audiovae.safetensors or audiovae.pth
tokenizer.json
tokenizer_config.json
```

The verified `config.json` marker is a top-level `architecture` string whose case-folded value is `voxcpm2`; whitespace is not trimmed. The resolver also checks the pinned structural sections `lm_config`, `encoder_config`, `dit_config`, and `audio_vae_config` and their required fields. The actual pinned mirror uses `model.safetensors` and `audiovae.pth`.

Tokenizer validation is based on the real inventory and pinned fast-tokenizer format: `tokenizer_config.json` identifies a tokenizer class, and `tokenizer.json` contains a non-empty version `1.0` BPE model. The real mirror also contains `special_tokens_map.json` and `tokenization_voxcpm2.py`; those files are inventoried but are not required by the core contract.

Only metadata and bounded file sizes are read. The config file is hashed with SHA-256 for identity. Weight files are stat'ed but never hashed or deserialized.

## Inventory and errors

The top-level inventory is sorted and bounded to 4,096 entries. Each entry reports relative path, byte size, and a role such as `config`, `tokenizer`, `main-weight`, `audiovae-weight`, `metadata`, or `other`. JSON metadata reads are bounded, and malformed paths, candidates, manifests, configs, tokenizers, and missing artifacts produce typed structured errors.

## Verification CLI

From a checkout:

```bash
VOXCPM_OFFLINE=1 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python scripts/verify_model.py
```

The installed equivalent is:

```bash
voxcpm-verify-model
```

The CLI emits deterministic JSON, exits zero only after successful metadata resolution, and exits non-zero for expected validation failures. It reports the resolved path, source kind, model ID, config identity, optional revision, required artifacts, inventory, and `remote_acquisition_count: 0`. It never emits `VOXCPM_API_TOKEN`.

## Scope

M2 does not provide model acquisition, tensor loading, inference, an API, streaming, scheduling, workers, or deployment qualification beyond local model metadata resolution.
