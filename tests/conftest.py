import json
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest


_DEFAULT_CONFIG = {
    "architecture": "voxcpm2",
    "lm_config": {
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "num_hidden_layers": 1,
        "vocab_size": 16,
    },
    "encoder_config": {
        "hidden_dim": 8,
        "ffn_dim": 16,
        "num_heads": 2,
        "num_layers": 1,
    },
    "dit_config": {
        "hidden_dim": 8,
        "ffn_dim": 16,
        "num_heads": 2,
        "num_layers": 1,
        "cfm_config": {},
    },
    "audio_vae_config": {
        "encoder_dim": 8,
        "encoder_rates": [2],
        "decoder_dim": 8,
        "decoder_rates": [2],
        "sample_rate": 16000,
        "out_sample_rate": 48000,
    },
}
_DEFAULT_TOKENIZER = {
    "version": "1.0",
    "model": {
        "type": "BPE",
        "vocab": {
            "<unk>": 0,
            "<s>": 1,
            "</s>": 2,
        },
    },
}
_CONFIG_UNSET = object()


@pytest.fixture
def model_factory() -> Callable[..., Path]:
    def create(
        path: Path,
        *,
        config: object = _CONFIG_UNSET,
        main_weight: str | None = "model.safetensors",
        main_weight_bytes: bytes | None = b"main",
        audiovae_weight: str | None = "audiovae.safetensors",
        audiovae_weight_bytes: bytes | None = b"audio",
        tokenizer: tuple[str, ...] | None = (
            "tokenizer.json",
            "tokenizer_config.json",
        ),
        extra_files: Mapping[str, bytes | str] | None = None,
    ) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        if config is _CONFIG_UNSET:
            config = _DEFAULT_CONFIG
        if config is not None:
            if isinstance(config, str):
                (path / "config.json").write_text(config, encoding="utf-8")
            else:
                (path / "config.json").write_text(
                    json.dumps(config),
                    encoding="utf-8",
                )
        if main_weight is not None and main_weight_bytes is not None:
            (path / main_weight).write_bytes(main_weight_bytes)
        if audiovae_weight is not None and audiovae_weight_bytes is not None:
            (path / audiovae_weight).write_bytes(audiovae_weight_bytes)
        if tokenizer is not None:
            for filename in tokenizer:
                if filename == "tokenizer.json":
                    content = json.dumps(_DEFAULT_TOKENIZER)
                elif filename == "tokenizer_config.json":
                    content = json.dumps({"tokenizer_class": "VoxCPM2Tokenizer"})
                else:
                    content = "tokenizer"
                (path / filename).write_text(content, encoding="utf-8")
        for filename, content in (extra_files or {}).items():
            if isinstance(content, bytes):
                (path / filename).write_bytes(content)
            else:
                (path / filename).write_text(content, encoding="utf-8")
        return path

    return create
