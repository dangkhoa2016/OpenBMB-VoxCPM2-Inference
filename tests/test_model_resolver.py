import hashlib
import itertools
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from voxcpm_runtime import model_resolver
from voxcpm_runtime.config import RuntimeConfig
from voxcpm_runtime.model_resolver import (
    DeploymentCandidate,
    InvalidModelConfigError,
    InvalidModelPathError,
    InvalidTokenizerMetadataError,
    MissingModelArtifactError,
    ModelCandidateError,
    ModelInventoryError,
    ModelNotFoundError,
    ModelResolver,
    OfflineResolutionError,
    RemoteAcquisitionError,
    UnexpectedModelArchitectureError,
)


class FakeRemoteAcquirer:
    def __init__(self, path: Path | None = None, error: Exception | None = None) -> None:
        self.path = path
        self.error = error
        self.calls = 0

    def acquire(self, config: RuntimeConfig) -> Path:
        self.calls += 1
        if self.error is not None:
            raise self.error
        assert config.offline is False
        assert self.path is not None
        return self.path


class InvalidPathRemoteAcquirer:
    def acquire(self, config: RuntimeConfig) -> object:
        assert config.offline is False
        return 1


class SecretPathRemoteAcquirer:
    def acquire(self, config: RuntimeConfig) -> str:
        assert config.offline is False
        return "https://example.invalid/model?token=remote-path-secret"


def _network_explosion(*args: object, **kwargs: object) -> None:
    raise AssertionError("network access is forbidden")


def _forbid_network(monkeypatch) -> None:
    monkeypatch.setattr(socket, "create_connection", _network_explosion)
    monkeypatch.setattr(socket, "getaddrinfo", _network_explosion)
    monkeypatch.setattr(socket.socket, "connect", _network_explosion)
    monkeypatch.setattr(socket.socket, "connect_ex", _network_explosion)
    monkeypatch.setattr(socket.socket, "sendto", _network_explosion)


def test_explicit_path_wins_without_other_resolution(tmp_path, model_factory):
    explicit = model_factory(tmp_path / "explicit")
    deployment = model_factory(tmp_path / "deployment")
    cache = model_factory(tmp_path / "cache")
    remote_path = model_factory(tmp_path / "remote")
    remote = FakeRemoteAcquirer(remote_path)
    config = RuntimeConfig(
        model_path=str(explicit),
        cache_dir=str(cache),
        offline=True,
    )

    resolved = ModelResolver().resolve(
        config,
        deployment_candidates=[DeploymentCandidate(deployment)],
        remote_acquirer=remote,
    )

    assert resolved.path == explicit.resolve()
    assert resolved.source_kind == "explicit-path"
    assert remote.calls == 0


def test_invalid_explicit_path_fails_closed(tmp_path, model_factory):
    deployment = model_factory(tmp_path / "deployment")
    remote = FakeRemoteAcquirer(tmp_path / "remote")
    config = RuntimeConfig(model_path=str(tmp_path / "missing"), offline=True)

    with pytest.raises(ModelNotFoundError):
        ModelResolver().resolve(
            config,
            deployment_candidates=[deployment],
            remote_acquirer=remote,
        )

    assert remote.calls == 0


def test_explicit_regular_file_is_invalid(tmp_path):
    path = tmp_path / "model.txt"
    path.write_text("not a model", encoding="utf-8")

    with pytest.raises(InvalidModelPathError, match="must be a directory"):
        ModelResolver().resolve(RuntimeConfig(model_path=str(path)))


def test_deployment_candidate_beats_cache(tmp_path, model_factory):
    deployment = model_factory(tmp_path / "deployment")
    cache = model_factory(tmp_path / "cache")
    candidate = DeploymentCandidate(
        deployment,
        revision="3" * 40,
        metadata={"mount": "fixture"},
    )
    remote = FakeRemoteAcquirer(tmp_path / "remote")

    resolved = ModelResolver().resolve(
        RuntimeConfig(cache_dir=str(cache), offline=True),
        deployment_candidates=[candidate],
        remote_acquirer=remote,
    )

    assert resolved.path == deployment.resolve()
    assert resolved.source_kind == "deployment-mount"
    assert resolved.revision == "3" * 40
    assert resolved.metadata["mount"] == "fixture"
    assert remote.calls == 0


def test_invalid_deployment_candidates_are_skipped_in_order(tmp_path, model_factory):
    wrong = model_factory(
        tmp_path / "a-wrong",
        config={"architecture": "voxcpm"},
    )
    valid = model_factory(tmp_path / "b-valid")
    remote = FakeRemoteAcquirer(tmp_path / "remote")

    resolved = ModelResolver().resolve(
        RuntimeConfig(offline=True),
        deployment_candidates=[wrong, valid],
        remote_acquirer=remote,
    )

    assert resolved.path == valid.resolve()
    assert remote.calls == 0


def test_wrong_candidate_model_identity_is_skipped(tmp_path, model_factory):
    wrong = DeploymentCandidate(
        model_factory(tmp_path / "wrong"),
        model_id="openbmb/other",
    )
    valid = model_factory(tmp_path / "valid")

    resolved = ModelResolver().resolve(
        RuntimeConfig(offline=True),
        deployment_candidates=[wrong, valid],
    )

    assert resolved.path == valid.resolve()


def test_candidate_model_identity_must_be_exact(tmp_path, model_factory):
    wrong = DeploymentCandidate(
        model_factory(tmp_path / "wrong"),
        model_id="OpenBMB/VoxCPM2",
    )
    valid = model_factory(tmp_path / "valid")

    resolved = ModelResolver().resolve(
        RuntimeConfig(offline=True),
        deployment_candidates=[wrong, valid],
    )

    assert resolved.path == valid.resolve()


def test_cache_fallback_works_offline(tmp_path, model_factory):
    cache = model_factory(tmp_path / "cache")
    remote = FakeRemoteAcquirer(tmp_path / "remote")

    resolved = ModelResolver().resolve(
        RuntimeConfig(cache_dir=str(cache), offline=True),
        remote_acquirer=remote,
    )

    assert resolved.path == cache.resolve()
    assert resolved.source_kind == "local-cache"
    assert remote.calls == 0


def test_invalid_cache_fails_offline_then_can_use_remote(tmp_path, model_factory):
    invalid_cache = model_factory(
        tmp_path / "invalid-cache",
        config={"architecture": "voxcpm"},
    )
    remote_path = model_factory(tmp_path / "remote")
    remote = FakeRemoteAcquirer(remote_path)

    with pytest.raises(OfflineResolutionError) as caught:
        ModelResolver().resolve(
            RuntimeConfig(cache_dir=str(invalid_cache), offline=True),
            remote_acquirer=remote,
        )

    resolved = ModelResolver().resolve(
        RuntimeConfig(cache_dir=str(invalid_cache), offline=False),
        remote_acquirer=remote,
    )

    assert caught.value.details["attempts"][0]["code"] == "unexpected_model_architecture"
    assert resolved.source_kind == "remote-acquired"
    assert remote.calls == 1


def test_remote_is_last_and_used_only_online(tmp_path, model_factory):
    cache = model_factory(tmp_path / "cache")
    remote_path = model_factory(tmp_path / "remote")
    remote = FakeRemoteAcquirer(remote_path)
    resolver = ModelResolver()

    cached = resolver.resolve(
        RuntimeConfig(cache_dir=str(cache), offline=False),
        remote_acquirer=remote,
    )
    acquired = resolver.resolve(
        RuntimeConfig(cache_dir=str(tmp_path / "missing"), offline=False),
        remote_acquirer=remote,
    )

    assert cached.source_kind == "local-cache"
    assert acquired.source_kind == "remote-acquired"
    assert acquired.path == remote_path.resolve()
    assert remote.calls == 1


def test_online_without_remote_provider_fails_clearly(tmp_path):
    with pytest.raises(RemoteAcquisitionError, match="no remote acquirer"):
        ModelResolver().resolve(
            RuntimeConfig(cache_dir=str(tmp_path / "missing"), offline=False)
        )


def test_offline_missing_model_never_calls_remote_or_network(
    tmp_path,
    monkeypatch,
):
    remote = FakeRemoteAcquirer(tmp_path / "remote")
    _forbid_network(monkeypatch)

    with pytest.raises(OfflineResolutionError) as caught:
        ModelResolver().resolve(
            RuntimeConfig(cache_dir=str(tmp_path / "missing"), offline=True),
            remote_acquirer=remote,
        )

    assert remote.calls == 0
    assert caught.value.details["attempts"][0]["source_kind"] == "local-cache"


def test_offline_valid_fixture_uses_no_network(tmp_path, model_factory, monkeypatch):
    cache = model_factory(tmp_path / "cache")
    _forbid_network(monkeypatch)

    resolved = ModelResolver().resolve(
        RuntimeConfig(cache_dir=str(cache), offline=True)
    )

    assert resolved.source_kind == "local-cache"


def test_remote_provider_exception_does_not_leak_message(tmp_path):
    secret = "remote-secret-placeholder"
    remote = FakeRemoteAcquirer(error=RuntimeError(secret))

    with pytest.raises(RemoteAcquisitionError) as caught:
        ModelResolver().resolve(
            RuntimeConfig(offline=False),
            remote_acquirer=remote,
        )

    assert secret not in str(caught.value)
    assert caught.value.details["provider_error_type"] == "RuntimeError"
    assert remote.calls == 1


def test_remote_provider_returned_path_is_redacted():
    secret = "remote-path-secret"

    with pytest.raises(RemoteAcquisitionError) as caught:
        ModelResolver().resolve(
            RuntimeConfig(offline=False),
            remote_acquirer=SecretPathRemoteAcquirer(),
        )

    serialized = json.dumps(caught.value.to_dict())
    assert secret not in str(caught.value)
    assert secret not in serialized
    assert caught.value.path is None
    assert caught.value.details["invalid_model_code"] == "model_not_found"


def test_remote_provider_invalid_result_is_wrapped(tmp_path):
    remote = FakeRemoteAcquirer(tmp_path / "missing")

    with pytest.raises(RemoteAcquisitionError, match="invalid model"):
        ModelResolver().resolve(
            RuntimeConfig(offline=False),
            remote_acquirer=remote,
        )


def test_remote_provider_invalid_path_type_is_wrapped():
    with pytest.raises(RemoteAcquisitionError, match="invalid path"):
        ModelResolver().resolve(
            RuntimeConfig(offline=False),
            remote_acquirer=InvalidPathRemoteAcquirer(),
        )


def test_paths_with_nul_are_structured_errors():
    resolver = ModelResolver()

    with pytest.raises(InvalidModelPathError):
        resolver.resolve(RuntimeConfig(model_path="bad\x00path"))
    with pytest.raises(OfflineResolutionError) as caught:
        resolver.resolve(RuntimeConfig(cache_dir="bad\x00path", offline=True))
    with pytest.raises(OfflineResolutionError):
        resolver.resolve(
            RuntimeConfig(offline=True),
            deployment_candidates=["bad\x00path"],
        )

    assert caught.value.details["attempts"][0]["code"] == "invalid_model_path"


def test_unmatched_user_paths_are_structured_errors():
    resolver = ModelResolver()
    path = "~voxcpm2-user-that-does-not-exist"

    with pytest.raises(InvalidModelPathError):
        resolver.resolve(RuntimeConfig(model_path=path))
    with pytest.raises(OfflineResolutionError) as caught:
        resolver.resolve(RuntimeConfig(cache_dir=path, offline=True))
    with pytest.raises(OfflineResolutionError):
        resolver.resolve(
            RuntimeConfig(offline=True),
            deployment_candidates=[DeploymentCandidate(path)],
        )

    assert caught.value.details["attempts"][0]["code"] == "invalid_model_path"


@pytest.mark.parametrize(
    "config",
    [
        {"architecture": "voxcpm"},
        {"model_type": "voxcpm2"},
        {"architecture": " voxcpm2"},
        {"architecture": 2},
    ],
)
def test_wrong_architecture_fails(tmp_path, model_factory, config):
    path = model_factory(tmp_path / "model", config=config)

    with pytest.raises(UnexpectedModelArchitectureError):
        ModelResolver().resolve(RuntimeConfig(model_path=str(path)))


def test_malformed_or_non_object_config_fails(tmp_path, model_factory):
    malformed = model_factory(tmp_path / "malformed", config="{")
    array = model_factory(tmp_path / "array", config=[])

    with pytest.raises(InvalidModelConfigError, match="bounded UTF-8 JSON"):
        ModelResolver().resolve(RuntimeConfig(model_path=str(malformed)))
    with pytest.raises(InvalidModelConfigError, match="JSON object"):
        ModelResolver().resolve(RuntimeConfig(model_path=str(array)))


def test_incomplete_pinned_config_structure_fails(tmp_path, model_factory):
    path = model_factory(tmp_path / "model")
    config_path = path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    del config["lm_config"]["hidden_size"]
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(InvalidModelConfigError, match="lm_config") as caught:
        ModelResolver().resolve(RuntimeConfig(model_path=str(path)))

    assert caught.value.details["missing_fields"] == ["hidden_size"]


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("tokenizer.json", "{"),
        ("tokenizer.json", '{"version":"1.0","model":{"type":"BPE","vocab":{}}}'),
        ("tokenizer_config.json", "{}"),
    ],
)
def test_malformed_tokenizer_metadata_fails(
    tmp_path,
    model_factory,
    filename,
    content,
):
    path = model_factory(
        tmp_path / "model",
        extra_files={filename: content},
    )

    with pytest.raises(InvalidTokenizerMetadataError):
        ModelResolver().resolve(RuntimeConfig(model_path=str(path)))


def test_architecture_error_does_not_echo_config_value(tmp_path, model_factory):
    secret = "architecture-secret-placeholder"
    path = model_factory(
        tmp_path / "model",
        config={"architecture": secret},
    )

    with pytest.raises(UnexpectedModelArchitectureError) as caught:
        ModelResolver().resolve(RuntimeConfig(model_path=str(path)))

    assert secret not in str(caught.value)
    assert secret not in json.dumps(caught.value.to_dict())


def test_missing_config_fails(tmp_path, model_factory):
    path = model_factory(tmp_path / "model", config=None)

    with pytest.raises(MissingModelArtifactError, match="Model config"):
        ModelResolver().resolve(RuntimeConfig(model_path=str(path)))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"main_weight": None}, "main weight"),
        ({"audiovae_weight": None}, "AudioVAE weight"),
        ({"tokenizer": ()}, "tokenizer"),
        ({"tokenizer": ("tokenizer.json",)}, "tokenizer_config.json"),
    ],
)
def test_required_artifact_groups_fail(tmp_path, model_factory, kwargs, message):
    path = model_factory(tmp_path / "model", **kwargs)

    with pytest.raises(MissingModelArtifactError, match=message):
        ModelResolver().resolve(RuntimeConfig(model_path=str(path)))


def test_empty_preferred_weight_does_not_fall_back(tmp_path, model_factory):
    path = model_factory(
        tmp_path / "model",
        main_weight="model.safetensors",
        main_weight_bytes=b"",
        extra_files={"pytorch_model.bin": b"fallback"},
    )

    with pytest.raises(MissingModelArtifactError, match="main weight file is empty"):
        ModelResolver().resolve(RuntimeConfig(model_path=str(path)))


@pytest.mark.parametrize(
    ("main_weight", "audiovae_weight"),
    [
        ("model.safetensors", "audiovae.safetensors"),
        ("pytorch_model.bin", "audiovae.pth"),
    ],
)
def test_supported_weight_alternatives_resolve(
    tmp_path,
    model_factory,
    main_weight,
    audiovae_weight,
):
    path = model_factory(
        tmp_path / "model",
        main_weight=main_weight,
        audiovae_weight=audiovae_weight,
    )

    resolved = ModelResolver().resolve(RuntimeConfig(model_path=str(path)))

    assert resolved.required_artifacts.main_weight == main_weight
    assert resolved.required_artifacts.audiovae_weight == audiovae_weight


def test_inventory_identity_and_roles_are_deterministic(tmp_path, model_factory):
    path = model_factory(
        tmp_path / "model",
        extra_files={
            "README.md": "fixture",
            "special_tokens_map.json": "{}",
            "notes.txt": "metadata",
        },
    )
    config_bytes = (path / "config.json").read_bytes()
    candidate = DeploymentCandidate(
        path,
        metadata={"z": "last", "a": "first"},
    )
    metadata_before = dict(candidate.metadata)
    candidates_before = (candidate,)

    resolved = ModelResolver().resolve(
        RuntimeConfig(offline=True),
        deployment_candidates=[candidate],
    )

    assert [entry.relative_path for entry in resolved.inventory] == sorted(
        entry.relative_path for entry in resolved.inventory
    )
    roles = {entry.relative_path: entry.role for entry in resolved.inventory}
    assert roles["config.json"] == "config"
    assert roles["model.safetensors"] == "main-weight"
    assert roles["audiovae.safetensors"] == "audiovae-weight"
    assert roles["tokenizer.json"] == "tokenizer"
    assert roles["special_tokens_map.json"] == "tokenizer"
    assert roles["notes.txt"] == "other"
    assert resolved.config_identity.sha256 == hashlib.sha256(config_bytes).hexdigest()
    assert resolved.config_identity.size_bytes == len(config_bytes)
    assert dict(candidate.metadata) == metadata_before
    assert (candidate,) == candidates_before


def test_inventory_is_bounded(tmp_path, model_factory, monkeypatch):
    path = model_factory(tmp_path / "model")
    monkeypatch.setattr(model_resolver, "DEFAULT_MAX_INVENTORY_ENTRIES", 3)

    with pytest.raises(ModelInventoryError, match="bounded inventory limit"):
        ModelResolver().resolve(RuntimeConfig(model_path=str(path)))


def test_candidate_input_validation(tmp_path):
    resolver = ModelResolver()

    with pytest.raises(TypeError, match="ordered iterable"):
        resolver.resolve(RuntimeConfig(), deployment_candidates=str(tmp_path))
    with pytest.raises(TypeError, match="paths"):
        resolver.resolve(RuntimeConfig(), deployment_candidates=[object()])
    with pytest.raises(TypeError, match="ordered iterable"):
        resolver.resolve(RuntimeConfig(), deployment_candidates={tmp_path})


def test_first_valid_candidate_stops_iteration(tmp_path, model_factory):
    valid = model_factory(tmp_path / "valid")

    def candidates():
        yield valid
        raise AssertionError("candidate iteration continued after success")

    resolved = ModelResolver().resolve(
        RuntimeConfig(offline=True),
        deployment_candidates=candidates(),
    )

    assert resolved.path == valid.resolve()


def test_candidate_iteration_failure_is_structured():
    def candidates():
        raise RuntimeError("candidate-source-secret")
        yield

    with pytest.raises(ModelCandidateError) as caught:
        ModelResolver().resolve(
            RuntimeConfig(offline=True),
            deployment_candidates=candidates(),
        )

    assert caught.value.details["source_error_type"] == "RuntimeError"
    assert "candidate-source-secret" not in str(caught.value)


def test_candidate_count_is_bounded(tmp_path, model_factory):
    wrong = model_factory(
        tmp_path / "wrong",
        config={"architecture": "voxcpm"},
    )

    with pytest.raises(ModelCandidateError, match="limit exceeded") as caught:
        ModelResolver().resolve(
            RuntimeConfig(offline=True),
            deployment_candidates=itertools.repeat(wrong),
        )

    assert caught.value.details["limit"] == 256


def test_candidate_revision_length_is_exact(tmp_path, model_factory):
    candidate = DeploymentCandidate(
        model_factory(tmp_path / "wrong"),
        revision="a" * 41,
    )
    valid = model_factory(tmp_path / "valid")

    resolved = ModelResolver().resolve(
        RuntimeConfig(offline=True),
        deployment_candidates=[candidate, valid],
    )

    assert resolved.path == valid.resolve()


def test_candidate_config_hash_mismatch_is_skipped(tmp_path, model_factory):
    wrong = DeploymentCandidate(
        model_factory(tmp_path / "wrong"),
        expected_config_sha256="a" * 64,
    )
    valid = model_factory(tmp_path / "valid")

    resolved = ModelResolver().resolve(
        RuntimeConfig(offline=True),
        deployment_candidates=[wrong, valid],
    )

    assert resolved.path == valid.resolve()


def test_model_resolver_import_has_no_optional_model_dependencies():
    code = (
        "import sys; import voxcpm_runtime.model_resolver; "
        "assert 'torch' not in sys.modules; "
        "assert 'transformers' not in sys.modules; "
        "assert 'safetensors' not in sys.modules"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr


def test_config_json_is_not_modified(tmp_path, model_factory):
    path = model_factory(tmp_path / "model")
    config_path = path / "config.json"
    before = config_path.read_bytes()

    ModelResolver().resolve(RuntimeConfig(model_path=str(path)))

    assert config_path.read_bytes() == before
    assert json.loads(before)["architecture"] == "voxcpm2"
