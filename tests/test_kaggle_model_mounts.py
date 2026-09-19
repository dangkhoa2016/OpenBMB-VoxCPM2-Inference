import hashlib
import json

import pytest

from deploy.kaggle.model_mounts import (
    DEFAULT_MODEL_REVISION,
    MAX_MANIFEST_BYTES,
    KaggleDiscoveryError,
    KaggleModelMountAdapter,
    discover_kaggle_model_candidates,
)
from voxcpm_runtime.config import RuntimeConfig
from voxcpm_runtime.model_resolver import ModelResolver


def _discover_unpinned(root):
    return discover_kaggle_model_candidates(
        root,
        expected_revision=None,
        expected_config_sha256=None,
    )


def test_discovers_valid_nested_model_and_ignores_unrelated_mounts(
    tmp_path,
    model_factory,
):
    model = model_factory(
        tmp_path / "models" / "owner" / "mirror" / "pytorch" / "default" / "1"
    )
    unrelated = tmp_path / "models" / "other" / "dataset"
    unrelated.mkdir(parents=True)
    (unrelated / "README.txt").write_text("unrelated", encoding="utf-8")

    candidates = _discover_unpinned(tmp_path / "models")

    assert [candidate.path for candidate in candidates] == [model.resolve()]
    assert candidates[0].metadata["adapter"] == "kaggle-model-mounts"


def test_invalid_candidate_before_valid_is_skipped(tmp_path, model_factory):
    root = tmp_path / "models"
    invalid = model_factory(
        root / "a-invalid",
        config={"architecture": "voxcpm"},
    )
    valid = model_factory(root / "b-valid")
    candidates = _discover_unpinned(root)

    resolved = ModelResolver().resolve(
        RuntimeConfig(offline=True),
        deployment_candidates=candidates,
    )

    assert [candidate.path for candidate in candidates] == [
        invalid.resolve(),
        valid.resolve(),
    ]
    assert resolved.path == valid.resolve()


def test_multiple_valid_candidates_are_selected_deterministically(
    tmp_path,
    model_factory,
):
    root = tmp_path / "models"
    second = model_factory(root / "z-model")
    first = model_factory(root / "a-model")

    candidates = _discover_unpinned(root)

    assert [candidate.path for candidate in candidates] == [
        first.resolve(),
        second.resolve(),
    ]


def test_no_model_found_returns_empty(tmp_path):
    (tmp_path / "models" / "unrelated").mkdir(parents=True)

    assert _discover_unpinned(tmp_path / "models") == ()


def test_missing_root_returns_empty(tmp_path):
    assert discover_kaggle_model_candidates(tmp_path / "missing") == ()


def test_root_must_be_directory(tmp_path):
    root = tmp_path / "models"
    root.write_text("not a directory", encoding="utf-8")

    with pytest.raises(KaggleDiscoveryError, match="must be a directory"):
        _discover_unpinned(root)


def test_discovery_depth_is_bounded(tmp_path, model_factory):
    model_factory(tmp_path / "models" / "nested" / "model")

    adapter = KaggleModelMountAdapter(tmp_path / "models", max_depth=0)

    assert adapter.discover() == ()


def test_discovery_entries_are_bounded(tmp_path):
    root = tmp_path / "models"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir(parents=True)
    adapter = KaggleModelMountAdapter(root, max_entries=2)

    with pytest.raises(KaggleDiscoveryError, match="exceeded 2 entries"):
        adapter.discover()


def test_mirror_manifest_attaches_model_identity(tmp_path, model_factory):
    model = model_factory(tmp_path / "models" / "mirror" / "1")
    revision = "4" * 40
    config_sha256 = hashlib.sha256(
        (model / "config.json").read_bytes()
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "source": {
            "repo_id": "openbmb/VoxCPM2",
            "resolved_commit": revision,
        },
        "target": {
            "handle": "dangkhoa2016/openbmb-voxcpm2/pyTorch/default",
        },
        "upstream_tree": {
            "files": [
                {
                    "path": "config.json",
                    "sha256": config_sha256,
                }
            ]
        },
    }
    (model / "_kaggle_mirror_provenance.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    candidate = discover_kaggle_model_candidates(
        tmp_path / "models",
        expected_revision=revision,
        expected_config_sha256=config_sha256,
    )[0]
    resolved = ModelResolver().resolve(
        RuntimeConfig(offline=True),
        deployment_candidates=[candidate],
    )

    assert candidate.model_id == "openbmb/VoxCPM2"
    assert candidate.revision == revision
    assert resolved.revision == revision
    assert resolved.metadata["mirror_handle"] == (
        "dangkhoa2016/openbmb-voxcpm2/pyTorch/default"
    )
    assert resolved.metadata["manifest_config_sha256"] == config_sha256
    assert resolved.metadata["model_revision"] == revision


def test_default_manifest_expectations_reject_unpinned_fixture(tmp_path, model_factory):
    model = model_factory(tmp_path / "models" / "mirror")
    manifest = {
        "schema_version": 1,
        "source": {
            "repo_id": "openbmb/VoxCPM2",
            "resolved_commit": DEFAULT_MODEL_REVISION,
        },
        "target": {
            "handle": "dangkhoa2016/openbmb-voxcpm2/pyTorch/default",
        },
        "upstream_tree": {
            "files": [
                {
                    "path": "config.json",
                    "sha256": hashlib.sha256(
                        (model / "config.json").read_bytes()
                    ).hexdigest(),
                }
            ]
        },
    }
    (model / "_kaggle_mirror_provenance.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(KaggleDiscoveryError, match="pinned model"):
        discover_kaggle_model_candidates(tmp_path / "models")


def test_ancestor_config_does_not_hide_nested_model(tmp_path, model_factory):
    root = tmp_path / "models"
    ancestor = model_factory(
        root / "container",
        config={"architecture": "voxcpm"},
    )
    nested = model_factory(root / "container" / "nested" / "model")
    candidates = _discover_unpinned(root)

    resolved = ModelResolver().resolve(
        RuntimeConfig(offline=True),
        deployment_candidates=candidates,
    )

    assert [candidate.path for candidate in candidates] == [
        ancestor.resolve(),
        nested.resolve(),
    ]
    assert resolved.path == nested.resolve()


def test_wrong_manifest_model_identity_is_not_defaulted(tmp_path, model_factory):
    wrong = model_factory(tmp_path / "models" / "wrong")
    valid = model_factory(tmp_path / "models" / "valid")
    manifest = {
        "schema_version": 1,
        "source": {
            "repo_id": "openbmb/not-voxcpm2",
            "resolved_commit": "6" * 40,
        }
    }
    (wrong / "_kaggle_mirror_provenance.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    candidates = _discover_unpinned(tmp_path / "models")

    resolved = ModelResolver().resolve(
        RuntimeConfig(offline=True),
        deployment_candidates=[*candidates, valid],
    )

    assert resolved.path == valid.resolve()


def test_manifest_config_hash_mismatch_is_not_accepted(tmp_path, model_factory):
    wrong = model_factory(tmp_path / "models" / "wrong")
    valid = model_factory(tmp_path / "models" / "valid")
    manifest = {
        "schema_version": 1,
        "source": {
            "repo_id": "openbmb/VoxCPM2",
            "resolved_commit": "7" * 40,
        },
        "target": {
            "handle": "dangkhoa2016/openbmb-voxcpm2/pyTorch/default",
        },
        "upstream_tree": {
            "files": [
                {
                    "path": "config.json",
                    "sha256": "8" * 64,
                }
            ]
        },
    }
    (wrong / "_kaggle_mirror_provenance.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    config_sha256 = hashlib.sha256(
        (valid / "config.json").read_bytes()
    ).hexdigest()
    candidates = discover_kaggle_model_candidates(
        tmp_path / "models",
        expected_revision="7" * 40,
        expected_config_sha256=config_sha256,
    )

    resolved = ModelResolver().resolve(
        RuntimeConfig(offline=True),
        deployment_candidates=[*candidates, valid],
    )

    assert resolved.path == valid.resolve()


def test_adapter_rejects_impossible_revision_length(tmp_path, model_factory):
    model = model_factory(tmp_path / "models" / "model")
    manifest = {
        "schema_version": 1,
        "source": {
            "repo_id": "openbmb/VoxCPM2",
            "resolved_commit": "9" * 41,
        },
        "target": {
            "handle": "dangkhoa2016/openbmb-voxcpm2/pyTorch/default",
        },
        "upstream_tree": {
            "files": [
                {
                    "path": "config.json",
                    "sha256": hashlib.sha256(
                        (model / "config.json").read_bytes()
                    ).hexdigest(),
                }
            ]
        },
    }
    (model / "_kaggle_mirror_provenance.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(KaggleDiscoveryError, match="revision"):
        _discover_unpinned(tmp_path / "models")


def test_manifest_read_is_bounded(tmp_path, model_factory):
    model = model_factory(tmp_path / "models" / "model")
    (model / "_kaggle_mirror_provenance.json").write_bytes(
        b" " * (MAX_MANIFEST_BYTES + 1)
    )

    with pytest.raises(KaggleDiscoveryError, match="exceeds"):
        _discover_unpinned(tmp_path / "models")


def test_invalid_root_path_is_structured():
    with pytest.raises(KaggleDiscoveryError, match="cannot be inspected"):
        discover_kaggle_model_candidates("bad\x00root")


def test_malformed_manifest_fails_clearly(tmp_path, model_factory):
    model = model_factory(tmp_path / "models" / "mirror")
    (model / "_kaggle_mirror_provenance.json").write_text(
        "{",
        encoding="utf-8",
    )

    with pytest.raises(KaggleDiscoveryError, match="not valid JSON"):
        _discover_unpinned(tmp_path / "models")


def test_malformed_manifest_does_not_hide_later_valid_model(tmp_path, model_factory):
    root = tmp_path / "models"
    invalid = model_factory(root / "a-invalid")
    valid = model_factory(root / "b-valid")
    (invalid / "_kaggle_mirror_provenance.json").write_text(
        "{",
        encoding="utf-8",
    )

    candidates = _discover_unpinned(root)

    assert [candidate.path for candidate in candidates] == [valid.resolve()]
