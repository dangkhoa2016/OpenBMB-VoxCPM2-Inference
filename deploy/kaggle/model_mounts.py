from __future__ import annotations

import hmac
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

from voxcpm_runtime.model_resolver import MODEL_ID, DeploymentCandidate


DEFAULT_KAGGLE_MODEL_ROOT = Path("/kaggle/input/models")
DEFAULT_MAX_DEPTH = 8
DEFAULT_MAX_ENTRIES = 10000
DEFAULT_MODEL_REVISION = "32279effe8c19989596f05d353d1447f51d9e915"
DEFAULT_CONFIG_SHA256 = "405f0dcd92f7feba6011ed4eac5c8d4f74cba9712f07fd5cfa3063bbdd95402c"
MAX_MANIFEST_BYTES = 1024 * 1024
_REVISION_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z", re.ASCII)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_MAX_IDENTITY_STRING = 512


class KaggleDiscoveryError(RuntimeError):
    pass


class KaggleModelMountAdapter:
    def __init__(
        self,
        root: str | os.PathLike[str] = DEFAULT_KAGGLE_MODEL_ROOT,
        *,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        expected_revision: str | None = DEFAULT_MODEL_REVISION,
        expected_config_sha256: str | None = DEFAULT_CONFIG_SHA256,
    ) -> None:
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if expected_revision is not None and (
            not isinstance(expected_revision, str)
            or _REVISION_PATTERN.fullmatch(expected_revision) is None
        ):
            raise ValueError("expected_revision must be a lowercase hexadecimal object ID")
        if expected_config_sha256 is not None and (
            not isinstance(expected_config_sha256, str)
            or _SHA256_PATTERN.fullmatch(expected_config_sha256) is None
        ):
            raise ValueError("expected_config_sha256 must be a lowercase SHA-256 digest")
        try:
            self.root = Path(root).expanduser()
        except (TypeError, ValueError, RuntimeError):
            raise KaggleDiscoveryError("Kaggle model root is not a valid path") from None
        self.max_depth = max_depth
        self.max_entries = max_entries
        self.expected_revision = expected_revision
        self.expected_config_sha256 = expected_config_sha256

    def discover(self) -> tuple[DeploymentCandidate, ...]:
        try:
            status = self.root.stat()
        except FileNotFoundError:
            return ()
        except (OSError, ValueError, RuntimeError):
            raise KaggleDiscoveryError(
                f"Kaggle model root cannot be inspected: {self.root}"
            ) from None
        if not stat.S_ISDIR(status.st_mode):
            raise KaggleDiscoveryError(
                f"Kaggle model root must be a directory: {self.root}"
            )
        candidates: list[DeploymentCandidate] = []
        candidate_errors: list[KaggleDiscoveryError] = []
        budget = [0]
        self._visit(
            self.root,
            depth=0,
            candidates=candidates,
            candidate_errors=candidate_errors,
            budget=budget,
        )
        if not candidates and candidate_errors:
            raise candidate_errors[0]
        return tuple(sorted(candidates, key=lambda item: str(item.path)))

    def _visit(
        self,
        directory: Path,
        *,
        depth: int,
        candidates: list[DeploymentCandidate],
        candidate_errors: list[KaggleDiscoveryError],
        budget: list[int],
    ) -> None:
        self._consume(budget)
        if self._has_config(directory):
            try:
                candidates.append(self._candidate(directory))
            except KaggleDiscoveryError as error:
                candidate_errors.append(error)
        if depth >= self.max_depth:
            return
        children: list[Path] = []
        try:
            with os.scandir(directory) as iterator:
                for entry in iterator:
                    self._consume(budget)
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            children.append(Path(entry.path))
                    except (OSError, ValueError, RuntimeError):
                        raise KaggleDiscoveryError(
                            f"Kaggle mount entry cannot be inspected: {entry.path}"
                        ) from None
        except KaggleDiscoveryError:
            raise
        except (OSError, ValueError, RuntimeError):
            raise KaggleDiscoveryError(
                f"Kaggle model directory cannot be read: {directory}"
            ) from None
        for child in sorted(children, key=lambda item: item.name):
            self._visit(
                child,
                depth=depth + 1,
                candidates=candidates,
                candidate_errors=candidate_errors,
                budget=budget,
            )

    def _consume(self, budget: list[int]) -> None:
        budget[0] += 1
        if budget[0] > self.max_entries:
            raise KaggleDiscoveryError(
                f"Kaggle discovery exceeded {self.max_entries} entries"
            )

    def _has_config(self, directory: Path) -> bool:
        config_path = directory / "config.json"
        try:
            status = config_path.stat()
        except (FileNotFoundError, NotADirectoryError):
            return False
        except (OSError, ValueError, RuntimeError):
            raise KaggleDiscoveryError(
                f"Kaggle model config cannot be inspected: {config_path}"
            ) from None
        return stat.S_ISREG(status.st_mode)

    def _candidate(self, directory: Path) -> DeploymentCandidate:
        revision, manifest_hash, metadata = self._manifest_metadata(directory)
        try:
            resolved = directory.resolve()
        except (OSError, RuntimeError):
            raise KaggleDiscoveryError(
                f"Kaggle model directory cannot be resolved: {directory}"
            ) from None
        if manifest_hash is not None:
            metadata["manifest_config_sha256"] = manifest_hash
            metadata["manifest_file"] = "_kaggle_mirror_provenance.json"
        return DeploymentCandidate(
            path=resolved,
            model_id=MODEL_ID,
            revision=revision,
            expected_config_sha256=self.expected_config_sha256,
            metadata=metadata,
        )

    def _manifest_metadata(
        self,
        directory: Path,
    ) -> tuple[str | None, str | None, dict[str, str]]:
        metadata = {"adapter": "kaggle-model-mounts"}
        manifest_path = directory / "_kaggle_mirror_provenance.json"
        try:
            status = manifest_path.stat()
        except (FileNotFoundError, NotADirectoryError):
            return None, None, metadata
        except (OSError, ValueError, RuntimeError):
            raise KaggleDiscoveryError(
                f"Kaggle mirror manifest cannot be inspected: {manifest_path}"
            ) from None
        if not stat.S_ISREG(status.st_mode):
            raise KaggleDiscoveryError(
                f"Kaggle mirror manifest must be a regular file: {manifest_path}"
            )
        if status.st_size > MAX_MANIFEST_BYTES:
            raise KaggleDiscoveryError(
                f"Kaggle mirror manifest exceeds {MAX_MANIFEST_BYTES} bytes: {manifest_path}"
            )
        try:
            with manifest_path.open("rb") as stream:
                data = stream.read(MAX_MANIFEST_BYTES + 1)
        except (OSError, ValueError, RuntimeError):
            raise KaggleDiscoveryError(
                f"Kaggle mirror manifest cannot be read: {manifest_path}"
            ) from None
        if len(data) > MAX_MANIFEST_BYTES:
            raise KaggleDiscoveryError(
                f"Kaggle mirror manifest exceeds {MAX_MANIFEST_BYTES} bytes: {manifest_path}"
            )
        try:
            payload = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
            raise KaggleDiscoveryError(
                f"Kaggle mirror manifest is not valid JSON: {manifest_path}"
            ) from None
        if not isinstance(payload, dict):
            raise KaggleDiscoveryError(
                f"Kaggle mirror manifest must contain a JSON object: {manifest_path}"
            )
        if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
            raise KaggleDiscoveryError(
                f"Kaggle mirror manifest schema_version must be 1: {manifest_path}"
            )
        source = self._required_mapping(payload, "source", manifest_path)
        target = self._required_mapping(payload, "target", manifest_path)
        upstream_tree = self._required_mapping(payload, "upstream_tree", manifest_path)
        model_id = self._string(source.get("repo_id"))
        if model_id != MODEL_ID:
            raise KaggleDiscoveryError(
                f"Kaggle mirror manifest identifies an unexpected model: {manifest_path}"
            )
        revision = self._string(source.get("resolved_commit"))
        if revision is None or _REVISION_PATTERN.fullmatch(revision) is None:
            raise KaggleDiscoveryError(
                f"Kaggle mirror manifest revision is invalid: {manifest_path}"
            )
        if self.expected_revision is not None and not hmac.compare_digest(
            revision,
            self.expected_revision,
        ):
            raise KaggleDiscoveryError(
                f"Kaggle mirror manifest revision does not match the pinned model: {manifest_path}"
            )
        handle = self._string(target.get("handle"))
        if handle is None:
            raise KaggleDiscoveryError(
                f"Kaggle mirror manifest target handle is invalid: {manifest_path}"
            )
        files = upstream_tree.get("files")
        if not isinstance(files, list):
            raise KaggleDiscoveryError(
                f"Kaggle mirror manifest file inventory is invalid: {manifest_path}"
            )
        config_hash = self._config_sha256(files, manifest_path)
        if self.expected_config_sha256 is not None and not hmac.compare_digest(
            config_hash,
            self.expected_config_sha256,
        ):
            raise KaggleDiscoveryError(
                f"Kaggle mirror manifest config identity does not match the pinned model: {manifest_path}"
            )
        metadata["mirror_handle"] = handle
        metadata["source_model_id"] = model_id
        metadata["model_revision"] = revision
        return revision, config_hash, metadata

    def _required_mapping(
        self,
        payload: dict[str, Any],
        key: str,
        manifest_path: Path,
    ) -> dict[str, Any]:
        value = payload.get(key)
        if not isinstance(value, dict):
            raise KaggleDiscoveryError(
                f"Kaggle mirror manifest section must be an object: {key}"
            )
        return value

    def _config_sha256(self, files: list[Any], manifest_path: Path) -> str:
        matches: list[str] = []
        for item in files:
            if not isinstance(item, dict):
                continue
            if item.get("path") != "config.json":
                continue
            digest = self._string(item.get("sha256"))
            if digest is None or _SHA256_PATTERN.fullmatch(digest) is None:
                raise KaggleDiscoveryError(
                    f"Kaggle mirror manifest config hash is invalid: {manifest_path}"
                )
            matches.append(digest)
        if len(matches) != 1:
            raise KaggleDiscoveryError(
                f"Kaggle mirror manifest must contain one config identity: {manifest_path}"
            )
        return matches[0]

    def _string(self, value: Any) -> str | None:
        if not isinstance(value, str) or not value or len(value) > _MAX_IDENTITY_STRING:
            return None
        return value


def discover_kaggle_model_candidates(
    root: str | os.PathLike[str] = DEFAULT_KAGGLE_MODEL_ROOT,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_entries: int = DEFAULT_MAX_ENTRIES,
    expected_revision: str | None = DEFAULT_MODEL_REVISION,
    expected_config_sha256: str | None = DEFAULT_CONFIG_SHA256,
) -> tuple[DeploymentCandidate, ...]:
    return KaggleModelMountAdapter(
        root,
        max_depth=max_depth,
        max_entries=max_entries,
        expected_revision=expected_revision,
        expected_config_sha256=expected_config_sha256,
    ).discover()


__all__ = [
    "DEFAULT_CONFIG_SHA256",
    "DEFAULT_KAGGLE_MODEL_ROOT",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_ENTRIES",
    "DEFAULT_MODEL_REVISION",
    "MAX_MANIFEST_BYTES",
    "KaggleDiscoveryError",
    "KaggleModelMountAdapter",
    "discover_kaggle_model_candidates",
]
