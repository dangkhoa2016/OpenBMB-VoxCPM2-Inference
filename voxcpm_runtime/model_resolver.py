from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol

from voxcpm_runtime.config import RuntimeConfig


MODEL_ID = "openbmb/VoxCPM2"
EXPECTED_ARCHITECTURE = "voxcpm2"
CONFIG_FILENAME = "config.json"
MAIN_WEIGHT_FILENAMES = ("model.safetensors", "pytorch_model.bin")
AUDIOVAE_WEIGHT_FILENAMES = ("audiovae.safetensors", "audiovae.pth")
REQUIRED_TOKENIZER_FILENAMES = ("tokenizer.json", "tokenizer_config.json")
OPTIONAL_TOKENIZER_FILENAMES = frozenset(
    {"special_tokens_map.json", "tokenization_voxcpm2.py"}
)
DEFAULT_MAX_INVENTORY_ENTRIES = 4096
DEFAULT_MAX_DEPLOYMENT_CANDIDATES = 256
MAX_CONFIG_BYTES = 1024 * 1024
MAX_TOKENIZER_CONFIG_BYTES = 1024 * 1024
MAX_TOKENIZER_JSON_BYTES = 16 * 1024 * 1024
_REQUIRED_CONFIG_SECTIONS = {
    "lm_config": frozenset(
        {
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "num_hidden_layers",
            "vocab_size",
        }
    ),
    "encoder_config": frozenset(
        {"hidden_dim", "ffn_dim", "num_heads", "num_layers"}
    ),
    "dit_config": frozenset(
        {"hidden_dim", "ffn_dim", "num_heads", "num_layers", "cfm_config"}
    ),
    "audio_vae_config": frozenset(
        {
            "encoder_dim",
            "encoder_rates",
            "decoder_dim",
            "decoder_rates",
            "sample_rate",
            "out_sample_rate",
        }
    ),
}

SourceKind = Literal[
    "explicit-path",
    "deployment-mount",
    "local-cache",
    "remote-acquired",
]
_REVISION_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z", re.ASCII)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)


class ModelResolutionError(Exception):
    code = "model_resolution_error"

    def __init__(
        self,
        message: str,
        *,
        path: str | os.PathLike[str] | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            self.path = Path(path) if path is not None else None
        except (TypeError, ValueError, RuntimeError):
            self.path = None
        self.details = dict(details or {})
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "message": str(self),
            "type": type(self).__name__,
        }
        if self.path is not None:
            result["path"] = str(self.path)
        if self.details:
            result["details"] = self.details
        return result


class ModelNotFoundError(ModelResolutionError):
    code = "model_not_found"


class InvalidModelPathError(ModelResolutionError):
    code = "invalid_model_path"


class MissingModelArtifactError(ModelResolutionError):
    code = "missing_model_artifact"


class InvalidModelConfigError(ModelResolutionError):
    code = "invalid_model_config"


class UnexpectedModelArchitectureError(ModelResolutionError):
    code = "unexpected_model_architecture"


class InvalidTokenizerMetadataError(ModelResolutionError):
    code = "invalid_tokenizer_metadata"


class OfflineResolutionError(ModelResolutionError):
    code = "offline_resolution_failed"


class RemoteAcquisitionError(ModelResolutionError):
    code = "remote_acquisition_failed"


class ModelInventoryError(ModelResolutionError):
    code = "model_inventory_failed"


class ModelCandidateError(ModelResolutionError):
    code = "invalid_model_candidate"


class RemoteAcquirer(Protocol):
    def acquire(self, config: RuntimeConfig) -> Path:
        ...


@dataclass(frozen=True, slots=True)
class DeploymentCandidate:
    path: str | os.PathLike[str]
    model_id: str | None = None
    revision: str | None = None
    expected_config_sha256: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            candidate = Path(self.path)
        except (TypeError, ValueError, RuntimeError):
            candidate = self.path
        else:
            try:
                candidate = candidate.expanduser()
            except RuntimeError:
                pass
        object.__setattr__(self, "path", candidate)
        normalized = {str(key): str(value) for key, value in self.metadata.items()}
        object.__setattr__(self, "metadata", MappingProxyType(dict(sorted(normalized.items()))))


@dataclass(frozen=True, slots=True)
class ConfigIdentity:
    architecture: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "architecture": self.architecture,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class RequiredArtifacts:
    config: str
    main_weight: str
    audiovae_weight: str
    tokenizer: tuple[str, ...]

    def to_dict(self) -> dict[str, str | list[str]]:
        return {
            "config": self.config,
            "main_weight": self.main_weight,
            "audiovae_weight": self.audiovae_weight,
            "tokenizer": list(self.tokenizer),
        }


@dataclass(frozen=True, slots=True)
class FileEntry:
    relative_path: str
    size_bytes: int
    role: str

    def to_dict(self) -> dict[str, str | int]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "role": self.role,
        }


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    path: Path
    source_kind: SourceKind
    model_id: str
    config_identity: ConfigIdentity
    revision: str | None
    inventory: tuple[FileEntry, ...]
    required_artifacts: RequiredArtifacts
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        normalized = {str(key): str(value) for key, value in self.metadata.items()}
        object.__setattr__(self, "metadata", MappingProxyType(dict(sorted(normalized.items()))))

    @property
    def file_count(self) -> int:
        return len(self.inventory)

    @property
    def total_bytes(self) -> int:
        return sum(entry.size_bytes for entry in self.inventory)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "source_kind": self.source_kind,
            "model_id": self.model_id,
            "config_identity": self.config_identity.to_dict(),
            "revision": self.revision,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "required_artifacts": self.required_artifacts.to_dict(),
            "inventory": [entry.to_dict() for entry in self.inventory],
            "metadata": dict(self.metadata),
        }


class ModelResolver:
    def resolve(
        self,
        config: RuntimeConfig,
        *,
        deployment_candidates: Iterable[
            str | os.PathLike[str] | DeploymentCandidate
        ]
        = (),
        remote_acquirer: RemoteAcquirer | None = None,
    ) -> ResolvedModel:
        if not isinstance(config, RuntimeConfig):
            raise TypeError("config must be a RuntimeConfig")
        if config.model_path is not None:
            return self._resolve_path(
                config.model_path,
                source_kind="explicit-path",
            )
        attempts: list[dict[str, Any]] = []
        for candidate in self._candidate_iterator(deployment_candidates):
            try:
                return self._resolve_candidate(candidate)
            except ModelResolutionError as error:
                attempts.append(error.to_dict())
        if config.cache_dir is not None:
            try:
                cache_path = Path(config.cache_dir).expanduser()
                cache_path.stat()
            except FileNotFoundError:
                attempts.append(
                    {
                        "code": ModelNotFoundError.code,
                        "path": str(cache_path),
                        "source_kind": "local-cache",
                        "type": ModelNotFoundError.__name__,
                    }
                )
            except (OSError, TypeError, ValueError, RuntimeError) as error:
                attempts.append(
                    {
                        "code": InvalidModelPathError.code,
                        "path": str(config.cache_dir),
                        "source_kind": "local-cache",
                        "type": type(error).__name__,
                    }
                )
            else:
                try:
                    return self._resolve_path(
                        cache_path,
                        source_kind="local-cache",
                    )
                except ModelResolutionError as error:
                    attempts.append(error.to_dict())
        if config.offline:
            raise OfflineResolutionError(
                "Offline mode is enabled and no valid local VoxCPM2 model was found",
                details={"attempts": attempts},
            )
        if remote_acquirer is None:
            raise RemoteAcquisitionError(
                "No valid local model was found and no remote acquirer was provided",
                details={"attempts": attempts},
            )
        try:
            acquired_path = remote_acquirer.acquire(config)
        except Exception as error:
            raise RemoteAcquisitionError(
                "The remote model acquirer failed",
                details={
                    "attempts": attempts,
                    "provider_error_type": type(error).__name__,
                },
            ) from None
        try:
            return self._resolve_path(
                acquired_path,
                source_kind="remote-acquired",
            )
        except ModelResolutionError as error:
            message = (
                "The remote model acquirer returned an invalid path"
                if isinstance(error, InvalidModelPathError)
                else "The remote model acquirer returned an invalid model"
            )
            raise RemoteAcquisitionError(
                message,
                details={
                    "attempts": attempts,
                    "invalid_model_code": error.code,
                    "invalid_model_type": type(error).__name__,
                },
            ) from None
        except (TypeError, ValueError):
            raise RemoteAcquisitionError(
                "The remote model acquirer returned an invalid path",
                details={
                    "attempts": attempts,
                    "remote_value_type": type(acquired_path).__name__,
                },
            ) from None

    def _candidate_iterator(
        self,
        candidates: Iterable[str | os.PathLike[str] | DeploymentCandidate],
    ) -> Iterator[DeploymentCandidate]:
        if isinstance(candidates, (str, bytes, os.PathLike, Mapping, set, frozenset)):
            raise TypeError("deployment_candidates must be an ordered iterable")
        try:
            iterator = iter(candidates)
        except TypeError:
            raise TypeError("deployment_candidates must be an ordered iterable") from None
        for count in range(DEFAULT_MAX_DEPLOYMENT_CANDIDATES + 1):
            try:
                value = next(iterator)
            except StopIteration:
                return
            except Exception as error:
                raise ModelCandidateError(
                    "Deployment candidate iteration failed",
                    details={"source_error_type": type(error).__name__},
                ) from None
            if count >= DEFAULT_MAX_DEPLOYMENT_CANDIDATES:
                raise ModelCandidateError(
                    "Deployment candidate limit exceeded",
                    details={"limit": DEFAULT_MAX_DEPLOYMENT_CANDIDATES},
                )
            if isinstance(value, DeploymentCandidate):
                yield value
            elif isinstance(value, (str, os.PathLike)):
                yield DeploymentCandidate(value)
            else:
                raise TypeError("deployment candidates must be paths or DeploymentCandidate objects")

    def _resolve_candidate(self, candidate: DeploymentCandidate) -> ResolvedModel:
        model_id = MODEL_ID if candidate.model_id is None else candidate.model_id
        if not isinstance(model_id, str) or model_id != MODEL_ID:
            raise UnexpectedModelArchitectureError(
                "Deployment metadata does not identify openbmb/VoxCPM2",
                path=candidate.path,
            )
        revision = candidate.revision
        if revision is not None and (
            not isinstance(revision, str)
            or _REVISION_PATTERN.fullmatch(revision) is None
        ):
            raise ModelCandidateError(
                "Deployment revision must be a 40- or 64-character lowercase hexadecimal object ID",
                path=candidate.path,
            )
        expected_hash = candidate.expected_config_sha256
        if expected_hash is not None and (
            not isinstance(expected_hash, str)
            or _SHA256_PATTERN.fullmatch(expected_hash) is None
        ):
            raise ModelCandidateError(
                "Deployment config identity must be a lowercase SHA-256 digest",
                path=candidate.path,
            )
        return self._resolve_path(
            candidate.path,
            source_kind="deployment-mount",
            model_id=model_id,
            revision=revision,
            expected_config_sha256=expected_hash,
            metadata=candidate.metadata,
        )

    def _resolve_path(
        self,
        path: str | os.PathLike[str],
        *,
        source_kind: SourceKind,
        model_id: str = MODEL_ID,
        revision: str | None = None,
        expected_config_sha256: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> ResolvedModel:
        root = self._validated_directory(path)
        config_identity = self._validate_config(root)
        if expected_config_sha256 is not None and not hmac.compare_digest(
            config_identity.sha256,
            expected_config_sha256,
        ):
            raise InvalidModelConfigError(
                "Model config does not match deployment metadata identity",
                path=root / CONFIG_FILENAME,
            )
        main_weight = self._select_weight(
            root,
            role="main weight",
            filenames=MAIN_WEIGHT_FILENAMES,
        )
        audiovae_weight = self._select_weight(
            root,
            role="AudioVAE weight",
            filenames=AUDIOVAE_WEIGHT_FILENAMES,
        )
        tokenizer = self._validate_tokenizer(root)
        required_artifacts = RequiredArtifacts(
            config=CONFIG_FILENAME,
            main_weight=main_weight,
            audiovae_weight=audiovae_weight,
            tokenizer=tokenizer,
        )
        inventory = self._build_inventory(root, required_artifacts)
        return ResolvedModel(
            path=root,
            source_kind=source_kind,
            model_id=model_id,
            config_identity=config_identity,
            revision=revision,
            inventory=inventory,
            required_artifacts=required_artifacts,
            metadata=metadata or {},
        )

    def _validated_directory(self, path: str | os.PathLike[str]) -> Path:
        try:
            candidate = Path(path).expanduser()
            status = candidate.stat()
        except FileNotFoundError:
            raise ModelNotFoundError(
                "Model path does not exist",
                path=path,
            ) from None
        except (OSError, TypeError, ValueError, RuntimeError):
            raise InvalidModelPathError(
                "Model path is invalid or cannot be inspected",
                path=path,
            ) from None
        if not stat.S_ISDIR(status.st_mode):
            raise InvalidModelPathError(
                "Model path must be a directory",
                path=candidate,
            )
        try:
            return candidate.resolve()
        except (OSError, RuntimeError):
            raise InvalidModelPathError(
                "Model path cannot be resolved",
                path=candidate,
            ) from None

    def _validate_config(self, root: Path) -> ConfigIdentity:
        config_path = root / CONFIG_FILENAME
        self._require_nonempty_regular_file(config_path, "Model config")
        try:
            with config_path.open("rb") as stream:
                data = stream.read(MAX_CONFIG_BYTES + 1)
        except (OSError, ValueError):
            raise InvalidModelConfigError(
                "Model config cannot be read",
                path=config_path,
            ) from None
        if len(data) > MAX_CONFIG_BYTES:
            raise InvalidModelConfigError(
                f"Model config exceeds {MAX_CONFIG_BYTES} bytes",
                path=config_path,
            )
        try:
            payload = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
            raise InvalidModelConfigError(
                "Model config is not valid bounded UTF-8 JSON",
                path=config_path,
            ) from None
        if not isinstance(payload, dict):
            raise InvalidModelConfigError(
                "Model config must contain a JSON object",
                path=config_path,
            )
        architecture = payload.get("architecture")
        if not isinstance(architecture, str) or architecture.casefold() != EXPECTED_ARCHITECTURE:
            raise UnexpectedModelArchitectureError(
                'Model config must contain architecture "voxcpm2"',
                path=config_path,
                details={"architecture_value_is_string": isinstance(architecture, str)},
            )
        self._validate_config_structure(payload, config_path)
        return ConfigIdentity(
            architecture=architecture,
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
        )

    def _validate_config_structure(
        self,
        payload: Mapping[str, Any],
        config_path: Path,
    ) -> None:
        for section, required_fields in _REQUIRED_CONFIG_SECTIONS.items():
            value = payload.get(section)
            if not isinstance(value, dict):
                raise InvalidModelConfigError(
                    f"Model config section must be an object: {section}",
                    path=config_path,
                )
            missing = sorted(required_fields.difference(value))
            if missing:
                raise InvalidModelConfigError(
                    f"Model config section is incomplete: {section}",
                    path=config_path,
                    details={"missing_fields": missing},
                )
        cfm_config = payload["dit_config"]["cfm_config"]
        if not isinstance(cfm_config, dict):
            raise InvalidModelConfigError(
                "Model config section must be an object: dit_config.cfm_config",
                path=config_path,
            )

    def _select_weight(self, root: Path, *, role: str, filenames: tuple[str, ...]) -> str:
        expected = " or ".join(filenames)
        for filename in filenames:
            path = root / filename
            try:
                status = path.stat()
            except FileNotFoundError:
                continue
            except (OSError, ValueError):
                raise InvalidModelPathError(
                    f"{role} candidate cannot be inspected: {filename}",
                    path=path,
                ) from None
            if not stat.S_ISREG(status.st_mode):
                raise MissingModelArtifactError(
                    f"{role} candidate must be a regular file: {filename}",
                    path=path,
                )
            if status.st_size == 0:
                raise MissingModelArtifactError(
                    f"{role} file is empty: {filename}",
                    path=path,
                )
            return filename
        raise MissingModelArtifactError(
            f"Missing non-empty {role}; expected {expected}",
            path=root,
        )

    def _validate_tokenizer(self, root: Path) -> tuple[str, ...]:
        for filename in REQUIRED_TOKENIZER_FILENAMES:
            self._require_nonempty_regular_file(
                root / filename,
                "Required tokenizer metadata",
            )
        tokenizer_config = self._read_tokenizer_json(
            root / "tokenizer_config.json",
            MAX_TOKENIZER_CONFIG_BYTES,
        )
        tokenizer_class = tokenizer_config.get("tokenizer_class")
        if not isinstance(tokenizer_class, str) or not tokenizer_class:
            raise InvalidTokenizerMetadataError(
                "Tokenizer config must identify tokenizer_class",
                path=root / "tokenizer_config.json",
            )
        tokenizer = self._read_tokenizer_json(
            root / "tokenizer.json",
            MAX_TOKENIZER_JSON_BYTES,
        )
        model = tokenizer.get("model")
        vocab = model.get("vocab") if isinstance(model, dict) else None
        valid_vocab = (
            isinstance(vocab, dict)
            and bool(vocab)
            and all(
                isinstance(token, str) and isinstance(index, int)
                for token, index in vocab.items()
            )
        )
        if (
            tokenizer.get("version") != "1.0"
            or not isinstance(model, dict)
            or model.get("type") != "BPE"
            or not valid_vocab
        ):
            raise InvalidTokenizerMetadataError(
                "Tokenizer JSON must contain a non-empty version 1.0 BPE model",
                path=root / "tokenizer.json",
            )
        return REQUIRED_TOKENIZER_FILENAMES

    def _read_tokenizer_json(
        self,
        path: Path,
        limit: int,
    ) -> dict[str, Any]:
        try:
            with path.open("rb") as stream:
                data = stream.read(limit + 1)
        except (OSError, ValueError):
            raise InvalidTokenizerMetadataError(
                "Tokenizer metadata cannot be read",
                path=path,
            ) from None
        if len(data) > limit:
            raise InvalidTokenizerMetadataError(
                f"Tokenizer metadata exceeds {limit} bytes",
                path=path,
            )
        try:
            payload = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
            raise InvalidTokenizerMetadataError(
                "Tokenizer metadata is not valid bounded UTF-8 JSON",
                path=path,
            ) from None
        if not isinstance(payload, dict):
            raise InvalidTokenizerMetadataError(
                "Tokenizer metadata must contain a JSON object",
                path=path,
            )
        return payload

    def _require_nonempty_regular_file(self, path: Path, label: str) -> None:
        try:
            status = path.stat()
        except FileNotFoundError:
            raise MissingModelArtifactError(
                f"Missing {label}: {path.name}",
                path=path,
            ) from None
        except (OSError, ValueError):
            raise InvalidModelPathError(
                f"{label} cannot be inspected: {path.name}",
                path=path,
            ) from None
        if not stat.S_ISREG(status.st_mode):
            raise MissingModelArtifactError(
                f"{label} must be a regular file: {path.name}",
                path=path,
            )
        if status.st_size == 0:
            raise MissingModelArtifactError(
                f"{label} is empty: {path.name}",
                path=path,
            )

    def _build_inventory(
        self,
        root: Path,
        required_artifacts: RequiredArtifacts,
    ) -> tuple[FileEntry, ...]:
        entries: list[FileEntry] = []
        try:
            with os.scandir(root) as iterator:
                for count, entry in enumerate(iterator):
                    if count >= DEFAULT_MAX_INVENTORY_ENTRIES:
                        raise ModelInventoryError(
                            "Model directory exceeds the bounded inventory limit",
                            path=root,
                            details={"limit": DEFAULT_MAX_INVENTORY_ENTRIES},
                        )
                    try:
                        if not entry.is_file(follow_symlinks=True):
                            continue
                        size = entry.stat(follow_symlinks=True).st_size
                    except OSError:
                        raise ModelInventoryError(
                            f"Inventory entry cannot be inspected: {entry.name}",
                            path=root / entry.name,
                        ) from None
                    entries.append(
                        FileEntry(
                            relative_path=entry.name,
                            size_bytes=size,
                            role=self._inventory_role(entry.name, required_artifacts),
                        )
                    )
        except ModelInventoryError:
            raise
        except OSError:
            raise ModelInventoryError(
                "Model directory inventory cannot be read",
                path=root,
            ) from None
        return tuple(sorted(entries, key=lambda item: item.relative_path))

    def _inventory_role(
        self,
        filename: str,
        required_artifacts: RequiredArtifacts,
    ) -> str:
        if filename == required_artifacts.config:
            return "config"
        if filename in MAIN_WEIGHT_FILENAMES:
            return "main-weight"
        if filename in AUDIOVAE_WEIGHT_FILENAMES:
            return "audiovae-weight"
        if filename in REQUIRED_TOKENIZER_FILENAMES or filename in OPTIONAL_TOKENIZER_FILENAMES:
            return "tokenizer"
        if filename.endswith(".json") or filename == "README.md":
            return "metadata"
        return "other"


__all__ = [
    "AUDIOVAE_WEIGHT_FILENAMES",
    "CONFIG_FILENAME",
    "ConfigIdentity",
    "DEFAULT_MAX_DEPLOYMENT_CANDIDATES",
    "DEFAULT_MAX_INVENTORY_ENTRIES",
    "DeploymentCandidate",
    "EXPECTED_ARCHITECTURE",
    "FileEntry",
    "InvalidModelConfigError",
    "InvalidModelPathError",
    "InvalidTokenizerMetadataError",
    "MAIN_WEIGHT_FILENAMES",
    "MODEL_ID",
    "MissingModelArtifactError",
    "ModelCandidateError",
    "ModelInventoryError",
    "ModelNotFoundError",
    "ModelResolutionError",
    "ModelResolver",
    "OfflineResolutionError",
    "REQUIRED_TOKENIZER_FILENAMES",
    "RemoteAcquisitionError",
    "RemoteAcquirer",
    "RequiredArtifacts",
    "ResolvedModel",
    "UnexpectedModelArchitectureError",
]
