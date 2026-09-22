from __future__ import annotations

import hmac
import ipaddress
from typing import Final

from voxcpm_runtime.config import RuntimeConfig
from voxcpm_runtime.api_errors import ApiError


def is_loopback_host(host: str) -> bool:
    value = host.strip().lower()
    if value == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def validate_bind_auth_policy(config: RuntimeConfig) -> None:
    if config.require_auth and not config.api_token_configured:
        raise ApiError(
            "api_token_required",
            "Authentication is required but no API token is configured.",
            status_code=500,
        )
    if (
        not is_loopback_host(config.host)
        and not config.require_auth
        and not config.allow_unauthenticated_external
    ):
        raise ApiError(
            "unauthenticated_external_bind_forbidden",
            "Unauthenticated external binding is not allowed.",
            status_code=500,
        )


def verify_bearer_token(config: RuntimeConfig, authorization: str | None) -> None:
    if not config.require_auth:
        return
    expected = config.api_token
    if expected is None:
        raise ApiError("api_token_required", "Authentication is unavailable.", status_code=500)
    prefix = "Bearer "
    if authorization is None or not authorization.startswith(prefix):
        raise ApiError("unauthorized", "Authentication is required.", status_code=401)
    supplied = authorization[len(prefix) :]
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise ApiError("unauthorized", "Authentication is required.", status_code=401)


__all__: Final[tuple[str, ...]] = (
    "is_loopback_host",
    "validate_bind_auth_policy",
    "verify_bearer_token",
)
