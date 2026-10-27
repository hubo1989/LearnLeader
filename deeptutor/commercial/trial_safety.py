"""Helpers for capturing trial-claim abuse signals without storing PII.

Only SHA-256 digests of request-side signals ever reach the control plane.
The salted domain prefix keeps trial-claim hashes from being cross-usable
with any other hash of the same raw value.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping

_CLAIM_SALT = "deeptutor-trial-claim:v1"

_MAX_RAW_LENGTH = 512


def hash_claim_value(value: str) -> str:
    """Return the salted SHA-256 digest of one raw claim signal."""

    clean = value.strip()[:_MAX_RAW_LENGTH]
    if not clean:
        raise ValueError("claim value must be a non-empty string")
    return sha256(f"{_CLAIM_SALT}\x00{clean}".encode("utf-8")).hexdigest()


def build_trial_claim_context(
    ip: str | None,
    user_agent: str | None = None,
) -> dict[str, Any] | None:
    """Hash request-side signals into a storable claim context.

    Returns ``None`` when neither signal is usable so callers simply skip
    recording rather than writing an empty row.
    """

    context: dict[str, Any] = {}
    if ip is not None and ip.strip():
        context["ip_hash"] = hash_claim_value(ip)
    if user_agent is not None and user_agent.strip():
        context["ua_hash"] = hash_claim_value(user_agent)
    return context or None


def claim_context_from_mapping(values: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Pass through already-hashed values, rejecting raw-looking input."""

    if values is None:
        return None
    context: dict[str, Any] = {}
    for key in ("ip_hash", "ua_hash"):
        value = values.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"claim context {key} must be a non-empty hash string")
        cleaned = value.strip()
        if len(cleaned) != 64 or any(character not in "0123456789abcdef" for character in cleaned):
            raise ValueError(f"claim context {key} must be a sha256 hex digest")
        context[key] = cleaned
    return context or None


__all__ = [
    "build_trial_claim_context",
    "claim_context_from_mapping",
    "hash_claim_value",
]
