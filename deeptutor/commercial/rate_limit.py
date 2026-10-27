"""Per-owner sliding-window request/token rate limiting for risk control.

The first risk gate: a lightweight check before ``reserve_usage`` so an
over-limit caller never consumes quota reservations or provider capacity.
Entitlement keys ``risk.rpm`` (LLM requests per minute) and ``risk.tpm``
(estimated LLM tokens per minute) enable the limits; absent keys mean
unlimited (backward compatible).

Storage is an in-process sliding window — matching the single-process
deployment contract used by the turn concurrency slots.  A multi-replica
deployment must move the counters to Redis or PostgreSQL without changing
the public ``risk_rate_limited`` rejection code.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Any, Mapping

RISK_RATE_LIMITED = "risk_rate_limited"

WINDOW_SECONDS = 60.0


class CommercialRateLimitDenied(RuntimeError):
    """Stable immediate-rejection error for an exhausted RPM/TPM window."""

    code = RISK_RATE_LIMITED

    def __init__(self, message: str = RISK_RATE_LIMITED) -> None:
        super().__init__(message)


@dataclass(slots=True)
class _Window:
    requests: deque[float]
    tokens: deque[tuple[float, int]]


class SlidingWindowRateLimiter:
    """In-memory sliding-window counter; process-local by design (v1)."""

    def __init__(self, *, window_seconds: float = WINDOW_SECONDS) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._window_seconds = window_seconds
        self._guard = threading.Lock()
        self._windows: dict[str, _Window] = {}

    def record(
        self,
        owner_id: str,
        *,
        requests: int = 1,
        tokens: int = 0,
        now: float | None = None,
    ) -> None:
        """Record consumption without an external limit check."""

        clean_owner = owner_id.strip()
        if not clean_owner:
            raise ValueError("owner_id is required")
        request_count = max(0, int(requests))
        token_count = max(0, int(tokens))
        instant = time.monotonic() if now is None else float(now)
        cutoff = instant - self._window_seconds
        with self._guard:
            window = self._windows.get(clean_owner)
            if window is None:
                window = _Window(requests=deque(), tokens=deque())
                self._windows[clean_owner] = window
            self._prune(window, cutoff)
            for _ in range(request_count):
                window.requests.append(instant)
            if token_count:
                window.tokens.append((instant, token_count))

    def consume(
        self,
        owner_id: str,
        *,
        rpm_limit: int | None = None,
        tpm_limit: int | None = None,
        requests: int = 1,
        tokens: int = 0,
        now: float | None = None,
    ) -> None:
        """Prune, check both limits, and record — all in one critical section.

        A separate check-then-record pair lets concurrent callers observe the
        same prior usage and both pass; this atomic form is what callers that
        enforce limits must use. ``record``/``usage`` stay for accounting-only
        paths.
        """

        clean_owner = owner_id.strip()
        if not clean_owner:
            raise ValueError("owner_id is required")
        request_count = max(0, int(requests))
        token_count = max(0, int(tokens))
        instant = time.monotonic() if now is None else float(now)
        cutoff = instant - self._window_seconds
        with self._guard:
            window = self._windows.get(clean_owner)
            if window is None:
                window = _Window(requests=deque(), tokens=deque())
                self._windows[clean_owner] = window
            self._prune(window, cutoff)
            if rpm_limit is not None and len(window.requests) + request_count > rpm_limit:
                raise CommercialRateLimitDenied(
                    f"{RISK_RATE_LIMITED}: risk.rpm exceeded: "
                    f"limit={rpm_limit}, used={len(window.requests)}, "
                    f"requested={request_count}"
                )
            token_total = sum(amount for _, amount in window.tokens)
            if tpm_limit is not None and token_total + token_count > tpm_limit:
                raise CommercialRateLimitDenied(
                    f"{RISK_RATE_LIMITED}: risk.tpm exceeded: "
                    f"limit={tpm_limit}, used={token_total}, requested={token_count}"
                )
            for _ in range(request_count):
                window.requests.append(instant)
            if token_count:
                window.tokens.append((instant, token_count))

    def usage(
        self,
        owner_id: str,
        *,
        now: float | None = None,
    ) -> tuple[int, int]:
        """Return ``(requests, tokens)`` consumed inside the current window."""

        clean_owner = owner_id.strip()
        instant = time.monotonic() if now is None else float(now)
        cutoff = instant - self._window_seconds
        with self._guard:
            window = self._windows.get(clean_owner)
            if window is None:
                return 0, 0
            self._prune(window, cutoff)
            token_total = sum(amount for _, amount in window.tokens)
            return len(window.requests), token_total

    def reset(self, owner_id: str | None = None) -> None:
        """Clear one owner's window (or all windows) — intended for tests."""

        with self._guard:
            if owner_id is None:
                self._windows.clear()
            else:
                self._windows.pop(owner_id.strip(), None)

    @staticmethod
    def _prune(window: _Window, cutoff: float) -> None:
        while window.requests and window.requests[0] <= cutoff:
            window.requests.popleft()
        while window.tokens and window.tokens[0][0] <= cutoff:
            window.tokens.popleft()


_default_limiter = SlidingWindowRateLimiter()


def get_rate_limiter() -> SlidingWindowRateLimiter:
    """Process-wide limiter used by the metering path."""
    return _default_limiter


def parse_risk_limit(values: Mapping[str, Any], key: str) -> int | None:
    """Parse ``risk.rpm``/``risk.tpm``; ``None`` means unlimited.

    Malformed values fail closed as ``0`` is not silently substituted —
    callers raise ``ValueError`` so the metering layer denies the request.
    """

    if key not in values:
        return None
    raw = values[key]
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return raw


def enforce_rate_limits(
    values: Mapping[str, Any],
    *,
    owner_id: str,
    requests: int = 1,
    tokens: int = 0,
    limiter: SlidingWindowRateLimiter | None = None,
    now: float | None = None,
) -> None:
    """Check RPM/TPM windows and record the call when both pass.

    Missing ``risk.*`` keys skip that dimension entirely so plans without the
    entitlements keep their pre-limit behavior.
    """

    rpm_limit = parse_risk_limit(values, "risk.rpm")
    tpm_limit = parse_risk_limit(values, "risk.tpm")
    if rpm_limit is None and tpm_limit is None:
        return
    active = limiter or _default_limiter
    active.consume(
        owner_id,
        rpm_limit=rpm_limit,
        tpm_limit=tpm_limit,
        requests=requests,
        tokens=tokens,
        now=now,
    )


__all__ = [
    "CommercialRateLimitDenied",
    "RISK_RATE_LIMITED",
    "SlidingWindowRateLimiter",
    "enforce_rate_limits",
    "get_rate_limiter",
    "parse_risk_limit",
]
