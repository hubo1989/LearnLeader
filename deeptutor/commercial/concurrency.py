"""Owner-scoped commercial turn concurrency with durable lease support.

The Phase-2 deployment contract runs one API process, and this module's
in-memory slot registry still serves that mode (and every test that fakes a
runtime without a repository).  When the commercial runtime holds a live
repository — commercial PostgreSQL mode — the slot becomes a durable
``commercial.turn_lease`` row instead, so multi-replica deployments count the
same owner limit across processes and a crashed replica's slots expire by
TTL instead of leaking forever.  Either way the public rejection code stays
``concurrent_turn_limit``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import logging
import threading
from typing import AsyncIterator
import uuid

from deeptutor.multi_user.context import get_current_user_or_none

from .entitlement_context import (
    CommercialAccessDenied,
    integer_limit,
    require_active_commercial_access,
    require_capability,
)
from .models import TURN_LEASE_TTL_SECONDS, TurnLease
from .runtime import get_commercial_runtime

CONCURRENT_TURN_LIMIT = "concurrent_turn_limit"

logger = logging.getLogger(__name__)


class CommercialConcurrencyLimitDenied(RuntimeError):
    """Stable immediate-rejection error for an exhausted owner turn limit."""

    code = CONCURRENT_TURN_LIMIT

    def __init__(self) -> None:
        super().__init__(self.code)


_slots_guard = threading.Lock()
_active_slots: dict[str, int] = {}


def _durable_lease_repository():
    """Return the runtime repository when durable leases should be used."""

    try:
        runtime = get_commercial_runtime()
    except Exception:  # pragma: no cover - defensive: runtime construction is env-driven
        return None
    if not getattr(runtime.settings, "enabled", False):
        return None
    repository = getattr(runtime, "repository", None)
    if repository is None:
        # Commercial mode is on but the control plane has not finished
        # startup; the memory fallback keeps single-process behavior.
        return None
    return repository


@dataclass(slots=True)
class CommercialTurnLease:
    """One idempotently releasable owner slot; bypass leases have no owner."""

    owner_id: str | None = None
    lease_id: str | None = None
    repository: object | None = None
    capability: str | None = None
    _released: bool = field(default=False, init=False, repr=False)

    @property
    def reserved(self) -> bool:
        return self.owner_id is not None and not self._released

    async def heartbeat(self, *, ttl_seconds: float = TURN_LEASE_TTL_SECONDS) -> bool:
        """Extend a durable lease's TTL; a no-op for memory leases."""

        if self.repository is None or self.lease_id is None or self._released:
            return False
        record = await self.repository.heartbeat_turn_lease(
            self.lease_id,
            now=datetime.now(timezone.utc),
            ttl_seconds=ttl_seconds,
        )
        return record is not None

    async def release(self) -> None:
        if self.owner_id is None or self._released:
            return
        if self.repository is not None and self.lease_id is not None:
            try:
                await self.repository.release_turn_lease(self.lease_id)
            except Exception:
                # The lease expires by TTL even if the release write fails;
                # never let cleanup mask the turn's own outcome.
                logger.exception("failed to release turn lease %s", self.lease_id)
            self._released = True
            return
        with _slots_guard:
            if self._released:
                return
            active = _active_slots.get(self.owner_id, 0)
            if active <= 1:
                _active_slots.pop(self.owner_id, None)
            else:
                _active_slots[self.owner_id] = active - 1
            self._released = True

    async def __aenter__(self) -> CommercialTurnLease:
        return self

    async def __aexit__(self, _exc_type, _exc, _tb) -> None:
        await self.release()


async def acquire_commercial_turn_lease(
    *,
    capability: str | None = None,
) -> CommercialTurnLease:
    """Validate fresh request context and immediately reserve one owner slot.

    ``capability`` is supplied by the unified runtime, whose capability names
    map directly to versioned plan entitlements.  Legacy LLM surfaces have no
    one-to-one capability key yet, so they pass ``None`` and still require an
    active Trial/paid subscription.  Disabled commercial mode and platform
    administrators receive a no-op lease.
    """

    access = (
        require_capability(capability)
        if capability is not None
        else require_active_commercial_access()
    )
    if access is None:
        return CommercialTurnLease()

    user = get_current_user_or_none()
    if user is None or user.id != access.owner_id:
        raise CommercialAccessDenied(
            "subscription_required",
            "The active subscription does not belong to the current user.",
        )

    limit = integer_limit("concurrent_turns")
    if limit is None:
        return CommercialTurnLease()

    owner_id = access.owner_id
    repository = _durable_lease_repository()
    if repository is not None:
        now = datetime.now(timezone.utc)
        record = TurnLease(
            id=f"lease_{uuid.uuid4().hex}",
            owner_id=owner_id,
            capability=capability,
            turn_id=None,
            acquired_at=now,
            heartbeat_at=now,
            expires_at=now + timedelta(seconds=TURN_LEASE_TTL_SECONDS),
        )
        acquired = await repository.acquire_turn_lease(record, limit=limit, now=now)
        if acquired is None:
            raise CommercialConcurrencyLimitDenied()
        return CommercialTurnLease(
            owner_id=owner_id,
            lease_id=record.id,
            repository=repository,
            capability=capability,
        )

    with _slots_guard:
        active = _active_slots.get(owner_id, 0)
        if active >= limit:
            raise CommercialConcurrencyLimitDenied()
        _active_slots[owner_id] = active + 1
    return CommercialTurnLease(owner_id=owner_id, capability=capability)


@asynccontextmanager
async def commercial_turn_lease(
    *,
    capability: str | None = None,
) -> AsyncIterator[CommercialTurnLease]:
    """Acquire and unconditionally release a consuming-operation lease."""

    lease = await acquire_commercial_turn_lease(capability=capability)
    try:
        yield lease
    finally:
        await lease.release()


__all__ = [
    "CONCURRENT_TURN_LIMIT",
    "CommercialConcurrencyLimitDenied",
    "CommercialTurnLease",
    "acquire_commercial_turn_lease",
    "commercial_turn_lease",
]
