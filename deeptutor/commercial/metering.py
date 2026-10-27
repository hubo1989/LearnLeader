"""Live commercial usage reservations at platform provider boundaries.

Cost attribution happens at finalize time through the versioned price
catalog (``platform_cost`` ledger): the reservation's meter and the
customer's model tier resolve to a catalog row, and the event records that
row's ``price_version`` and computed ``cost_micros``.  With no catalog row
the event stays ``unpriced-v1`` with zero cost — prices are never invented,
so the platform-cost budget gate cannot be silently satisfied by fiction.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, TypeVar
import uuid

from deeptutor.multi_user.context import get_current_user_or_none

from .entitlement_context import get_commercial_access
from .errors import CommercialError, EntitlementDenied
from .models import UNPRICED_PRICE_VERSION, UsageEvent, UsageReservation
from .postgres import PostgresCommercialRepository
from .runtime import CommercialRuntime, get_commercial_runtime
from .service import CommercialControlPlane


class CommercialMeteringError(RuntimeError):
    """Base class for fail-closed live metering failures."""


class CommercialUsageLimitExceeded(CommercialMeteringError):
    """The current subscription cannot reserve the requested quantity."""


class CommercialUsageRateLimited(CommercialUsageLimitExceeded):
    """The owner exhausted a risk-control RPM/TPM window (HTTP 429)."""

    code = "risk_rate_limited"


class CommercialUsageUnavailable(CommercialMeteringError):
    """The durable commercial usage ledger could not be reached safely."""


@dataclass(frozen=True, slots=True)
class UsageMeasurement:
    """Normalized provider usage or a conservative reservation fallback."""

    quantity: int
    units: Mapping[str, int | float]
    is_estimated: bool


@dataclass(slots=True)
class CommercialUsageLease:
    """An asynchronous reservation finalized or released exactly once."""

    control_plane: CommercialControlPlane
    reservation: UsageReservation
    provider: str | None
    model: str | None
    _state: str = "active"
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def finalize(
        self,
        actual_quantity: int,
        *,
        usage_units: Mapping[str, int | float] | None = None,
        is_estimated: bool,
    ) -> UsageEvent | None:
        """Commit measured usage once; a successful retry returns a no-op."""

        async with self._lock:
            if self._state != "active":
                return None
            try:
                event = await self.control_plane.finalize_metered_usage(
                    self.reservation.id,
                    actual_quantity=max(0, int(actual_quantity)),
                    provider=self.provider,
                    model=self.model,
                    usage_units=usage_units,
                    is_estimated=is_estimated,
                )
            except Exception as exc:
                raise _translate_error(exc) from exc
            self._state = "finalized"
            return event

    async def release(self) -> None:
        """Release unused capacity once after provider failure or cancellation."""

        async with self._lock:
            if self._state != "active":
                return
            try:
                await self.control_plane.release_usage(self.reservation.id)
            except Exception as exc:
                raise _translate_error(exc) from exc
            self._state = "released"


@dataclass(slots=True)
class SyncCommercialUsageLease:
    """Synchronous facade used by the existing MinerU parser contract."""

    runtime: CommercialRuntime
    reservation: UsageReservation
    provider: str | None
    model: str | None
    _state: str = "active"

    def finalize(
        self,
        actual_quantity: int,
        *,
        usage_units: Mapping[str, int | float] | None = None,
        is_estimated: bool,
    ) -> UsageEvent | None:
        if self._state != "active":
            return None

        async def operation(control_plane: CommercialControlPlane) -> UsageEvent:
            return await control_plane.finalize_metered_usage(
                self.reservation.id,
                actual_quantity=max(0, int(actual_quantity)),
                provider=self.provider,
                model=self.model,
                usage_units=usage_units,
                is_estimated=is_estimated,
            )

        event = _run_control_plane_sync(self.runtime, operation)
        self._state = "finalized"
        return event

    def release(self) -> None:
        if self._state != "active":
            return

        async def operation(control_plane: CommercialControlPlane) -> None:
            await control_plane.release_usage(self.reservation.id)

        _run_control_plane_sync(self.runtime, operation)
        self._state = "released"


def commercial_usage_required(*, source: str) -> bool:
    """Whether a call must enter the commercial ledger before provider work."""

    if source not in {"platform", "byok"}:
        return False
    try:
        runtime = get_commercial_runtime()
    except Exception:
        # Keep the provider wrapper installed so reservation fails closed later.
        return True
    if not runtime.settings.enabled:
        return False
    user = get_current_user_or_none()
    return user is None or not user.is_admin


async def reserve_commercial_usage(
    *,
    meter: str,
    quantity: int,
    source: str,
    provider: str | None,
    model: str | None,
    request_id: str | None = None,
    count_byok: bool = False,
) -> CommercialUsageLease | None:
    """Validate access and optionally reserve a meter before provider work.

    BYOK calls always pass the subscription gate, but normally keep their
    user-funded tokens out of platform usage totals.  Resource limits that are
    source-independent (currently MinerU pages) opt in with ``count_byok``.
    """

    context = _metering_context(source=source, count_byok=count_byok)
    if context is None:
        return None
    runtime, customer_id, access = context
    control_plane = runtime.control_plane
    if control_plane is None:
        raise CommercialUsageUnavailable("commercial usage ledger has not started")
    reservation = await _reserve(
        control_plane,
        customer_id=customer_id,
        meter=meter,
        quantity=quantity,
        provider=provider,
        model=model,
        source=source,
        request_id=request_id,
        owner_id=access.owner_id,
        plan_values=access.resolved.values,
    )
    return CommercialUsageLease(
        control_plane=control_plane,
        reservation=reservation,
        provider=_optional(provider),
        model=_optional(model),
    )


def reserve_commercial_usage_sync(
    *,
    meter: str,
    quantity: int,
    source: str,
    provider: str | None,
    model: str | None,
    request_id: str | None = None,
    count_byok: bool = False,
) -> SyncCommercialUsageLease | None:
    """Reserve usage for a synchronous provider boundary such as MinerU.

    If the parser is running on the same event-loop thread that owns asyncpg,
    a short-lived one-connection repository performs only the ledger operation.
    This avoids both an unsafe cross-loop pool call and provider work occurring
    before the reservation.  Worker-thread calls reuse the lifespan pool via
    ``run_coroutine_threadsafe``.
    """

    context = _metering_context(source=source, count_byok=count_byok)
    if context is None:
        return None
    runtime, customer_id, access = context

    async def operation(control_plane: CommercialControlPlane) -> UsageReservation:
        return await _reserve(
            control_plane,
            customer_id=customer_id,
            meter=meter,
            quantity=quantity,
            provider=provider,
            model=model,
            source=source,
            request_id=request_id,
            owner_id=access.owner_id,
            plan_values=access.resolved.values,
        )

    reservation = _run_control_plane_sync(runtime, operation)
    return SyncCommercialUsageLease(
        runtime=runtime,
        reservation=reservation,
        provider=_optional(provider),
        model=_optional(model),
    )


def measurement_from_usage(
    usage: Any,
    *,
    fallback_quantity: int,
    meter: str,
) -> UsageMeasurement:
    """Normalize common provider schemas, otherwise book the reservation bound."""

    units: dict[str, int | float] = {}
    for key in (
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "tokens",
        "pages",
    ):
        value = _usage_value(usage, key)
        if value is not None:
            units[key] = value

    total = _positive_int(units.get("total_tokens"))
    if total <= 0:
        total = _positive_int(units.get("tokens"))
    if total <= 0:
        total = _positive_int(units.get("pages"))
    if total <= 0:
        total = max(
            _positive_int(units.get("input_tokens")) + _positive_int(units.get("output_tokens")),
            _positive_int(units.get("prompt_tokens"))
            + _positive_int(units.get("completion_tokens")),
        )
    if total > 0:
        return UsageMeasurement(total, units or {meter: total}, False)
    fallback = max(1, int(fallback_quantity))
    return UsageMeasurement(fallback, {meter: fallback}, True)


async def _reserve(
    control_plane: CommercialControlPlane,
    *,
    customer_id: str,
    meter: str,
    quantity: int,
    provider: str | None,
    model: str | None,
    source: str,
    request_id: str | None,
    owner_id: str | None = None,
    plan_values: Mapping[str, Any] | None = None,
) -> UsageReservation:
    requested = max(1, int(quantity))
    stable_request_id = request_id or f"{meter}:{uuid.uuid4().hex}"
    metadata = {
        "provider": _optional(provider),
        "model": _optional(model),
        "price_version": UNPRICED_PRICE_VERSION,
        "source": "byok" if source == "byok" else "platform",
    }
    if owner_id is not None and plan_values is not None and meter == "llm_tokens":
        # Risk gate before quota: an over-limit caller must not consume a
        # reservation or provider capacity.  Only the LLM meter carries
        # RPM/TPM semantics; embedding and page meters have their own quotas.
        from .rate_limit import CommercialRateLimitDenied, enforce_rate_limits

        try:
            enforce_rate_limits(
                plan_values,
                owner_id=owner_id,
                requests=1,
                tokens=requested,
            )
        except CommercialRateLimitDenied as exc:
            raise CommercialUsageRateLimited(str(exc)) from exc
        except ValueError as exc:
            raise CommercialUsageRateLimited(f"risk_rate_limited: {exc}") from exc
    try:
        return await control_plane.reserve_usage(
            customer_id,
            meter,
            requested,
            stable_request_id,
            metadata=metadata,
        )
    except Exception as exc:
        raise _translate_error(exc) from exc


def _metering_context(
    *,
    source: str,
    count_byok: bool = False,
) -> tuple[CommercialRuntime, str, Any] | None:
    try:
        runtime = get_commercial_runtime()
    except Exception as exc:
        raise CommercialUsageUnavailable("commercial usage ledger configuration failed") from exc
    if not runtime.settings.enabled:
        return None
    user = get_current_user_or_none()
    if user is None:
        raise CommercialUsageUnavailable("commercial usage requires an authenticated owner")
    if user.is_admin:
        return None
    access = get_commercial_access()
    if access is None or access.owner_id != user.id or not access.is_current:
        raise CommercialUsageLimitExceeded("an active subscription or trial is required")
    customer_id = access.resolved.customer_id.strip()
    if not customer_id:
        raise CommercialUsageUnavailable("commercial access has no billing customer")
    if source == "byok" and not count_byok:
        return None
    return runtime, customer_id, access


T = TypeVar("T")
_ControlPlaneOperation = Callable[[CommercialControlPlane], Awaitable[T]]


def _run_control_plane_sync(
    runtime: CommercialRuntime,
    operation: _ControlPlaneOperation[T],
) -> T:
    control_plane = runtime.control_plane
    if control_plane is None:
        raise CommercialUsageUnavailable("commercial usage ledger has not started")

    repository = getattr(runtime, "repository", None)
    pool_loop = getattr(getattr(repository, "pool", None), "_loop", None)
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    try:
        if pool_loop is not None and pool_loop.is_running() and pool_loop is not running_loop:
            return asyncio.run_coroutine_threadsafe(operation(control_plane), pool_loop).result()
        if isinstance(repository, PostgresCommercialRepository):
            return _run_in_helper_thread(lambda: _run_with_temporary_postgres(runtime, operation))
        if running_loop is None:
            return asyncio.run(operation(control_plane))
        return _run_in_helper_thread(lambda: asyncio.run(operation(control_plane)))
    except Exception as exc:
        raise _translate_error(exc) from exc


def _run_with_temporary_postgres(
    runtime: CommercialRuntime,
    operation: _ControlPlaneOperation[T],
) -> T:
    database_url = str(runtime.settings.database_url or "").strip()
    if not database_url:
        raise CommercialUsageUnavailable("commercial database URL is unavailable")

    async def runner() -> T:
        repository = await PostgresCommercialRepository.connect(
            database_url,
            min_size=1,
            max_size=1,
            run_migrations=False,
        )
        try:
            return await operation(CommercialControlPlane(repository))
        finally:
            await repository.close()

    return asyncio.run(runner())


def _run_in_helper_thread(call: Callable[[], T]) -> T:
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="commercial-metering") as executor:
        return executor.submit(call).result()


def _translate_error(exc: Exception) -> CommercialMeteringError:
    if isinstance(exc, CommercialMeteringError):
        return exc
    if isinstance(exc, EntitlementDenied):
        return CommercialUsageLimitExceeded(str(exc))
    if isinstance(exc, CommercialError):
        return CommercialUsageUnavailable(str(exc))
    return CommercialUsageUnavailable(f"commercial usage ledger unavailable: {exc}")


def _optional(value: str | None) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def _usage_value(usage: Any, key: str) -> int | float | None:
    if usage is None:
        return None
    value = usage.get(key) if isinstance(usage, Mapping) else getattr(usage, key, None)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return value


def _positive_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


__all__ = [
    "CommercialMeteringError",
    "CommercialUsageLease",
    "CommercialUsageLimitExceeded",
    "CommercialUsageUnavailable",
    "SyncCommercialUsageLease",
    "UNPRICED_PRICE_VERSION",
    "UsageMeasurement",
    "commercial_usage_required",
    "measurement_from_usage",
    "reserve_commercial_usage",
    "reserve_commercial_usage_sync",
]
