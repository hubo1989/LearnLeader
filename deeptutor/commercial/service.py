"""Application service for commercial subscriptions, entitlements, and usage."""

from __future__ import annotations

import calendar
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any
import uuid

from .errors import (
    CommercialConflict,
    CommercialNotFound,
    EntitlementDenied,
    IdempotencyConflict,
    InvalidOrderTransition,
)
from .keys import (
    canonical_json,
    payload_sha256,
    request_dedupe_key,
    webhook_event_dedupe_key,
)
from .models import (
    DEFAULT_MODEL_TIER,
    ESTIMATED_COST_METADATA_KEY,
    PLATFORM_COST_CREDIT_MICROS,
    PRICE_KINDS,
    UNPRICED_PRICE_VERSION,
    BillingCustomer,
    Entitlement,
    OrderStatus,
    OrderType,
    PlanVersion,
    PriceEntry,
    PriceEntryDraft,
    PriceKind,
    PurchaseOrder,
    QuotaWindow,
    ReconciliationReport,
    ResolvedEntitlements,
    Subscription,
    SubscriptionStatus,
    UsageEvent,
    UsageReservation,
    UsageReservationState,
    WebhookEvent,
    reservation_metadata_fingerprint,
    thaw_json,
)
from .repository import CommercialRepository

DEFAULT_TRIAL_DAYS = 7
DEFAULT_RESERVATION_TTL = timedelta(minutes=30)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def default_id_factory(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class CommercialControlPlane:
    """Coordinates domain rules while repositories own atomic write boundaries."""

    def __init__(
        self,
        repository: CommercialRepository,
        *,
        clock: Callable[[], datetime] = utc_now,
        id_factory: Callable[[str], str] = default_id_factory,
        trial_days: int = DEFAULT_TRIAL_DAYS,
    ) -> None:
        if trial_days <= 0:
            raise ValueError("trial_days must be positive")
        self.repository = repository
        self._clock = clock
        self._id_factory = id_factory
        self.trial_days = trial_days

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("commercial control-plane clock must be timezone-aware")
        return value

    async def ensure_billing_customer(
        self,
        *,
        owner_id: str,
        provider: str = "internal",
        external_customer_id: str | None = None,
    ) -> BillingCustomer:
        now = self._now()
        clean_owner = _required(owner_id, "owner_id")
        clean_provider = _required(provider, "provider").lower()
        clean_external = _optional(external_customer_id)
        return await self.repository.ensure_billing_customer(
            BillingCustomer(
                id=self._id_factory("cus"),
                owner_id=clean_owner,
                provider=clean_provider,
                external_customer_id=clean_external,
                trial_claimed_at=None,
                trial_subscription_id=None,
                created_at=now,
                updated_at=now,
            )
        )

    async def publish_plan_version(
        self,
        *,
        plan_code: str,
        version: int,
        name: str,
        price_minor: int,
        currency: str,
        entitlements: Mapping[str, Any],
        billing_interval: str = "month",
    ) -> PlanVersion:
        if version < 1:
            raise ValueError("plan version must be at least 1")
        if isinstance(price_minor, bool) or price_minor < 0:
            raise ValueError("price_minor must be a non-negative integer")
        clean_currency = _required(currency, "currency").upper()
        if len(clean_currency) != 3:
            raise ValueError("currency must be a three-letter ISO code")
        now = self._now()
        plan_id = self._id_factory("planv")
        plan = PlanVersion(
            id=plan_id,
            plan_code=_required(plan_code, "plan_code").lower(),
            version=version,
            name=_required(name, "name"),
            price_minor=price_minor,
            currency=clean_currency,
            billing_interval=_required(billing_interval, "billing_interval").lower(),
            trial_days=self.trial_days,
            created_at=now,
        )
        entitlement_rows: list[Entitlement] = []
        for key, value in sorted(entitlements.items()):
            clean_key = _required(str(key), "entitlement key")
            canonical_json(value)
            entitlement_rows.append(
                Entitlement(
                    id=self._id_factory("ent"),
                    plan_version_id=plan_id,
                    key=clean_key,
                    value=value,
                    created_at=now,
                )
            )
        return await self.repository.publish_plan_version(plan, tuple(entitlement_rows))

    async def list_plan_versions(self) -> tuple[PlanVersion, ...]:
        """Every published plan version, ordered for stable pricing pages."""

        return await self.repository.list_plan_versions()

    async def publish_price_version(
        self,
        *,
        price_version: str,
        kind: str,
        effective_from: datetime,
        entries: Sequence[PriceEntryDraft],
    ) -> tuple[PriceEntry, ...]:
        """Publish immutable price-catalog rows; identical replays are no-ops.

        ``kind`` selects the ledger: ``platform_cost`` prices what a metered
        unit costs the platform (budget gate and finalize-time cost
        attribution), ``list_price`` prices what it is sold for (billing
        preview).  Republishing an existing (price_version, kind, meter,
        model_tier) row with different content is rejected; price changes
        require a new ``price_version``.  The reserved ``unpriced-v1``
        sentinel is rejected as a version name, and the whole
        ``platform_cost`` ledger must stay single-currency — budget
        accounting sums raw micros, so a second currency would compare
        unrelated amounts.
        """

        clean_version = _required(price_version, "price_version")
        if clean_version == UNPRICED_PRICE_VERSION:
            raise ValueError(f"{UNPRICED_PRICE_VERSION} is reserved for unpriced events")
        clean_kind = _required(kind, "kind").lower()
        if clean_kind not in PRICE_KINDS:
            raise ValueError("kind must be platform_cost or list_price")
        _require_aware(effective_from, "effective_from")
        drafts = tuple(entries)
        if not drafts:
            raise ValueError("at least one price entry is required")
        now = self._now()
        seen_rows: set[tuple[str, str]] = set()
        rows: list[PriceEntry] = []
        for draft in drafts:
            meter = _required(draft.meter, "meter")
            tier = _required(draft.model_tier, "model_tier").lower()
            if (meter, tier) in seen_rows:
                raise ValueError(f"duplicate price row for {meter}/{tier} in one catalog version")
            seen_rows.add((meter, tier))
            if (
                isinstance(draft.unit_price_micros, bool)
                or not isinstance(draft.unit_price_micros, int)
                or draft.unit_price_micros < 0
            ):
                raise ValueError("unit_price_micros must be a non-negative integer")
            currency = _required(draft.currency, "currency").upper()
            if len(currency) != 3:
                raise ValueError("currency must be a three-letter ISO code")
            rows.append(
                PriceEntry(
                    id=self._id_factory("price"),
                    price_version=clean_version,
                    kind=clean_kind,
                    meter=meter,
                    model_tier=tier,
                    unit_price_micros=draft.unit_price_micros,
                    currency=currency,
                    effective_from=effective_from,
                    created_at=now,
                )
            )
        if clean_kind == PriceKind.PLATFORM_COST.value:
            existing = await self.repository.list_price_entries()
            currencies = {entry.currency for entry in existing if entry.kind == clean_kind}
            currencies.update(row.currency for row in rows)
            if len(currencies) > 1:
                raise ValueError("platform_cost catalog rows must share a single currency")
        return await self.repository.publish_price_entries(tuple(rows))

    async def list_price_entries(
        self,
        *,
        kind: str | None = None,
    ) -> tuple[PriceEntry, ...]:
        """Every published catalog row, newest effective_from first."""

        clean_kind = None if kind is None else _required(kind, "kind").lower()
        if clean_kind is not None and clean_kind not in PRICE_KINDS:
            raise ValueError("kind must be platform_cost or list_price")
        entries = await self.repository.list_price_entries()
        if clean_kind is not None:
            entries = tuple(entry for entry in entries if entry.kind == clean_kind)
        return entries

    async def ensure_trial(
        self,
        customer_id: str,
        plan_version_id: str,
        *,
        claim_context: Mapping[str, Any] | None = None,
    ) -> Subscription:
        now = self._now()
        plan = await self.repository.get_plan_version(plan_version_id)
        if plan is None:
            raise CommercialNotFound(f"plan version not found: {plan_version_id}")
        trial_end = now + timedelta(days=plan.trial_days)
        candidate = Subscription(
            id=self._id_factory("sub"),
            customer_id=_required(customer_id, "customer_id"),
            plan_version_id=plan.id,
            status=SubscriptionStatus.TRIALING,
            provider="internal",
            external_subscription_id=None,
            current_period_start=now,
            current_period_end=trial_end,
            trial_started_at=now,
            trial_ends_at=trial_end,
            canceled_at=None,
            version=1,
            created_at=now,
            updated_at=now,
        )
        clean_context = _clean_claim_context(claim_context)
        return await self.repository.ensure_trial(candidate, now, clean_context)

    async def suspend_subscription(self, subscription_id: str) -> Subscription:
        """Freeze effective access while keeping the live subscription slot."""

        return await self.transition_subscription(
            _required(subscription_id, "subscription_id"),
            SubscriptionStatus.SUSPENDED,
        )

    async def reinstate_subscription(self, subscription_id: str) -> Subscription:
        """Restore a suspended subscription to active access."""

        return await self.transition_subscription(
            _required(subscription_id, "subscription_id"),
            SubscriptionStatus.ACTIVE,
        )

    async def suspend_customer(self, owner_id: str) -> Subscription:
        """Suspend the owner's current subscription; used by admin controls."""

        customer = await self.get_billing_customer_by_owner(owner_id)
        if customer is None:
            raise CommercialNotFound(f"billing customer not found for owner: {owner_id}")
        subscription = await self._live_subscription(customer.id)
        if subscription is None:
            raise CommercialNotFound(f"no suspendable subscription for owner: {owner_id}")
        return await self.suspend_subscription(subscription.id)

    async def reinstate_customer(self, owner_id: str) -> Subscription:
        """Reinstate the owner's suspended subscription; used by admin controls.

        Retrying after a successful reinstate is a no-op that returns the
        already-active subscription instead of a confusing not-found.
        """

        customer = await self.get_billing_customer_by_owner(owner_id)
        if customer is None:
            raise CommercialNotFound(f"billing customer not found for owner: {owner_id}")
        subscriptions = await self.repository.list_subscriptions(customer.id)
        suspended = [item for item in subscriptions if item.status is SubscriptionStatus.SUSPENDED]
        if suspended:
            target = max(suspended, key=lambda item: (item.updated_at, item.created_at, item.id))
            return await self.reinstate_subscription(target.id)
        live = await self._live_subscription(customer.id)
        if live is not None and live.status is SubscriptionStatus.ACTIVE:
            return live
        raise CommercialNotFound(f"no suspended subscription for owner: {owner_id}")

    async def _live_subscription(self, customer_id: str) -> Subscription | None:
        now = self._now()
        live = await self.repository.get_entitled_subscription(customer_id, now)
        if live is not None:
            return live
        subscriptions = await self.repository.list_subscriptions(customer_id)
        candidates = [
            item
            for item in subscriptions
            if item.status
            in {
                SubscriptionStatus.TRIALING,
                SubscriptionStatus.ACTIVE,
                SubscriptionStatus.PAST_DUE,
                SubscriptionStatus.SUSPENDED,
            }
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item.updated_at, item.created_at, item.id))

    async def create_active_subscription(
        self,
        customer_id: str,
        plan_version_id: str,
        *,
        external_subscription_id: str,
        current_period_start: datetime,
        current_period_end: datetime,
        provider: str = "manual",
    ) -> Subscription:
        _require_aware(current_period_start, "current_period_start")
        _require_aware(current_period_end, "current_period_end")
        if current_period_end <= current_period_start:
            raise ValueError("current_period_end must be after current_period_start")
        now = self._now()
        candidate = Subscription(
            id=self._id_factory("sub"),
            customer_id=_required(customer_id, "customer_id"),
            plan_version_id=_required(plan_version_id, "plan_version_id"),
            status=SubscriptionStatus.ACTIVE,
            provider=_required(provider, "provider").lower(),
            external_subscription_id=_required(
                external_subscription_id,
                "external_subscription_id",
            ),
            current_period_start=current_period_start,
            current_period_end=current_period_end,
            trial_started_at=None,
            trial_ends_at=None,
            canceled_at=None,
            version=1,
            created_at=now,
            updated_at=now,
        )
        return await self.repository.create_subscription(candidate)

    async def activate_paid_subscription(
        self,
        customer_id: str,
        plan_version_id: str,
        *,
        provider: str,
        external_subscription_id: str,
        current_period_start: datetime,
        current_period_end: datetime,
    ) -> Subscription:
        """Atomically replace effective access with a verified provider subscription."""
        _require_aware(current_period_start, "current_period_start")
        _require_aware(current_period_end, "current_period_end")
        if current_period_end <= current_period_start:
            raise ValueError("current_period_end must be after current_period_start")
        now = self._now()
        candidate = Subscription(
            id=self._id_factory("sub"),
            customer_id=_required(customer_id, "customer_id"),
            plan_version_id=_required(plan_version_id, "plan_version_id"),
            status=SubscriptionStatus.ACTIVE,
            provider=_required(provider, "provider").lower(),
            external_subscription_id=_required(
                external_subscription_id,
                "external_subscription_id",
            ),
            current_period_start=current_period_start,
            current_period_end=current_period_end,
            trial_started_at=None,
            trial_ends_at=None,
            canceled_at=None,
            version=1,
            created_at=now,
            updated_at=now,
        )
        return await self.repository.activate_paid_subscription(candidate, now)

    async def transition_subscription(
        self,
        subscription_id: str,
        target: SubscriptionStatus,
    ) -> Subscription:
        return await self.repository.transition_subscription(
            _required(subscription_id, "subscription_id"),
            target,
            self._now(),
        )

    async def resolve_entitlements(
        self,
        customer_id: str,
        *,
        at: datetime | None = None,
    ) -> ResolvedEntitlements:
        resolved_at = at or self._now()
        _require_aware(resolved_at, "resolved_at")
        clean_customer_id = _required(customer_id, "customer_id")
        subscription = await self.repository.get_entitled_subscription(
            clean_customer_id,
            resolved_at,
        )
        if subscription is None:
            return ResolvedEntitlements(
                customer_id=clean_customer_id,
                subscription_id=None,
                plan_version_id=None,
                status=None,
                active=False,
                resolved_at=resolved_at,
                valid_until=None,
                values={},
            )
        entitlements = await self.repository.list_entitlements(subscription.plan_version_id)
        values = {item.key: thaw_json(item.value) for item in entitlements}
        valid_until = subscription.current_period_end
        if subscription.trial_ends_at is not None:
            valid_until = min(valid_until, subscription.trial_ends_at)
        return ResolvedEntitlements(
            customer_id=clean_customer_id,
            subscription_id=subscription.id,
            plan_version_id=subscription.plan_version_id,
            status=subscription.status,
            active=True,
            resolved_at=resolved_at,
            valid_until=valid_until,
            values=values,
        )

    async def record_webhook_event(
        self,
        *,
        provider: str,
        external_event_id: str,
        event_type: str,
        payload: Any,
        occurred_at: datetime | None = None,
        customer_id: str | None = None,
    ) -> WebhookEvent:
        if occurred_at is not None:
            _require_aware(occurred_at, "occurred_at")
        clean_provider = _required(provider, "provider").lower()
        clean_external_id = _required(external_event_id, "external_event_id")
        canonical_json(payload)
        now = self._now()
        return await self.repository.record_webhook_event(
            WebhookEvent(
                id=self._id_factory("wh"),
                customer_id=_optional(customer_id),
                provider=clean_provider,
                external_event_id=clean_external_id,
                dedupe_key=webhook_event_dedupe_key(clean_provider, clean_external_id),
                event_type=_required(event_type, "event_type"),
                payload=payload,
                payload_sha256=payload_sha256(payload),
                occurred_at=occurred_at,
                received_at=now,
                processed_at=None,
                processing_error=None,
                processing_attempts=0,
            )
        )

    async def mark_webhook_processed(
        self,
        event_id: str,
        *,
        processing_error: str | None = None,
    ) -> WebhookEvent:
        return await self.repository.mark_webhook_processed(
            _required(event_id, "event_id"),
            self._now(),
            _optional(processing_error),
        )

    async def reserve_usage(
        self,
        customer_id: str,
        meter: str,
        quantity: int,
        request_id: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        expires_in: timedelta = DEFAULT_RESERVATION_TTL,
    ) -> UsageReservation:
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
            raise ValueError("usage reservation quantity must be a positive integer")
        if expires_in <= timedelta(0):
            raise ValueError("expires_in must be positive")
        clean_customer_id = _required(customer_id, "customer_id")
        clean_meter = _required(meter, "meter")
        clean_request_id = _required(request_id, "request_id")
        metadata_value = dict(metadata or {})
        # The budget estimate is a server-owned reserved key: caller input
        # is dropped here and the value is injected only after pricing, so
        # a replay's fingerprint matches regardless of when it was taken.
        metadata_value.pop(ESTIMATED_COST_METADATA_KEY, None)
        canonical_json(metadata_value)
        dedupe_key = request_dedupe_key(
            "usage-reserve",
            clean_customer_id,
            clean_request_id,
        )

        # Return a durable prior result before checking today's entitlement.
        # This preserves retry safety after a timeout crosses a subscription end.
        existing = await self.repository.get_usage_reservation_by_dedupe_key(dedupe_key)
        if existing is not None:
            if (
                existing.customer_id,
                existing.meter,
                existing.quantity,
                reservation_metadata_fingerprint(existing.metadata),
            ) != (
                clean_customer_id,
                clean_meter,
                quantity,
                reservation_metadata_fingerprint(metadata_value),
            ):
                raise IdempotencyConflict(
                    "usage request id was replayed with different reservation semantics"
                )
            return existing

        resolved = await self.resolve_entitlements(clean_customer_id)
        if not resolved.active or resolved.subscription_id is None:
            raise EntitlementDenied("an active or trialing subscription is required")
        try:
            quota_policy = resolved.quota(clean_meter)
        except (KeyError, ValueError) as exc:
            raise EntitlementDenied(f"missing or invalid quota.{clean_meter} entitlement") from exc
        try:
            budget_limit_micros = _platform_budget_limit_micros(resolved.values)
        except ValueError as exc:
            raise EntitlementDenied(f"invalid platform-cost budget entitlement: {exc}") from exc
        now = self._now()
        subscription = await self._require_subscription(resolved.subscription_id)
        estimated_cost_micros = 0
        if budget_limit_micros is not None:
            model = metadata_value.get("model")
            catalog_entry = await self.repository.resolve_price(
                PriceKind.PLATFORM_COST.value,
                clean_meter,
                _resolve_model_tier(resolved.values, model if isinstance(model, str) else None),
                now,
            )
            if catalog_entry is not None:
                estimated_cost_micros = catalog_entry.unit_price_micros * quantity
                # Snapshot the estimate on the reservation so the budget
                # gate can also count in-flight reservations atomically.
                metadata_value[ESTIMATED_COST_METADATA_KEY] = estimated_cost_micros
        period_start, period_end = _quota_period(
            quota_policy.window,
            now,
            subscription.current_period_start,
            resolved.valid_until or subscription.current_period_end,
        )
        candidate = UsageReservation(
            id=self._id_factory("res"),
            dedupe_key=dedupe_key,
            customer_id=clean_customer_id,
            subscription_id=resolved.subscription_id,
            meter=clean_meter,
            quantity=quantity,
            state=UsageReservationState.RESERVED,
            metadata=metadata_value,
            period_start=period_start,
            period_end=period_end,
            expires_at=min(now + expires_in, period_end),
            created_at=now,
            finalized_at=None,
            released_at=None,
            actual_quantity=None,
            usage_event_id=None,
        )
        return await self.repository.reserve_usage(
            candidate,
            quota_policy.limit,
            budget_limit_micros=budget_limit_micros,
            estimated_cost_micros=estimated_cost_micros,
        )

    async def usage_summary(self, customer_id: str) -> dict[str, dict[str, Any]]:
        """Read-only per-meter usage inside each quota's current window.

        Mirrors the enforcement accounting (``InMemoryCommercialRepository.
        reserve_usage``): finalized usage events plus still-held reservations
        share one period, so a finalized reservation is never double-counted.
        """
        clean_customer_id = _required(customer_id, "customer_id")
        resolved = await self.resolve_entitlements(clean_customer_id)
        if resolved.subscription_id is None:
            return {}
        subscription = await self._require_subscription(resolved.subscription_id)
        now = self._now()
        events = await self.repository.list_usage_events(clean_customer_id)
        reservations = await self.repository.list_usage_reservations(clean_customer_id)

        summary: dict[str, dict[str, Any]] = {}
        for key in resolved.values:
            if not isinstance(key, str) or not key.startswith("quota."):
                continue
            meter = key[len("quota.") :]
            try:
                policy = resolved.quota(meter)
            except (KeyError, ValueError):
                continue
            period_start, period_end = _quota_period(
                policy.window,
                now,
                subscription.current_period_start,
                resolved.valid_until or subscription.current_period_end,
            )

            def _in_period(item: Any, *, _meter: str = meter) -> bool:
                return (
                    item.subscription_id == resolved.subscription_id
                    and item.meter == _meter
                    and item.period_start == period_start
                    and item.period_end == period_end
                )

            used = sum(event.quantity for event in events if _in_period(event))
            held = sum(
                item.quantity
                for item in reservations
                if item.state is UsageReservationState.RESERVED and _in_period(item)
            )
            summary[meter] = {
                "limit": policy.limit,
                "window": policy.window.value,
                "used": used,
                "reserved": held,
                "period_start": period_start,
                "period_end": period_end,
            }
        return summary

    async def usage_details(
        self,
        customer_id: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        meter: str | None = None,
        limit: int = 500,
    ) -> dict[str, Any]:
        """Filtered usage-event read model with per-day/per-model aggregates.

        A per-user accounting view over the durable ledger: raw events plus
        rollups that stay derived (never a billing source of truth). Filters
        are pushed into the repository query so a long-lived ledger never
        fully loads; ``limit`` caps the newest-first event page and the
        ``truncated`` flag says whether older events exist in the window.
        Estimated shares are per meter — mixing tokens with pages in one
        ratio would be dimensionally meaningless — plus an event-count share
        for a quick metering-health read.
        """

        clean_customer_id = _required(customer_id, "customer_id")
        if since is not None:
            _require_aware(since, "since")
        if until is not None:
            _require_aware(until, "until")
        clean_meter = _optional(meter)
        if isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        events = list(
            await self.repository.list_usage_events(
                clean_customer_id,
                since=since,
                until=until,
                meter=clean_meter,
                limit=limit + 1,
            )
        )
        truncated = len(events) > limit
        if truncated:
            events = events[:limit]
        totals: dict[str, int] = {}
        by_day: dict[str, dict[str, int]] = {}
        by_model: dict[str, dict[str, int]] = {}
        estimated_by_meter: dict[str, int] = {}
        estimated_events = 0
        for event in events:
            quantity = int(event.quantity)
            totals[event.meter] = totals.get(event.meter, 0) + quantity
            day = event.occurred_at.astimezone(timezone.utc).date().isoformat()
            by_day.setdefault(day, {}).setdefault(event.meter, 0)
            by_day[day][event.meter] += quantity
            model_key = event.model or "unknown"
            by_model.setdefault(event.meter, {}).setdefault(model_key, 0)
            by_model[event.meter][model_key] += quantity
            if event.is_estimated:
                estimated_events += 1
                estimated_by_meter[event.meter] = estimated_by_meter.get(event.meter, 0) + quantity
        share_by_meter = {
            meter_key: (quantity / totals[meter_key]) if totals.get(meter_key) else 0.0
            for meter_key, quantity in estimated_by_meter.items()
        }
        total_events = len(events)
        return {
            "events": [
                {
                    "id": event.id,
                    "meter": event.meter,
                    "quantity": int(event.quantity),
                    "provider": event.provider,
                    "model": event.model,
                    "usage_units": thaw_json(event.usage_units),
                    "price_version": event.price_version,
                    "cost_micros": int(event.cost_micros),
                    "is_estimated": event.is_estimated,
                    "metadata": thaw_json(event.metadata),
                    "occurred_at": event.occurred_at.isoformat(),
                }
                for event in events
            ],
            "totals": totals,
            "by_day": by_day,
            "by_model": by_model,
            "estimated": {
                "events": estimated_events,
                "event_share": (estimated_events / total_events if total_events else 0.0),
                "by_meter": share_by_meter,
            },
            "truncated": truncated,
            "since": since.isoformat() if since is not None else None,
            "until": until.isoformat() if until is not None else None,
        }

    async def billing_preview(self, customer_id: str) -> dict[str, Any]:
        """Read-only cost estimate for the caller's current billing position.

        The fixed charge snapshots the subscription's immutable plan-version
        price for the subscription period.  Overage is usage beyond each
        quota's plan limit **inside that quota's enforcement window** —
        today for ``utc_day`` meters, the subscription period otherwise —
        mirroring the window ``reserve_usage`` enforces, so filling a daily
        quota every day never reads as period overage.  Overage units
        consume the limit oldest-event-first and price at the ``list_price``
        catalog row effective when each event occurred, so publishing a new
        catalog row never reprices history.  Catalog misses surface as
        ``unpriced: true`` rows instead of a guessed price.  Nothing is
        written; this is a derived read model, never a billing source of
        truth.  ``totals.overage_micros`` is ``null`` when priced rows carry
        mixed currencies or a currency other than the plan's.
        """

        clean_customer_id = _required(customer_id, "customer_id")
        resolved = await self.resolve_entitlements(clean_customer_id)
        if not resolved.active or resolved.subscription_id is None:
            raise CommercialNotFound(f"no active subscription for customer: {clean_customer_id}")
        subscription = await self._require_subscription(resolved.subscription_id)
        plan = await self.repository.get_plan_version(subscription.plan_version_id)
        if plan is None:
            raise CommercialConflict("subscription references a missing plan version")
        # Extended renewals share one subscription row across paid periods, so
        # the row's current_period_* spans the whole lifetime. Use the derived
        # billing period (fulfilled-order chain) to delimit one cycle.
        from .billing_period import resolve_billing_period

        at = self._now()
        billing_period = await resolve_billing_period(self.repository, subscription, at)
        period_start = billing_period.period_start
        period_end = billing_period.period_end
        events = [
            event
            for event in await self.repository.list_usage_events(
                clean_customer_id,
                since=period_start,
                until=period_end,
            )
            if event.subscription_id == subscription.id
        ]
        # Load the list_price ledger once and resolve locally: pricing each
        # overage event at its own instant would otherwise be one catalog
        # query per event on this user-facing endpoint.
        catalog = [
            entry
            for entry in await self.repository.list_price_entries()
            if entry.kind == PriceKind.LIST_PRICE.value
        ]
        usage_rows: list[dict[str, Any]] = []
        overage_rows: list[dict[str, Any]] = []
        unpriced_meters: list[str] = []
        priced_currencies: set[str] = set()
        overage_total_micros = 0
        for key in sorted(resolved.values):
            if not isinstance(key, str) or not key.startswith("quota."):
                continue
            meter = key[len("quota.") :]
            try:
                policy = resolved.quota(meter)
            except (KeyError, ValueError):
                continue
            # Each quota is evaluated in its own enforcement window (today
            # for utc_day meters, the subscription period otherwise) — the
            # same window reserve_usage enforces, so a customer who fills a
            # daily quota every day is never reported as period overage.
            window_start, window_end = _quota_period(policy.window, at, period_start, period_end)
            meter_events = sorted(
                (
                    event
                    for event in events
                    if event.meter == meter and window_start <= event.occurred_at < window_end
                ),
                key=lambda event: (event.occurred_at, event.id),
            )
            used = sum(int(event.quantity) for event in meter_events)
            usage_rows.append(
                {
                    "meter": meter,
                    "limit": policy.limit,
                    "window": policy.window.value,
                    "window_start": window_start.isoformat(),
                    "window_end": window_end.isoformat(),
                    "used": used,
                    "excess": max(0, used - policy.limit) if policy.limit is not None else 0,
                }
            )
            if policy.limit is None:
                continue
            # Consume the limit oldest-event-first and price each overage
            # unit at the catalog row effective when that event occurred, so
            # publishing a new catalog row never reprices history.
            remaining = policy.limit
            buckets: dict[tuple[str, str | None], dict[str, Any]] = {}
            for event in meter_events:
                free = min(int(event.quantity), max(remaining, 0))
                remaining -= free
                chargeable = int(event.quantity) - free
                if chargeable <= 0:
                    continue
                tier = _resolve_model_tier(resolved.values, event.model)
                entry = _resolve_price_from(catalog, meter, tier, event.occurred_at)
                bucket_key = (tier, entry.id if entry is not None else None)
                bucket = buckets.get(bucket_key)
                if bucket is None:
                    bucket = {
                        "meter": meter,
                        "model_tier": tier,
                        "quantity": 0,
                        "unit_price_micros": entry.unit_price_micros if entry else None,
                        "cost_micros": 0,
                        "currency": entry.currency if entry else None,
                        "unpriced": entry is None,
                    }
                    buckets[bucket_key] = bucket
                    overage_rows.append(bucket)
                    if entry is None and meter not in unpriced_meters:
                        unpriced_meters.append(meter)
                bucket["quantity"] += chargeable
                if entry is not None:
                    bucket["cost_micros"] += entry.unit_price_micros * chargeable
        for row in overage_rows:
            if not row["unpriced"]:
                overage_total_micros += row["cost_micros"]
                priced_currencies.add(row["currency"])
        currency_mismatch = bool(priced_currencies) and priced_currencies != {plan.currency}
        single_currency = next(iter(priced_currencies)) if len(priced_currencies) == 1 else None
        return {
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "subscription": {
                "id": subscription.id,
                "plan_version_id": subscription.plan_version_id,
                "status": subscription.status.value,
            },
            "fixed": {"price_minor": plan.price_minor, "currency": plan.currency},
            "usage": usage_rows,
            "overage": overage_rows,
            "totals": {
                "fixed_minor": plan.price_minor,
                "fixed_currency": plan.currency,
                "overage_micros": None if currency_mismatch else overage_total_micros,
                "overage_currency": None if currency_mismatch else single_currency,
                "currency_mismatch": currency_mismatch,
                "unpriced_meters": unpriced_meters,
            },
        }

    async def finalize_usage(
        self,
        reservation_id: str,
        *,
        actual_quantity: int,
        provider: str | None = None,
        model: str | None = None,
        usage_units: Mapping[str, int | float] | None = None,
        price_version: str = "unpriced",
        cost_micros: int = 0,
        is_estimated: bool = True,
        reservation: UsageReservation | None = None,
    ) -> UsageEvent:
        """Commit final usage; ``reservation`` may preload the row to reuse.

        Callers that already fetched the reservation (for example to derive
        pricing) pass it here so the reservation is read once per finalize.
        """

        if (
            isinstance(actual_quantity, bool)
            or not isinstance(actual_quantity, int)
            or actual_quantity < 0
        ):
            raise ValueError("actual_quantity must be a non-negative integer")
        if isinstance(cost_micros, bool) or not isinstance(cost_micros, int) or cost_micros < 0:
            raise ValueError("cost_micros must be a non-negative integer")
        if not isinstance(is_estimated, bool):
            raise ValueError("is_estimated must be a boolean")
        clean_price_version = _required(price_version, "price_version")
        clean_reservation_id = _required(reservation_id, "reservation_id")
        if reservation is None:
            reservation = await self.repository.get_usage_reservation(clean_reservation_id)
        elif reservation.id != clean_reservation_id:
            raise ValueError("preloaded reservation does not match reservation_id")
        if reservation is None:
            raise CommercialNotFound(f"usage reservation not found: {clean_reservation_id}")
        units = dict(usage_units or {reservation.meter: actual_quantity})
        for unit, value in units.items():
            _required(str(unit), "usage unit")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError("usage_units values must be non-negative numbers")
        canonical_json(units)
        now = self._now()
        candidate = UsageEvent(
            id=self._id_factory("use"),
            dedupe_key=request_dedupe_key("usage-finalize", reservation.id, reservation.id),
            reservation_id=reservation.id,
            customer_id=reservation.customer_id,
            subscription_id=reservation.subscription_id,
            meter=reservation.meter,
            quantity=actual_quantity,
            provider=_optional(provider),
            model=_optional(model),
            usage_units=units,
            price_version=clean_price_version,
            cost_micros=cost_micros,
            is_estimated=is_estimated,
            metadata=thaw_json(reservation.metadata),
            period_start=reservation.period_start,
            period_end=reservation.period_end,
            occurred_at=now,
            created_at=now,
        )
        return await self.repository.finalize_usage(
            reservation.id,
            actual_quantity,
            candidate,
            now,
        )

    async def finalize_metered_usage(
        self,
        reservation_id: str,
        *,
        actual_quantity: int,
        provider: str | None = None,
        model: str | None = None,
        usage_units: Mapping[str, int | float] | None = None,
        is_estimated: bool,
    ) -> UsageEvent:
        """Finalize a reservation with catalog pricing (``platform_cost``).

        Tier and price are snapshotted to the reservation: the model tier
        comes from the ``pricing.model_tiers`` mapping of the plan the
        reserved subscription belongs to (not whatever subscription is
        current at finalize time), and the catalog row is the one effective
        when the reservation was created — a stable instant, so a retry
        after the response was lost resolves the same immutable row instead
        of conflicting with the recorded attribution.  With no matching row
        the event stays ``unpriced-v1`` with zero cost — prices are never
        invented, so the platform-cost budget stays honest until a catalog
        row exists.
        """

        clean_reservation_id = _required(reservation_id, "reservation_id")
        reservation = await self.repository.get_usage_reservation(clean_reservation_id)
        if reservation is None:
            raise CommercialNotFound(f"usage reservation not found: {clean_reservation_id}")
        subscription = await self.repository.get_subscription(reservation.subscription_id)
        if subscription is None:
            raise CommercialConflict(
                f"usage reservation references a missing subscription: "
                f"{reservation.subscription_id}"
            )
        entitlements = await self.repository.list_entitlements(subscription.plan_version_id)
        plan_values = {item.key: thaw_json(item.value) for item in entitlements}
        tier = _resolve_model_tier(plan_values, model)
        catalog_entry = await self.repository.resolve_price(
            PriceKind.PLATFORM_COST.value,
            reservation.meter,
            tier,
            reservation.created_at,
        )
        if catalog_entry is None:
            price_version, cost_micros = UNPRICED_PRICE_VERSION, 0
        else:
            price_version = catalog_entry.price_version
            cost_micros = catalog_entry.unit_price_micros * max(0, int(actual_quantity))
        return await self.finalize_usage(
            clean_reservation_id,
            actual_quantity=actual_quantity,
            provider=provider,
            model=model,
            usage_units=usage_units,
            price_version=price_version,
            cost_micros=cost_micros,
            is_estimated=is_estimated,
            reservation=reservation,
        )

    async def release_usage(self, reservation_id: str) -> UsageReservation:
        return await self.repository.release_usage(
            _required(reservation_id, "reservation_id"),
            self._now(),
        )

    async def create_order(
        self,
        customer_id: str,
        plan_version_id: str,
        *,
        order_type: OrderType = OrderType.SUBSCRIPTION_NEW,
        idempotency_key: str,
    ) -> PurchaseOrder:
        """Snapshot the plan's price and period into a pending payment order.

        Payment capture is deliberately out of scope here: the order just
        records what was agreed. Until a gateway exists, an administrator
        moves it to fulfilled after offline collection; a future webhook
        does the same through ``pay_and_fulfill_order``.
        """

        if not isinstance(order_type, OrderType):
            raise ValueError("order_type must be an OrderType")
        clean_customer_id = _required(customer_id, "customer_id")
        clean_plan_id = _required(plan_version_id, "plan_version_id")
        clean_idempotency_key = _required(idempotency_key, "idempotency_key")
        plan = await self.repository.get_plan_version(clean_plan_id)
        if plan is None:
            raise CommercialNotFound(f"plan version not found: {clean_plan_id}")
        if plan.price_minor <= 0:
            raise EntitlementDenied(
                "this plan version cannot be ordered; contact support to change plans"
            )
        now = self._now()
        period_start = now
        if order_type is OrderType.SUBSCRIPTION_RENEW:
            live = await self.repository.get_entitled_subscription(clean_customer_id, now)
            if live is not None and live.current_period_end > now:
                period_start = live.current_period_end
        period_end = _plan_period_end(plan.billing_interval, period_start)
        candidate = PurchaseOrder(
            id=self._id_factory("ord"),
            dedupe_key=request_dedupe_key("order-create", clean_customer_id, clean_idempotency_key),
            customer_id=clean_customer_id,
            plan_version_id=plan.id,
            order_type=order_type,
            price_minor=plan.price_minor,
            currency=plan.currency,
            status=OrderStatus.PENDING_PAYMENT,
            provider="manual",
            external_order_id=None,
            subscription_id=None,
            period_start=period_start,
            period_end=period_end,
            paid_at=None,
            fulfilled_at=None,
            canceled_at=None,
            refunded_at=None,
            metadata={"order_type": order_type.value},
            created_at=now,
            updated_at=now,
        )
        return await self.repository.create_order(candidate)

    async def get_order(self, order_id: str) -> PurchaseOrder:
        order = await self.repository.get_order(_required(order_id, "order_id"))
        if order is None:
            raise CommercialNotFound(f"purchase order not found: {order_id}")
        return order

    async def list_orders(
        self,
        customer_id: str,
        *,
        status: str | None = None,
    ) -> tuple[PurchaseOrder, ...]:
        return await self.repository.list_orders(
            _required(customer_id, "customer_id"),
            status=status,
        )

    async def pay_and_fulfill_order(
        self,
        order_id: str,
        *,
        operator: str,
        payment_reference: str | None = None,
    ) -> tuple[PurchaseOrder, Subscription]:
        """Confirm payment and activate the ordered access without a gap.

        An early renewal of the same plan *extends* the live subscription to
        the order's period end — replacing it immediately would cancel the
        customer's remaining paid time, and entitlement resolution ignores
        subscriptions whose period has not started. Every other fulfillment
        replaces effective access through the atomic activate path, with the
        new period starting no later than now. The replacement subscription
        is keyed to the order id (``external_subscription_id``), so a retried
        fulfillment converges on the same row instead of stacking rows.
        """

        clean_order_id = _required(order_id, "order_id")
        clean_operator = _required(operator, "operator")
        order = await self.get_order(clean_order_id)
        if order.status not in {
            OrderStatus.PENDING_PAYMENT,
            OrderStatus.PAID,
            OrderStatus.FULFILLED,
        }:
            raise InvalidOrderTransition(
                f"order in status {order.status.value} cannot be fulfilled"
            )
        now = self._now()
        if order.period_end <= now:
            raise InvalidOrderTransition(
                "the purchased period has already ended; create a new order"
            )
        live = await self.repository.get_entitled_subscription(order.customer_id, now)
        if (
            live is not None
            and live.plan_version_id == order.plan_version_id
            and order.period_start >= live.current_period_end
        ):
            return await self.repository.pay_and_fulfill_order(
                order.id,
                None,
                now,
                operator=clean_operator,
                payment_reference=_optional(payment_reference),
                extend_subscription_id=live.id,
            )
        candidate = Subscription(
            id=self._id_factory("sub"),
            customer_id=order.customer_id,
            plan_version_id=order.plan_version_id,
            status=SubscriptionStatus.ACTIVE,
            provider="manual",
            external_subscription_id=order.id,
            current_period_start=min(order.period_start, now),
            current_period_end=order.period_end,
            trial_started_at=None,
            trial_ends_at=None,
            canceled_at=None,
            version=1,
            created_at=now,
            updated_at=now,
        )
        return await self.repository.pay_and_fulfill_order(
            order.id,
            candidate,
            now,
            operator=clean_operator,
            payment_reference=_optional(payment_reference),
        )

    async def cancel_order(self, order_id: str) -> PurchaseOrder:
        return await self.repository.cancel_order(
            _required(order_id, "order_id"),
            self._now(),
        )

    async def reconcile(self, *, repair: bool = False) -> ReconciliationReport:
        """Audit derived lifecycle state and optionally apply deterministic repair."""
        return await self.repository.reconcile(self._now(), repair=repair)

    async def get_billing_customer_by_owner(
        self,
        owner_id: str,
    ) -> BillingCustomer | None:
        return await self.repository.get_billing_customer_by_owner(_required(owner_id, "owner_id"))

    async def export_customer_data(self, owner_id: str) -> dict[str, Any]:
        """Build a portable export without webhook payloads or provider customer ids."""
        customer = await self.get_billing_customer_by_owner(owner_id)
        if customer is None:
            raise CommercialNotFound(f"billing customer not found for owner: {owner_id}")
        subscriptions = await self.repository.list_subscriptions(customer.id)
        reservations = await self.repository.list_usage_reservations(customer.id)
        usage_events = await self.repository.list_usage_events(customer.id)
        orders = await self.repository.list_orders(customer.id)
        return {
            "schema_version": 1,
            "exported_at": self._now().isoformat(),
            "customer": {
                "id": customer.id,
                "owner_id": customer.owner_id,
                "trial_claimed_at": _iso(customer.trial_claimed_at),
                "created_at": customer.created_at.isoformat(),
            },
            "orders": [
                {
                    "id": order.id,
                    "plan_version_id": order.plan_version_id,
                    "order_type": order.order_type.value,
                    "price_minor": order.price_minor,
                    "currency": order.currency,
                    "status": order.status.value,
                    "period_start": order.period_start.isoformat(),
                    "period_end": order.period_end.isoformat(),
                    "paid_at": _iso(order.paid_at),
                    "fulfilled_at": _iso(order.fulfilled_at),
                    "canceled_at": _iso(order.canceled_at),
                    "created_at": order.created_at.isoformat(),
                }
                for order in orders
            ],
            "subscriptions": [
                {
                    "id": item.id,
                    "plan_version_id": item.plan_version_id,
                    "status": item.status.value,
                    "current_period_start": item.current_period_start.isoformat(),
                    "current_period_end": item.current_period_end.isoformat(),
                    "trial_started_at": _iso(item.trial_started_at),
                    "trial_ends_at": _iso(item.trial_ends_at),
                    "canceled_at": _iso(item.canceled_at),
                    "version": item.version,
                }
                for item in subscriptions
            ],
            "usage_reservations": [
                {
                    "id": item.id,
                    "subscription_id": item.subscription_id,
                    "meter": item.meter,
                    "quantity": item.quantity,
                    "state": item.state.value,
                    "period_start": item.period_start.isoformat(),
                    "period_end": item.period_end.isoformat(),
                    "created_at": item.created_at.isoformat(),
                }
                for item in reservations
            ],
            "usage_events": [
                {
                    "id": item.id,
                    "subscription_id": item.subscription_id,
                    "meter": item.meter,
                    "quantity": item.quantity,
                    "provider": item.provider,
                    "model": item.model,
                    "usage_units": thaw_json(item.usage_units),
                    "price_version": item.price_version,
                    "cost_micros": item.cost_micros,
                    "is_estimated": item.is_estimated,
                    "occurred_at": item.occurred_at.isoformat(),
                }
                for item in usage_events
            ],
        }

    async def erase_customer(self, owner_id: str) -> bool:
        """Erase pre-payment commercial data; future finance policy may anonymize instead."""
        return await self.repository.erase_customer(_required(owner_id, "owner_id"))

    async def _require_subscription(self, subscription_id: str) -> Subscription:
        subscription = await self.repository.get_subscription(subscription_id)
        if subscription is None:
            raise CommercialConflict(
                f"resolved subscription disappeared during reservation: {subscription_id}"
            )
        return subscription


def _clean_claim_context(claim_context: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Normalize trial claim signals to a hash-only JSON object.

    Only ``ip_hash``/``ua_hash`` survive, and each must be a non-empty string.
    Raw addresses or user-agent strings are never stored — callers hash them
    with SHA-256 before they reach the control plane.
    """

    if claim_context is None:
        return None
    cleaned: dict[str, Any] = {}
    for key in ("ip_hash", "ua_hash"):
        value = claim_context.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"trial claim context {key} must be a non-empty hash string")
        cleaned[key] = value.strip()
    return cleaned or None


def _resolve_model_tier(values: Mapping[str, Any], model: str | None) -> str:
    """Map a model name to its pricing tier; ``default`` when unmapped.

    The ``pricing.model_tiers`` entitlement maps model ids to tier names.
    Absent models, a missing mapping, or non-string tier values all resolve
    to the ``default`` tier: pricing falls back leniently because an
    unpriced event (cost zero) is recoverable while a dropped usage event
    is not.
    """

    tiers = values.get("pricing.model_tiers")
    if isinstance(tiers, Mapping) and model:
        tier = tiers.get(model)
        if isinstance(tier, str):
            cleaned = tier.strip().lower()
            if cleaned:
                return cleaned
    return DEFAULT_MODEL_TIER


def _resolve_price_from(
    entries: Sequence[PriceEntry],
    meter: str,
    model_tier: str,
    at: datetime,
) -> PriceEntry | None:
    """Resolve a price from an in-memory catalog slice (see ``resolve_price``).

    Mirrors the repositories' resolution rule so read models can price many
    events without a query each: latest ``effective_from`` at or before
    ``at``, ties broken by ``created_at`` then ``price_version``.
    """

    eligible = [
        entry
        for entry in entries
        if entry.meter == meter and entry.model_tier == model_tier and entry.effective_from <= at
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda entry: (entry.effective_from, entry.created_at, entry.price_version),
    )


def _platform_budget_limit_micros(values: Mapping[str, Any]) -> int | None:
    """Resolve the platform-cost budget gate; ``None`` means the gate is off.

    ``budget.platform_cost_micros`` (integer micros) wins when present;
    ``0`` intentionally blocks all platform spend.  Otherwise
    ``budget.platform_cost_credits`` applies with the fixed conversion
    **1 credit = 1000 micros** (``PLATFORM_COST_CREDIT_MICROS``), matching
    the Trial v1 entitlement that predates the micros key.  Plans without
    either key keep their pre-gate behavior; malformed values deny rather
    than silently skipping the budget.
    """

    if "budget.platform_cost_micros" in values:
        micros = values["budget.platform_cost_micros"]
        if isinstance(micros, bool) or not isinstance(micros, int) or micros < 0:
            raise ValueError("budget.platform_cost_micros must be a non-negative integer")
        return micros
    if "budget.platform_cost_credits" in values:
        credits = values["budget.platform_cost_credits"]
        if isinstance(credits, bool) or not isinstance(credits, int) or credits <= 0:
            raise ValueError("budget.platform_cost_credits must be a positive integer")
        return credits * PLATFORM_COST_CREDIT_MICROS
    return None


def _required(value: str, field: str) -> str:
    clean = value.strip()
    if not clean:
        raise ValueError(f"{field} is required")
    return clean


def _optional(value: str | None) -> str | None:
    if value is None:
        return None
    clean = value.strip()
    return clean or None


def _require_aware(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")


def _quota_period(
    window: QuotaWindow,
    now: datetime,
    subscription_start: datetime,
    subscription_end: datetime,
) -> tuple[datetime, datetime]:
    if window is QuotaWindow.SUBSCRIPTION:
        return subscription_start, subscription_end
    utc = now.astimezone(timezone.utc)
    day_start = utc.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)
    return max(day_start, subscription_start), min(day_end, subscription_end)


def _plan_period_end(interval: str, start: datetime) -> datetime:
    """Calendar-accurate next period boundary (months clamp to their end)."""

    clean = _required(interval, "billing_interval").lower()
    if clean == "month":
        year = start.year + (1 if start.month == 12 else 0)
        month = 1 if start.month == 12 else start.month + 1
        _, last_day = calendar.monthrange(year, month)
        return start.replace(year=year, month=month, day=min(start.day, last_day))
    if clean == "year":
        year = start.year + 1
        _, feb_last = calendar.monthrange(year, 2)
        return start.replace(year=year, day=min(start.day, feb_last if start.month == 2 else 31))
    raise ValueError(f"unsupported billing interval: {interval}")


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None
