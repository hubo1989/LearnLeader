"""Application service for commercial subscriptions, entitlements, and usage."""

from __future__ import annotations

import calendar
from collections.abc import Callable, Mapping
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
    BillingCustomer,
    Entitlement,
    OrderStatus,
    OrderType,
    PlanVersion,
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

    async def ensure_trial(self, customer_id: str, plan_version_id: str) -> Subscription:
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
        return await self.repository.ensure_trial(candidate, now)

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
                canonical_json(thaw_json(existing.metadata)),
            ) != (
                clean_customer_id,
                clean_meter,
                quantity,
                canonical_json(metadata_value),
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
        now = self._now()
        subscription = await self._require_subscription(resolved.subscription_id)
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
        return await self.repository.reserve_usage(candidate, quota_policy.limit)

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
    ) -> dict[str, Any]:
        """Filtered usage-event read model with per-day/per-model aggregates.

        A per-user accounting view over the durable ledger: raw events plus
        rollups that stay derived (never a billing source of truth). The
        estimated share flags provider calls whose usage the provider never
        reported, so a rising value is a metering-health signal.
        """

        clean_customer_id = _required(customer_id, "customer_id")
        if since is not None:
            _require_aware(since, "since")
        if until is not None:
            _require_aware(until, "until")
        clean_meter = _optional(meter)
        events = [
            event
            for event in await self.repository.list_usage_events(clean_customer_id)
            if (since is None or event.occurred_at >= since)
            and (until is None or event.occurred_at < until)
            and (clean_meter is None or event.meter == clean_meter)
        ]
        totals: dict[str, int] = {}
        by_day: dict[str, dict[str, int]] = {}
        by_model: dict[str, dict[str, int]] = {}
        estimated_events = 0
        estimated_quantity = 0
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
                estimated_quantity += quantity
        total_quantity = sum(totals.values())
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
                "quantity": estimated_quantity,
                "share": (estimated_quantity / total_quantity) if total_quantity else 0.0,
            },
            "since": since.isoformat() if since is not None else None,
            "until": until.isoformat() if until is not None else None,
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
    ) -> UsageEvent:
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
        reservation = await self.repository.get_usage_reservation(
            _required(reservation_id, "reservation_id")
        )
        if reservation is None:
            raise CommercialNotFound(f"usage reservation not found: {reservation_id}")
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
