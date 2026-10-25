"""Concurrency-safe in-memory repository used by tests and local composition."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime
from typing import Any

from .errors import (
    ActiveSubscriptionExists,
    CommercialConflict,
    CommercialNotFound,
    EntitlementDenied,
    IdempotencyConflict,
    ImmutablePlanVersion,
    ImmutablePriceVersion,
    InvalidOrderTransition,
    InvalidSubscriptionTransition,
    PendingOrderExists,
    TrialAlreadyClaimed,
    UsageAmountExceeded,
)
from .keys import canonical_json
from .models import (
    ALLOWED_SUBSCRIPTION_TRANSITIONS,
    ESTIMATED_USAGE_MIN_EVENTS,
    ESTIMATED_USAGE_SHARE_THRESHOLD,
    BillingCustomer,
    Entitlement,
    OrderStatus,
    PlanVersion,
    PriceEntry,
    PurchaseOrder,
    ReconciliationFinding,
    ReconciliationReport,
    Subscription,
    SubscriptionStatus,
    UsageEvent,
    UsageReservation,
    UsageReservationState,
    WebhookEvent,
    thaw_json,
)


class InMemoryCommercialRepository:
    """Reference implementation of the repository's atomicity contract.

    The single lock intentionally models transaction boundaries. It makes the
    fake useful for concurrency tests instead of merely being a dict-shaped
    stub that hides races present in production.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._customers: dict[str, BillingCustomer] = {}
        self._customer_by_owner: dict[str, str] = {}
        self._plans: dict[str, PlanVersion] = {}
        self._plan_by_key: dict[tuple[str, int], str] = {}
        self._entitlements: dict[str, tuple[Entitlement, ...]] = {}
        self._prices: dict[str, PriceEntry] = {}
        self._price_by_row: dict[tuple[str, str, str, str], str] = {}
        self._subscriptions: dict[str, Subscription] = {}
        self._subscription_by_external: dict[tuple[str, str], str] = {}
        self._webhooks: dict[str, WebhookEvent] = {}
        self._webhook_by_dedupe: dict[str, str] = {}
        self._reservations: dict[str, UsageReservation] = {}
        self._reservation_by_dedupe: dict[str, str] = {}
        self._usage_events: dict[str, UsageEvent] = {}
        self._usage_event_by_reservation: dict[str, str] = {}
        self._orders: dict[str, PurchaseOrder] = {}
        self._order_by_dedupe: dict[str, str] = {}
        self._order_by_external: dict[tuple[str, str], str] = {}

    async def ensure_billing_customer(self, candidate: BillingCustomer) -> BillingCustomer:
        async with self._lock:
            existing_id = self._customer_by_owner.get(candidate.owner_id)
            if existing_id is not None:
                existing = self._customers[existing_id]
                if (
                    existing.provider != candidate.provider
                    or existing.external_customer_id != candidate.external_customer_id
                ):
                    raise CommercialConflict(
                        "billing customer owner was reused with different provider identity"
                    )
                return existing
            if candidate.id in self._customers:
                raise CommercialConflict(f"billing customer id already exists: {candidate.id}")
            self._customers[candidate.id] = candidate
            self._customer_by_owner[candidate.owner_id] = candidate.id
            return candidate

    async def get_billing_customer(self, customer_id: str) -> BillingCustomer | None:
        async with self._lock:
            return self._customers.get(customer_id)

    async def get_billing_customer_by_owner(self, owner_id: str) -> BillingCustomer | None:
        async with self._lock:
            customer_id = self._customer_by_owner.get(owner_id)
            return self._customers.get(customer_id) if customer_id else None

    async def publish_plan_version(
        self,
        candidate: PlanVersion,
        entitlements: tuple[Entitlement, ...],
    ) -> PlanVersion:
        async with self._lock:
            key = (candidate.plan_code, candidate.version)
            existing_id = self._plan_by_key.get(key)
            if existing_id is None and candidate.id in self._plans:
                existing_id = candidate.id
            if existing_id is not None:
                existing = self._plans[existing_id]
                existing_entitlements = self._entitlements[existing.id]
                if not self._same_plan(existing, candidate) or not self._same_entitlements(
                    existing_entitlements, entitlements
                ):
                    raise ImmutablePlanVersion(
                        f"plan version {candidate.plan_code}@{candidate.version} is immutable"
                    )
                return existing
            self._plans[candidate.id] = candidate
            self._plan_by_key[key] = candidate.id
            self._entitlements[candidate.id] = tuple(
                sorted(entitlements, key=lambda item: item.key)
            )
            return candidate

    @staticmethod
    def _same_plan(existing: PlanVersion, candidate: PlanVersion) -> bool:
        return (
            existing.plan_code,
            existing.version,
            existing.name,
            existing.price_minor,
            existing.currency,
            existing.billing_interval,
            existing.trial_days,
        ) == (
            candidate.plan_code,
            candidate.version,
            candidate.name,
            candidate.price_minor,
            candidate.currency,
            candidate.billing_interval,
            candidate.trial_days,
        )

    @staticmethod
    def _same_entitlements(
        existing: tuple[Entitlement, ...],
        candidate: tuple[Entitlement, ...],
    ) -> bool:
        def normalized(values: tuple[Entitlement, ...]) -> dict[str, str]:
            return {item.key: canonical_json(thaw_json(item.value)) for item in values}

        return normalized(existing) == normalized(candidate)

    async def get_plan_version(self, plan_version_id: str) -> PlanVersion | None:
        async with self._lock:
            return self._plans.get(plan_version_id)

    async def list_plan_versions(self) -> tuple[PlanVersion, ...]:
        async with self._lock:
            return tuple(
                sorted(
                    self._plans.values(),
                    key=lambda item: (item.plan_code, item.version),
                )
            )

    async def list_entitlements(self, plan_version_id: str) -> tuple[Entitlement, ...]:
        async with self._lock:
            return self._entitlements.get(plan_version_id, ())

    async def publish_price_entries(
        self,
        entries: tuple[PriceEntry, ...],
    ) -> tuple[PriceEntry, ...]:
        async with self._lock:
            # Validate the whole batch before mutating any state, matching
            # the Postgres transaction: a rejected row can never leave
            # earlier rows of the same batch published.
            published: list[PriceEntry] = []
            pending: list[PriceEntry] = []
            for candidate in entries:
                row_key = (
                    candidate.price_version,
                    candidate.kind,
                    candidate.meter,
                    candidate.model_tier,
                )
                existing_id = self._price_by_row.get(row_key)
                if existing_id is not None:
                    existing = self._prices[existing_id]
                    if not _same_price_entry(existing, candidate):
                        raise ImmutablePriceVersion(
                            "price catalog row is immutable; publish a new price_version"
                        )
                    published.append(existing)
                    continue
                if candidate.id in self._prices:
                    raise CommercialConflict(f"price entry id already exists: {candidate.id}")
                published.append(candidate)
                pending.append(candidate)
            for candidate in pending:
                self._prices[candidate.id] = candidate
                self._price_by_row[
                    (
                        candidate.price_version,
                        candidate.kind,
                        candidate.meter,
                        candidate.model_tier,
                    )
                ] = candidate.id
            return tuple(published)

    async def resolve_price(
        self,
        kind: str,
        meter: str,
        model_tier: str,
        at: datetime,
    ) -> PriceEntry | None:
        async with self._lock:
            eligible = [
                entry
                for entry in self._prices.values()
                if entry.kind == kind
                and entry.meter == meter
                and entry.model_tier == model_tier
                and entry.effective_from <= at
            ]
            if not eligible:
                return None
            return max(
                eligible,
                key=lambda entry: (entry.effective_from, entry.created_at, entry.price_version),
            )

    async def list_price_entries(self) -> tuple[PriceEntry, ...]:
        async with self._lock:
            # Match the Postgres ORDER BY: the temporal keys descending,
            # then kind/meter/model_tier ascending (stable two-pass sort).
            rows = sorted(
                self._prices.values(),
                key=lambda entry: (entry.kind, entry.meter, entry.model_tier),
            )
            return tuple(
                sorted(
                    rows,
                    key=lambda entry: (entry.effective_from, entry.created_at, entry.price_version),
                    reverse=True,
                )
            )

    async def ensure_trial(
        self,
        candidate: Subscription,
        claimed_at: datetime,
    ) -> Subscription:
        async with self._lock:
            customer = self._require_customer(candidate.customer_id)
            self._require_plan(candidate.plan_version_id)
            if customer.trial_subscription_id is not None:
                previous = self._subscriptions.get(customer.trial_subscription_id)
                if previous is None:
                    raise CommercialConflict(
                        "billing customer references a missing trial subscription"
                    )
                if previous.plan_version_id != candidate.plan_version_id:
                    raise TrialAlreadyClaimed("the lifetime trial was claimed for another plan")
                return previous
            self._assert_no_live_subscription(candidate.customer_id)
            self._subscriptions[candidate.id] = candidate
            self._customers[customer.id] = replace(
                customer,
                trial_claimed_at=claimed_at,
                trial_subscription_id=candidate.id,
                updated_at=claimed_at,
            )
            return candidate

    async def create_subscription(self, candidate: Subscription) -> Subscription:
        async with self._lock:
            self._require_customer(candidate.customer_id)
            self._require_plan(candidate.plan_version_id)
            if candidate.external_subscription_id:
                external_key = (candidate.provider, candidate.external_subscription_id)
                existing_id = self._subscription_by_external.get(external_key)
                if existing_id is not None:
                    existing = self._subscriptions[existing_id]
                    if not self._same_subscription_identity(existing, candidate):
                        raise IdempotencyConflict(
                            "external subscription id was reused with different semantics"
                        )
                    return existing
            existing = self._subscriptions.get(candidate.id)
            if existing is not None:
                if not self._same_subscription_identity(existing, candidate):
                    raise IdempotencyConflict("subscription id was reused with different semantics")
                return existing
            if candidate.status in {
                SubscriptionStatus.TRIALING,
                SubscriptionStatus.ACTIVE,
                SubscriptionStatus.PAST_DUE,
            }:
                self._assert_no_live_subscription(candidate.customer_id)
            self._subscriptions[candidate.id] = candidate
            if candidate.external_subscription_id:
                self._subscription_by_external[
                    (candidate.provider, candidate.external_subscription_id)
                ] = candidate.id
            return candidate

    async def activate_paid_subscription(
        self,
        candidate: Subscription,
        changed_at: datetime,
    ) -> Subscription:
        """Atomically replace the effective trial/paid row with a provider row."""
        async with self._lock:
            self._require_customer(candidate.customer_id)
            self._require_plan(candidate.plan_version_id)
            if candidate.external_subscription_id:
                key = (candidate.provider, candidate.external_subscription_id)
                existing_id = self._subscription_by_external.get(key)
                if existing_id is not None:
                    existing = self._subscriptions[existing_id]
                    if not self._same_subscription_identity(existing, candidate):
                        raise IdempotencyConflict(
                            "external subscription id was reused with different semantics"
                        )
                    return existing
            for subscription in tuple(self._subscriptions.values()):
                if subscription.customer_id != candidate.customer_id:
                    continue
                if subscription.status in {
                    SubscriptionStatus.TRIALING,
                    SubscriptionStatus.ACTIVE,
                    SubscriptionStatus.PAST_DUE,
                }:
                    self._subscriptions[subscription.id] = replace(
                        subscription,
                        status=SubscriptionStatus.CANCELED,
                        canceled_at=changed_at,
                        version=subscription.version + 1,
                        updated_at=changed_at,
                    )
            self._subscriptions[candidate.id] = candidate
            if candidate.external_subscription_id:
                self._subscription_by_external[
                    (candidate.provider, candidate.external_subscription_id)
                ] = candidate.id
            return candidate

    @staticmethod
    def _same_subscription_identity(
        existing: Subscription,
        candidate: Subscription,
    ) -> bool:
        return (
            existing.customer_id,
            existing.plan_version_id,
            existing.provider,
            existing.external_subscription_id,
            existing.current_period_start,
            existing.current_period_end,
        ) == (
            candidate.customer_id,
            candidate.plan_version_id,
            candidate.provider,
            candidate.external_subscription_id,
            candidate.current_period_start,
            candidate.current_period_end,
        )

    async def get_subscription(self, subscription_id: str) -> Subscription | None:
        async with self._lock:
            return self._subscriptions.get(subscription_id)

    async def list_subscriptions(self, customer_id: str) -> tuple[Subscription, ...]:
        async with self._lock:
            return tuple(
                sorted(
                    (
                        subscription
                        for subscription in self._subscriptions.values()
                        if subscription.customer_id == customer_id
                    ),
                    key=lambda item: (item.created_at, item.id),
                )
            )

    async def transition_subscription(
        self,
        subscription_id: str,
        target: SubscriptionStatus,
        changed_at: datetime,
    ) -> Subscription:
        async with self._lock:
            existing = self._require_subscription(subscription_id)
            if existing.status is target:
                return existing
            if target not in ALLOWED_SUBSCRIPTION_TRANSITIONS[existing.status]:
                raise InvalidSubscriptionTransition(
                    f"cannot transition subscription from {existing.status.value} to {target.value}"
                )
            if target is SubscriptionStatus.ACTIVE:
                self._assert_no_live_subscription(existing.customer_id, exclude_id=existing.id)
            updated = replace(
                existing,
                status=target,
                canceled_at=(
                    changed_at
                    if target is SubscriptionStatus.CANCELED
                    else None
                    if target is SubscriptionStatus.ACTIVE
                    else existing.canceled_at
                ),
                version=existing.version + 1,
                updated_at=changed_at,
            )
            self._subscriptions[subscription_id] = updated
            return updated

    async def get_entitled_subscription(
        self,
        customer_id: str,
        at: datetime,
    ) -> Subscription | None:
        async with self._lock:
            eligible = [
                item
                for item in self._subscriptions.values()
                if item.customer_id == customer_id
                and item.status in {SubscriptionStatus.TRIALING, SubscriptionStatus.ACTIVE}
                and item.current_period_start <= at < item.current_period_end
                and (
                    item.status is not SubscriptionStatus.TRIALING
                    or (item.trial_ends_at is not None and at < item.trial_ends_at)
                )
            ]
            if len(eligible) > 1:
                raise CommercialConflict("multiple entitled subscriptions violate the invariant")
            return eligible[0] if eligible else None

    async def record_webhook_event(self, candidate: WebhookEvent) -> WebhookEvent:
        async with self._lock:
            if candidate.customer_id is not None:
                self._require_customer(candidate.customer_id)
            existing_id = self._webhook_by_dedupe.get(candidate.dedupe_key)
            if existing_id is not None:
                existing = self._webhooks[existing_id]
                if (
                    existing.customer_id != candidate.customer_id
                    or existing.provider != candidate.provider
                    or existing.external_event_id != candidate.external_event_id
                    or existing.event_type != candidate.event_type
                    or existing.payload_sha256 != candidate.payload_sha256
                ):
                    raise IdempotencyConflict(
                        "webhook event id was replayed with different content"
                    )
                return existing
            self._webhooks[candidate.id] = candidate
            self._webhook_by_dedupe[candidate.dedupe_key] = candidate.id
            return candidate

    async def mark_webhook_processed(
        self,
        event_id: str,
        processed_at: datetime,
        processing_error: str | None = None,
    ) -> WebhookEvent:
        async with self._lock:
            existing = self._webhooks.get(event_id)
            if existing is None:
                raise CommercialNotFound(f"webhook event not found: {event_id}")
            if existing.processed_at is not None:
                if processing_error is None:
                    return existing
                raise IdempotencyConflict("processed webhook cannot be marked failed")
            updated = replace(
                existing,
                processed_at=processed_at if processing_error is None else None,
                processing_error=processing_error,
                processing_attempts=existing.processing_attempts + 1,
            )
            self._webhooks[event_id] = updated
            return updated

    async def get_usage_reservation_by_dedupe_key(
        self,
        dedupe_key: str,
    ) -> UsageReservation | None:
        async with self._lock:
            reservation_id = self._reservation_by_dedupe.get(dedupe_key)
            return self._reservations.get(reservation_id) if reservation_id else None

    async def reserve_usage(
        self,
        candidate: UsageReservation,
        quota_limit: int | None,
        *,
        budget_limit_micros: int | None = None,
        estimated_cost_micros: int = 0,
    ) -> UsageReservation:
        async with self._lock:
            existing_id = self._reservation_by_dedupe.get(candidate.dedupe_key)
            if existing_id is not None:
                existing = self._reservations[existing_id]
                self._assert_same_reservation(existing, candidate)
                return existing
            self._require_customer(candidate.customer_id)
            subscription = self._require_subscription(candidate.subscription_id)
            if subscription.customer_id != candidate.customer_id:
                raise CommercialConflict("usage reservation subscription owner mismatch")
            if (
                subscription.status not in {SubscriptionStatus.TRIALING, SubscriptionStatus.ACTIVE}
                or not (
                    subscription.current_period_start
                    <= candidate.created_at
                    < subscription.current_period_end
                )
                or (
                    subscription.status is SubscriptionStatus.TRIALING
                    and (
                        subscription.trial_ends_at is None
                        or candidate.created_at >= subscription.trial_ends_at
                    )
                )
            ):
                raise EntitlementDenied(
                    "subscription became ineligible before usage reservation committed"
                )
            used = sum(
                event.quantity
                for event in self._usage_events.values()
                if self._same_usage_period(event, candidate)
            )
            reserved = sum(
                item.quantity
                for item in self._reservations.values()
                if item.state is UsageReservationState.RESERVED
                and self._same_usage_period(item, candidate)
            )
            if quota_limit is not None and used + reserved + candidate.quantity > quota_limit:
                raise EntitlementDenied(
                    f"quota.{candidate.meter} exceeded: limit={quota_limit}, "
                    f"used={used}, reserved={reserved}, requested={candidate.quantity}"
                )
            if budget_limit_micros is not None:
                # Platform-cost budget: attributed cost from finalized
                # events plus the snapshotted estimates of in-flight
                # (active) reservations, so overlapping priced calls cannot
                # each pass against the same remaining budget.
                spent = sum(
                    event.cost_micros
                    for event in self._usage_events.values()
                    if event.customer_id == candidate.customer_id
                    and event.subscription_id == candidate.subscription_id
                )
                in_flight = sum(
                    _reservation_estimate_micros(item)
                    for item in self._reservations.values()
                    if item.state is UsageReservationState.RESERVED
                    and item.customer_id == candidate.customer_id
                    and item.subscription_id == candidate.subscription_id
                )
                estimated = max(0, estimated_cost_micros)
                if spent + in_flight + estimated > budget_limit_micros:
                    raise EntitlementDenied(
                        "budget.platform_cost exceeded: "
                        f"limit_micros={budget_limit_micros}, "
                        f"spent_micros={spent}, "
                        f"in_flight_micros={in_flight}, "
                        f"estimated_micros={estimated}"
                    )
            self._reservations[candidate.id] = candidate
            self._reservation_by_dedupe[candidate.dedupe_key] = candidate.id
            return candidate

    @staticmethod
    def _same_usage_period(item: Any, candidate: UsageReservation) -> bool:
        return (
            item.customer_id == candidate.customer_id
            and item.subscription_id == candidate.subscription_id
            and item.meter == candidate.meter
            and item.period_start == candidate.period_start
            and item.period_end == candidate.period_end
        )

    @staticmethod
    def _assert_same_reservation(
        existing: UsageReservation,
        candidate: UsageReservation,
    ) -> None:
        if (
            existing.customer_id,
            existing.subscription_id,
            existing.meter,
            existing.quantity,
            canonical_json(thaw_json(existing.metadata)),
        ) != (
            candidate.customer_id,
            candidate.subscription_id,
            candidate.meter,
            candidate.quantity,
            canonical_json(thaw_json(candidate.metadata)),
        ):
            raise IdempotencyConflict(
                "usage request id was replayed with different reservation semantics"
            )

    async def get_usage_reservation(
        self,
        reservation_id: str,
    ) -> UsageReservation | None:
        async with self._lock:
            return self._reservations.get(reservation_id)

    async def list_usage_reservations(
        self,
        customer_id: str,
    ) -> tuple[UsageReservation, ...]:
        async with self._lock:
            return tuple(
                sorted(
                    (
                        item
                        for item in self._reservations.values()
                        if item.customer_id == customer_id
                    ),
                    key=lambda item: (item.created_at, item.id),
                )
            )

    async def list_usage_events(
        self,
        customer_id: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        meter: str | None = None,
        limit: int | None = None,
    ) -> tuple[UsageEvent, ...]:
        async with self._lock:
            selected = sorted(
                (
                    item
                    for item in self._usage_events.values()
                    if item.customer_id == customer_id
                    and (since is None or item.occurred_at >= since)
                    and (until is None or item.occurred_at < until)
                    and (meter is None or item.meter == meter)
                ),
                key=lambda item: (item.occurred_at, item.id),
                reverse=True,
            )
            return tuple(selected[:limit] if limit is not None else selected)

    async def finalize_usage(
        self,
        reservation_id: str,
        actual_quantity: int,
        event_candidate: UsageEvent,
        finalized_at: datetime,
    ) -> UsageEvent:
        async with self._lock:
            reservation = self._require_reservation(reservation_id)
            if reservation.state is UsageReservationState.FINALIZED:
                if reservation.actual_quantity != actual_quantity:
                    raise IdempotencyConflict("finalize retry changed the actual usage quantity")
                event_id = self._usage_event_by_reservation[reservation_id]
                existing_event = self._usage_events[event_id]
                if not _same_usage_event_accounting(existing_event, event_candidate):
                    raise IdempotencyConflict(
                        "finalize retry changed immutable usage cost attribution"
                    )
                return existing_event
            if reservation.state is UsageReservationState.RELEASED:
                raise IdempotencyConflict("released usage reservation cannot be finalized")
            if actual_quantity > reservation.quantity:
                raise UsageAmountExceeded(
                    f"actual usage {actual_quantity} exceeds reserved {reservation.quantity}"
                )
            self._usage_events[event_candidate.id] = event_candidate
            self._usage_event_by_reservation[reservation_id] = event_candidate.id
            self._reservations[reservation_id] = replace(
                reservation,
                state=UsageReservationState.FINALIZED,
                finalized_at=finalized_at,
                actual_quantity=actual_quantity,
                usage_event_id=event_candidate.id,
            )
            return event_candidate

    async def release_usage(
        self,
        reservation_id: str,
        released_at: datetime,
    ) -> UsageReservation:
        async with self._lock:
            reservation = self._require_reservation(reservation_id)
            if reservation.state is UsageReservationState.RELEASED:
                return reservation
            if reservation.state is UsageReservationState.FINALIZED:
                raise IdempotencyConflict("finalized usage reservation cannot be released")
            updated = replace(
                reservation,
                state=UsageReservationState.RELEASED,
                released_at=released_at,
            )
            self._reservations[reservation_id] = updated
            return updated

    async def reconcile(self, at: datetime, *, repair: bool) -> ReconciliationReport:
        async with self._lock:
            findings: list[ReconciliationFinding] = []
            repaired = 0
            for subscription in tuple(self._subscriptions.values()):
                if (
                    subscription.status is SubscriptionStatus.TRIALING
                    and subscription.trial_ends_at is not None
                    and subscription.trial_ends_at <= at
                ):
                    findings.append(
                        ReconciliationFinding(
                            code="expired_trial_not_materialized",
                            entity_type="subscription",
                            entity_id=subscription.id,
                            detail="trial end timestamp passed while status remained trialing",
                            repairable=True,
                        )
                    )
                    if repair:
                        self._subscriptions[subscription.id] = replace(
                            subscription,
                            status=SubscriptionStatus.EXPIRED,
                            version=subscription.version + 1,
                            updated_at=at,
                        )
                        repaired += 1
            for reservation in tuple(self._reservations.values()):
                if (
                    reservation.state is UsageReservationState.RESERVED
                    and reservation.expires_at <= at
                ):
                    findings.append(
                        ReconciliationFinding(
                            code="stale_usage_reservation",
                            entity_type="usage_reservation",
                            entity_id=reservation.id,
                            detail="reservation lease expired without finalize or release",
                            repairable=True,
                        )
                    )
                    if repair:
                        self._reservations[reservation.id] = replace(
                            reservation,
                            state=UsageReservationState.RELEASED,
                            released_at=at,
                        )
                        repaired += 1
            totals: dict[tuple[str, str], list[int]] = {}
            for event in self._usage_events.values():
                key = (event.customer_id, event.subscription_id)
                counts = totals.setdefault(key, [0, 0])
                counts[0] += 1
                if event.is_estimated:
                    counts[1] += 1
            for (customer_id, subscription_id), (total, estimated) in sorted(totals.items()):
                if (
                    total < ESTIMATED_USAGE_MIN_EVENTS
                    or estimated / total <= ESTIMATED_USAGE_SHARE_THRESHOLD
                ):
                    continue
                findings.append(
                    ReconciliationFinding(
                        code="estimated_usage_share_high",
                        entity_type="subscription",
                        entity_id=subscription_id,
                        detail=(
                            f"customer {customer_id}: {estimated}/{total} usage events "
                            f"are estimated ({estimated / total:.0%}); provider usage "
                            "reporting looks unhealthy"
                        ),
                        repairable=False,
                    )
                )
            return ReconciliationReport(
                checked_at=at,
                findings=tuple(findings),
                repaired=repaired,
            )

    async def create_order(self, candidate: PurchaseOrder) -> PurchaseOrder:
        async with self._lock:
            self._require_customer(candidate.customer_id)
            self._require_plan(candidate.plan_version_id)
            existing_id = self._order_by_dedupe.get(candidate.dedupe_key)
            if existing_id is not None:
                existing = self._orders[existing_id]
                if not self._same_order_identity(existing, candidate):
                    raise IdempotencyConflict(
                        "order idempotency key was replayed with different semantics"
                    )
                return existing
            if candidate.id in self._orders:
                raise CommercialConflict(f"purchase order id already exists: {candidate.id}")
            if candidate.external_order_id:
                external_key = (candidate.provider, candidate.external_order_id)
                if external_key in self._order_by_external:
                    raise CommercialConflict("external order id already exists for this provider")
            for order in self._orders.values():
                if (
                    order.customer_id == candidate.customer_id
                    and order.status is OrderStatus.PENDING_PAYMENT
                ):
                    raise PendingOrderExists(
                        "cancel or fulfill the open order before creating another"
                    )
            self._orders[candidate.id] = candidate
            self._order_by_dedupe[candidate.dedupe_key] = candidate.id
            if candidate.external_order_id:
                self._order_by_external[(candidate.provider, candidate.external_order_id)] = (
                    candidate.id
                )
            return candidate

    async def get_order(self, order_id: str) -> PurchaseOrder | None:
        async with self._lock:
            return self._orders.get(order_id)

    async def list_orders(
        self,
        customer_id: str,
        *,
        status: str | None = None,
    ) -> tuple[PurchaseOrder, ...]:
        async with self._lock:
            orders = [
                order
                for order in self._orders.values()
                if order.customer_id == customer_id
                and (status is None or order.status.value == status)
            ]
            return tuple(sorted(orders, key=lambda item: (item.created_at, item.id)))

    async def list_all_orders(
        self,
        *,
        status: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> tuple[PurchaseOrder, ...]:
        async with self._lock:
            orders = sorted(
                (
                    order
                    for order in self._orders.values()
                    if status is None or order.status.value == status
                ),
                key=lambda item: (item.created_at, item.id),
                reverse=True,
            )
            return tuple(orders[max(0, offset) : max(0, offset) + max(1, limit)])

    async def pay_and_fulfill_order(
        self,
        order_id: str,
        subscription_candidate: Subscription | None,
        changed_at: datetime,
        *,
        operator: str,
        payment_reference: str | None = None,
        extend_subscription_id: str | None = None,
    ) -> tuple[PurchaseOrder, Subscription]:
        async with self._lock:
            order = self._require_order(order_id)
            if order.status is OrderStatus.FULFILLED:
                subscription = self._subscriptions.get(order.subscription_id or "")
                if subscription is None:
                    raise CommercialConflict("fulfilled order lost its subscription")
                return order, subscription
            if order.status not in {OrderStatus.PENDING_PAYMENT, OrderStatus.PAID}:
                raise InvalidOrderTransition(
                    f"order in status {order.status.value} cannot be fulfilled"
                )
            if extend_subscription_id is not None:
                # Early renewal: extend the live subscription instead of
                # replacing it, so paid time is never interrupted.
                live = self._subscriptions.get(extend_subscription_id)
                if live is None or live.status not in {
                    SubscriptionStatus.TRIALING,
                    SubscriptionStatus.ACTIVE,
                    SubscriptionStatus.PAST_DUE,
                }:
                    raise CommercialConflict("renewal target subscription is not live anymore")
                if order.period_end <= live.current_period_end:
                    raise CommercialConflict("renewal order does not extend the live period")
                activated = replace(
                    live,
                    current_period_end=order.period_end,
                    version=live.version + 1,
                    updated_at=changed_at,
                )
                self._subscriptions[live.id] = activated
            else:
                assert subscription_candidate is not None
                # The activation below deduplicates on the order's external
                # subscription id, so a concurrent retry converges on one row.
                activated = self._activate_subscription_locked(
                    subscription_candidate,
                    changed_at,
                )
            metadata = dict(thaw_json(order.metadata) or {})
            metadata["fulfilled_by"] = operator
            if payment_reference:
                metadata["payment_reference"] = payment_reference
            updated = replace(
                order,
                status=OrderStatus.FULFILLED,
                paid_at=order.paid_at or changed_at,
                fulfilled_at=changed_at,
                subscription_id=activated.id,
                metadata=metadata,
                updated_at=changed_at,
            )
            self._orders[order.id] = updated
            return updated, activated

    def _activate_subscription_locked(
        self,
        candidate: Subscription,
        changed_at: datetime,
    ) -> Subscription:
        """In-lock, in-memory variant of ``activate_paid_subscription``."""

        if candidate.external_subscription_id:
            key = (candidate.provider, candidate.external_subscription_id)
            existing_id = self._subscription_by_external.get(key)
            if existing_id is not None:
                existing = self._subscriptions[existing_id]
                if not self._same_subscription_identity(existing, candidate):
                    raise IdempotencyConflict(
                        "external subscription id was reused with different semantics"
                    )
                return existing
        for subscription in tuple(self._subscriptions.values()):
            if subscription.customer_id != candidate.customer_id:
                continue
            if subscription.status in {
                SubscriptionStatus.TRIALING,
                SubscriptionStatus.ACTIVE,
                SubscriptionStatus.PAST_DUE,
            }:
                self._subscriptions[subscription.id] = replace(
                    subscription,
                    status=SubscriptionStatus.CANCELED,
                    canceled_at=changed_at,
                    version=subscription.version + 1,
                    updated_at=changed_at,
                )
        self._subscriptions[candidate.id] = candidate
        if candidate.external_subscription_id:
            self._subscription_by_external[
                (candidate.provider, candidate.external_subscription_id)
            ] = candidate.id
        return candidate

    async def cancel_order(
        self,
        order_id: str,
        canceled_at: datetime,
    ) -> PurchaseOrder:
        async with self._lock:
            order = self._require_order(order_id)
            if order.status is OrderStatus.CANCELED:
                return order
            if order.status is not OrderStatus.PENDING_PAYMENT:
                raise InvalidOrderTransition(
                    f"order in status {order.status.value} cannot be canceled"
                )
            updated = replace(
                order,
                status=OrderStatus.CANCELED,
                canceled_at=canceled_at,
                updated_at=canceled_at,
            )
            self._orders[order.id] = updated
            return updated

    def _require_order(self, order_id: str) -> PurchaseOrder:
        order = self._orders.get(order_id)
        if order is None:
            raise CommercialNotFound(f"purchase order not found: {order_id}")
        return order

    @staticmethod
    def _same_order_identity(existing: PurchaseOrder, candidate: PurchaseOrder) -> bool:
        return (
            existing.customer_id,
            existing.plan_version_id,
            existing.order_type,
            existing.price_minor,
            existing.currency,
            existing.provider,
        ) == (
            candidate.customer_id,
            candidate.plan_version_id,
            candidate.order_type,
            candidate.price_minor,
            candidate.currency,
            candidate.provider,
        )

    async def erase_customer(self, owner_id: str) -> bool:
        async with self._lock:
            customer_id = self._customer_by_owner.get(owner_id)
            if customer_id is None:
                return False
            reservation_ids = {
                item.id for item in self._reservations.values() if item.customer_id == customer_id
            }
            event_ids = {
                item.id for item in self._usage_events.values() if item.customer_id == customer_id
            }
            subscription_ids = {
                item.id for item in self._subscriptions.values() if item.customer_id == customer_id
            }
            for event_id in event_ids:
                event = self._usage_events.pop(event_id)
                self._usage_event_by_reservation.pop(event.reservation_id, None)
            for reservation_id in reservation_ids:
                reservation = self._reservations.pop(reservation_id)
                self._reservation_by_dedupe.pop(reservation.dedupe_key, None)
            for subscription_id in subscription_ids:
                subscription = self._subscriptions.pop(subscription_id)
                if subscription.external_subscription_id:
                    self._subscription_by_external.pop(
                        (subscription.provider, subscription.external_subscription_id),
                        None,
                    )
            for webhook_id, webhook in tuple(self._webhooks.items()):
                if webhook.customer_id == customer_id:
                    self._webhooks.pop(webhook_id)
                    self._webhook_by_dedupe.pop(webhook.dedupe_key, None)
            for order_id, order in tuple(self._orders.items()):
                if order.customer_id == customer_id:
                    self._orders.pop(order_id)
                    self._order_by_dedupe.pop(order.dedupe_key, None)
                    if order.external_order_id:
                        self._order_by_external.pop(
                            (order.provider, order.external_order_id),
                            None,
                        )
            self._customers.pop(customer_id)
            self._customer_by_owner.pop(owner_id, None)
            return True

    def _require_customer(self, customer_id: str) -> BillingCustomer:
        customer = self._customers.get(customer_id)
        if customer is None:
            raise CommercialNotFound(f"billing customer not found: {customer_id}")
        return customer

    def _require_plan(self, plan_version_id: str) -> PlanVersion:
        plan = self._plans.get(plan_version_id)
        if plan is None:
            raise CommercialNotFound(f"plan version not found: {plan_version_id}")
        return plan

    def _require_subscription(self, subscription_id: str) -> Subscription:
        subscription = self._subscriptions.get(subscription_id)
        if subscription is None:
            raise CommercialNotFound(f"subscription not found: {subscription_id}")
        return subscription

    def _require_reservation(self, reservation_id: str) -> UsageReservation:
        reservation = self._reservations.get(reservation_id)
        if reservation is None:
            raise CommercialNotFound(f"usage reservation not found: {reservation_id}")
        return reservation

    def _assert_no_live_subscription(
        self,
        customer_id: str,
        *,
        exclude_id: str | None = None,
    ) -> None:
        if any(
            item.customer_id == customer_id
            and item.id != exclude_id
            and item.status
            in {
                SubscriptionStatus.TRIALING,
                SubscriptionStatus.ACTIVE,
                SubscriptionStatus.PAST_DUE,
            }
            for item in self._subscriptions.values()
        ):
            raise ActiveSubscriptionExists(
                "customer already has a trialing, active, or past-due subscription"
            )


def _reservation_estimate_micros(reservation: UsageReservation) -> int:
    """Read a reservation's snapshotted cost estimate; 0 when unpriced."""

    value = thaw_json(reservation.metadata).get("estimated_cost_micros")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _same_price_entry(existing: PriceEntry, candidate: PriceEntry) -> bool:
    """Semantic identity for replayed catalog rows; ids and timestamps differ."""

    return (
        existing.price_version,
        existing.kind,
        existing.meter,
        existing.model_tier,
        existing.unit_price_micros,
        existing.currency,
        existing.effective_from,
    ) == (
        candidate.price_version,
        candidate.kind,
        candidate.meter,
        candidate.model_tier,
        candidate.unit_price_micros,
        candidate.currency,
        candidate.effective_from,
    )


def _same_usage_event_accounting(existing: UsageEvent, candidate: UsageEvent) -> bool:
    return (
        existing.quantity,
        existing.provider,
        existing.model,
        canonical_json(thaw_json(existing.usage_units)),
        existing.price_version,
        existing.cost_micros,
        existing.is_estimated,
    ) == (
        candidate.quantity,
        candidate.provider,
        candidate.model,
        canonical_json(thaw_json(candidate.usage_units)),
        candidate.price_version,
        candidate.cost_micros,
        candidate.is_estimated,
    )
