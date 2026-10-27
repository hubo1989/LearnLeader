"""Persistence boundary for the commercial control plane."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from .models import (
    BillingCustomer,
    Entitlement,
    PlanVersion,
    PriceEntry,
    PurchaseOrder,
    ReconciliationReport,
    Subscription,
    SubscriptionStatus,
    TurnLease,
    UsageEvent,
    UsageReservation,
    WebhookEvent,
)


@runtime_checkable
class CommercialRepository(Protocol):
    async def ensure_billing_customer(self, candidate: BillingCustomer) -> BillingCustomer: ...

    async def get_billing_customer(self, customer_id: str) -> BillingCustomer | None: ...

    async def get_billing_customer_by_owner(self, owner_id: str) -> BillingCustomer | None: ...

    async def publish_plan_version(
        self,
        candidate: PlanVersion,
        entitlements: tuple[Entitlement, ...],
    ) -> PlanVersion: ...

    async def get_plan_version(self, plan_version_id: str) -> PlanVersion | None: ...

    async def list_plan_versions(self) -> tuple[PlanVersion, ...]: ...

    async def list_entitlements(self, plan_version_id: str) -> tuple[Entitlement, ...]: ...

    async def publish_price_entries(
        self,
        entries: tuple[PriceEntry, ...],
    ) -> tuple[PriceEntry, ...]: ...

    async def resolve_price(
        self,
        kind: str,
        meter: str,
        model_tier: str,
        at: datetime,
    ) -> PriceEntry | None: ...

    async def list_price_entries(self) -> tuple[PriceEntry, ...]: ...

    async def ensure_trial(
        self,
        candidate: Subscription,
        claimed_at: datetime,
        claim_context: dict | None = None,
    ) -> Subscription: ...

    async def create_subscription(self, candidate: Subscription) -> Subscription: ...

    async def activate_paid_subscription(
        self,
        candidate: Subscription,
        changed_at: datetime,
    ) -> Subscription: ...

    async def get_subscription(self, subscription_id: str) -> Subscription | None: ...

    async def list_subscriptions(self, customer_id: str) -> tuple[Subscription, ...]: ...

    async def transition_subscription(
        self,
        subscription_id: str,
        target: SubscriptionStatus,
        changed_at: datetime,
    ) -> Subscription: ...

    async def get_entitled_subscription(
        self,
        customer_id: str,
        at: datetime,
    ) -> Subscription | None: ...

    async def record_webhook_event(self, candidate: WebhookEvent) -> WebhookEvent: ...

    async def mark_webhook_processed(
        self,
        event_id: str,
        processed_at: datetime,
        processing_error: str | None = None,
    ) -> WebhookEvent: ...

    async def get_usage_reservation_by_dedupe_key(
        self,
        dedupe_key: str,
    ) -> UsageReservation | None: ...

    async def reserve_usage(
        self,
        candidate: UsageReservation,
        quota_limit: int | None,
        *,
        budget_limit_micros: int | None = None,
        estimated_cost_micros: int = 0,
    ) -> UsageReservation: ...

    async def get_usage_reservation(
        self,
        reservation_id: str,
    ) -> UsageReservation | None: ...

    async def list_usage_reservations(
        self,
        customer_id: str,
    ) -> tuple[UsageReservation, ...]: ...

    async def list_usage_events(
        self,
        customer_id: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        meter: str | None = None,
        limit: int | None = None,
    ) -> tuple[UsageEvent, ...]: ...

    async def finalize_usage(
        self,
        reservation_id: str,
        actual_quantity: int,
        event_candidate: UsageEvent,
        finalized_at: datetime,
    ) -> UsageEvent: ...

    async def release_usage(
        self,
        reservation_id: str,
        released_at: datetime,
    ) -> UsageReservation: ...

    async def reconcile(self, at: datetime, *, repair: bool) -> ReconciliationReport: ...

    async def create_order(self, candidate: PurchaseOrder) -> PurchaseOrder: ...

    async def get_order(self, order_id: str) -> PurchaseOrder | None: ...

    async def list_orders(
        self,
        customer_id: str,
        *,
        status: str | None = None,
    ) -> tuple[PurchaseOrder, ...]: ...

    async def list_all_orders(
        self,
        *,
        status: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> tuple[PurchaseOrder, ...]: ...

    async def pay_and_fulfill_order(
        self,
        order_id: str,
        subscription_candidate: Subscription | None,
        changed_at: datetime,
        *,
        operator: str,
        payment_reference: str | None = None,
        extend_subscription_id: str | None = None,
    ) -> tuple[PurchaseOrder, Subscription]: ...

    async def cancel_order(
        self,
        order_id: str,
        canceled_at: datetime,
    ) -> PurchaseOrder: ...

    async def acquire_turn_lease(
        self,
        candidate: TurnLease,
        *,
        limit: int,
        now: datetime,
    ) -> TurnLease | None:
        """Insert one owner turn lease, or ``None`` when the limit is full.

        Expired rows never count against ``limit``: callers may sweep first,
        but the acquire itself must ignore leases past ``expires_at``.
        """
        ...

    async def heartbeat_turn_lease(
        self,
        lease_id: str,
        *,
        now: datetime,
        ttl_seconds: float,
    ) -> TurnLease | None:
        """Extend a live lease's TTL; ``None`` when it expired or is unknown."""
        ...

    async def release_turn_lease(self, lease_id: str) -> bool:
        """Remove one lease; returns ``False`` when it was already gone."""
        ...

    async def sweep_turn_leases(self, now: datetime) -> int:
        """Delete expired leases and return how many were removed."""
        ...

    async def erase_customer(self, owner_id: str) -> bool: ...
