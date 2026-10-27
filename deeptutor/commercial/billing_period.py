"""Billing-period derivation from the fulfilled purchase-order chain.

M2's extend-renewal semantics share one subscription row across paid periods,
so a subscription's ``current_period_*`` spans the whole extended lifetime and
cannot delimit a billing cycle.  Fulfilled orders are the durable period
evidence: each carries ``period_start``/``period_end`` and, once fulfilled,
points at the subscription it pays for.

The period covering an instant is the fulfilled order of that subscription
whose window contains the instant (latest ``period_start`` wins).  When no
order covers the instant — trials and provider-activated subscriptions with no
manual order — the subscription's own period window is the fallback, which is
exact for those single-window rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Sequence

from .models import OrderStatus, PurchaseOrder, Subscription

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .repository import CommercialRepository


@dataclass(frozen=True, slots=True)
class BillingPeriod:
    """One derived billing window for a subscription at a point in time."""

    subscription_id: str
    period_start: datetime
    period_end: datetime
    source: str
    order_id: str | None = None


def billing_period_from_orders(
    subscription: Subscription,
    orders: Sequence[PurchaseOrder],
    at: datetime,
) -> BillingPeriod:
    """Return the billing window covering ``at`` for ``subscription``.

    Only fulfilled orders bound to this subscription count as period evidence.
    A pending or canceled order never opens a period, and an order that pays
    for a different subscription is irrelevant to this one.
    """

    candidates = [
        order
        for order in orders
        if order.status is OrderStatus.FULFILLED
        and order.subscription_id == subscription.id
        and order.period_start <= at < order.period_end
    ]
    if candidates:
        order = max(candidates, key=lambda item: (item.period_start, item.id))
        return BillingPeriod(
            subscription_id=subscription.id,
            period_start=order.period_start,
            period_end=order.period_end,
            source="order",
            order_id=order.id,
        )
    return BillingPeriod(
        subscription_id=subscription.id,
        period_start=subscription.current_period_start,
        period_end=subscription.current_period_end,
        source="subscription",
        order_id=None,
    )


async def resolve_billing_period(
    repository: "CommercialRepository",
    subscription: Subscription,
    at: datetime,
) -> BillingPeriod:
    """Repository-backed variant of :func:`billing_period_from_orders`."""

    orders = await repository.list_orders(subscription.customer_id)
    return billing_period_from_orders(subscription, orders, at)


__all__ = [
    "BillingPeriod",
    "billing_period_from_orders",
    "resolve_billing_period",
]
