"""Purchase-order lifecycle: creation snapshot, manual fulfillment, cancel."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from deeptutor.commercial.errors import (
    CommercialNotFound,
    IdempotencyConflict,
    InvalidOrderTransition,
    PendingOrderExists,
)
from deeptutor.commercial.models import OrderStatus, OrderType

from .conftest import seed_customer_and_plan


@pytest.mark.asyncio
async def test_create_order_snapshots_price_and_period(commercial_harness):
    service, _repository, clock = commercial_harness
    customer, plan = await seed_customer_and_plan(service)

    order = await service.create_order(
        customer.id,
        plan.id,
        idempotency_key="checkout-1",
    )

    assert order.status is OrderStatus.PENDING_PAYMENT
    assert order.price_minor == plan.price_minor
    assert order.currency == plan.currency
    assert order.provider == "manual"
    assert order.period_start == clock.value
    # conftest's plan defaults to a monthly interval: Aug 5 -> Sep 5.
    assert order.period_end == datetime(2026, 9, 5, 9, 0, tzinfo=timezone.utc)

    replayed = await service.create_order(
        customer.id,
        plan.id,
        idempotency_key="checkout-1",
    )
    assert replayed.id == order.id


@pytest.mark.asyncio
async def test_create_order_rejects_open_duplicate_and_semantic_replay(
    commercial_harness,
):
    service, _repository, clock = commercial_harness
    customer, plan = await seed_customer_and_plan(service)

    await service.create_order(customer.id, plan.id, idempotency_key="checkout-1")

    with pytest.raises(PendingOrderExists):
        await service.create_order(
            customer.id,
            plan.id,
            idempotency_key="checkout-2",
        )

    with pytest.raises(IdempotencyConflict):
        await service.create_order(
            customer.id,
            plan.id,
            idempotency_key="checkout-1",
            order_type=OrderType.SUBSCRIPTION_RENEW,
        )


@pytest.mark.asyncio
async def test_renewal_order_extends_from_live_period_end(commercial_harness):
    service, _repository, clock = commercial_harness
    customer, plan = await seed_customer_and_plan(service)
    subscription = await service.create_active_subscription(
        customer.id,
        plan.id,
        external_subscription_id="sub-live",
        current_period_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
        current_period_end=datetime(2026, 8, 20, tzinfo=timezone.utc),
    )
    assert subscription.status.value == "active"

    order = await service.create_order(
        customer.id,
        plan.id,
        order_type=OrderType.SUBSCRIPTION_RENEW,
        idempotency_key="renew-1",
    )
    assert order.period_start == datetime(2026, 8, 20, tzinfo=timezone.utc)
    assert order.period_end == datetime(2026, 9, 20, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_pay_and_fulfill_replaces_trial_with_active_subscription(
    commercial_harness,
):
    service, _repository, clock = commercial_harness
    customer, plan = await seed_customer_and_plan(service)
    await service.ensure_trial(customer.id, plan.id)

    order = await service.create_order(customer.id, plan.id, idempotency_key="pay-1")
    fulfilled, subscription = await service.pay_and_fulfill_order(
        order.id,
        operator="admin@local",
        payment_reference="bank-transfer-42",
    )

    assert fulfilled.status is OrderStatus.FULFILLED
    assert fulfilled.paid_at == clock.value
    assert fulfilled.subscription_id == subscription.id
    assert subscription.status.value == "active"
    assert subscription.external_subscription_id == order.id
    assert subscription.current_period_start == order.period_start
    assert fulfilled.metadata["fulfilled_by"] == "admin@local"
    assert fulfilled.metadata["payment_reference"] == "bank-transfer-42"

    resolved = await service.resolve_entitlements(customer.id)
    assert resolved.active is True
    assert resolved.subscription_id == subscription.id

    # Retrying fulfillment converges on the same rows.
    retried, retried_subscription = await service.pay_and_fulfill_order(
        order.id,
        operator="admin@local",
    )
    assert retried.id == fulfilled.id
    assert retried_subscription.id == subscription.id
    assert (
        len(await service.repository.list_subscriptions(customer.id)) == 2
    )  # canceled trial + active


@pytest.mark.asyncio
async def test_cancel_order_only_from_pending(commercial_harness):
    service, _repository, clock = commercial_harness
    customer, plan = await seed_customer_and_plan(service)

    order = await service.create_order(customer.id, plan.id, idempotency_key="cancel-1")
    canceled = await service.cancel_order(order.id)
    assert canceled.status is OrderStatus.CANCELED
    assert canceled.canceled_at == clock.value

    # Cancellation is idempotent; fulfillment and re-cancel stay rejected.
    again = await service.cancel_order(order.id)
    assert again.status is OrderStatus.CANCELED
    with pytest.raises(InvalidOrderTransition):
        await service.pay_and_fulfill_order(order.id, operator="admin@local")

    # After cancellation a new order can be opened.
    fresh = await service.create_order(customer.id, plan.id, idempotency_key="cancel-2")
    assert fresh.status is OrderStatus.PENDING_PAYMENT


@pytest.mark.asyncio
async def test_unknown_order_and_plan_fail_loudly(commercial_harness):
    service, _repository, _clock = commercial_harness
    customer, plan = await seed_customer_and_plan(service)

    with pytest.raises(CommercialNotFound):
        await service.get_order("ord_missing")
    with pytest.raises(CommercialNotFound):
        await service.create_order(customer.id, "planv_missing", idempotency_key="x")
    with pytest.raises(ValueError):
        await service.create_order(customer.id, plan.id, idempotency_key=" ")


@pytest.mark.asyncio
async def test_idempotent_replay_across_clock_advance(commercial_harness):
    """A replay at a later clock must return the original order, not conflict."""
    service, _repository, clock = commercial_harness
    customer, plan = await seed_customer_and_plan(service)

    first = await service.create_order(customer.id, plan.id, idempotency_key="replay-1")
    clock.value = clock.value + timedelta(days=3)
    replayed = await service.create_order(customer.id, plan.id, idempotency_key="replay-1")
    assert replayed.id == first.id
    assert replayed.period_start == first.period_start


@pytest.mark.asyncio
async def test_early_renewal_extends_live_subscription_without_gap(commercial_harness):
    service, _repository, clock = commercial_harness
    customer, plan = await seed_customer_and_plan(service)
    live = await service.create_active_subscription(
        customer.id,
        plan.id,
        external_subscription_id="sub-live-renew",
        current_period_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
        current_period_end=datetime(2026, 8, 20, tzinfo=timezone.utc),
    )

    order = await service.create_order(
        customer.id,
        plan.id,
        order_type=OrderType.SUBSCRIPTION_RENEW,
        idempotency_key="early-renew-1",
    )
    fulfilled, subscription = await service.pay_and_fulfill_order(order.id, operator="admin@local")

    assert subscription.id == live.id  # extended, not replaced
    assert subscription.current_period_end == order.period_end
    assert subscription.status.value == "active"
    assert fulfilled.subscription_id == live.id

    resolved = await service.resolve_entitlements(customer.id)
    assert resolved.active is True
    assert resolved.subscription_id == live.id


@pytest.mark.asyncio
async def test_fulfilling_an_expired_period_is_rejected(commercial_harness):
    service, _repository, clock = commercial_harness
    customer, plan = await seed_customer_and_plan(service)
    order = await service.create_order(customer.id, plan.id, idempotency_key="stale-1")
    clock.value = order.period_end + timedelta(seconds=1)

    with pytest.raises(InvalidOrderTransition, match="already ended"):
        await service.pay_and_fulfill_order(order.id, operator="admin@local")
