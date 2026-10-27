"""M4 risk-control domain tests: billing periods, leases, limits, abuse."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from deeptutor.commercial import (
    CommercialControlPlane,
    EntitlementDenied,
    InMemoryCommercialRepository,
    InvalidSubscriptionTransition,
    PriceEntryDraft,
    SubscriptionStatus,
    TurnLease,
    UsageReservationState,
)
from deeptutor.commercial.billing_period import billing_period_from_orders
from deeptutor.commercial.models import OrderStatus, OrderType, PurchaseOrder
from deeptutor.commercial.rate_limit import (
    CommercialRateLimitDenied,
    SlidingWindowRateLimiter,
    enforce_rate_limits,
)

NOW = datetime(2026, 10, 27, 9, 0, tzinfo=timezone.utc)


def _service():
    import itertools

    counter = itertools.count(1)
    service = CommercialControlPlane(
        InMemoryCommercialRepository(),
        clock=lambda: NOW,
        id_factory=lambda prefix: f"{prefix}_{next(counter):04d}",
    )
    return service


def _order(
    order_id: str,
    subscription_id: str,
    start: datetime,
    end: datetime,
    *,
    status: OrderStatus = OrderStatus.FULFILLED,
) -> PurchaseOrder:
    return PurchaseOrder(
        id=order_id,
        dedupe_key=f"dedupe-{order_id}",
        customer_id="cus_1",
        plan_version_id="planv_1",
        order_type="subscription_renew",
        price_minor=4900,
        currency="CNY",
        status=status,
        provider="manual",
        external_order_id=None,
        subscription_id=subscription_id,
        period_start=start,
        period_end=end,
        paid_at=None,
        fulfilled_at=None,
        canceled_at=None,
        refunded_at=None,
        metadata={},
        created_at=start,
        updated_at=start,
    )


class TestBillingPeriodFromOrders:
    def _subscription(self, start: datetime, end: datetime):
        from deeptutor.commercial import Subscription

        return Subscription(
            id="sub_1",
            customer_id="cus_1",
            plan_version_id="planv_1",
            status=SubscriptionStatus.ACTIVE,
            provider="manual",
            external_subscription_id=None,
            current_period_start=start,
            current_period_end=end,
            trial_started_at=None,
            trial_ends_at=None,
            canceled_at=None,
            version=1,
            created_at=start,
            updated_at=start,
        )

    def test_covering_fulfilled_order_wins_over_subscription_window(self):
        subscription = self._subscription(NOW - timedelta(days=60), NOW + timedelta(days=60))
        orders = [
            _order("ord_1", "sub_1", NOW - timedelta(days=60), NOW),
            _order("ord_2", "sub_1", NOW, NOW + timedelta(days=60)),
        ]
        period = billing_period_from_orders(subscription, orders, NOW + timedelta(hours=1))
        assert period.order_id == "ord_2"
        assert period.source == "order"
        assert period.period_start == NOW
        assert period.period_end == NOW + timedelta(days=60)

    def test_non_fulfilled_and_foreign_orders_are_ignored(self):
        subscription = self._subscription(NOW - timedelta(days=5), NOW + timedelta(days=25))
        orders = [
            _order(
                "ord_pending",
                "sub_1",
                NOW,
                NOW + timedelta(days=30),
                status=OrderStatus.PENDING_PAYMENT,
            ),
            _order("ord_other", "sub_other", NOW - timedelta(days=1), NOW + timedelta(days=29)),
        ]
        period = billing_period_from_orders(subscription, orders, NOW)
        assert period.source == "subscription"
        assert period.order_id is None

    def test_latest_covering_order_breaks_ties(self):
        subscription = self._subscription(NOW - timedelta(days=60), NOW + timedelta(days=60))
        overlapping = [
            _order("ord_a", "sub_1", NOW - timedelta(days=2), NOW + timedelta(days=28)),
            _order("ord_b", "sub_1", NOW - timedelta(days=1), NOW + timedelta(days=29)),
        ]
        period = billing_period_from_orders(subscription, overlapping, NOW)
        assert period.order_id == "ord_b"


@pytest.mark.asyncio
async def test_budget_resets_in_new_billing_period_after_extend_renewal():
    """Extend-renewal keeps one subscription row; the budget must not."""
    import itertools

    clock = type("Clock", (), {"value": NOW})()
    sequence = itertools.count(1)
    service = CommercialControlPlane(
        InMemoryCommercialRepository(),
        clock=lambda: clock.value,
        id_factory=lambda prefix: f"{prefix}_{next(sequence):04d}",
    )
    customer = await service.ensure_billing_customer(owner_id="u_cycle")
    plan = await service.publish_plan_version(
        plan_code="cycle",
        version=1,
        name="Cycle Monthly",
        price_minor=4900,
        currency="CNY",
        entitlements={
            "quota.llm_tokens": {"limit": 100_000, "window": "subscription"},
            "budget.platform_cost_micros": 300,
        },
    )
    await service.publish_price_version(
        price_version="pv-1",
        kind="platform_cost",
        effective_from=NOW - timedelta(days=1),
        entries=[
            PriceEntryDraft(
                meter="llm_tokens", model_tier="default", unit_price_micros=3, currency="CNY"
            )
        ],
    )
    # Period 1 starts with the first fulfillment.
    order = await service.create_order(customer.id, plan.id, idempotency_key="cycle-1")
    await service.pay_and_fulfill_order(order.id, operator="admin")

    # Exhaust the period-1 budget inside [NOW, NOW+30d).
    clock.value = NOW + timedelta(days=1)
    first = await service.reserve_usage(customer.id, "llm_tokens", 100, "spend-1")
    await service.finalize_metered_usage(first.id, actual_quantity=100, is_estimated=True)
    with pytest.raises(EntitlementDenied, match="budget.platform_cost exceeded"):
        await service.reserve_usage(customer.id, "llm_tokens", 1, "spend-blocked")

    # Early renewal: order 2's period starts at the live period end.
    renewal = await service.create_order(
        customer.id,
        plan.id,
        order_type=OrderType.SUBSCRIPTION_RENEW,
        idempotency_key="cycle-2",
    )
    clock.value = NOW + timedelta(days=29)
    await service.pay_and_fulfill_order(renewal.id, operator="admin")

    # Inside the new billing period the budget starts over.
    clock.value = NOW + timedelta(days=31)
    fresh = await service.reserve_usage(customer.id, "llm_tokens", 100, "spend-2")
    assert fresh.metadata["estimated_cost_micros"] == 300
    await service.finalize_metered_usage(fresh.id, actual_quantity=100, is_estimated=True)
    with pytest.raises(EntitlementDenied, match="budget.platform_cost exceeded"):
        await service.reserve_usage(customer.id, "llm_tokens", 1, "spend-3")


@pytest.mark.asyncio
async def test_turn_lease_acquire_limit_expiry_and_sweep():
    repository = InMemoryCommercialRepository()

    def lease(lease_id: str, owner: str, *, minutes: float, base: datetime = NOW) -> TurnLease:
        return TurnLease(
            id=lease_id,
            owner_id=owner,
            capability="chat",
            turn_id=None,
            acquired_at=base,
            heartbeat_at=base,
            expires_at=base + timedelta(minutes=minutes),
        )

    first = await repository.acquire_turn_lease(
        lease("l1", "owner-a", minutes=15), limit=2, now=NOW
    )
    second = await repository.acquire_turn_lease(
        lease("l2", "owner-a", minutes=15), limit=2, now=NOW
    )
    assert first is not None and second is not None
    denied = await repository.acquire_turn_lease(
        lease("l3", "owner-a", minutes=15), limit=2, now=NOW
    )
    assert denied is None

    # Other owners have independent slots.
    other = await repository.acquire_turn_lease(
        lease("l4", "owner-b", minutes=15), limit=1, now=NOW
    )
    assert other is not None

    # Release frees the slot immediately.
    assert await repository.release_turn_lease("l1") is True
    assert await repository.release_turn_lease("l1") is False
    third = await repository.acquire_turn_lease(
        lease("l5", "owner-a", minutes=15), limit=2, now=NOW
    )
    assert third is not None

    # TTL expiry frees slots lazily on acquire; the late lease must carry
    # its own future expiry or it would be born expired.
    late = NOW + timedelta(minutes=30)
    expired_slot = await repository.acquire_turn_lease(
        lease("l6", "owner-a", base=late, minutes=15), limit=1, now=late
    )
    assert expired_slot is not None
    # A lease that expires later, untouched by any acquire, is what sweep
    # exists to reclaim (the janitor path).
    short = await repository.acquire_turn_lease(
        lease("l7", "owner-b", base=late, minutes=1), limit=1, now=late
    )
    assert short is not None
    later = late + timedelta(minutes=2)
    assert await repository.sweep_turn_leases(later) == 1
    assert await repository.sweep_turn_leases(later) == 0

    # Heartbeat extends a live lease.
    heartbeated = await repository.heartbeat_turn_lease("l6", now=later, ttl_seconds=900.0)
    assert heartbeated is not None
    assert heartbeated.expires_at > later


def test_rate_limits_window_rolls_and_missing_keys_are_unlimited():
    limiter = SlidingWindowRateLimiter(window_seconds=60.0)

    # No keys: no enforcement at any volume.
    enforce_rate_limits({}, owner_id="u_free", requests=1000, tokens=10**9, limiter=limiter)

    enforce_rate_limits(
        {"risk.rpm": 2, "risk.tpm": 10},
        owner_id="u_limited",
        requests=1,
        tokens=4,
        limiter=limiter,
        now=100.0,
    )
    enforce_rate_limits(
        {"risk.rpm": 2, "risk.tpm": 10},
        owner_id="u_limited",
        requests=1,
        tokens=4,
        limiter=limiter,
        now=110.0,
    )
    with pytest.raises(CommercialRateLimitDenied, match="risk.tpm exceeded"):
        enforce_rate_limits(
            {"risk.tpm": 10},
            owner_id="u_limited",
            requests=1,
            tokens=3,
            limiter=limiter,
            now=120.0,
        )
    # The failed check did not consume the window.
    with pytest.raises(CommercialRateLimitDenied, match="risk.rpm exceeded"):
        enforce_rate_limits(
            {"risk.rpm": 2},
            owner_id="u_limited",
            requests=1,
            tokens=0,
            limiter=limiter,
            now=125.0,
        )
    # The 60s window slides: the earliest call falls out.
    enforce_rate_limits(
        {"risk.rpm": 2, "risk.tpm": 10},
        owner_id="u_limited",
        requests=1,
        tokens=2,
        limiter=limiter,
        now=161.0,
    )
    # Owners never share windows.
    enforce_rate_limits(
        {"risk.rpm": 1},
        owner_id="u_other",
        requests=1,
        tokens=0,
        limiter=limiter,
        now=100.0,
    )

    with pytest.raises(ValueError, match="risk.rpm"):
        enforce_rate_limits({"risk.rpm": -1}, owner_id="u_bad", limiter=limiter, now=100.0)


@pytest.mark.asyncio
async def test_trial_claim_context_is_hashed_and_abuse_is_flagged():
    service = _service()
    from deeptutor.commercial.trial_safety import hash_claim_value

    plan = await service.publish_plan_version(
        plan_code="trialplan",
        version=1,
        name="Trial Plan",
        price_minor=0,
        currency="CNY",
        entitlements={"capability.chat": True},
    )
    shared_ip_hash = hash_claim_value("203.0.113.9")
    for index in range(3):
        customer = await service.ensure_billing_customer(owner_id=f"u_abuse_{index}")
        await service.ensure_trial(
            customer.id,
            plan.id,
            claim_context={"ip_hash": shared_ip_hash},
        )
    lone = await service.ensure_billing_customer(owner_id="u_clean")
    await service.ensure_trial(
        lone.id,
        plan.id,
        claim_context={"ip_hash": hash_claim_value("198.51.100.2")},
    )

    report = await service.reconcile(repair=False)
    abuse = [f for f in report.findings if f.code == "trial_abuse_suspected"]
    assert len(abuse) == 1
    assert abuse[0].entity_id == shared_ip_hash
    assert "3 trials" in abuse[0].detail
    assert abuse[0].repairable is False

    stored = await service.get_billing_customer_by_owner("u_abuse_0")
    assert stored.trial_claim_context["ip_hash"] == shared_ip_hash
    # Only hashes are stored: the raw address never appears anywhere.
    assert "203.0.113.9" not in str(stored.trial_claim_context)


@pytest.mark.asyncio
async def test_suspended_round_trip_denies_access_and_blocks_new_subs():
    service = _service()
    customer = await service.ensure_billing_customer(owner_id="u_suspend")
    plan = await service.publish_plan_version(
        plan_code="sus",
        version=1,
        name="Suspend Monthly",
        price_minor=4900,
        currency="CNY",
        entitlements={"capability.chat": True},
    )
    paid = await service.create_active_subscription(
        customer.id,
        plan.id,
        external_subscription_id="sub_sus_1",
        current_period_start=NOW,
        current_period_end=NOW + timedelta(days=30),
    )

    suspended = await service.transition_subscription(paid.id, SubscriptionStatus.SUSPENDED)
    assert suspended.status is SubscriptionStatus.SUSPENDED

    # Entitlement resolution denies while suspended.
    resolved = await service.resolve_entitlements(customer.id)
    assert resolved.active is False
    assert resolved.subscription_id is None

    # A suspended subscription still owns the live slot.
    with pytest.raises(Exception, match="already has"):
        await service.create_active_subscription(
            customer.id,
            plan.id,
            external_subscription_id="sub_sus_2",
            current_period_start=NOW,
            current_period_end=NOW + timedelta(days=30),
        )

    # Suspension is reversible and idempotent.
    active = await service.transition_subscription(paid.id, SubscriptionStatus.ACTIVE)
    assert active.status is SubscriptionStatus.ACTIVE
    again = await service.transition_subscription(paid.id, SubscriptionStatus.SUSPENDED)
    assert again.status is SubscriptionStatus.SUSPENDED

    # Trials cannot jump straight to suspended.
    trial_customer = await service.ensure_billing_customer(owner_id="u_trial_only")
    trial = await service.ensure_trial(trial_customer.id, plan.id)
    with pytest.raises(InvalidSubscriptionTransition):
        await service.transition_subscription(trial.id, SubscriptionStatus.SUSPENDED)


@pytest.mark.asyncio
async def test_turn_concurrency_uses_durable_lease_when_repository_available(tmp_path, monkeypatch):
    """The concurrency module routes through the repository when commercial
    mode has a live repository, and falls back to memory otherwise."""
    from types import SimpleNamespace

    from deeptutor.commercial import concurrency
    from deeptutor.commercial.access import CommercialAccess, set_commercial_access
    from deeptutor.commercial.entitlement_context import reset_commercial_access
    from deeptutor.commercial.models import ResolvedEntitlements
    from deeptutor.multi_user.context import reset_current_user, set_current_user
    from deeptutor.multi_user.models import CurrentUser, UserScope

    repository = InMemoryCommercialRepository()
    runtime = SimpleNamespace(
        settings=SimpleNamespace(enabled=True),
        repository=repository,
    )
    monkeypatch.setattr(concurrency, "get_commercial_runtime", lambda: runtime)

    owner_id = "owner-durable"
    user = CurrentUser(
        id=owner_id,
        username=f"{owner_id}@example.com",
        role="user",
        scope=UserScope(kind="user", user_id=owner_id, root=tmp_path / owner_id),
    )
    resolved = ResolvedEntitlements(
        customer_id=f"cus-{owner_id}",
        subscription_id=f"sub-{owner_id}",
        plan_version_id="plan-trial-v1",
        status=SubscriptionStatus.TRIALING,
        active=True,
        resolved_at=NOW,
        valid_until=NOW + timedelta(days=1),
        values={"capability.chat": True, "limits.concurrent_turns": 1},
    )
    monkeypatch.setattr(
        concurrency,
        "require_capability",
        lambda name: CommercialAccess(owner_id, resolved),
    )
    monkeypatch.setattr(concurrency, "get_current_user_or_none", lambda: user)
    monkeypatch.setattr(concurrency, "integer_limit", lambda key: 1)

    token = set_current_user(user)
    access_token = set_commercial_access(CommercialAccess(owner_id, resolved))
    try:
        lease = await concurrency.acquire_commercial_turn_lease(capability="chat")
        assert lease.reserved is True
        assert lease.lease_id is not None
        assert lease.repository is repository
        # The durable row blocks a second acquire with the same stable code.
        with pytest.raises(
            concurrency.CommercialConcurrencyLimitDenied, match="concurrent_turn_limit"
        ):
            await concurrency.acquire_commercial_turn_lease(capability="chat")
        await lease.release()
        assert await repository.release_turn_lease(lease.lease_id) is False
        renewed = await concurrency.acquire_commercial_turn_lease(capability="chat")
        assert renewed.reserved is True
        await renewed.release()
    finally:
        reset_commercial_access(access_token)
        reset_current_user(token)

    # Memory fallback: enabled runtime without a repository keeps the
    # historical in-process slot behavior.
    memory_runtime = SimpleNamespace(settings=SimpleNamespace(enabled=True), repository=None)
    monkeypatch.setattr(concurrency, "get_commercial_runtime", lambda: memory_runtime)
    token = set_current_user(user)
    access_token = set_commercial_access(CommercialAccess(owner_id, resolved))
    try:
        first = await concurrency.acquire_commercial_turn_lease(capability="chat")
        assert first.reserved is True
        assert first.lease_id is None
        with pytest.raises(
            concurrency.CommercialConcurrencyLimitDenied, match="concurrent_turn_limit"
        ):
            await concurrency.acquire_commercial_turn_lease(capability="chat")
        await first.release()
    finally:
        reset_commercial_access(access_token)
        reset_current_user(token)


@pytest.mark.asyncio
async def test_llm_metering_enforces_risk_limits_before_reservation():
    """The risk gate fires before any reservation is created."""
    from deeptutor.commercial.metering import CommercialUsageRateLimited, _reserve

    service = _service()
    customer = await service.ensure_billing_customer(owner_id="owner-risk")
    plan = await service.publish_plan_version(
        plan_code="riskplan",
        version=1,
        name="Risk Plan",
        price_minor=0,
        currency="CNY",
        entitlements={
            "capability.chat": True,
            "quota.llm_tokens": {"limit": 100_000, "window": "subscription"},
            "quota.embedding_tokens": {"limit": 100_000, "window": "subscription"},
            "risk.rpm": 1,
            "risk.tpm": 10,
        },
    )
    await service.ensure_trial(customer.id, plan.id)

    async def reserve(meter: str, tokens: int):
        return await _reserve(
            service,
            customer_id=customer.id,
            meter=meter,
            quantity=tokens,
            provider=None,
            model=None,
            source="platform",
            request_id=f"risk-{meter}-{tokens}",
            owner_id="owner-risk",
            plan_values={"risk.rpm": 1, "risk.tpm": 10},
        )

    first = await reserve("llm_tokens", 4)
    assert first.quantity == 4
    with pytest.raises(CommercialUsageRateLimited, match="risk_rate_limited"):
        await reserve("llm_tokens", 4)

    # Other meters are not RPM/TPM gated.
    other = await reserve("embedding_tokens", 99_999)
    assert other.meter == "embedding_tokens"
