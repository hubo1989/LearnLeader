"""Pricing catalog, cost attribution, budget gate, and billing preview."""

from __future__ import annotations

from datetime import timedelta

import pytest

from deeptutor.commercial import (
    CommercialNotFound,
    EntitlementDenied,
    ImmutablePriceVersion,
    PriceEntryDraft,
)
from deeptutor.commercial.models import UNPRICED_PRICE_VERSION

pytestmark = pytest.mark.asyncio


async def _publish_platform_catalog(service, *, price_version: str = "prices-2026-10"):
    return await service.publish_price_version(
        price_version=price_version,
        kind="platform_cost",
        effective_from=service._now() - timedelta(days=1),
        entries=[
            PriceEntryDraft(
                meter="llm_tokens",
                model_tier="default",
                unit_price_micros=3,
                currency="CNY",
            ),
            PriceEntryDraft(
                meter="llm_tokens",
                model_tier="premium",
                unit_price_micros=9,
                currency="CNY",
            ),
        ],
    )


async def _metered_customer(
    service,
    *,
    owner: str = "user-pricing",
    token_limit: int = 100,
    token_window: str = "subscription",
    price_minor: int = 0,
    extra: dict | None = None,
):
    customer = await service.ensure_billing_customer(owner_id=owner)
    entitlements = {
        "quota.llm_tokens": {"limit": token_limit, "window": token_window},
        "quota.parse_pages": {"limit": 8, "window": "utc_day"},
    }
    entitlements.update(extra or {})
    plan = await service.publish_plan_version(
        plan_code=f"metered-{owner}",
        version=1,
        name="Metered pricing test",
        price_minor=price_minor,
        currency="CNY",
        entitlements=entitlements,
    )
    subscription = await service.ensure_trial(customer.id, plan.id)
    assert subscription.status.value == "trialing"
    return customer, plan


async def _reserve_and_finalize(
    service,
    customer_id: str,
    *,
    meter: str = "llm_tokens",
    quantity: int,
    request_id: str,
    model: str | None = None,
    is_estimated: bool = False,
):
    reservation = await service.reserve_usage(customer_id, meter, quantity, request_id)
    return await service.finalize_metered_usage(
        reservation.id,
        actual_quantity=quantity,
        model=model,
        is_estimated=is_estimated,
    )


async def test_publish_price_version_is_idempotent_and_immutable(commercial_harness):
    service, repository, clock = commercial_harness
    del repository, clock

    published = await _publish_platform_catalog(service)
    assert [entry.model_tier for entry in published] == ["default", "premium"]
    assert all(entry.kind == "platform_cost" for entry in published)

    replay = await _publish_platform_catalog(service)
    assert [entry.id for entry in replay] == [entry.id for entry in published]

    with pytest.raises(ImmutablePriceVersion):
        await service.publish_price_version(
            price_version="prices-2026-10",
            kind="platform_cost",
            effective_from=service._now() - timedelta(days=1),
            entries=[
                PriceEntryDraft(
                    meter="llm_tokens",
                    model_tier="default",
                    unit_price_micros=4,
                    currency="CNY",
                )
            ],
        )

    with pytest.raises(ValueError, match="kind must be"):
        await service.publish_price_version(
            price_version="x",
            kind="wholesale",
            effective_from=service._now(),
            entries=[
                PriceEntryDraft(
                    meter="m", model_tier="default", unit_price_micros=1, currency="CNY"
                )
            ],
        )
    with pytest.raises(ValueError, match="three-letter"):
        await service.publish_price_version(
            price_version="x",
            kind="list_price",
            effective_from=service._now(),
            entries=[
                PriceEntryDraft(
                    meter="m", model_tier="default", unit_price_micros=1, currency="RMBX"
                )
            ],
        )
    with pytest.raises(ValueError, match="duplicate price row"):
        await service.publish_price_version(
            price_version="x",
            kind="list_price",
            effective_from=service._now(),
            entries=[
                PriceEntryDraft(
                    meter="m", model_tier="default", unit_price_micros=1, currency="CNY"
                ),
                PriceEntryDraft(
                    meter="m", model_tier="default", unit_price_micros=2, currency="CNY"
                ),
            ],
        )
    with pytest.raises(ValueError, match="at least one"):
        await service.publish_price_version(
            price_version="x",
            kind="list_price",
            effective_from=service._now(),
            entries=[],
        )


async def test_price_ledgers_are_independent_and_listed(commercial_harness):
    service, repository, clock = commercial_harness
    del clock

    await _publish_platform_catalog(service, price_version="same-name")
    listed = await service.publish_price_version(
        price_version="same-name",
        kind="list_price",
        effective_from=service._now() - timedelta(days=1),
        entries=[
            PriceEntryDraft(
                meter="llm_tokens", model_tier="default", unit_price_micros=30, currency="CNY"
            )
        ],
    )
    assert listed[0].kind == "list_price"
    platform = await repository.resolve_price(
        "platform_cost", "llm_tokens", "default", service._now()
    )
    sale = await repository.resolve_price("list_price", "llm_tokens", "default", service._now())
    assert platform is not None and platform.unit_price_micros == 3
    assert sale is not None and sale.unit_price_micros == 30

    all_entries = await service.list_price_entries(kind="list_price")
    assert [entry.kind for entry in all_entries] == ["list_price"]


async def test_resolve_price_picks_latest_effective_row(commercial_harness):
    service, repository, clock = commercial_harness
    now = clock.value
    await service.publish_price_version(
        price_version="p-v1",
        kind="platform_cost",
        effective_from=now - timedelta(days=10),
        entries=[
            PriceEntryDraft(
                meter="llm_tokens", model_tier="default", unit_price_micros=2, currency="CNY"
            )
        ],
    )
    await service.publish_price_version(
        price_version="p-v2",
        kind="platform_cost",
        effective_from=now - timedelta(days=1),
        entries=[
            PriceEntryDraft(
                meter="llm_tokens", model_tier="default", unit_price_micros=5, currency="CNY"
            )
        ],
    )

    latest = await repository.resolve_price("platform_cost", "llm_tokens", "default", now)
    assert latest is not None
    assert latest.price_version == "p-v2"
    assert latest.unit_price_micros == 5

    earlier = await repository.resolve_price(
        "platform_cost", "llm_tokens", "default", now - timedelta(days=5)
    )
    assert earlier is not None
    assert earlier.price_version == "p-v1"

    # Before any row, or the wrong ledger: no price is invented.
    assert (
        await repository.resolve_price(
            "platform_cost", "llm_tokens", "default", now - timedelta(days=30)
        )
        is None
    )
    assert await repository.resolve_price("list_price", "llm_tokens", "default", now) is None


async def test_finalize_prices_default_tier_then_mapped_tier(commercial_harness):
    service, repository, clock = commercial_harness
    del repository, clock
    await _publish_platform_catalog(service)

    customer, plan = await _metered_customer(service)
    event = await _reserve_and_finalize(
        service, customer.id, quantity=40, request_id="turn-1", model="gpt-test"
    )
    # No pricing.model_tiers mapping: the default tier applies.
    assert event.price_version == "prices-2026-10"
    assert event.cost_micros == 3 * 40

    tiered_customer, _tiered_plan = await _metered_customer(
        service,
        owner="user-tiered",
        extra={"pricing.model_tiers": {"gpt-premium": "premium"}},
    )
    event = await _reserve_and_finalize(
        service, tiered_customer.id, quantity=10, request_id="turn-premium", model="gpt-premium"
    )
    assert event.price_version == "prices-2026-10"
    assert event.cost_micros == 9 * 10


async def test_finalize_stays_unpriced_without_catalog_hit(commercial_harness):
    service, repository, clock = commercial_harness
    del repository, clock
    customer, plan = await _metered_customer(service)

    event = await _reserve_and_finalize(
        service, customer.id, quantity=50, request_id="turn-2", model="unknown-model"
    )
    assert event.price_version == UNPRICED_PRICE_VERSION
    assert event.cost_micros == 0


async def test_budget_gate_blocks_when_platform_cost_exhausted(commercial_harness):
    service, repository, clock = commercial_harness
    del repository, clock
    customer, plan = await _metered_customer(service, extra={"budget.platform_cost_micros": 150})
    await _publish_platform_catalog(service)

    # Default tier at 3 micros/token: 50 tokens cost exactly the 150 limit.
    event = await _reserve_and_finalize(
        service, customer.id, quantity=50, request_id="budget-1", model="gpt-test"
    )
    assert event.cost_micros == 150

    # Any further priced request exceeds the exhausted budget.
    with pytest.raises(EntitlementDenied, match="budget.platform_cost exceeded"):
        await service.reserve_usage(customer.id, "llm_tokens", 10, "budget-2")

    # An unpriced meter (no catalog row) still reserves: no price invented.
    other = await service.reserve_usage(customer.id, "parse_pages", 5, "budget-3")
    assert other.meter == "parse_pages"


async def test_budget_gate_converts_credits_and_skips_old_plans(commercial_harness):
    service, repository, clock = commercial_harness
    del repository, clock
    # 1 credit = 1000 micros: 2 credits cap platform cost at 2000 micros.
    customer, plan = await _metered_customer(
        service,
        owner="user-credits",
        token_limit=10_000,
        extra={"budget.platform_cost_credits": 2},
    )
    await _publish_platform_catalog(service)

    reservation = await service.reserve_usage(customer.id, "llm_tokens", 600, "credits-1")
    assert reservation.quantity == 600  # estimated 600 * 3 = 1800 <= 2000
    with pytest.raises(EntitlementDenied, match="budget.platform_cost exceeded"):
        # estimated 700 * 3 = 2100 > 2000 denies before any provider work.
        await service.reserve_usage(customer.id, "llm_tokens", 700, "credits-2")

    # Plans without either budget key keep their pre-gate behavior.
    plain_customer, _plain_plan = await _metered_customer(
        service, owner="user-plain", token_limit=10_000
    )
    plain = await service.reserve_usage(plain_customer.id, "llm_tokens", 10_000, "plain-1")
    assert plain.quantity == 10_000

    # Malformed budget values deny (fail closed) instead of skipping the gate.
    bad_customer, _bad_plan = await _metered_customer(
        service, owner="user-bad", extra={"budget.platform_cost_credits": -5}
    )
    with pytest.raises(EntitlementDenied, match="invalid platform-cost budget"):
        await service.reserve_usage(bad_customer.id, "llm_tokens", 5, "bad-1")


async def test_billing_preview_prices_overage_at_event_time_by_tier(commercial_harness):
    """Overage consumes the limit oldest-first and prices per event instant.

    The quota system normally prevents usage beyond a window's limit, so
    this scenario plants ledger rows directly — the defensive pricing path
    for inconsistent history or future gateway-fed meters.
    """

    from deeptutor.commercial.models import UsageEvent

    service, repository, clock = commercial_harness
    customer, plan = await _metered_customer(
        service,
        owner="user-preview",
        token_limit=10,
        token_window="subscription",
        price_minor=4_900,
        extra={"pricing.model_tiers": {"gpt-premium": "premium"}},
    )
    t0 = clock.value
    await service.publish_price_version(
        price_version="list-v1",
        kind="list_price",
        effective_from=t0 - timedelta(days=2),
        entries=[
            PriceEntryDraft(
                meter="llm_tokens", model_tier="default", unit_price_micros=2, currency="CNY"
            ),
            PriceEntryDraft(
                meter="llm_tokens", model_tier="premium", unit_price_micros=10, currency="CNY"
            ),
        ],
    )
    # A later catalog version reprices only events after its effective_from.
    await service.publish_price_version(
        price_version="list-v2",
        kind="list_price",
        effective_from=t0 + timedelta(minutes=30),
        entries=[
            PriceEntryDraft(
                meter="llm_tokens", model_tier="default", unit_price_micros=5, currency="CNY"
            )
        ],
    )

    subscription = await service.repository.get_entitled_subscription(customer.id, t0)
    assert subscription is not None

    def _event(name: str, quantity: int, model: str, minutes: int) -> None:
        occurred = t0 + timedelta(minutes=minutes)
        repository._usage_events[f"use_{name}"] = UsageEvent(
            id=f"use_{name}",
            dedupe_key=f"dedupe-{name}",
            reservation_id=f"res_{name}",
            customer_id=customer.id,
            subscription_id=subscription.id,
            meter="llm_tokens",
            quantity=quantity,
            provider="openai",
            model=model,
            usage_units={"total_tokens": quantity},
            price_version="unpriced-v1",
            cost_micros=0,
            is_estimated=False,
            metadata={},
            period_start=subscription.current_period_start,
            period_end=subscription.current_period_end,
            occurred_at=occurred,
            created_at=occurred,
        )

    # Limit 10, oldest-first: 6 free + 4 free, then 2 default overage before
    # the v2 cutoff (priced 2), 4 premium overage after it (still v1 premium
    # at 10 — v2 only repriced default), and 3 more default units after the
    # cutoff priced at the new 5.
    _event("a", 6, "gpt-test", 10)
    _event("b", 6, "gpt-test", 20)  # 4 free + 2 overage at list-v1
    _event("c", 4, "gpt-premium", 40)  # all overage at list-v1 premium
    _event("d", 3, "gpt-test", 50)  # all overage at list-v2 default
    # parse_pages overage with no list_price row at all -> unpriced.
    repository._usage_events["use_pages"] = UsageEvent(
        id="use_pages",
        dedupe_key="dedupe-pages",
        reservation_id="res_pages",
        customer_id=customer.id,
        subscription_id=subscription.id,
        meter="parse_pages",
        quantity=9,
        provider="mineru",
        model=None,
        usage_units={"pages": 9},
        price_version="unpriced-v1",
        cost_micros=0,
        is_estimated=False,
        metadata={},
        period_start=subscription.current_period_start,
        period_end=subscription.current_period_end,
        occurred_at=t0,
        created_at=t0,
    )

    preview = await service.billing_preview(customer.id)
    assert preview["fixed"] == {"price_minor": 4_900, "currency": "CNY"}
    usage_by_meter = {row["meter"]: row for row in preview["usage"]}
    assert usage_by_meter["llm_tokens"]["used"] == 19
    assert usage_by_meter["llm_tokens"]["excess"] == 9
    assert usage_by_meter["llm_tokens"]["window"] == "subscription"
    assert usage_by_meter["parse_pages"]["excess"] == 1

    overage_by_key = {
        (row["meter"], row["model_tier"], row["unit_price_micros"]): row
        for row in preview["overage"]
    }
    llm_v1 = overage_by_key[("llm_tokens", "default", 2)]
    llm_v2 = overage_by_key[("llm_tokens", "default", 5)]
    llm_premium = overage_by_key[("llm_tokens", "premium", 10)]
    pages_row = overage_by_key[("parse_pages", "default", None)]
    assert llm_v1["quantity"] == 2 and llm_v1["cost_micros"] == 4
    assert llm_v2["quantity"] == 3 and llm_v2["cost_micros"] == 15
    assert llm_premium["quantity"] == 4 and llm_premium["cost_micros"] == 40
    assert all(row["unpriced"] is False for row in (llm_v1, llm_v2, llm_premium))
    assert pages_row["unpriced"] is True
    assert pages_row["cost_micros"] == 0

    totals = preview["totals"]
    assert totals["fixed_minor"] == 4_900
    assert totals["overage_micros"] == 4 + 15 + 40
    assert totals["overage_currency"] == "CNY"
    assert totals["currency_mismatch"] is False
    assert totals["unpriced_meters"] == ["parse_pages"]

    # Without a subscription there is nothing to preview.
    stranger = await service.ensure_billing_customer(owner_id="user-stranger")
    with pytest.raises(CommercialNotFound):
        await service.billing_preview(stranger.id)


async def test_billing_preview_never_flags_compliant_daily_usage(commercial_harness):
    """Filling a daily quota every day is not period overage."""

    service, repository, clock = commercial_harness
    del repository
    customer, plan = await _metered_customer(
        service,
        owner="user-compliant",
        token_limit=100,
        token_window="utc_day",
        price_minor=4_900,
    )
    await service.publish_price_version(
        price_version="list-2026-10",
        kind="list_price",
        effective_from=clock.value - timedelta(days=1),
        entries=[
            PriceEntryDraft(
                meter="llm_tokens", model_tier="default", unit_price_micros=2, currency="CNY"
            )
        ],
    )
    # Three days, each exactly at the daily cap.
    for day in range(3):
        await _reserve_and_finalize(
            service, customer.id, quantity=100, request_id=f"full-{day}", model="gpt-test"
        )
        clock.value += timedelta(days=1)

    preview = await service.billing_preview(customer.id)
    usage_by_meter = {row["meter"]: row for row in preview["usage"]}
    # The preview evaluates the current (today's) window, not the period.
    assert usage_by_meter["llm_tokens"]["used"] == 0
    assert usage_by_meter["llm_tokens"]["excess"] == 0
    assert preview["overage"] == []
    assert preview["totals"]["overage_micros"] == 0
    assert preview["totals"]["unpriced_meters"] == []


async def test_billing_preview_without_overage_has_no_list_rows(commercial_harness):
    service, repository, clock = commercial_harness
    del repository, clock
    customer, plan = await _metered_customer(service, owner="user-nolimit")

    await _reserve_and_finalize(
        service, customer.id, quantity=15, request_id="unpriced-1", model=None
    )

    preview = await service.billing_preview(customer.id)
    assert preview["totals"]["overage_micros"] == 0
    assert preview["totals"]["unpriced_meters"] == []
    assert preview["overage"] == []
    usage_by_meter = {row["meter"]: row for row in preview["usage"]}
    assert usage_by_meter["llm_tokens"]["used"] == 15
    assert usage_by_meter["llm_tokens"]["excess"] == 0


async def test_reconcile_flags_high_estimated_share_without_repair(commercial_harness):
    service, repository, clock = commercial_harness
    del repository, clock
    customer, plan = await _metered_customer(service)

    for index in range(6):
        await _reserve_and_finalize(
            service,
            customer.id,
            quantity=1,
            request_id=f"estimate-{index}",
            model="gpt-test",
            is_estimated=index < 4,  # 4 of 6 estimated > 50%.
        )

    report = await service.reconcile(repair=True)
    matches = [
        finding for finding in report.findings if finding.code == "estimated_usage_share_high"
    ]
    assert len(matches) == 1
    finding = matches[0]
    # M4: the share is grouped per billing period (here the trial's
    # subscription window, since no fulfilled order exists yet).
    assert finding.entity_type == "billing_period"
    assert finding.entity_id is not None
    assert finding.repairable is False
    assert "4/6" in finding.detail

    # Repair does not remove estimated events; the signal is informational.
    events = await service.repository.list_usage_events(customer.id)
    assert len(events) == 6


async def test_reconcile_ignores_small_samples_and_boundary_share(commercial_harness):
    service, repository, clock = commercial_harness
    del repository, clock
    customer, plan = await _metered_customer(service, owner="user-small")

    # Fewer than 5 events: never flagged, even at 100% estimated.
    for index in range(3):
        await _reserve_and_finalize(
            service,
            customer.id,
            quantity=1,
            request_id=f"small-{index}",
            model="gpt-test",
            is_estimated=True,
        )
    report = await service.reconcile(repair=False)
    assert not [f for f in report.findings if f.code == "estimated_usage_share_high"]

    # Exactly at the 50% threshold: the strictly-greater rule does not flag.
    other, _other_plan = await _metered_customer(service, owner="user-half")
    for index in range(6):
        await _reserve_and_finalize(
            service,
            other.id,
            quantity=1,
            request_id=f"half-{index}",
            model="gpt-test",
            is_estimated=index < 3,
        )
    report = await service.reconcile(repair=False)
    assert not [f for f in report.findings if f.code == "estimated_usage_share_high"]


async def test_publish_price_version_rejects_naive_effective_from(commercial_harness):
    service, repository, clock = commercial_harness
    del repository, clock
    from datetime import datetime

    with pytest.raises(ValueError, match="timezone-aware"):
        await service.publish_price_version(
            price_version="naive",
            kind="platform_cost",
            effective_from=datetime(2026, 10, 25, 9, 0),
            entries=[
                PriceEntryDraft(
                    meter="llm_tokens", model_tier="default", unit_price_micros=1, currency="CNY"
                )
            ],
        )


async def test_overlapping_reservations_share_the_budget(commercial_harness):
    """In-flight reservations count their snapshotted estimate, not zero."""

    service, repository, clock = commercial_harness
    del repository, clock
    customer, plan = await _metered_customer(
        service, token_limit=10_000, extra={"budget.platform_cost_micros": 300}
    )
    await _publish_platform_catalog(service)  # default tier: 3 micros/token

    # 100 tokens estimate exactly 300 micros: reserved but not finalized.
    held = await service.reserve_usage(customer.id, "llm_tokens", 100, "shared-1")
    assert held.metadata["estimated_cost_micros"] == 300

    # A second priced call must see the first reservation's estimate.
    with pytest.raises(EntitlementDenied, match="budget.platform_cost exceeded"):
        await service.reserve_usage(customer.id, "llm_tokens", 1, "shared-2")

    # Unpriced meters are still free to reserve while it is held.
    other = await service.reserve_usage(customer.id, "parse_pages", 5, "shared-3")
    assert other.meter == "parse_pages"


async def test_reserve_replay_matches_despite_snapshot_key(commercial_harness):
    """The server-injected estimate must not break idempotent replays."""

    service, repository, clock = commercial_harness
    del repository, clock
    customer, plan = await _metered_customer(
        service, token_limit=10_000, extra={"budget.platform_cost_credits": 5}
    )
    await _publish_platform_catalog(service)

    original = await service.reserve_usage(
        customer.id,
        "llm_tokens",
        10,
        "replay-1",
        metadata={"provider": "openai", "model": "gpt-test"},
    )
    assert original.metadata["estimated_cost_micros"] == 30

    # A retry carries caller metadata only — the estimate is injected after
    # the fingerprint comparison — so it must return the stored reservation.
    replay = await service.reserve_usage(
        customer.id,
        "llm_tokens",
        10,
        "replay-1",
        metadata={"provider": "openai", "model": "gpt-test"},
    )
    assert replay.id == original.id

    # Caller-supplied values for the reserved key are dropped, not trusted.
    scrubbed = await service.reserve_usage(
        customer.id,
        "llm_tokens",
        5,
        "replay-2",
        metadata={"estimated_cost_micros": "not-a-number"},
    )
    assert "estimated_cost_micros" not in scrubbed.metadata or isinstance(
        scrubbed.metadata["estimated_cost_micros"], int
    )


async def test_repository_rejects_duplicate_batch_rows_and_second_currency(
    commercial_harness,
):
    """Repository-level guards mirror the service checks and the DB trigger."""

    from deeptutor.commercial.errors import CommercialConflict
    from deeptutor.commercial.models import PriceEntry

    service, repository, clock = commercial_harness
    del clock
    now = service._now()

    def _entry(entry_id: str, price_version: str, currency: str = "CNY") -> PriceEntry:
        return PriceEntry(
            id=entry_id,
            price_version=price_version,
            kind="platform_cost",
            meter="llm_tokens",
            model_tier="default",
            unit_price_micros=1,
            currency=currency,
            effective_from=now,
            created_at=now,
        )

    # Duplicate row key inside one batch.
    with pytest.raises(CommercialConflict, match="duplicate price row"):
        await repository.publish_price_entries(
            (_entry("p1", "dup-batch"), _entry("p2", "dup-batch"))
        )

    # A second platform_cost currency is rejected under the repository lock.
    await repository.publish_price_entries((_entry("p3", "cny-v1"),))
    with pytest.raises(CommercialConflict, match="single currency"):
        await repository.publish_price_entries((_entry("p4", "usd-v1", "USD"),))


async def test_publish_rejects_sentinel_version_and_second_platform_currency(
    commercial_harness,
):
    service, repository, clock = commercial_harness
    del repository, clock
    await _publish_platform_catalog(service)  # platform_cost in CNY

    from deeptutor.commercial.models import UNPRICED_PRICE_VERSION

    with pytest.raises(ValueError, match="reserved for unpriced events"):
        await service.publish_price_version(
            price_version=UNPRICED_PRICE_VERSION,
            kind="list_price",
            effective_from=service._now(),
            entries=[
                PriceEntryDraft(
                    meter="llm_tokens", model_tier="default", unit_price_micros=1, currency="CNY"
                )
            ],
        )

    # A second platform_cost currency would sum unrelated micros.
    with pytest.raises(ValueError, match="single currency"):
        await service.publish_price_version(
            price_version="prices-usd",
            kind="platform_cost",
            effective_from=service._now(),
            entries=[
                PriceEntryDraft(
                    meter="llm_tokens", model_tier="default", unit_price_micros=1, currency="USD"
                )
            ],
        )

    # list_price keeps its mixed-currency tolerance (preview flags it).
    await service.publish_price_version(
        price_version="list-usd",
        kind="list_price",
        effective_from=service._now(),
        entries=[
            PriceEntryDraft(
                meter="llm_tokens", model_tier="default", unit_price_micros=1, currency="USD"
            )
        ],
    )
    entries = await service.list_price_entries(kind="list_price")
    assert {entry.currency for entry in entries} == {"USD"}


async def test_finalize_prices_with_the_reserved_plans_tier_mapping(commercial_harness):
    """A late finalize keeps the reserved plan's tiers, not the new plan's."""

    service, repository, clock = commercial_harness
    del repository
    await _publish_platform_catalog(service)
    customer, plain_plan = await _metered_customer(service, owner="user-switch")
    reservation = await service.reserve_usage(customer.id, "llm_tokens", 10, "switch-1")

    # Replace effective access with a plan mapping the model to premium.
    premium_plan = await service.publish_plan_version(
        plan_code="premium-mapped",
        version=1,
        name="Premium mapped",
        price_minor=0,
        currency="CNY",
        entitlements={
            "quota.llm_tokens": {"limit": 1_000, "window": "subscription"},
            "pricing.model_tiers": {"gpt-test": "premium"},
        },
    )
    await service.activate_paid_subscription(
        customer.id,
        premium_plan.id,
        provider="manual",
        external_subscription_id="ext-switch-1",
        current_period_start=clock.value,
        current_period_end=clock.value + timedelta(days=30),
    )

    event = await service.finalize_metered_usage(
        reservation.id, actual_quantity=10, model="gpt-test", is_estimated=False
    )
    # The reservation belongs to the plain plan: default tier at 3 micros,
    # never the replacement plan's premium tier at 9.
    assert event.cost_micros == 3 * 10
    del plain_plan


async def test_finalize_retry_after_catalog_update_keeps_recorded_price(commercial_harness):
    """Retries resolve the reservation-time instant, so attribution is stable."""

    service, repository, clock = commercial_harness
    del repository
    customer, plan = await _metered_customer(service, token_limit=1_000)
    await _publish_platform_catalog(service)  # default tier at 3 micros
    reservation = await service.reserve_usage(customer.id, "llm_tokens", 10, "retry-1")

    # A repriced catalog becomes effective after the reservation exists.
    clock.value += timedelta(minutes=5)
    await service.publish_price_version(
        price_version="prices-raised",
        kind="platform_cost",
        effective_from=clock.value,
        entries=[
            PriceEntryDraft(
                meter="llm_tokens", model_tier="default", unit_price_micros=50, currency="CNY"
            )
        ],
    )

    event = await service.finalize_metered_usage(
        reservation.id, actual_quantity=10, model="gpt-test", is_estimated=False
    )
    assert event.price_version == "prices-2026-10"
    assert event.cost_micros == 3 * 10

    # A retried finalize converges on the same event instead of conflicting.
    replay = await service.finalize_metered_usage(
        reservation.id, actual_quantity=10, model="gpt-test", is_estimated=False
    )
    assert replay.id == event.id
    assert replay.cost_micros == event.cost_micros


async def test_memory_batch_publish_is_atomic_and_consistently_ordered(commercial_harness):
    """A rejected row cannot leave earlier rows of the batch published."""

    from deeptutor.commercial.models import PriceEntry

    service, repository, clock = commercial_harness
    del clock
    published = await _publish_platform_catalog(service)
    original = published[0]
    now = service._now()

    # Batch via the repository interface: a fresh row first, then a drifted
    # replay of an already-published row. The whole batch fails and the
    # fresh row must not remain published (Postgres rolls back the same way).
    fresh = PriceEntry(
        id="price_fresh",
        price_version="prices-batch",
        kind="platform_cost",
        meter="fresh_meter",
        model_tier="default",
        unit_price_micros=1,
        currency="CNY",
        effective_from=now,
        created_at=now,
    )
    drifted = PriceEntry(
        id="price_drift",
        price_version=original.price_version,
        kind=original.kind,
        meter=original.meter,
        model_tier=original.model_tier,
        unit_price_micros=99,
        currency=original.currency,
        effective_from=original.effective_from,
        created_at=now,
    )
    with pytest.raises(ImmutablePriceVersion):
        await repository.publish_price_entries((fresh, drifted))
    assert await repository.resolve_price("platform_cost", "fresh_meter", "default", now) is None

    # Listing matches the Postgres ordering: temporal keys descending, then
    # kind/meter/model_tier ascending.
    await service.publish_price_version(
        price_version="prices-older",
        kind="platform_cost",
        effective_from=now - timedelta(days=5),
        entries=[
            PriceEntryDraft(
                meter="zz_meter", model_tier="default", unit_price_micros=1, currency="CNY"
            ),
            PriceEntryDraft(
                meter="aa_meter", model_tier="default", unit_price_micros=1, currency="CNY"
            ),
        ],
    )
    listed = await repository.list_price_entries()
    temporal = [(entry.effective_from, entry.created_at, entry.price_version) for entry in listed]
    assert temporal == sorted(temporal, reverse=True)
    newest = [entry for entry in listed if entry.price_version == "prices-2026-10"]
    assert [entry.model_tier for entry in newest] == ["default", "premium"]
    oldest = [entry for entry in listed if entry.price_version == "prices-older"]
    assert [entry.meter for entry in oldest] == ["aa_meter", "zz_meter"]
