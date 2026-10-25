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


async def test_billing_preview_splits_overage_across_tiers(commercial_harness):
    service, repository, clock = commercial_harness
    del repository
    customer, plan = await _metered_customer(
        service,
        owner="user-preview",
        token_limit=100,
        token_window="utc_day",
        price_minor=4_900,
        extra={"pricing.model_tiers": {"gpt-premium": "premium"}},
    )
    await service.publish_price_version(
        price_version="list-2026-10",
        kind="list_price",
        effective_from=service._now() - timedelta(days=1),
        entries=[
            PriceEntryDraft(
                meter="llm_tokens", model_tier="default", unit_price_micros=2, currency="CNY"
            ),
            PriceEntryDraft(
                meter="llm_tokens", model_tier="premium", unit_price_micros=10, currency="CNY"
            ),
        ],
    )

    # Day one fills 60 default tokens; day two adds 60 premium tokens.
    # Daily quotas hold, but the subscription period totals 120 > 100.
    await _reserve_and_finalize(
        service, customer.id, quantity=60, request_id="preview-1", model="gpt-test"
    )
    clock.value += timedelta(days=1)
    await _reserve_and_finalize(
        service, customer.id, quantity=60, request_id="preview-2", model="gpt-premium"
    )
    # parse_pages: 5 + 5 across two days exceeds its 8/day-plan cap in the
    # period and has no list_price row at all -> unpriced overage.
    await _reserve_and_finalize(
        service, customer.id, meter="parse_pages", quantity=5, request_id="preview-3"
    )
    clock.value += timedelta(days=1)
    await _reserve_and_finalize(
        service, customer.id, meter="parse_pages", quantity=5, request_id="preview-4"
    )

    preview = await service.billing_preview(customer.id)
    assert preview["fixed"] == {"price_minor": 4_900, "currency": "CNY"}
    usage_by_meter = {row["meter"]: row for row in preview["usage"]}
    assert usage_by_meter["llm_tokens"]["used"] == 120
    assert usage_by_meter["llm_tokens"]["excess"] == 20
    assert usage_by_meter["parse_pages"]["excess"] == 2

    overage_by_key = {(row["meter"], row["model_tier"]): row for row in preview["overage"]}
    llm_default = overage_by_key[("llm_tokens", "default")]
    llm_premium = overage_by_key[("llm_tokens", "premium")]
    pages_row = overage_by_key[("parse_pages", "default")]
    assert llm_default["quantity"] == 10 and llm_default["cost_micros"] == 20
    assert llm_premium["quantity"] == 10 and llm_premium["cost_micros"] == 100
    assert llm_default["unpriced"] is False and llm_premium["unpriced"] is False
    assert pages_row["unpriced"] is True
    assert pages_row["cost_micros"] == 0

    totals = preview["totals"]
    assert totals["fixed_minor"] == 4_900
    assert totals["overage_micros"] == 120
    assert totals["overage_currency"] == "CNY"
    assert totals["currency_mismatch"] is False
    assert totals["unpriced_meters"] == ["parse_pages"]

    # Without a subscription there is nothing to preview.
    stranger = await service.ensure_billing_customer(owner_id="user-stranger")
    with pytest.raises(CommercialNotFound):
        await service.billing_preview(stranger.id)


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
    assert finding.entity_type == "subscription"
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
