"""Per-user usage accounting read model over the durable ledger."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from .conftest import seed_customer_and_plan

_SEQ = {"n": 0}


def _request_id(label: str) -> str:
    _SEQ["n"] += 1
    return f"{label}:{_SEQ['n']}"


async def _finalized_event(service, customer, *, meter, quantity, actual, model, estimated):
    reservation = await service.reserve_usage(customer.id, meter, quantity, _request_id(meter))
    return await service.finalize_usage(
        reservation.id,
        actual_quantity=actual,
        provider="openai",
        model=model,
        usage_units={meter: actual},
        is_estimated=estimated,
    )


async def _seed_usage(service):
    customer, plan = await seed_customer_and_plan(service)
    subscription = await service.create_active_subscription(
        customer.id,
        plan.id,
        external_subscription_id=f"sub_usage_{_request_id('x')}",
        current_period_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
        current_period_end=datetime(2026, 9, 1, tzinfo=timezone.utc),
    )
    return customer, subscription


@pytest.mark.asyncio
async def test_usage_details_aggregates_by_day_model_and_estimated_share(
    commercial_harness,
):
    service, _repository, clock = commercial_harness
    customer, _subscription = await _seed_usage(service)

    day_one = datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc)
    clock.value = day_one
    await _finalized_event(
        service,
        customer,
        meter="llm_tokens",
        quantity=60,
        actual=50,
        model="gpt-x",
        estimated=False,
    )
    await _finalized_event(
        service,
        customer,
        meter="parse_pages",
        quantity=5,
        actual=5,
        model="mineru",
        estimated=False,
    )
    clock.value = day_one + timedelta(days=2, hours=1)
    await _finalized_event(
        service,
        customer,
        meter="llm_tokens",
        quantity=30,
        actual=30,
        model="gpt-y",
        estimated=True,
    )

    details = await service.usage_details(customer.id)

    assert details["totals"] == {"llm_tokens": 80, "parse_pages": 5}
    assert set(details["by_day"]) == {"2026-08-05", "2026-08-07"}
    assert details["by_day"]["2026-08-07"] == {"llm_tokens": 30}
    assert details["by_model"]["llm_tokens"] == {"gpt-x": 50, "gpt-y": 30}
    assert details["estimated"]["events"] == 1
    assert details["estimated"]["event_share"] == pytest.approx(1 / 3)
    assert details["estimated"]["by_meter"] == {"llm_tokens": pytest.approx(30 / 80)}
    assert details["truncated"] is False
    assert [event["model"] for event in details["events"]] == ["gpt-y", "mineru", "gpt-x"]
    assert details["events"][0]["usage_units"] == {"llm_tokens": 30}
    assert details["events"][0]["price_version"]


@pytest.mark.asyncio
async def test_usage_details_filters_by_window_and_meter(commercial_harness):
    service, _repository, clock = commercial_harness
    customer, _subscription = await _seed_usage(service)

    day_one = datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc)
    clock.value = day_one
    await _finalized_event(
        service,
        customer,
        meter="llm_tokens",
        quantity=60,
        actual=50,
        model="gpt-x",
        estimated=False,
    )
    clock.value = day_one + timedelta(days=3)
    await _finalized_event(
        service,
        customer,
        meter="llm_tokens",
        quantity=40,
        actual=40,
        model="gpt-y",
        estimated=False,
    )
    await _finalized_event(
        service,
        customer,
        meter="parse_pages",
        quantity=5,
        actual=5,
        model="mineru",
        estimated=False,
    )

    window = await service.usage_details(
        customer.id,
        since=day_one + timedelta(days=1),
        until=day_one + timedelta(days=4),
    )
    assert window["totals"] == {"llm_tokens": 40, "parse_pages": 5}

    page = await service.usage_details(customer.id, limit=2)
    assert page["truncated"] is True
    assert len(page["events"]) == 2

    only_llm = await service.usage_details(customer.id, meter="llm_tokens")
    assert only_llm["totals"] == {"llm_tokens": 90}
    assert {event["model"] for event in only_llm["events"]} == {"gpt-x", "gpt-y"}


@pytest.mark.asyncio
async def test_usage_details_rejects_naive_windows(commercial_harness):
    service, _repository, _clock = commercial_harness
    customer, _subscription = await _seed_usage(service)
    with pytest.raises(ValueError):
        await service.usage_details(customer.id, since=datetime(2026, 8, 5, 10, 0))


@pytest.mark.asyncio
async def test_usage_details_on_empty_ledger_returns_zero_share(commercial_harness):
    service, _repository, _clock = commercial_harness
    customer, _subscription = await _seed_usage(service)
    details = await service.usage_details(customer.id)
    assert details["events"] == []
    assert details["totals"] == {}
    assert details["estimated"] == {"events": 0, "event_share": 0.0, "by_meter": {}}
