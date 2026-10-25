"""HTTP surface for the price catalog and the billing preview.

Each scenario runs inside one asyncio loop (httpx ASGITransport), so the
in-memory repository's locks never cross event loops.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from fastapi import Depends, FastAPI
import httpx

from deeptutor.api.routers.auth import require_admin
from deeptutor.commercial import (
    CommercialControlPlane,
    InMemoryCommercialRepository,
    PriceEntryDraft,
)
from deeptutor.commercial.access import CommercialAccess, set_commercial_access
from deeptutor.commercial.admin_api import router as admin_router
from deeptutor.commercial.api import require_commercial_access, router
from deeptutor.commercial.models import ResolvedEntitlements, SubscriptionStatus
from deeptutor.commercial.runtime import CommercialSettings
from deeptutor.multi_user.context import set_current_user
from deeptutor.multi_user.models import CurrentUser, UserScope
from deeptutor.services.auth import TokenPayload

NOW = datetime(2026, 8, 5, 9, 0, tzinfo=timezone.utc)


class _Runtime:
    def __init__(self, control_plane, enabled: bool = True) -> None:
        self.settings = CommercialSettings(
            enabled=enabled,
            database_url="postgresql://unused/test",
            trial_model_binding=None,
        )
        self.control_plane = control_plane


def _user(root, *, admin: bool = False) -> CurrentUser:
    user_id = "u_admin" if admin else "u_me"
    return CurrentUser(
        id=user_id,
        username="admin@example.com" if admin else "me@example.com",
        role="admin" if admin else "user",
        scope=UserScope(kind="user", user_id=user_id, root=root),
    )


def _build_app(monkeypatch, tmp_path, *, runtime, admin: bool = False):
    monkeypatch.setattr("deeptutor.commercial.api.get_commercial_runtime", lambda: runtime)
    monkeypatch.setattr("deeptutor.commercial.admin_api.get_commercial_runtime", lambda: runtime)
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    # Mirror production: the adapter mounts the admin router behind
    # require_admin; the override below swaps in the scenario's user.
    app.include_router(
        admin_router,
        prefix="/api/v1",
        dependencies=[Depends(require_admin)],
    )
    user = _user(tmp_path, admin=admin)

    async def fake_access():
        set_current_user(user)
        resolved = ResolvedEntitlements(
            customer_id="cus_me",
            subscription_id="sub_1",
            plan_version_id="planv_1",
            status=SubscriptionStatus.TRIALING,
            active=True,
            resolved_at=NOW,
            valid_until=NOW + timedelta(days=7),
            values={},
        )
        set_commercial_access(CommercialAccess(user.id, resolved))
        return TokenPayload(username=user.username, role=user.role, user_id=user.id)

    app.dependency_overrides[require_commercial_access] = fake_access
    app.dependency_overrides[require_admin] = fake_access
    return app, user


class _MutableClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


def _service() -> tuple[CommercialControlPlane, _MutableClock]:
    import itertools

    counter = itertools.count(1)

    def id_factory(prefix: str) -> str:
        # The derived access context pins customer_id to cus_me; the ledger
        # row must match so the API's control-plane calls find the customer.
        return "cus_me" if prefix == "cus" else f"{prefix}_{next(counter):04d}"

    clock = _MutableClock(NOW)
    service = CommercialControlPlane(
        InMemoryCommercialRepository(),
        clock=clock,
        id_factory=id_factory,
    )
    return service, clock


async def _seed_metered_customer(service: CommercialControlPlane):
    await service.ensure_billing_customer(owner_id="u_me")
    plan = await service.publish_plan_version(
        plan_code="pro",
        version=1,
        name="Pro Monthly",
        price_minor=4900,
        currency="CNY",
        entitlements={
            # Daily quotas hold per day; the subscription-period total can
            # exceed the limit, which is what the preview's overage prices.
            "quota.llm_tokens": {"limit": 10, "window": "utc_day"},
        },
    )
    await service.ensure_trial("cus_me", plan.id)
    return plan


def _price_payload(**overrides) -> dict:
    payload = {
        "price_version": "prices-2026-10",
        "kind": "platform_cost",
        "effective_from": (NOW - timedelta(days=1)).isoformat(),
        "entries": [
            {
                "meter": "llm_tokens",
                "model_tier": "default",
                "unit_price_micros": 3,
                "currency": "CNY",
            }
        ],
    }
    payload.update(overrides)
    return payload


def test_admin_publishes_price_catalog_idempotently(tmp_path, monkeypatch) -> None:
    service, _clock = _service()
    runtime = _Runtime(service)
    audit_calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "deeptutor.commercial.admin_api.log_admin_action",
        lambda action, **kwargs: audit_calls.append((action, kwargs.get("summary") or {})),
    )

    async def scenario() -> None:
        app, _user_ctx = _build_app(monkeypatch, tmp_path, runtime=runtime, admin=True)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            created = await client.post(
                "/api/v1/admin/commercial/prices",
                json=_price_payload(),
            )
            assert created.status_code == 200
            body = created.json()
            assert body[0]["price_version"] == "prices-2026-10"
            assert body[0]["unit_price_micros"] == 3

            replay = await client.post(
                "/api/v1/admin/commercial/prices",
                json=_price_payload(),
            )
            assert replay.status_code == 200
            assert replay.json()[0]["id"] == body[0]["id"]

            # Same row key, different price: immutable -> conflict.
            drifted = _price_payload(
                entries=[
                    {
                        "meter": "llm_tokens",
                        "model_tier": "default",
                        "unit_price_micros": 4,
                        "currency": "CNY",
                    }
                ]
            )
            conflict = await client.post("/api/v1/admin/commercial/prices", json=drifted)
            assert conflict.status_code == 409

            listed = await client.get(
                "/api/v1/admin/commercial/prices",
                params={"kind": "list_price"},
            )
            assert listed.status_code == 200
            assert listed.json() == []

    asyncio.run(scenario())
    assert audit_calls and audit_calls[0][0] == "commercial.price.publish"
    assert audit_calls[0][1]["rows"] == 1


def test_billing_preview_returns_fixed_plus_priced_overage(tmp_path, monkeypatch) -> None:
    service, clock = _service()
    runtime = _Runtime(service)

    async def scenario() -> None:
        await _seed_metered_customer(service)
        await service.publish_price_version(
            price_version="list-2026-10",
            kind="list_price",
            effective_from=NOW - timedelta(days=1),
            entries=[
                PriceEntryDraft(
                    meter="llm_tokens",
                    model_tier="default",
                    unit_price_micros=2,
                    currency="CNY",
                )
            ],
        )
        # 10 tokens on day one, 5 more on day two: period total 15 exceeds
        # the 10/day plan cap by 5, priced at 2 micros each.
        first = await service.reserve_usage("cus_me", "llm_tokens", 10, "preview-1")
        await service.finalize_metered_usage(
            first.id, actual_quantity=10, model="gpt-test", is_estimated=False
        )
        clock.value += timedelta(days=1)
        second = await service.reserve_usage("cus_me", "llm_tokens", 5, "preview-2")
        await service.finalize_metered_usage(
            second.id, actual_quantity=5, model="gpt-test", is_estimated=False
        )

        app, _user_ctx = _build_app(monkeypatch, tmp_path, runtime=runtime)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            preview = await client.get("/api/v1/commercial/billing/preview")
            assert preview.status_code == 200
            body = preview.json()
            assert body["fixed"] == {"price_minor": 4900, "currency": "CNY"}
            assert body["totals"]["overage_micros"] == 10  # (15 - 10) * 2
            assert body["totals"]["unpriced_meters"] == []

    asyncio.run(scenario())


def test_billing_preview_marks_unpriced_overage(tmp_path, monkeypatch) -> None:
    service, clock = _service()
    runtime = _Runtime(service)

    async def scenario() -> None:
        await _seed_metered_customer(service)
        first = await service.reserve_usage("cus_me", "llm_tokens", 10, "unpriced-1")
        await service.finalize_metered_usage(
            first.id, actual_quantity=10, model="gpt-test", is_estimated=False
        )
        clock.value += timedelta(days=1)
        second = await service.reserve_usage("cus_me", "llm_tokens", 2, "unpriced-2")
        await service.finalize_metered_usage(
            second.id, actual_quantity=2, model="gpt-test", is_estimated=False
        )

        app, _user_ctx = _build_app(monkeypatch, tmp_path, runtime=runtime)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            preview = await client.get("/api/v1/commercial/billing/preview")
            assert preview.status_code == 200
            body = preview.json()
            assert body["totals"]["overage_micros"] == 0
            assert body["totals"]["unpriced_meters"] == ["llm_tokens"]
            row = body["overage"][0]
            assert row["unpriced"] is True
            assert row["quantity"] == 2

    asyncio.run(scenario())


def test_billing_preview_guards_disabled_and_admin(tmp_path, monkeypatch) -> None:
    disabled = _Runtime(_service()[0], enabled=False)

    async def scenario_disabled() -> None:
        app, _user_ctx = _build_app(monkeypatch, tmp_path, runtime=disabled)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get("/api/v1/commercial/billing/preview")
            assert response.status_code == 404

    asyncio.run(scenario_disabled())

    enabled = _Runtime(_service()[0])

    async def scenario_admin() -> None:
        app, _user_ctx = _build_app(monkeypatch, tmp_path, runtime=enabled, admin=True)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get("/api/v1/commercial/billing/preview")
            assert response.status_code == 403

    asyncio.run(scenario_admin())
