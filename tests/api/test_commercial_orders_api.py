"""HTTP surface for plans, orders, and manual fulfillment.

Each scenario runs inside one asyncio loop (httpx ASGITransport), so the
in-memory repository's locks never cross event loops.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from fastapi import Depends, FastAPI
import httpx

from deeptutor.api.routers.auth import require_admin
from deeptutor.commercial import CommercialControlPlane, InMemoryCommercialRepository
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


def _build_app(monkeypatch, tmp_path, *, runtime):
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
    user = _user(tmp_path)

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


def _service() -> CommercialControlPlane:
    import itertools

    counter = itertools.count(1)

    def id_factory(prefix: str) -> str:
        # The derived access context pins customer_id to cus_me; the ledger
        # row must match so the API's control-plane calls find the customer.
        return "cus_me" if prefix == "cus" else f"{prefix}_{next(counter):04d}"

    return CommercialControlPlane(
        InMemoryCommercialRepository(),
        clock=lambda: NOW,
        id_factory=id_factory,
    )


def test_user_order_flow_creates_lists_and_cancels(tmp_path, monkeypatch) -> None:
    service = _service()
    runtime = _Runtime(service)

    async def scenario() -> None:
        await service.ensure_billing_customer(owner_id="u_me")
        plan = await service.publish_plan_version(
            plan_code="pro",
            version=1,
            name="Pro Monthly",
            price_minor=4900,
            currency="CNY",
            entitlements={"capability.chat": True},
        )
        app, _user_ctx = _build_app(monkeypatch, tmp_path, runtime=runtime)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            plans = await client.get("/api/v1/commercial/plans")
            assert plans.status_code == 200
            assert plans.json()[0]["price_minor"] == 4900

            created = await client.post(
                "/api/v1/commercial/orders",
                json={
                    "plan_version_id": plan.id,
                    "idempotency_key": "checkout-http-1",
                },
            )
            assert created.status_code == 200
            order = created.json()
            assert order["status"] == "pending_payment"

            replay = await client.post(
                "/api/v1/commercial/orders",
                json={
                    "plan_version_id": plan.id,
                    "idempotency_key": "checkout-http-1",
                },
            )
            assert replay.json()["id"] == order["id"]

            listed = await client.get("/api/v1/commercial/orders")
            assert [item["id"] for item in listed.json()] == [order["id"]]

            canceled = await client.post(f"/api/v1/commercial/orders/{order['id']}/cancel")
            assert canceled.status_code == 200
            assert canceled.json()["status"] == "canceled"

    asyncio.run(scenario())


def test_admin_mark_paid_fulfills_order_and_audits(tmp_path, monkeypatch) -> None:
    service = _service()
    runtime = _Runtime(service)
    audit_calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "deeptutor.commercial.admin_api.log_admin_action",
        lambda action, **kwargs: audit_calls.append((action, kwargs.get("summary") or {})),
    )

    async def scenario() -> None:
        await service.ensure_billing_customer(owner_id="u_me")
        plan = await service.publish_plan_version(
            plan_code="pro",
            version=1,
            name="Pro Monthly",
            price_minor=4900,
            currency="CNY",
            entitlements={"capability.chat": True},
        )
        app, _user_ctx = _build_app(monkeypatch, tmp_path, runtime=runtime)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            created = await client.post(
                "/api/v1/commercial/orders",
                json={"plan_version_id": plan.id, "idempotency_key": "pay-http-1"},
            )
            order = created.json()

            fulfilled = await client.post(
                f"/api/v1/admin/commercial/orders/{order['id']}/mark-paid",
                json={"payment_reference": "wire-42"},
            )
            assert fulfilled.status_code == 200
            body = fulfilled.json()
            assert body["status"] == "fulfilled"
            assert body["subscription_id"]

            replay = await client.post(
                f"/api/v1/admin/commercial/orders/{order['id']}/mark-paid",
                json={},
            )
            assert replay.status_code == 200
            assert replay.json()["subscription_id"] == body["subscription_id"]

    asyncio.run(scenario())
    assert audit_calls and audit_calls[0][0] == "commercial.order.mark_paid"
    assert audit_calls[0][1]["payment_reference"] == "wire-42"


def test_disabled_commercial_mode_hides_the_surfaces(tmp_path, monkeypatch) -> None:
    runtime = _Runtime(_service(), enabled=False)

    async def scenario() -> None:
        app, _user_ctx = _build_app(monkeypatch, tmp_path, runtime=runtime)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            plans = await client.get("/api/v1/commercial/plans")
            assert plans.status_code == 200
            assert plans.json() == []
            orders = await client.get("/api/v1/commercial/orders")
            assert orders.status_code == 404

    asyncio.run(scenario())
