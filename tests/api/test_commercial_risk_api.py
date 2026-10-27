"""Admin risk-control endpoints: suspend/reinstate with audit trail."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from fastapi import Depends, FastAPI
import httpx

from deeptutor.api.routers.auth import require_admin
from deeptutor.commercial import CommercialControlPlane, InMemoryCommercialRepository
from deeptutor.commercial.admin_api import router as admin_router
from deeptutor.commercial.api import require_commercial_access, router
from deeptutor.commercial.models import SubscriptionStatus
from deeptutor.commercial.runtime import CommercialSettings
from deeptutor.multi_user.context import set_current_user
from deeptutor.multi_user.models import CurrentUser, UserScope
from deeptutor.services.auth import TokenPayload

NOW = datetime(2026, 10, 27, 9, 0, tzinfo=timezone.utc)


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
    user = _user(tmp_path, admin=True)

    async def fake_admin():
        set_current_user(user)
        return TokenPayload(username=user.username, role=user.role, user_id=user.id)

    app.dependency_overrides[require_admin] = fake_admin
    app.dependency_overrides[require_commercial_access] = fake_admin
    return app, user


def _service() -> CommercialControlPlane:
    import itertools

    counter = itertools.count(1)
    return CommercialControlPlane(
        InMemoryCommercialRepository(),
        clock=lambda: NOW,
        id_factory=lambda prefix: f"{prefix}_{next(counter):04d}",
    )


def test_suspend_and_reinstate_round_trip_with_audit(tmp_path, monkeypatch) -> None:
    service = _service()
    runtime = _Runtime(service)
    audit_calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "deeptutor.commercial.admin_api.log_admin_action",
        lambda action, **kwargs: audit_calls.append((action, kwargs.get("summary") or {})),
    )

    async def scenario() -> None:
        customer = await service.ensure_billing_customer(owner_id="u_me")
        plan = await service.publish_plan_version(
            plan_code="pro",
            version=1,
            name="Pro Monthly",
            price_minor=4900,
            currency="CNY",
            entitlements={"capability.chat": True},
        )
        await service.create_active_subscription(
            customer.id,
            plan.id,
            external_subscription_id="sub_risk_1",
            current_period_start=NOW,
            current_period_end=NOW + timedelta(days=30),
        )
        app, _user_ctx = _build_app(monkeypatch, tmp_path, runtime=runtime)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            suspended = await client.post("/api/v1/admin/commercial/customers/u_me/suspend")
            assert suspended.status_code == 200
            body = suspended.json()
            assert body["status"] == "suspended"
            assert body["owner_id"] == "u_me"
            assert body["subscription_id"]

            # While suspended, entitlement resolution denies access.
            resolved = await service.resolve_entitlements(customer.id)
            assert resolved.active is False

            # Idempotent suspension keeps the same state.
            again = await client.post("/api/v1/admin/commercial/customers/u_me/suspend")
            assert again.status_code == 200
            assert again.json()["status"] == "suspended"

            reinstated = await client.post("/api/v1/admin/commercial/customers/u_me/reinstate")
            assert reinstated.status_code == 200
            assert reinstated.json()["status"] == "active"

            # Reinstate is also idempotent.
            repeat = await client.post("/api/v1/admin/commercial/customers/u_me/reinstate")
            assert repeat.status_code == 200
            assert repeat.json()["status"] == "active"

            # Unknown owners are a clean 404.
            missing = await client.post("/api/v1/admin/commercial/customers/nobody/suspend")
            assert missing.status_code == 404

    asyncio.run(scenario())
    actions = [action for action, _ in audit_calls]
    assert "commercial.customer.suspend" in actions
    assert "commercial.customer.reinstate" in actions
    suspend_summary = next(
        summary for action, summary in audit_calls if action == "commercial.customer.suspend"
    )
    assert suspend_summary["owner_id"] == "u_me"


def test_suspend_requires_a_suspendable_subscription(tmp_path, monkeypatch) -> None:
    service = _service()
    runtime = _Runtime(service)
    monkeypatch.setattr(
        "deeptutor.commercial.admin_api.log_admin_action",
        lambda action, **kwargs: None,
    )

    async def scenario() -> None:
        # Only ACTIVE can suspend: a trialing subscription conflicts (409).
        customer = await service.ensure_billing_customer(owner_id="u_trial")
        plan = await service.publish_plan_version(
            plan_code="pro",
            version=1,
            name="Pro Monthly",
            price_minor=4900,
            currency="CNY",
            entitlements={"capability.chat": True},
        )
        await service.ensure_trial(customer.id, plan.id)
        # An owner without any subscription is a clean 404.
        await service.ensure_billing_customer(owner_id="u_empty")
        app, _user_ctx = _build_app(monkeypatch, tmp_path, runtime=runtime)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            denied = await client.post("/api/v1/admin/commercial/customers/u_trial/suspend")
            assert denied.status_code == 409

            missing = await client.post("/api/v1/admin/commercial/customers/u_empty/suspend")
            assert missing.status_code == 404

    asyncio.run(scenario())
