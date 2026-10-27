from __future__ import annotations

from datetime import datetime, timedelta, timezone
import itertools
from types import SimpleNamespace

import pytest

from deeptutor.commercial.access import (
    CommercialAccess,
    get_commercial_access,
    project_grant_for_commercial,
    require_commercial_access,
    reset_commercial_access,
    set_commercial_access,
)
from deeptutor.commercial.entitlement_context import (
    CommercialAccessDenied,
    integer_limit,
    require_capability,
)
from deeptutor.commercial.errors import CommercialConfigurationError
from deeptutor.commercial.integration import (
    reconcile_existing_commercial_customers,
    startup_commercial_foundation,
    validate_commercial_deployment,
)
from deeptutor.commercial.memory import InMemoryCommercialRepository
from deeptutor.commercial.models import (
    ResolvedEntitlements,
    SubscriptionStatus,
)
from deeptutor.commercial.runtime import CommercialRuntime, CommercialSettings
from deeptutor.commercial.seed import TrialV1ModelBinding, seed_trial_v1
from deeptutor.commercial.service import CommercialControlPlane
from deeptutor.multi_user.context import reset_current_user, set_current_user
from deeptutor.multi_user.models import CurrentUser, UserScope
from deeptutor.services.auth import TokenPayload


def _settings(*, enabled: bool = True) -> CommercialSettings:
    return CommercialSettings(
        enabled=enabled,
        database_url="postgresql://unused/test" if enabled else None,
        trial_model_binding=(
            TrialV1ModelBinding("trial-profile", ("trial-model",)) if enabled else None
        ),
    )


def _catalog(*, owner_bound: bool = False, include_model: bool = True):
    return {
        "services": {
            "llm": {
                "profiles": [
                    {
                        "id": "trial-profile",
                        "owner_bound": owner_bound,
                        "models": ([{"id": "trial-model"}] if include_model else []),
                    }
                ]
            }
        }
    }


def _user(tmp_path, *, user_id: str = "u_1", role: str = "user") -> CurrentUser:
    return CurrentUser(
        id=user_id,
        username="user@example.com" if role == "user" else "admin@example.com",
        role=role,
        scope=UserScope(
            kind="user" if role == "user" else "admin",
            user_id=user_id,
            root=tmp_path,
        ),
    )


@pytest.mark.parametrize(
    ("auth", "integrations", "environ", "catalog", "message"),
    [
        (
            {"enabled": False, "cookie_secure": True},
            {"pocketbase_url": ""},
            {"DEEPTUTOR_BOOTSTRAP_ADMIN_EMAIL": "admin@example.com"},
            _catalog(),
            "auth.enabled",
        ),
        (
            {"enabled": True, "cookie_secure": False},
            {"pocketbase_url": ""},
            {"DEEPTUTOR_BOOTSTRAP_ADMIN_EMAIL": "admin@example.com"},
            _catalog(),
            "cookie_secure",
        ),
        (
            {"enabled": True, "cookie_secure": True},
            {"pocketbase_url": "http://pocketbase:8090"},
            {"DEEPTUTOR_BOOTSTRAP_ADMIN_EMAIL": "admin@example.com"},
            _catalog(),
            "PocketBase",
        ),
        (
            {"enabled": True, "cookie_secure": True},
            {"pocketbase_url": ""},
            {"DEEPTUTOR_BOOTSTRAP_ADMIN_EMAIL": "not-an-email"},
            _catalog(),
            "BOOTSTRAP_ADMIN_EMAIL",
        ),
        (
            {"enabled": True, "cookie_secure": True},
            {"pocketbase_url": ""},
            {"DEEPTUTOR_BOOTSTRAP_ADMIN_EMAIL": "admin@example.com"},
            _catalog(owner_bound=True),
            "owner-bound",
        ),
        (
            {"enabled": True, "cookie_secure": True},
            {"pocketbase_url": ""},
            {"DEEPTUTOR_BOOTSTRAP_ADMIN_EMAIL": "admin@example.com"},
            _catalog(include_model=False),
            "missing",
        ),
    ],
)
def test_commercial_startup_security_gates_fail_closed(
    auth, integrations, environ, catalog, message
) -> None:
    with pytest.raises(CommercialConfigurationError, match=message):
        validate_commercial_deployment(
            CommercialRuntime(_settings()),
            auth_settings=auth,
            integrations_settings=integrations,
            environ=environ,
            catalog=catalog,
            users=[],
        )


def test_commercial_startup_gate_accepts_explicit_secure_builtin_configuration() -> None:
    validate_commercial_deployment(
        CommercialRuntime(_settings()),
        auth_settings={"enabled": True, "cookie_secure": True},
        integrations_settings={"pocketbase_url": ""},
        environ={"DEEPTUTOR_BOOTSTRAP_ADMIN_EMAIL": "Admin@Example.com"},
        catalog=_catalog(),
        users=[],
    )


def test_disabled_commercial_gate_is_inert() -> None:
    validate_commercial_deployment(
        CommercialRuntime(_settings(enabled=False)),
        auth_settings={},
        integrations_settings={"pocketbase_url": "http://unsupported"},
        environ={},
        catalog={},
        users=[],
    )


def test_existing_users_require_matching_active_bootstrap_admin() -> None:
    runtime = CommercialRuntime(_settings())
    common = {
        "auth_settings": {"enabled": True, "cookie_secure": True},
        "integrations_settings": {"pocketbase_url": ""},
        "environ": {"DEEPTUTOR_BOOTSTRAP_ADMIN_EMAIL": "admin@example.com"},
        "catalog": _catalog(),
    }
    with pytest.raises(CommercialConfigurationError, match="no active verified"):
        validate_commercial_deployment(
            runtime,
            users=[{"username": "user@example.com", "role": "user", "email_verified": True}],
            **common,
        )
    with pytest.raises(CommercialConfigurationError, match="must match"):
        validate_commercial_deployment(
            runtime,
            users=[
                {
                    "username": "other-admin@example.com",
                    "role": "admin",
                    "email_verified": True,
                    "disabled": False,
                }
            ],
            **common,
        )
    validate_commercial_deployment(
        runtime,
        users=[
            {
                "username": "ADMIN@example.com",
                "role": "admin",
                "email_verified": True,
                "disabled": False,
            }
        ],
        **common,
    )


@pytest.mark.asyncio
async def test_existing_user_reconciliation_maps_customer_without_starting_trial() -> None:
    class FakeControlPlane:
        def __init__(self, owners: list[str]) -> None:
            self.owners = owners

        async def ensure_billing_customer(self, *, owner_id: str):
            self.owners.append(owner_id)

    class FakeRuntime:
        settings = _settings()

        def __init__(self) -> None:
            self.owners: list[str] = []
            self.control_plane = FakeControlPlane(self.owners)

    runtime = FakeRuntime()
    count = await reconcile_existing_commercial_customers(
        runtime,  # type: ignore[arg-type]
        users=[
            {"id": "u_ok", "role": "user", "email_verified": True, "disabled": False},
            {"id": "u_unverified", "role": "user", "email_verified": False},
            {"id": "u_disabled", "role": "user", "email_verified": True, "disabled": True},
            {"id": "u_admin", "role": "admin", "email_verified": True},
        ],
    )
    assert count == 1
    assert runtime.owners == ["u_ok"]


@pytest.mark.asyncio
async def test_startup_runs_validate_start_and_reconcile(monkeypatch) -> None:
    events: list[str] = []

    class FakeControlPlane:
        async def reconcile(self, *, repair: bool):
            assert repair is True
            events.append("repair")
            return SimpleNamespace(repaired=2)

        async def ensure_billing_customer(self, *, owner_id: str):
            events.append(f"customer:{owner_id}")

    class FakeRuntime:
        settings = _settings()
        control_plane = FakeControlPlane()

        async def startup(self):
            events.append("startup")

        async def shutdown(self):
            events.append("shutdown")

    runtime = FakeRuntime()
    monkeypatch.setattr(
        "deeptutor.commercial.integration.validate_commercial_deployment",
        lambda _runtime: events.append("validate"),
    )
    monkeypatch.setattr(
        "deeptutor.commercial.integration.list_users",
        lambda: [
            {
                "id": "u_existing",
                "role": "user",
                "email_verified": True,
                "disabled": False,
            }
        ],
    )
    assert await startup_commercial_foundation(runtime) is runtime  # type: ignore[arg-type]
    assert events == ["validate", "startup", "repair", "customer:u_existing"]


async def _active_runtime(now: datetime) -> tuple[CommercialRuntime, CommercialControlPlane]:
    sequence = itertools.count(1)
    repository = InMemoryCommercialRepository()
    service = CommercialControlPlane(
        repository,
        clock=lambda: now,
        id_factory=lambda prefix: f"{prefix}_{next(sequence)}",
    )
    plan = await seed_trial_v1(
        service,
        TrialV1ModelBinding("trial-profile", ("trial-model",)),
    )
    runtime = CommercialRuntime(_settings())
    runtime.repository = repository  # type: ignore[assignment]
    runtime.control_plane = service
    runtime.trial_plan = plan
    return runtime, service


@pytest.mark.asyncio
async def test_real_reconciliation_does_not_consume_dormant_user_trial() -> None:
    runtime, service = await _active_runtime(datetime.now(timezone.utc))
    assert (
        await reconcile_existing_commercial_customers(
            runtime,
            users=[
                {
                    "id": "u_dormant",
                    "role": "user",
                    "email_verified": True,
                    "disabled": False,
                }
            ],
        )
        == 1
    )
    customer = await service.get_billing_customer_by_owner("u_dormant")
    assert customer is not None
    assert customer.trial_claimed_at is None
    assert await service.repository.list_subscriptions(customer.id) == ()


@pytest.mark.asyncio
async def test_dependency_resolves_trial_and_installs_request_context(
    tmp_path, monkeypatch
) -> None:
    runtime, _service = await _active_runtime(datetime.now(timezone.utc))
    user = _user(tmp_path)
    user_token = set_current_user(user)
    access_token = set_commercial_access(None)
    monkeypatch.setattr("deeptutor.commercial.access.get_commercial_runtime", lambda: runtime)
    try:
        payload = TokenPayload(username=user.username, role="user", user_id=user.id)
        assert await require_commercial_access(payload) is payload
        access = get_commercial_access()
        assert access is not None
        assert access.owner_id == user.id
        assert access.resolved.status is SubscriptionStatus.TRIALING
    finally:
        reset_commercial_access(access_token)
        reset_current_user(user_token)


@pytest.mark.asyncio
async def test_expired_trial_and_missing_subscription_install_inactive_state(
    tmp_path, monkeypatch
) -> None:
    outer_access_token = set_commercial_access(None)
    now = datetime.now(timezone.utc)
    runtime, service = await _active_runtime(now - timedelta(days=8))
    await runtime.ensure_trial_for_owner("u_expired")
    service._clock = lambda: now  # type: ignore[attr-defined]

    user = _user(tmp_path, user_id="u_expired")
    user_token = set_current_user(user)
    monkeypatch.setattr("deeptutor.commercial.access.get_commercial_runtime", lambda: runtime)
    try:
        payload = TokenPayload(username=user.username, role="user", user_id=user.id)
        assert await require_commercial_access(payload) is payload
        expired = get_commercial_access()
        assert expired is not None
        assert expired.resolved.status is SubscriptionStatus.EXPIRED
        assert expired.is_current is False
    finally:
        reset_current_user(user_token)

    class NoSubscriptionRuntime:
        settings = _settings()

        async def ensure_trial_for_owner(self, _owner_id: str, **_kwargs):
            return None

        async def resolve_for_owner(self, owner_id: str):
            return ResolvedEntitlements(
                customer_id=owner_id,
                subscription_id=None,
                plan_version_id=None,
                status=None,
                active=False,
                resolved_at=now,
                valid_until=None,
                values={},
            )

    missing_user = _user(tmp_path, user_id="u_missing")
    user_token = set_current_user(missing_user)
    monkeypatch.setattr(
        "deeptutor.commercial.access.get_commercial_runtime",
        lambda: NoSubscriptionRuntime(),
    )
    try:
        payload = TokenPayload(
            username=missing_user.username,
            role="user",
            user_id=missing_user.id,
        )
        assert await require_commercial_access(payload) is payload
        missing = get_commercial_access()
        assert missing is not None
        assert missing.resolved.status is None
        assert missing.is_current is False
    finally:
        reset_current_user(user_token)
        reset_commercial_access(outer_access_token)


@pytest.mark.asyncio
async def test_admin_bypasses_commercial_resolution(tmp_path, monkeypatch) -> None:
    class ExplodingRuntime:
        settings = _settings()

        async def ensure_trial_for_owner(self, _owner_id: str, **_kwargs):
            raise AssertionError("admin must not enter subscription resolution")

    admin = _user(tmp_path, user_id="u_admin", role="admin")
    user_token = set_current_user(admin)
    monkeypatch.setattr(
        "deeptutor.commercial.access.get_commercial_runtime", lambda: ExplodingRuntime()
    )
    try:
        payload = TokenPayload(username=admin.username, role="admin", user_id=admin.id)
        assert await require_commercial_access(payload) is payload
        assert get_commercial_access() is None
    finally:
        reset_current_user(user_token)


@pytest.mark.asyncio
async def test_trial_entitlements_project_onto_grant_without_persistence() -> None:
    runtime, _service = await _active_runtime(datetime.now(timezone.utc))
    await runtime.ensure_trial_for_owner("u_trial")
    resolved = await runtime.resolve_for_owner("u_trial")
    assert resolved is not None
    token = set_commercial_access(CommercialAccess("u_trial", resolved))
    try:
        grant = project_grant_for_commercial(
            "u_trial",
            {
                "models": {"llm": []},
                "platform": {},
                "byok": {},
                "partners": [{"partner_id": "shared"}],
                "enabled_tools": None,
                "mcp_tools": ["host-tool"],
                "exec_enabled": None,
                "quota": {},
                "token_quota": {},
            },
        )
    finally:
        reset_commercial_access(token)

    assert grant["models"]["llm"] == [{"profile_id": "trial-profile", "model_ids": ["trial-model"]}]
    assert grant["platform"] == {
        "embedding": {"enabled": True},
        "mineru": {"enabled": True},
    }
    assert all(item["enabled"] for item in grant["byok"].values())
    assert grant["enabled_tools"] == []
    assert grant["mcp_tools"] == []
    assert grant["exec_enabled"] is False
    assert grant["partners"] == []
    assert grant["quota"]["llm"] == {
        "daily_tokens": 100_000,
        "monthly_tokens": 0,
    }
    assert grant["quota"]["embedding"]["monthly_tokens"] == 500_000
    assert grant["quota"]["mineru"] == {
        "daily_pages": 0,
        "monthly_pages": 20,
        "max_pages_per_file": 10,
    }


def test_stale_commercial_snapshot_projects_deny_all() -> None:
    now = datetime.now(timezone.utc)
    resolved = ResolvedEntitlements(
        customer_id="cus_1",
        subscription_id="sub_1",
        plan_version_id="plan_1",
        status=SubscriptionStatus.TRIALING,
        active=True,
        resolved_at=now - timedelta(days=8),
        valid_until=now - timedelta(seconds=1),
        values={"models.llm": {"profile_id": "p", "model_ids": ["m"]}},
    )
    token = set_commercial_access(CommercialAccess("u_1", resolved))
    try:
        projected = project_grant_for_commercial(
            "u_1",
            {"models": {"llm": [{"profile_id": "unsafe"}]}},
        )
    finally:
        reset_commercial_access(token)
    assert projected["models"]["llm"] == []
    assert projected["exec_enabled"] is False
    assert projected["mcp_tools"] == []


def test_missing_snapshot_fails_closed_for_commercial_regular_user(tmp_path, monkeypatch) -> None:
    class Runtime:
        settings = _settings()

    user_token = set_current_user(_user(tmp_path, user_id="u_missing_context"))
    access_token = set_commercial_access(None)
    monkeypatch.setattr(
        "deeptutor.commercial.entitlement_context.get_commercial_runtime",
        lambda: Runtime(),
    )
    try:
        projected = project_grant_for_commercial(
            "u_missing_context",
            {
                "models": {"llm": [{"profile_id": "leaked"}]},
                "exec_enabled": True,
                "mcp_tools": ["host"],
            },
        )
    finally:
        reset_commercial_access(access_token)
        reset_current_user(user_token)
    assert projected["models"]["llm"] == []
    assert projected["exec_enabled"] is False
    assert projected["mcp_tools"] == []


@pytest.mark.asyncio
async def test_consuming_guards_use_stable_denial_codes_and_limits(tmp_path, monkeypatch) -> None:
    runtime, _service = await _active_runtime(datetime.now(timezone.utc))
    await runtime.ensure_trial_for_owner("u_guard")
    resolved = await runtime.resolve_for_owner("u_guard")
    assert resolved is not None
    user_token = set_current_user(_user(tmp_path, user_id="u_guard"))
    access_token = set_commercial_access(CommercialAccess("u_guard", resolved))
    monkeypatch.setattr(
        "deeptutor.commercial.entitlement_context.get_commercial_runtime",
        lambda: runtime,
    )
    try:
        assert require_capability("deep_solve") is not None
        assert integer_limit("concurrent_turns") == 1
        with pytest.raises(CommercialAccessDenied) as denied:
            require_capability("math_animator")
        assert denied.value.code == "capability_not_in_plan"
    finally:
        reset_commercial_access(access_token)
        reset_current_user(user_token)


@pytest.mark.asyncio
async def test_load_grant_projects_the_active_trial_onto_users_without_a_grant_file(
    tmp_path, monkeypatch
):
    """Regression: load_grant must call project_grant_for_commercial.

    The projection is designed to be applied at load time; without the call a
    trial user (no grant file yet) resolves to empty_grant — no models — and
    the WS capability gate silently closes the chat with code 4003.
    """
    from deeptutor.multi_user import grants
    from deeptutor.multi_user.grants import load_grant

    monkeypatch.setattr(grants, "GRANTS_DIR", tmp_path / "grants")
    now = datetime.now(timezone.utc)
    runtime, _service = await _active_runtime(now)
    await runtime.ensure_trial_for_owner("u_trial")
    resolved = await runtime.resolve_for_owner("u_trial")
    assert resolved is not None

    user_token = set_current_user(_user(tmp_path, user_id="u_trial"))
    access_token = set_commercial_access(CommercialAccess("u_trial", resolved))
    try:
        grant = load_grant("u_trial")
    finally:
        reset_commercial_access(access_token)
        reset_current_user(user_token)

    assert grant["models"]["llm"] == [{"profile_id": "trial-profile", "model_ids": ["trial-model"]}]
    assert grant["quota"]["llm"]["daily_tokens"] == 100_000


def test_load_grant_fails_closed_without_a_commercial_snapshot(tmp_path, monkeypatch):
    from deeptutor.multi_user import grants
    from deeptutor.multi_user.grants import load_grant

    monkeypatch.setattr(grants, "GRANTS_DIR", tmp_path / "grants")
    monkeypatch.setattr(
        "deeptutor.commercial.entitlement_context.get_commercial_runtime",
        lambda: SimpleNamespace(settings=_settings()),
    )
    user_token = set_current_user(_user(tmp_path, user_id="u_no_snapshot"))
    access_token = set_commercial_access(None)
    try:
        grant = load_grant("u_no_snapshot")
    finally:
        reset_commercial_access(access_token)
        reset_current_user(user_token)

    assert grant["models"]["llm"] == []
    assert grant["exec_enabled"] is False
