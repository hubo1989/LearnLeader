from __future__ import annotations

import io
import zipfile

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import pytest


@pytest.fixture
def local_auth_env(auth_isolated_root, monkeypatch):
    from deeptutor.api.routers import auth as auth_router
    from deeptutor.services import auth as auth_service

    monkeypatch.setattr(auth_router, "AUTH_ENABLED", True)
    monkeypatch.setattr(auth_router, "POCKETBASE_ENABLED", False)
    monkeypatch.setattr(auth_service, "AUTH_ENABLED", True)
    monkeypatch.setattr(auth_service, "AUTH_SECRET", "route-test-secret")
    auth_service.add_user("user@example.com", "password1234", role="user")
    return auth_router, auth_service


def test_login_route_enforces_durable_username_ip_limit(local_auth_env, monkeypatch) -> None:
    auth_router, _auth_service = local_auth_env
    monkeypatch.setenv("DEEPTUTOR_LOGIN_MAX_FAILURES", "2")
    app = FastAPI()
    app.include_router(auth_router.router)

    with TestClient(app) as client:
        first = client.post("/login", json={"username": "user@example.com", "password": "wrong"})
        second = client.post("/login", json={"username": "USER@example.com", "password": "wrong"})
        blocked = client.post(
            "/login",
            json={"username": "user@example.com", "password": "password1234"},
        )

    assert first.status_code == 401
    assert second.status_code == 401
    assert blocked.status_code == 429
    assert int(blocked.headers["retry-after"]) > 0


def test_login_route_keeps_auth_disabled_local_mode(monkeypatch) -> None:
    from deeptutor.api.routers import auth as auth_router

    monkeypatch.setattr(auth_router, "AUTH_ENABLED", False)
    monkeypatch.setattr(
        auth_router.login_rate_limit,
        "check_login_allowed",
        lambda *_args, **_kwargs: pytest.fail("disabled auth must bypass limiter"),
    )
    app = FastAPI()
    app.include_router(auth_router.router)

    with TestClient(app) as client:
        response = client.post("/login", json={"username": "", "password": ""})

    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_login_route_rejects_overlong_username_before_limiter(local_auth_env, monkeypatch) -> None:
    auth_router, _auth_service = local_auth_env
    monkeypatch.setattr(
        auth_router.login_rate_limit,
        "check_login_allowed",
        lambda *_args, **_kwargs: pytest.fail("invalid payload must not reach the limiter"),
    )
    app = FastAPI()
    app.include_router(auth_router.router)

    with TestClient(app) as client:
        response = client.post(
            "/login",
            json={"username": "用" * 100_000, "password": "password1234"},
        )

    assert response.status_code == 422


def test_profile_export_route_returns_zip(local_auth_env) -> None:
    auth_router, auth_service = local_auth_env
    from deeptutor.multi_user import paths

    info = auth_service.get_user_info("user@example.com")
    assert info is not None
    workspace_file = paths.USERS_ROOT / info["id"] / "workspace" / "note.txt"
    workspace_file.parent.mkdir(parents=True, exist_ok=True)
    workspace_file.write_text("private note", encoding="utf-8")
    payload = auth_service.TokenPayload(
        username="user@example.com", role="user", user_id=info["id"]
    )
    app = FastAPI()
    app.include_router(auth_router.router)
    app.dependency_overrides[auth_router.require_auth] = lambda: payload

    with TestClient(app) as client:
        response = client.get("/profile/export")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["cache-control"] == "no-store"
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert "account/profile.json" in archive.namelist()
        assert "workspace/workspace/note.txt" in archive.namelist()


def test_self_delete_route_requires_password_and_erases_account(local_auth_env) -> None:
    auth_router, auth_service = local_auth_env

    info = auth_service.get_user_info("user@example.com")
    assert info is not None
    payload = auth_service.TokenPayload(
        username="user@example.com", role="user", user_id=info["id"]
    )
    app = FastAPI()
    app.include_router(auth_router.router)
    app.dependency_overrides[auth_router.require_auth] = lambda: payload

    with TestClient(app) as client:
        rejected = client.request("DELETE", "/profile", json={"password": "wrong-password"})
        deleted = client.request("DELETE", "/profile", json={"password": "password1234"})

    assert rejected.status_code == 401
    assert deleted.status_code == 200
    assert deleted.json() == {"ok": True, "already_deleted": False}
    assert "Max-Age=0" in deleted.headers.get("set-cookie", "")
    assert auth_service.get_user_info("user@example.com") is None


@pytest.mark.asyncio
async def test_admin_delete_route_is_idempotent(local_auth_env) -> None:
    auth_router, auth_service = local_auth_env

    first = await auth_router.remove_user(
        "user@example.com",
        current=auth_service.TokenPayload(
            username="owner@example.com", role="admin", user_id="u_owner"
        ),
    )
    second = await auth_router.remove_user(
        "user@example.com",
        current=auth_service.TokenPayload(
            username="owner@example.com", role="admin", user_id="u_owner"
        ),
    )

    assert first == {"ok": True, "already_deleted": False}
    assert second == {"ok": True, "already_deleted": True}


@pytest.mark.asyncio
async def test_commercial_erasure_precedes_identity_commit(local_auth_env, monkeypatch) -> None:
    auth_router, auth_service = local_auth_env
    from deeptutor.commercial import runtime as commercial_runtime

    target = auth_service.get_user_info("user@example.com")
    assert target is not None
    observed = []

    class ControlPlane:
        async def erase_customer(self, owner_id: str) -> bool:
            current = auth_service.get_user_info("user@example.com")
            observed.append((owner_id, current))
            assert current is not None
            assert current["disabled"] is True
            return True

    class Runtime:
        settings = type("Settings", (), {"enabled": True})()
        control_plane = ControlPlane()

    monkeypatch.setattr(commercial_runtime, "get_commercial_runtime", lambda: Runtime())
    result = await auth_router.remove_user(
        "user@example.com",
        current=auth_service.TokenPayload(
            username="owner@example.com", role="admin", user_id="u_owner"
        ),
    )

    assert result["ok"] is True
    assert observed[0][0] == target["id"]
    assert auth_service.get_user_info("user@example.com") is None


@pytest.mark.asyncio
async def test_commercial_erasure_failure_leaves_disabled_identity(
    local_auth_env, monkeypatch
) -> None:
    auth_router, auth_service = local_auth_env
    from deeptutor.commercial import runtime as commercial_runtime

    class ControlPlane:
        async def erase_customer(self, _owner_id: str) -> bool:
            raise OSError("postgres unavailable")

    class Runtime:
        settings = type("Settings", (), {"enabled": True})()
        control_plane = ControlPlane()

    monkeypatch.setattr(commercial_runtime, "get_commercial_runtime", lambda: Runtime())
    with pytest.raises(HTTPException) as exc:
        await auth_router.remove_user(
            "user@example.com",
            current=auth_service.TokenPayload(
                username="owner@example.com", role="admin", user_id="u_owner"
            ),
        )

    assert exc.value.status_code == 503
    current = auth_service.get_user_info("user@example.com")
    assert current is not None
    assert current["disabled"] is True


@pytest.mark.asyncio
async def test_admin_created_user_gets_trial_or_pending_marker(local_auth_env, monkeypatch) -> None:
    auth_router, auth_service = local_auth_env
    from deeptutor.commercial import runtime as commercial_runtime

    calls = []

    class Runtime:
        settings = type("Settings", (), {"enabled": True})()

        async def ensure_trial_for_owner(self, owner_id: str, **_kwargs):
            calls.append(owner_id)

    runtime = Runtime()
    monkeypatch.setattr(commercial_runtime, "get_commercial_runtime", lambda: runtime)
    current = auth_service.TokenPayload(
        username="owner@example.com", role="admin", user_id="u_owner"
    )
    created = await auth_router.admin_create_user(
        auth_router.RegisterRequest(username="trial-user", password="password1234"),
        current=current,
    )
    assert created["trial_pending"] is False
    assert calls == [created["user_id"]]

    async def fail_trial(_owner_id: str, **_kwargs):
        raise OSError("postgres unavailable")

    runtime.ensure_trial_for_owner = fail_trial
    pending = await auth_router.admin_create_user(
        auth_router.RegisterRequest(username="pending-user", password="password1234"),
        current=current,
    )
    assert pending["trial_pending"] is True
    assert auth_service.get_user_info("pending-user") is not None
