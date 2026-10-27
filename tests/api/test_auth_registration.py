from __future__ import annotations

from types import SimpleNamespace

from fastapi import BackgroundTasks, HTTPException, Response
import pytest

from deeptutor.api.routers import auth as auth_router
from deeptutor.services import email_verification


def _request(ip: str = "127.0.0.1"):
    return SimpleNamespace(client=SimpleNamespace(host=ip), headers={})


async def _request_code(
    body: auth_router.EmailRegistrationRequest,
    ip: str = "127.0.0.1",
):
    tasks = BackgroundTasks()
    result = await auth_router.request_registration_code(body, _request(ip), tasks)
    await tasks()
    return result


@pytest.fixture
def auth_registration_env(tmp_path, monkeypatch):
    from deeptutor.multi_user import identity
    from deeptutor.services import auth as auth_service

    system_root = tmp_path / "data" / "system"
    monkeypatch.setattr(identity, "AUTH_DIR", system_root / "auth")
    monkeypatch.setattr(identity, "USERS_FILE", system_root / "auth" / "users.json")
    monkeypatch.setattr(identity, "SECRET_FILE", system_root / "auth" / "auth_secret")
    monkeypatch.setattr(auth_router, "AUTH_ENABLED", True)
    monkeypatch.setattr(auth_router, "POCKETBASE_ENABLED", False)
    monkeypatch.setattr(auth_service, "AUTH_SECRET", "test-auth-secret")
    monkeypatch.setattr(
        auth_router,
        "load_auth_settings",
        lambda: {
            "self_registration_enabled": True,
            "email_verification_required": True,
        },
    )
    monkeypatch.setattr(email_verification, "email_delivery_configured", lambda: True)
    sent_challenges = []
    monkeypatch.setattr(
        email_verification,
        "send_verification_email",
        lambda challenge: sent_challenges.append(challenge),
    )
    monkeypatch.setattr(
        email_verification,
        "load_auth_settings",
        lambda: {
            "verification_code_ttl_minutes": 10,
            "verification_resend_cooldown_seconds": 60,
            "verification_max_attempts": 5,
            "verification_max_per_email_hour": 5,
            "verification_max_per_ip_hour": 30,
        },
    )
    return system_root, sent_challenges


@pytest.mark.asyncio
async def test_registration_requires_code_and_then_auto_logs_in(auth_registration_env) -> None:
    system_root, sent_challenges = auth_registration_env
    body = auth_router.EmailRegistrationRequest(email="User@Example.com", password="password1234")
    result = await _request_code(body)
    assert result["ok"] is True

    with pytest.raises(HTTPException) as invalid:
        await auth_router.register(
            auth_router.VerifyRegistrationRequest(email="user@example.com", code="000000"),
            _request(),
            Response(),
        )
    assert invalid.value.status_code == 400

    db = system_root / "auth" / "email_verification.sqlite3"
    import sqlite3

    with sqlite3.connect(db) as connection:
        row = connection.execute(
            "SELECT code_digest, issued_at FROM pending_registrations WHERE email = ?",
            ("user@example.com",),
        ).fetchone()
    assert row is not None

    challenge = sent_challenges[0]
    response = Response()
    registered = await auth_router.register(
        auth_router.VerifyRegistrationRequest(email="user@example.com", code=challenge.code),
        _request(),
        response,
    )
    assert registered["ok"] is True
    # Public registration never bootstraps deployment administration.  An
    # operator must explicitly nominate the bootstrap address instead.
    assert registered["role"] == "user"
    assert registered["is_first_user"] is False
    assert "dt_token" in response.headers.get("set-cookie", "")

    with pytest.raises(HTTPException) as replay:
        await auth_router.register(
            auth_router.VerifyRegistrationRequest(email="user@example.com", code=challenge.code),
            _request(),
            Response(),
        )
    assert replay.value.status_code == 400


@pytest.mark.asyncio
async def test_explicit_bootstrap_admin_email_is_promoted(
    auth_registration_env, monkeypatch
) -> None:
    _, sent_challenges = auth_registration_env
    from deeptutor.services import auth as auth_service

    monkeypatch.setenv("DEEPTUTOR_BOOTSTRAP_ADMIN_EMAIL", "owner@example.com")

    await _request_code(
        auth_router.EmailRegistrationRequest(email="Owner@Example.com", password="password1234"),
    )
    challenge = sent_challenges[0]
    registered = await auth_router.register(
        auth_router.VerifyRegistrationRequest(email="owner@example.com", code=challenge.code),
        _request(),
        Response(),
    )

    assert registered["role"] == "admin"
    assert registered["is_admin"] is True
    assert auth_service.get_user_info("owner@example.com")["role"] == "admin"


@pytest.mark.asyncio
async def test_bootstrap_admin_can_register_after_a_regular_user(
    auth_registration_env, monkeypatch
) -> None:
    _, sent_challenges = auth_registration_env

    await _request_code(
        auth_router.EmailRegistrationRequest(email="first@example.com", password="password1234"),
        "127.0.0.10",
    )
    first = await auth_router.register(
        auth_router.VerifyRegistrationRequest(
            email="first@example.com", code=sent_challenges[-1].code
        ),
        _request(),
        Response(),
    )
    assert first["role"] == "user"

    monkeypatch.setenv("DEEPTUTOR_BOOTSTRAP_ADMIN_EMAIL", "owner@example.com")
    await _request_code(
        auth_router.EmailRegistrationRequest(email="owner@example.com", password="password1234"),
        "127.0.0.11",
    )
    owner = await auth_router.register(
        auth_router.VerifyRegistrationRequest(
            email="owner@example.com", code=sent_challenges[-1].code
        ),
        _request(),
        Response(),
    )

    assert owner["role"] == "admin"
    assert owner["is_admin"] is True


def test_bootstrap_admin_configuration_must_be_an_email(monkeypatch) -> None:
    from deeptutor.services import auth as auth_service

    monkeypatch.setenv("DEEPTUTOR_BOOTSTRAP_ADMIN_EMAIL", "plain-admin")

    assert auth_service.bootstrap_admin_email() == ""


def test_configured_auth_json_admin_is_preserved_when_user_store_starts(
    auth_registration_env, monkeypatch
) -> None:
    from deeptutor.services import auth as auth_service

    monkeypatch.setattr(auth_service, "AUTH_USERNAME", "legacy-owner")
    monkeypatch.setattr(auth_service, "AUTH_PASSWORD_HASH", "configured-hash")

    created, regular = auth_service.add_verified_user("user@example.com", "user-hash", role="user")
    users = {item["username"]: item for item in auth_service.list_users()}

    assert created is True
    assert regular["role"] == "user"
    assert users["legacy-owner"]["role"] == "admin"
    assert users["user@example.com"]["role"] == "user"


@pytest.mark.asyncio
async def test_registration_request_does_not_send_for_existing_account(
    auth_registration_env,
) -> None:
    _, sent_challenges = auth_registration_env
    from deeptutor.services.auth import add_user

    add_user("existing@example.com", "password1234")
    sent = []

    # The route calls asyncio.to_thread with the function object, so replacing
    # it is enough to prove the existing-address branch stays side-effect free.
    import deeptutor.services.email_verification as verification_module

    original = verification_module.send_verification_email
    verification_module.send_verification_email = lambda challenge: sent.append(True)
    try:
        result = await _request_code(
            auth_router.EmailRegistrationRequest(
                email="EXISTING@example.com", password="password1234"
            ),
        )
    finally:
        verification_module.send_verification_email = original

    assert result["ok"] is True
    assert sent == []
    assert sent_challenges == []


@pytest.mark.asyncio
async def test_registration_request_queues_same_background_path_for_existing_and_missing(
    auth_registration_env,
) -> None:
    from deeptutor.services.auth import add_user

    add_user("existing@example.com", "password1234")
    existing_tasks = BackgroundTasks()
    missing_tasks = BackgroundTasks()
    existing = await auth_router.request_registration_code(
        auth_router.EmailRegistrationRequest(email="existing@example.com", password="password1234"),
        _request("127.0.0.2"),
        existing_tasks,
    )
    missing = await auth_router.request_registration_code(
        auth_router.EmailRegistrationRequest(email="missing@example.com", password="password1234"),
        _request("127.0.0.3"),
        missing_tasks,
    )

    assert existing == missing
    assert len(existing_tasks.tasks) == len(missing_tasks.tasks) == 1
    assert existing_tasks.tasks[0].func is missing_tasks.tasks[0].func
    await existing_tasks()
    await missing_tasks()


@pytest.mark.asyncio
async def test_public_registration_survives_deferred_trial_provisioning(
    auth_registration_env, monkeypatch
) -> None:
    _, sent_challenges = auth_registration_env
    from deeptutor.commercial import runtime as commercial_runtime
    from deeptutor.services import auth as auth_service

    class Runtime:
        settings = type("Settings", (), {"enabled": True})()

        async def ensure_trial_for_owner(self, _owner_id: str, **_kwargs):
            raise OSError("postgres unavailable")

    monkeypatch.setattr(commercial_runtime, "get_commercial_runtime", lambda: Runtime())
    await _request_code(
        auth_router.EmailRegistrationRequest(email="pending@example.com", password="password1234")
    )
    registered = await auth_router.register(
        auth_router.VerifyRegistrationRequest(
            email="pending@example.com", code=sent_challenges[0].code
        ),
        _request(),
        Response(),
    )

    assert registered["trial_pending"] is True
    assert auth_service.get_user_info("pending@example.com") is not None
