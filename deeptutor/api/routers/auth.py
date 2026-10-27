"""Auth router — login, logout, status, registration, profile, and user-management endpoints."""

import asyncio
from contextvars import Token as _CtxToken
import csv
from datetime import datetime, timedelta, timezone
import io
import ipaddress
import logging
import os
import re
import secrets
import time

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Cookie,
    Depends,
    File,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
    WebSocket,
    status,
)
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field, ValidationError, field_validator
from starlette.background import BackgroundTask

from deeptutor.commercial.runtime import commercial_mode_requested
from deeptutor.commercial.service import DEFAULT_TRIAL_DAYS
from deeptutor.services.config import load_auth_settings, load_system_settings
from deeptutor.services.config.origins import normalize_origin, normalize_origins


def _cookie_samesite(*, secure: bool, commercial: bool) -> str:
    """Choose the legacy cross-site policy or hosted SaaS CSRF boundary."""

    if commercial:
        # Hosted commercial deployments must keep the frontend and API on the
        # same site. Lax prevents ambient cookies on cross-site POSTs while the
        # explicit CORS and WebSocket-Origin checks cover script/socket access.
        return "lax"
    # Legacy/self-hosted secure deployments may intentionally split the web
    # and API origins. SameSite=None preserves that behavior outside SaaS mode.
    return "none" if secure else "lax"


_SECURE = bool(load_auth_settings()["cookie_secure"])
_COMMERCIAL = commercial_mode_requested()
_SAMESITE = _cookie_samesite(
    secure=_SECURE,
    commercial=_COMMERCIAL,
)

from deeptutor.multi_user.audit import log_admin_action, log_usage
from deeptutor.multi_user.context import (
    get_current_user_or_none,
    reset_current_user,
    set_current_user,
    user_from_token_payload,
)
from deeptutor.multi_user.device_credentials import (
    heartbeat_device_credential,
    issue_device_credential,
    list_device_credentials,
    revoke_device_credential,
)
from deeptutor.multi_user.identity import get_user_by_id
from deeptutor.multi_user.learning_access import learning_policy_for_user
from deeptutor.multi_user.models import AccountPreset
from deeptutor.multi_user.paths import local_admin_user
from deeptutor.multi_user.session_handoff import (
    CODE_LIFETIME_SECONDS,
    TICKET_LIFETIME_SECONDS,
    HandoffError,
    HandoffRateLimited,
    HandoffRejected,
    canonical_host,
    decrypt_ticket_payload,
    encrypt_ticket_payload,
    get_session_handoff_store,
    hash_secret,
    is_loopback_host,
    public_origin,
)
from deeptutor.services import (
    account_lifecycle,
    email_verification,
    login_rate_limit,
    password_reset,
)
from deeptutor.services.auth import (
    AUTH_ENABLED,
    POCKETBASE_ENABLED,
    TOKEN_EXPIRE_HOURS,
    TokenPayload,
    add_user,
    add_verified_user,
    authenticate,
    authenticate_device,
    authenticate_pb,
    create_token,
    decode_token,
    delete_user,
    get_user_info,
    hash_password,
    list_users,
    register_pb,
    reset_password,
    set_avatar,
    set_learner_profile,
    set_role,
    validate_password_bytes,
)
from deeptutor.services.auth import (
    get_learner_profile as load_learner_profile,
)
from deeptutor.services.codex_auth.contracts import CodexAuthError
from deeptutor.services.codex_auth.service import deliver_codex_oauth_callback

logger = logging.getLogger(__name__)

router = APIRouter()

_COOKIE_NAME = "dt_token"
AUTH_COOKIE_NAME = _COOKIE_NAME
_COOKIE_MAX_AGE = TOKEN_EXPIRE_HOURS * 3600
_USER_IMPORT_MAX_BYTES = 2 * 1024 * 1024
_USER_IMPORT_MAX_ROWS = 500
_USER_BATCH_MAX_ROWS = 500
_FRONTEND_HOST_HEADER = "x-deeptutor-frontend-host"
# The bundled frontend proxy asserts the visitor address it observed on its
# own socket (web/instrumentation.ts stamps it beyond client reach). A direct
# caller's copy of this header is never trusted: it is honored only from
# AUTH_TRUSTED_FRONTEND_PROXY_IPS peers.
_CLIENT_IP_ASSERTION_HEADER = "x-deeptutor-client-ip"
_AUTH_RUNTIME_SETTINGS = load_auth_settings()
PRIVATE_LOGIN_HOSTS = frozenset(
    str(host).lower().rstrip(".") for host in _AUTH_RUNTIME_SETTINGS.get("private_login_hosts", [])
)


# Only a frontend proxy on an explicitly trusted peer may assert the browser
# host or the client address. Never derive either identity from Host or
# X-Forwarded-* on a direct API call.
# The frontend's incoming Host is meaningful only when its public ingress
# rejects forged private hosts and direct access to its raw listener.
def _trusted_proxy_ips() -> frozenset[str]:
    raw = os.environ.get("AUTH_TRUSTED_FRONTEND_PROXY_IPS", "127.0.0.1,::1")
    trusted: set[str] = set()
    for item in raw.split(","):
        try:
            trusted.add(str(ipaddress.ip_address(item.strip())))
        except ValueError:
            continue
    return frozenset(trusted)


TRUSTED_FRONTEND_PROXY_IPS = _trusted_proxy_ips()


def _trusted_peer(request: Request) -> str:
    """Canonical IP of the connecting peer when it is a trusted proxy."""
    peer = request.client.host if request.client else ""
    try:
        peer_ip = str(ipaddress.ip_address(str(peer)))
    except ValueError:
        return ""
    return peer_ip if peer_ip in TRUSTED_FRONTEND_PROXY_IPS else ""


def _cookie_attrs() -> dict:
    """Attribute set shared by ``login``'s ``set_cookie`` and ``logout``'s
    ``delete_cookie``.

    The deletion ``Set-Cookie`` must carry the same attributes as the one
    that created the cookie — ``delete_cookie`` defaults ``secure=False``,
    which browsers reject when paired with ``SameSite=None``, silently
    keeping the old cookie. See #623. Reads the module globals at call time
    so tests can monkeypatch ``_SECURE``/``_SAMESITE``.
    """
    return {
        "key": _COOKIE_NAME,
        "httponly": True,
        "samesite": _SAMESITE,
        "secure": _SECURE,
    }


def _request_frontend_host(request: Request) -> str:
    peer_ip = _trusted_peer(request)
    if not peer_ip:
        return ""
    raw = request.headers.get(_FRONTEND_HOST_HEADER, "")
    try:
        return canonical_host(raw)
    except HandoffError:
        return ""


def _asserted_client_ip(request: Request) -> str:
    """Visitor address asserted by a trusted frontend proxy, or "".

    The bundled Next proxy copies its own connection-layer stamp
    (``x-deeptutor-connection-ip``, overwritten on the server socket in
    web/instrumentation.ts) into ``_CLIENT_IP_ASSERTION_HEADER``. Trusting
    that assertion is safe only because (a) the peer is an explicitly trusted
    proxy address and (b) the stamped value originates from the proxy's TCP
    peer, so a browser cannot inject it through headers. Anything unparseable
    degrades to "" and the caller falls back to the real peer address.
    """
    if not _trusted_peer(request):
        return ""
    raw = str(request.headers.get(_CLIENT_IP_ASSERTION_HEADER, "") or "").strip()[:128]
    try:
        asserted = ipaddress.ip_address(raw)
    except ValueError:
        return ""
    # A dual-stack proxy may present IPv4 visitors as IPv4-mapped addresses;
    # normalize so both forms share one limiter bucket.
    if isinstance(asserted, ipaddress.IPv6Address) and asserted.ipv4_mapped is not None:
        asserted = asserted.ipv4_mapped
    return str(asserted)


def _require_private_frontend(request: Request) -> None:
    """Restrict credential endpoints only when private hosts are configured."""

    if not PRIVATE_LOGIN_HOSTS:
        return
    host = _request_frontend_host(request)
    if is_loopback_host(host) or host in PRIVATE_LOGIN_HOSTS:
        return
    logger.warning(
        "Credential endpoint refused for non-private frontend host policy",
    )
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Sign-in is only available from a private DeepTutor origin",
    )


def _no_store(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class LoginRequest(BaseModel):
    """Payload for the POST /login endpoint."""

    username: str
    password: str

    @field_validator("username")
    @classmethod
    def username_bounded(cls, v: str) -> str:
        value = v.strip()
        if len(value.encode("utf-8")) > 254:
            raise ValueError("Username must be at most 254 UTF-8 bytes")
        return value

    @field_validator("password")
    @classmethod
    def password_bounded(cls, v: str) -> str:
        # Login intentionally has no minimum: legacy local accounts may use a
        # short password, and auth-disabled localhost mode accepts an empty one.
        return validate_password_bytes(v)


class DeviceLoginRequest(BaseModel):
    """Payload for the built-in device-credential login endpoint."""

    pairing_code: str = Field(min_length=8, max_length=128)
    pin: str = Field(min_length=6, max_length=6)


class DeviceCredentialCreateRequest(BaseModel):
    """Admin payload for issuing a local ordinary-user device credential."""

    user_id: str = Field(min_length=1, max_length=64)
    device_name: str = Field(min_length=1, max_length=80)
    expires_in_days: int = Field(ge=1, le=365)
    daily_limit_minutes: int = Field(ge=5, le=1440)


class RegisterRequest(BaseModel):
    """Payload for the POST /register endpoint."""

    username: str
    password: str
    preset: AccountPreset = "standard"

    @field_validator("username")
    @classmethod
    def username_valid(cls, v: str) -> str:
        import re

        v = v.strip()
        if not v:
            raise ValueError("Email cannot be empty")
        # Accept standard email addresses (used by PocketBase mode) or plain
        # usernames (used by the built-in SQLite/JSON auth mode).
        email_re = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
        plain_re = re.compile(r"^[A-Za-z0-9_\-.]{3,64}$")
        if len(v.encode("utf-8")) > 254 or (not email_re.match(v) and not plain_re.match(v)):
            raise ValueError("Enter a valid email address")
        return v

    @field_validator("password")
    @classmethod
    def password_valid(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return validate_password_bytes(v)


class EmailRegistrationRequest(BaseModel):
    """Start a public registration by requesting a mailbox verification code."""

    email: str
    password: str

    @field_validator("email")
    @classmethod
    def email_valid(cls, v: str) -> str:
        v = email_verification.normalize_email(v)
        if len(v) > 254 or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", v):
            raise ValueError("Enter a valid email address")
        return v

    @field_validator("password")
    @classmethod
    def password_valid(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return validate_password_bytes(v)


class VerifyRegistrationRequest(BaseModel):
    """Complete public registration with the one-time email code."""

    email: str
    code: str

    @field_validator("email")
    @classmethod
    def email_valid(cls, v: str) -> str:
        v = email_verification.normalize_email(v)
        if len(v) > 254 or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", v):
            raise ValueError("Enter a valid email address")
        return v

    @field_validator("code")
    @classmethod
    def code_valid(cls, v: str) -> str:
        v = v.strip()
        if not re.fullmatch(r"\d{6}", v):
            raise ValueError("Verification code must be 6 digits")
        return v


class PasswordResetRequest(BaseModel):
    """Request a one-time password-reset code for a mailbox."""

    email: str

    @field_validator("email")
    @classmethod
    def email_valid(cls, v: str) -> str:
        v = email_verification.normalize_email(v)
        if len(v) > 254 or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", v):
            raise ValueError("Enter a valid email address")
        return v


class ConfirmPasswordResetRequest(PasswordResetRequest):
    """Consume a reset code and choose a replacement password."""

    code: str
    new_password: str

    @field_validator("code")
    @classmethod
    def code_valid(cls, v: str) -> str:
        v = v.strip()
        if not re.fullmatch(r"\d{6}", v):
            raise ValueError("Verification code must be 6 digits")
        return v

    @field_validator("new_password")
    @classmethod
    def password_valid(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return validate_password_bytes(v)


class DeleteOwnAccountRequest(BaseModel):
    """Password confirmation for irreversible self-service deletion."""

    password: str

    @field_validator("password")
    @classmethod
    def password_bounded(cls, v: str) -> str:
        return validate_password_bytes(v)


class SessionHandoffCreateRequest(BaseModel):
    """Private request that starts a one-time public handoff."""

    public_origin: str = Field(min_length=8, max_length=2048)


class SessionHandoffExchangeRequest(BaseModel):
    """Public request that trades a pairing code for a JWE ticket."""

    code: str = Field(min_length=16, max_length=256)


class SessionHandoffCompleteRequest(BaseModel):
    """Public request that trades a JWE ticket for the normal cookie."""

    ticket: str = Field(min_length=32, max_length=8192)


class SetRoleRequest(BaseModel):
    """Payload for the PUT /users/{username}/role endpoint."""

    role: str

    @field_validator("role")
    @classmethod
    def role_valid(cls, v: str) -> str:
        # Validate against the same whitelist the identity store enforces so
        # the API layer and the storage layer share a single source of truth;
        # a hardcoded subset here would 422 roles the store itself accepts.
        from deeptutor.multi_user.models import VALID_ROLES

        if v not in VALID_ROLES:
            raise ValueError(f"Role must be one of {sorted(VALID_ROLES)}")
        return v


class AdminCreateUserRequest(RegisterRequest):
    """Admin user-creation payload.

    A preset configures an ordinary account; it never becomes a third role.
    """

    preset: AccountPreset = "standard"


class AdminBatchDeleteRequest(BaseModel):
    """Usernames for an admin-initiated batch deletion."""

    usernames: list[str] = Field(min_length=1, max_length=_USER_BATCH_MAX_ROWS)

    @field_validator("usernames")
    @classmethod
    def usernames_valid(cls, value: list[str]) -> list[str]:
        usernames: list[str] = []
        for raw_username in value:
            username = raw_username.strip()
            if not username:
                raise ValueError("Usernames cannot be empty")
            if username not in usernames:
                usernames.append(username)
        if not usernames:
            raise ValueError("At least one username is required")
        return usernames


class AuthStatusResponse(BaseModel):
    """Response body for the GET /status endpoint."""

    enabled: bool
    authenticated: bool
    user_id: str | None = None
    username: str | None = None
    role: str | None = None
    is_admin: bool = False
    avatar: str = ""
    commercial_enabled: bool = False
    trial_days: int | None = None
    preset: AccountPreset | None = None
    learning_policy: dict | None = None


class UserInfo(BaseModel):
    """Single user record returned by the GET /users and /profile endpoints."""

    id: str = ""
    username: str
    role: str
    created_at: str
    disabled: bool = False
    avatar: str = ""
    email_verified: bool = True

    preset: AccountPreset = "standard"


class LearnerProfileRequest(BaseModel):
    age: int | None = Field(default=None, ge=3, le=120)
    grade_level: str | None = Field(default=None, max_length=80)
    curriculum: str | None = Field(default=None, max_length=80)
    language: str | None = Field(default=None, max_length=80)
    reading_level: str | None = Field(default=None, max_length=80)
    explanation_style: str | None = Field(default=None, max_length=80)


# Markers settable through PUT /profile. Image markers ("img:<version>") are
# managed exclusively by the upload endpoint so users cannot point their
# avatar at a file that was never validated.
_ICON_MARKER_RE = re.compile(r"^icon:[a-z0-9-]{1,32}:[a-z0-9-]{1,32}$")

# User ids are generated as "u_<uuid hex>" (plus the "local-admin" /
# "env-admin" sentinels); reject anything else before it reaches the
# filesystem layer.
_USER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class UpdateProfileRequest(BaseModel):
    """Payload for the PUT /profile endpoint."""

    avatar: str

    @field_validator("avatar")
    @classmethod
    def avatar_valid(cls, v: str) -> str:
        v = v.strip()
        if v and not _ICON_MARKER_RE.match(v):
            raise ValueError("Avatar must be empty or 'icon:<name>:<color>'")
        return v


# ---------------------------------------------------------------------------
# Shared helper — extract token from cookie or Bearer header
# ---------------------------------------------------------------------------


def _bearer_token_from_header(authorization: str | None) -> str | None:
    """Parse ``Authorization: Bearer <token>`` without using ``HTTPBearer``.

    ``HTTPBearer`` is a class-based dependency whose ``__call__`` is annotated
    ``request: Request``. FastAPI doesn't inject a Request into WebSocket
    dependency resolution, which makes ``HTTPBearer`` raise ``TypeError`` the
    moment a router with this dep mounts a WS endpoint. Doing the parse by
    hand keeps ``require_auth`` HTTP/WS-symmetric.
    """
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        token = parts[1].strip()
        return token or None
    return None


def _extract_token(authorization: str | None, dt_token: str | None) -> str | None:
    return _bearer_token_from_header(authorization) or dt_token


def _client_ip(request: Request) -> str:
    """Rate-limit identity for this request.

    Through the trusted bundled frontend proxy, all visitors would otherwise
    share the proxy's address and one global limiter bucket, so the proxy's
    connection-level visitor assertion is used. Every other caller — including
    direct requests that try to forge the assertion header — is identified by
    its own socket address, which keeps direct-request rate limiting and
    cross-account protection exactly as strict as before.
    """
    return _asserted_client_ip(request) or (request.client.host if request.client else "unknown")


def _check_login_rate_limit(username: str, client_ip: str) -> None:
    try:
        login_rate_limit.check_login_allowed(username, client_ip)
    except login_rate_limit.LoginRateLimited as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts. Please try again later.",
            headers={"Retry-After": str(exc.retry_after)},
        ) from exc
    except login_rate_limit.LoginRateLimitUnavailable as exc:
        logger.error("Persistent login limiter is unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Login is temporarily unavailable.",
        ) from exc


def _record_login_result(
    username: str,
    client_ip: str,
    *,
    success: bool,
) -> None:
    try:
        login_rate_limit.record_login_result(username, client_ip, success=success)
    except login_rate_limit.LoginRateLimitUnavailable as exc:
        logger.error("Could not persist login attempt: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Login is temporarily unavailable.",
        ) from exc


async def _provision_commercial_trial(
    user_id: str,
    role: str,
    request: Request | None = None,
) -> bool:
    """Return whether a created regular account still needs trial repair.

    ``request`` supplies the visitor stamp for the trial's abuse-signal
    context (sha256 hashes only); without it the claim records no context.
    """
    if role == "admin" or not user_id:
        return False
    claim_token = None
    try:
        from deeptutor.commercial.entitlement_context import set_trial_claim_context
        from deeptutor.commercial.trial_safety import build_trial_claim_context

        claim = None
        if request is not None:
            claim = build_trial_claim_context(
                _client_ip(request),
                request.headers.get("user-agent"),
            )
        claim_token = set_trial_claim_context(claim)
        from deeptutor.commercial.runtime import get_commercial_runtime

        runtime = get_commercial_runtime()
        if not runtime.settings.enabled:
            return False
        await runtime.ensure_trial_for_owner(user_id)
        return False
    except Exception:
        # Identity creation is already committed.  Keep login usable and let
        # require_commercial_access's idempotent ensure call reconcile this
        # account when PostgreSQL becomes healthy again.
        logger.exception("Trial provisioning deferred for owner %s", user_id)
        return True
    finally:
        if claim_token is not None:
            from deeptutor.commercial.entitlement_context import reset_trial_claim_context

            reset_trial_claim_context(claim_token)


async def _commercial_export_data(user_id: str) -> dict | None:
    from deeptutor.commercial.errors import CommercialNotFound
    from deeptutor.commercial.runtime import get_commercial_runtime

    runtime = get_commercial_runtime()
    if not runtime.settings.enabled:
        return None
    if runtime.control_plane is None:
        raise RuntimeError("Commercial control plane is not available")
    try:
        return await runtime.control_plane.export_customer_data(user_id)
    except CommercialNotFound:
        # A legacy identity may not have reached reconciliation yet. There is
        # no commercial record to export, so the local export remains valid.
        return None
    except Exception as exc:
        raise RuntimeError("Commercial account export is unavailable") from exc


async def _erase_commercial_customer(user_id: str) -> None:
    from deeptutor.commercial.runtime import get_commercial_runtime

    runtime = get_commercial_runtime()
    if not runtime.settings.enabled:
        return
    if runtime.control_plane is None:
        raise RuntimeError("Commercial control plane is not available")
    await runtime.control_plane.erase_customer(user_id)


async def _delete_account_saga(
    username: str,
    *,
    expected_user_id: str | None,
    actor_id: str,
    actor_username: str,
) -> account_lifecycle.AccountDeletionResult:
    """Quiesce local work and erase billing before identity's commit point."""
    context = await asyncio.to_thread(
        account_lifecycle.begin_account_deletion,
        username,
        actor_id=actor_id,
        actor_username=actor_username,
        expected_user_id=expected_user_id,
    )
    if context.already_deleted:
        return await asyncio.to_thread(account_lifecycle.finish_account_deletion, context)
    try:
        from deeptutor.api.utils.task_id_manager import TaskIDManager
        from deeptutor.services.memory.consolidator.runs import get_run_manager
        from deeptutor.services.session.turn_runtime import cancel_turns_for_user

        await asyncio.gather(
            cancel_turns_for_user(context.user_id),
            TaskIDManager.get_instance().cancel_all_for_owner(context.user_id),
            get_run_manager().cancel_all_for_owner(context.user_id),
        )
        await asyncio.to_thread(
            account_lifecycle.remove_scheduled_account_jobs,
            context.user_id,
        )
        await _erase_commercial_customer(context.user_id)
    except Exception as exc:
        await asyncio.to_thread(
            account_lifecycle.fail_account_deletion,
            context,
            exc,
        )
        raise account_lifecycle.AccountLifecycleError(
            "Account deletion did not complete; the account remains disabled"
        ) from exc
    return await asyncio.to_thread(account_lifecycle.finish_account_deletion, context)


# ---------------------------------------------------------------------------
# Dependencies — reusable auth guards for other routers
# ---------------------------------------------------------------------------


def _install_current_user(payload: TokenPayload | None) -> _CtxToken:
    """Install the request-local current-user ContextVar from an auth result.

    Single point of truth for ``payload → CurrentUser`` so HTTP and WebSocket
    entry points produce identical user objects. ``payload is None`` means
    "no JWT was required" (AUTH_ENABLED=false) and resolves to the local
    admin user; a non-None payload resolves through ``user_from_token_payload``.

    Returns the ContextVar reset token. HTTP callers ignore it (the request
    ends with the task, so the var is GC'd with the task context). WebSocket
    callers keep it and call ``reset_current_user`` in their ``finally`` block,
    because a WS connection outlives the dependency-resolution task.

    ⚠ Invariant: every authenticated entry point MUST call this before the
    handler runs. Skipping it leaves ``get_current_path_service()`` falling
    back to the admin workspace — the silent-routing root cause of #481.
    """
    user = local_admin_user() if payload is None else user_from_token_payload(payload)
    return set_current_user(user)


async def require_auth(
    authorization: str | None = Header(default=None, alias="Authorization"),
    dt_token: str | None = Cookie(default=None, alias=_COOKIE_NAME),
    request: Request = None,
) -> TokenPayload | None:
    """
    FastAPI dependency that enforces authentication when AUTH_ENABLED=true.

    Accepts the JWT from either:
      - Authorization: Bearer <token> header
      - dt_token cookie

    ``Header`` and ``Cookie`` are kept here in place of ``HTTPBearer`` so the
    function stays usable from WebSocket call sites that don't go through
    FastAPI's standard HTTP request lifecycle.

    Returns the authenticated TokenPayload, or None if auth is disabled.
    Raises HTTP 401 if auth is enabled but the token is missing or invalid.

    Declared ``async def`` so the ``set_current_user`` call runs in the same
    asyncio context as the endpoint. A sync dependency is dispatched via
    ``anyio.to_thread.run_sync``, which executes the function in a worker
    thread under a *copy* of the request context; any ``ContextVar.set``
    inside that thread is discarded when the thread returns, leaving the
    endpoint to read the unset default. That regression was the root cause
    of #481.
    """
    if not AUTH_ENABLED:
        _install_current_user(None)
        _install_request_workspace(request)
        return None

    token = _extract_token(authorization, dt_token)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    _install_current_user(payload)
    _install_request_workspace(request)
    # Install the commercial snapshot for every authenticated HTTP request,
    # mirroring ws_revalidate_identity on the WebSocket side. Without it,
    # load_grant's projection fails closed (commercial enabled, no snapshot)
    # and grant-backed surfaces — model options, capability gates, turns —
    # see a deny-all grant even for an active trial.
    try:
        from deeptutor.commercial.entitlement_context import (
            install_commercial_access_for_current_user,
        )

        await install_commercial_access_for_current_user()
    except Exception:
        # Leave the snapshot unset: the fail-closed projection denies costly
        # work, and the error surfaces in logs rather than 500ing every route.
        logger.exception("Commercial access install failed in require_auth")
    return payload


def _install_request_workspace(request) -> None:
    from deeptutor.services.workspace.context import install_workspace_scope
    from deeptutor.services.workspace.models import WorkspaceError

    headers = getattr(request, "headers", {})
    params = getattr(request, "query_params", {})
    header = headers.get("x-deeptutor-workspace")
    query = params.get("dt_workspace")
    if header is not None and query is not None and header != query:
        raise HTTPException(status_code=400, detail="Conflicting workspace scopes.")
    try:
        path = getattr(getattr(request, "url", None), "path", "")
        from deeptutor.services.workspace.knowledge import library_request

        library_request.set(
            path.startswith(("/api/knowledge-bases", "/ws/knowledge-bases"))
            and params.get("resource_library") == "true"
        )
        from deeptutor.services.skill.runtime import library_workspace

        skill_workspace = (
            params.get("skill_workspace", "") if path.startswith("/api/skills") else ""
        )
        library_workspace.set(skill_workspace)
        catalog_management = library_request.get() or path.startswith(
            ("/api/skills", "/api/space/mcp")
        )
        management = path.startswith(
            ("/api/settings", "/api/auth", "/api/multi-user", "/api/task-board")
        )
        selected = install_workspace_scope(
            None if management or catalog_management else header if header is not None else query
        )
        if skill_workspace:
            from deeptutor.services.workspace import get_content_workspace_service

            service = get_content_workspace_service()
            service.validate_chat_binding(
                skill_workspace,
                existing=getattr(request, "method", "GET") in {"GET", "HEAD", "OPTIONS"},
            )
        if (
            not management
            and selected.archived
            and getattr(request, "method", "GET") not in {"GET", "HEAD", "OPTIONS"}
        ):
            raise WorkspaceError("Restore this workspace before changing its data.")
        # Management/migration requests acquire their own exclusive lease.
        task_board_request = path == "/api/task-board" or path.startswith("/api/task-board/")
        if getattr(request, "method", None) is not None and (
            not management or (task_board_request and path != "/api/task-board/events")
        ):
            from deeptutor.services.workspace.activity import acquire_activity

            state = getattr(request, "state", None)
            if state is not None and getattr(state, "workspace_activity", None) is None:
                state.workspace_activity = acquire_activity()
    except WorkspaceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


class _WsAuthFailed:
    """Sentinel: ws_require_auth failed and closed the WebSocket."""


ws_auth_failed: _WsAuthFailed = _WsAuthFailed()


def _configured_websocket_origins() -> set[str]:
    """Return the exact browser origins accepted by authenticated WS routes.

    WebSockets bypass ``CORSMiddleware``, so this intentionally mirrors
    ``api.main._build_cors_settings`` instead of relying on HTTP CORS headers.
    """
    system = load_system_settings()
    frontend_port = str(system["frontend_port"])
    origins = {
        f"http://localhost:{frontend_port}",
        f"http://127.0.0.1:{frontend_port}",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    }
    origins.update(normalize_origins([system["cors_origin"], system["cors_origins"]]))
    return origins


def _websocket_origin_allowed(ws: WebSocket) -> bool:
    """Reject browser cross-site upgrades while allowing non-browser clients.

    Browsers always send ``Origin`` for WebSocket handshakes. CLI clients that
    authenticate with a Bearer token commonly omit it, which remains allowed.
    """
    if not AUTH_ENABLED:
        return True
    origin = str(getattr(ws, "headers", {}).get("origin") or "").strip()
    if not origin:
        return True
    normalized = normalize_origin(origin)
    allowed = _configured_websocket_origins()
    # A wildcard is never safe with authenticated cookies: it would recreate
    # the cross-site WebSocket hijacking path this check is meant to close.
    return bool(normalized and normalized in allowed and normalized != "*")


def _websocket_auth_token(ws: WebSocket) -> str | None:
    query_token = ws.query_params.get("token")
    authorization = getattr(ws, "headers", {}).get("authorization")
    return query_token or _bearer_token_from_header(authorization) or ws.cookies.get(_COOKIE_NAME)


def _websocket_principal_is_current(ws: WebSocket) -> bool:
    """Revalidate the socket JWT against the currently bound principal."""
    if not AUTH_ENABLED:
        return True
    raw_token = _websocket_auth_token(ws)
    expected = get_current_user_or_none()
    if not raw_token or expected is None:
        return False
    try:
        payload = decode_token(raw_token)
        if payload is None:
            return False
        current = user_from_token_payload(payload)
    except Exception:
        logger.warning("WebSocket token revalidation failed", exc_info=True)
        return False
    return (
        current.id == expected.id
        and current.username == expected.username
        and current.role == expected.role
        and current.scope.cache_key == expected.scope.cache_key
    )


async def _close_websocket(ws: WebSocket, code: int) -> None:
    try:
        await ws.close(code=code)
    except Exception:
        logger.debug("Failed to close WebSocket with code %s", code, exc_info=True)


async def _refresh_websocket_commercial_access(ws: WebSocket) -> bool:
    """Install a fresh subscription snapshot in this socket task.

    A WebSocket can outlive both a seven-day Trial boundary and an account
    state change.  Refreshing on the handshake and before every inbound frame
    makes subsequent consuming handlers observe current entitlements while
    still allowing already-running work to finish under its copied context.
    """
    try:
        from deeptutor.commercial.entitlement_context import (
            install_commercial_access_for_current_user,
        )

        await install_commercial_access_for_current_user()
        return True
    except Exception:
        logger.exception("Commercial access refresh failed for WebSocket principal")
        await _close_websocket(ws, 1011)
        return False


async def ws_require_capability_access(ws: WebSocket, capability: str = "llm") -> bool:
    """Fail the WS upgrade when the current user has no usable capability."""
    from deeptutor.multi_user.model_access import has_capability_access

    if has_capability_access(capability):
        return True
    await _close_websocket(ws, 4003)
    return False


async def ws_revalidate_identity(ws: WebSocket) -> bool:
    """Apply account deletion/disable/password reset before an inbound frame."""
    if not _websocket_principal_is_current(ws):
        await _close_websocket(ws, 4001)
        return False
    return await _refresh_websocket_commercial_access(ws)


async def ws_authorize_message(ws: WebSocket, capability: str = "llm") -> bool:
    """Revalidate identity and model access before a consuming WS action."""
    if not await ws_revalidate_identity(ws):
        return False
    return await ws_require_capability_access(ws, capability)


async def ws_require_auth(ws: WebSocket) -> _CtxToken | _WsAuthFailed:
    """Authenticate a WebSocket connection and set the user ContextVar.

    Must be called **before** ``ws.accept()`` so the server can reject
    unauthenticated upgrades cleanly.

    Returns a ContextVar reset token on success, or ``ws_auth_failed``
    on failure (the WebSocket is already closed — the caller should
    ``return`` immediately).

    Usage::

        user_token = await ws_require_auth(ws)
        if user_token is ws_auth_failed:
            return
        await ws.accept()
        try:
            ...
        finally:
            reset_current_user(user_token)
    """
    payload = None
    if AUTH_ENABLED:
        if not _websocket_origin_allowed(ws):
            await _close_websocket(ws, 4003)
            return ws_auth_failed
        token = _websocket_auth_token(ws)
        payload = decode_token(token) if token else None
        if not payload:
            await ws.close(code=4001)
            return ws_auth_failed
    user_token = _install_current_user(payload)
    if await _refresh_websocket_commercial_access(ws):
        return user_token
    try:
        _install_request_workspace(ws)
    except HTTPException:
        reset_current_user(user_token)
        await ws.close(code=4004)
        return ws_auth_failed
    return user_token


async def require_admin(
    payload: TokenPayload | None = Depends(require_auth),
) -> TokenPayload:
    """
    FastAPI dependency that requires the caller to be an admin.

    Raises HTTP 403 if the authenticated user is not an admin.
    When AUTH_ENABLED=false, all requests are treated as admin.

    ``async def`` mirrors ``require_auth`` so the dependency chain stays on
    the event loop and the user ContextVar set by ``require_auth`` is visible
    to the endpoint.
    """
    if not AUTH_ENABLED:
        return _local_admin_token_payload()

    if payload is None or payload.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return payload


_LEARNER_KB_READ_ROUTES = frozenset(
    {
        "/api/knowledge-bases",
        "/api/knowledge-bases/list",
        "/api/knowledge-bases/{kb_name}",
        "/api/knowledge-bases/{kb_name}/files",
        "/api/knowledge-bases/{kb_name}/files/{filename:path}",
        "/api/knowledge-bases/{kb_name}/file-preview-text/{filename:path}",
        "/api/knowledge-bases/{kb_name}/visual-assets/{asset_id}",
        "/api/knowledge-bases/{kb_name}/indexing-run",
        "/api/knowledge-bases/{kb_name}/indexing-readiness",
        "/api/knowledge-bases/{kb_name}/progress",
    }
)


_LEARNER_SETTINGS_WRITE_ROUTES = frozenset(
    {
        # UI preferences resolve through the per-user path service into the
        # caller's OWN workspace (interface.json) — saving theme/language is
        # safe for learning accounts.
        ("PUT", "/api/settings/ui"),
        # Per-user settings draft document. Applying a draft to the runtime
        # catalog stays admin-gated in the endpoint itself.
        ("PUT", "/api/settings/draft"),
        ("DELETE", "/api/settings/draft"),
        # Workspace self-management: the caller's own registrations inside
        # their own scope directory. Data migration/export/root-migration
        # routes under the same prefix stay denied — they move data across
        # scopes and carry no internal admin gate.
        ("PUT", "/api/settings/workspace"),
        ("POST", "/api/settings/workspace/validate"),
        ("POST", "/api/settings/workspace/registrations"),
        ("PATCH", "/api/settings/workspace/registrations/{workspace_id}"),
    }
)


def _learning_surface_for_path(
    path: str, method: str = "GET", *, route_path: str | None = None
) -> str:
    normalized = "/" + str(path or "").lstrip("/")
    for root, surface in (
        ("/api/reading", "reading"),
        ("/api/courses", "reading"),
        ("/api/dashboard/learning-library/materials", "reading"),
        ("/api/dashboard/learning-library/reading", "reading"),
        ("/api/books", "books"),
        ("/api/dashboard/learning-library/books", "books"),
        ("/api/dashboard/learning-library/mastery", "chat"),
        ("/api/chat", "chat"),
        ("/api/question", "chat"),
        ("/api/question-notebook", "chat"),
        ("/api/sessions", "chat"),
        # Task cards are private to the learner's current content workspace.
        ("/api/task-board", "chat"),
        # Mastery Path progress/topics are the learner's own per-user data;
        # the router already scopes every record to the current account, so
        # all methods (including progress PATCH/POST) belong to "chat".
        ("/api/mastery-paths", "chat"),
    ):
        if normalized == root or normalized.startswith(f"{root}/"):
            return surface
    # Match the resolved route template, not just the URL prefix: this keeps
    # admin diagnostics and engine configuration GETs out of the learner shell.
    if (
        method.upper() == "GET"
        and (normalized == "/api/knowledge-bases" or normalized.startswith("/api/knowledge-bases/"))
        and route_path in _LEARNER_KB_READ_ROUTES
    ):
        return "reading"
    # Learner-safe settings/workspace writes, matched on the resolved route
    # template like the KB reads above: the same URL prefixes also host
    # admin-grade operations (catalog apply, data migration), so a plain
    # prefix match would open too much.
    if route_path and (method.upper(), route_path) in _LEARNER_SETTINGS_WRITE_ROUTES:
        return "chat"
    # Choosing the model for a chat turn is part of chat. The handler is
    # already grant-filtered. Match this exact path only: a prefix of
    # /api/settings would also open catalog writes on the same router (#1222).
    if method.upper() == "GET" and normalized == "/api/settings/llm-options":
        return "chat"
    return ""


def _resolved_route_path(request: Request) -> str | None:
    """Keep include-time prefixes across FastAPI's flat and lazy routers."""
    # Lazy router inclusion retains the original route in scope["route"].
    # Its path omits include_router prefixes; the effective context owns the
    # full matched template. Direct routes and older releases use the route.
    fastapi_scope = request.scope.get("fastapi")
    if isinstance(fastapi_scope, dict):
        context = fastapi_scope.get("effective_route_context")
        path = getattr(context, "path", None)
        if isinstance(path, str):
            return path
    path = getattr(request.scope.get("route"), "path", None)
    return path if isinstance(path, str) else None


async def require_learning_surface(
    request: Request,
    _: TokenPayload | None = Depends(require_auth),
) -> None:
    """Second-stage default-deny guard for configured learning accounts."""
    from deeptutor.multi_user.learning_access import (
        assert_learning_surface,
        learning_surface_allowed,
    )

    route_path = _resolved_route_path(request)
    # The landing page combines independent learning surfaces (#1987). Admit
    # any granted surface; the handler filters its readers with the same policy.
    if (
        request.method == "GET"
        and request.url.path == "/api/dashboard/learning-index"
        and route_path == "/api/dashboard/learning-index"
        and any(learning_surface_allowed(surface) for surface in ("books", "chat", "reading"))
    ):
        return

    try:
        assert_learning_surface(
            _learning_surface_for_path(
                request.url.path,
                request.method,
                route_path=route_path,
            )
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc


def _local_admin_token_payload() -> TokenPayload:
    """Synthetic admin payload used when AUTH_ENABLED=false.

    Mirrors the local admin identity (LOCAL_ADMIN_USERNAME / LOCAL_ADMIN_ID)
    so audit logs and self-reference checks behave the same as in multi-user
    mode. Values are kept aligned with ``local_admin_user()`` in
    ``deeptutor/multi_user/paths.py``.
    """
    from deeptutor.multi_user.models import LOCAL_ADMIN_ID, LOCAL_ADMIN_USERNAME

    return TokenPayload(
        username=LOCAL_ADMIN_USERNAME,
        role="admin",
        user_id=LOCAL_ADMIN_ID,
    )


# ---------------------------------------------------------------------------
# Public endpoints (no auth required)
# ---------------------------------------------------------------------------


@router.get("/openai-codex/callback")
async def receive_codex_oauth_callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
) -> HTMLResponse:
    headers = {"Cache-Control": "no-store"}
    try:
        callback_state = state if len(request.query_params.getlist("state")) == 1 else None
        await deliver_codex_oauth_callback(code, callback_state, error)
    except CodexAuthError as exc:
        return HTMLResponse(
            (
                "<!doctype html><title>LearnLeader Codex</title>"
                "<p>Authentication could not be received. Return to LearnLeader and try again.</p>"
            ),
            status_code=exc.http_status,
            headers=headers,
        )
    return HTMLResponse(
        (
            "<!doctype html><title>LearnLeader Codex</title>"
            "<p>Authentication received. You can return to LearnLeader.</p>"
        ),
        headers=headers,
    )


@router.get("/status", response_model=AuthStatusResponse)
async def auth_status(
    authorization: str | None = Header(default=None, alias="Authorization"),
    dt_token: str | None = Cookie(default=None, alias=_COOKIE_NAME),
) -> AuthStatusResponse:
    """Return whether auth is enabled and whether the current request is authenticated."""
    if not AUTH_ENABLED:
        return AuthStatusResponse(
            enabled=False,
            authenticated=True,
            user_id="local-admin",
            username="local",
            role="admin",
            is_admin=True,
            commercial_enabled=False,
            trial_days=None,
            preset="standard",
        )

    token = _extract_token(authorization, dt_token)
    payload = decode_token(token) if token else None
    avatar = ""
    preset: AccountPreset | None = None
    learning_policy = None
    if payload is not None:
        info = get_user_info(payload.username)
        if info:
            avatar = str(info.get("avatar") or "")
            raw_preset = str(info.get("preset") or "standard")
            if raw_preset == "learner":
                preset = "learner"
            elif raw_preset == "custom":
                preset = "custom"
            else:
                preset = "standard"
        learning_policy = learning_policy_for_user(
            payload.user_id,
            is_admin=payload.role == "admin",
        )
        # Guardians may attach a policy to a standard or custom account. That
        # policy, not the stored label, is what the runtime must enforce and
        # what clients need to render the scoped experience.
        if learning_policy is not None and preset != "learner":
            preset = "learner"
    return AuthStatusResponse(
        enabled=True,
        authenticated=payload is not None,
        user_id=payload.user_id if payload else None,
        username=payload.username if payload else None,
        role=payload.role if payload else None,
        is_admin=payload.role == "admin" if payload else False,
        avatar=avatar,
        commercial_enabled=_COMMERCIAL,
        trial_days=DEFAULT_TRIAL_DAYS if _COMMERCIAL else None,
        preset=preset,
        learning_policy=learning_policy,
    )


@router.post("/login")
async def login(body: LoginRequest, request: Request, response: Response) -> dict:
    """Validate credentials and set a JWT cookie."""
    if not AUTH_ENABLED:
        return {"ok": True, "message": "Auth is disabled — no login required."}

    client_ip = _client_ip(request)
    _check_login_rate_limit(body.username, client_ip)
    _require_private_frontend(request)
    _no_store(response)
    if POCKETBASE_ENABLED:
        # PocketBase mode: email = username field for backwards-compat with the
        # existing LoginRequest schema; users can pass their email as "username".
        pb_result = authenticate_pb(body.username, body.password)
        _record_login_result(body.username, client_ip, success=pb_result is not None)
        if not pb_result:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
            )
        payload, pb_token = pb_result
        response.set_cookie(value=pb_token, max_age=_COOKIE_MAX_AGE, **_cookie_attrs())
        logger.info(f"User '{payload.username}' logged in via PocketBase (role={payload.role!r})")
        return {
            "ok": True,
            "user_id": payload.user_id,
            "username": payload.username,
            "role": payload.role,
            "is_admin": payload.role == "admin",
        }

    # Standard JWT + bcrypt mode
    result = authenticate(body.username, body.password)
    _record_login_result(body.username, client_ip, success=result is not None)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

    token = create_token(result.username, result.role, result.user_id)
    response.set_cookie(value=token, max_age=_COOKIE_MAX_AGE, **_cookie_attrs())

    logger.info(f"User '{result.username}' logged in (role={result.role!r})")
    return {
        "ok": True,
        "user_id": result.user_id,
        "username": result.username,
        "role": result.role,
        "is_admin": result.role == "admin",
    }


_PASSWORD_RESET_PUBLIC_RESPONSE = {
    "ok": True,
    "message": "If this address has an account, a password reset code has been sent.",
}


@router.post("/password-reset/request-code", status_code=status.HTTP_202_ACCEPTED)
async def request_password_reset_code(
    body: PasswordResetRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict:
    """Issue a reset code without revealing whether the account exists."""
    if not AUTH_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Auth is disabled — password reset is not available.",
        )
    if POCKETBASE_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Use the configured identity provider to reset this password.",
        )
    try:
        challenge = password_reset.issue_challenge(body.email, _client_ip(request))
        info = get_user_info(body.email)
        deliverable = bool(
            password_reset.email_delivery_configured()
            and info
            and not bool(info.get("disabled", False))
            and bool(info.get("email_verified", True))
        )
        # Always enqueue the same background callable.  The HTTP response is
        # therefore independent of account existence and SMTP latency/failure.
        background_tasks.add_task(
            password_reset.deliver_password_reset_challenge,
            challenge,
            deliverable=deliverable,
        )
    except password_reset.PasswordResetRateLimited as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Please wait before requesting another password reset code.",
            headers={"Retry-After": "60"},
        ) from exc
    except password_reset.PasswordResetError as exc:
        logger.warning("Password reset challenge service unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Password reset is temporarily unavailable.",
        ) from exc
    return dict(_PASSWORD_RESET_PUBLIC_RESPONSE)


@router.post("/password-reset/confirm")
async def confirm_password_reset(body: ConfirmPasswordResetRequest) -> dict:
    """Consume a one-time code, replace the password, and revoke old JWTs."""
    if not AUTH_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Auth is disabled — password reset is not available.",
        )
    if POCKETBASE_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Use the configured identity provider to reset this password.",
        )
    try:
        valid = password_reset.consume_challenge(body.email, body.code)
    except password_reset.PasswordResetError as exc:
        logger.warning("Password reset verification unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Password reset is temporarily unavailable.",
        ) from exc
    if not valid or not reset_password(body.email, body.new_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired password reset code.",
        )
    try:
        login_rate_limit.clear_login_history(body.email)
    except login_rate_limit.LoginRateLimitUnavailable:
        # The password/version update is already committed. Old tokens are
        # revoked even if cleanup of failed-login history needs later repair.
        logger.exception("Could not clear login history after password reset")
    return {"ok": True}


def _require_builtin_device_auth() -> None:
    if not AUTH_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Device credentials require built-in authentication.",
        )
    if POCKETBASE_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Device credentials are not supported in PocketBase mode.",
        )


@router.post("/device-login")
async def device_login(body: DeviceLoginRequest, response: Response) -> dict:
    """Exchange a device pairing code and PIN for the account's normal cookie."""

    _require_builtin_device_auth()
    payload = authenticate_device(body.pairing_code, body.pin)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect device credentials",
        )

    token = create_token(
        payload.username,
        payload.role,
        payload.user_id,
        device_credential_id=payload.device_credential_id,
        device_session_nonce=payload.device_session_nonce,
    )
    response.set_cookie(value=token, max_age=_COOKIE_MAX_AGE, **_cookie_attrs())
    logger.info(f"User '{payload.username}' logged in with a device credential")
    return {
        "ok": True,
        "user_id": payload.user_id,
        "username": payload.username,
        "role": payload.role,
        "is_admin": payload.role == "admin",
        "device_credential_id": payload.device_credential_id,
    }


@router.post("/session-handoff")
async def create_session_handoff(
    body: SessionHandoffCreateRequest,
    request: Request,
    response: Response,
    authorization: str | None = Header(default=None, alias="Authorization"),
    dt_token: str | None = Cookie(default=None, alias=_COOKIE_NAME),
    payload: TokenPayload | None = Depends(require_auth),
) -> dict:
    """Create a host-bound, one-time pairing code from the private origin."""

    _no_store(response)
    if not AUTH_ENABLED or payload is None or not PRIVATE_LOGIN_HOSTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session handoff requires authentication and configured private login hosts.",
        )
    _require_private_frontend(request)

    try:
        target_origin = public_origin(body.public_origin)
        target_host = canonical_host(target_origin.removeprefix("https://"))
        now = int(time.time())
        ticket_claims: dict[str, object] = {
            "nonce": secrets.token_urlsafe(24),
            "host": target_host,
            # Rotated at code exchange; this initial upper bound cannot expire
            # before a valid pairing code is presented.
            "exp": now + CODE_LIFETIME_SECONDS + TICKET_LIFETIME_SECONDS,
        }
        if POCKETBASE_ENABLED:
            ticket_claims.update(
                {"mode": "pocketbase", "token": _extract_token(authorization, dt_token)}
            )
        else:
            ticket_claims.update(
                {
                    "mode": "builtin",
                    "username": payload.username,
                    "role": payload.role,
                    "user_id": payload.user_id,
                    "device_credential_id": payload.device_credential_id,
                    "device_session_nonce": payload.device_session_nonce,
                }
            )
        ticket = encrypt_ticket_payload(ticket_claims)
        record = get_session_handoff_store().create(
            encrypted_ticket=ticket,
            ticket_hash=hash_secret(ticket),
            public_host=target_host,
            rate_key=payload.user_id or payload.username,
        )
    except HandoffRateLimited:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many handoff requests. Try again later.",
        )
    except HandoffError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Enter a valid HTTPS public origin",
        )

    logger.info(
        "Created session handoff for user=%s target_host=%s",
        payload.username,
        target_host,
    )
    return {
        "ok": True,
        "code": record.code,
        "handoff_url": f"{target_origin}/handoff?code={record.code}",
        "expires_at": record.expires_at,
        "expires_in": CODE_LIFETIME_SECONDS,
    }


@router.post("/session-handoff/exchange")
async def exchange_session_handoff(
    body: SessionHandoffExchangeRequest,
    request: Request,
    response: Response,
) -> dict:
    """Consume a pairing code and return a short-lived ticket in the response body."""

    _no_store(response)
    host = _request_frontend_host(request)
    if not host:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Pairing request has no valid frontend host",
        )
    try:
        ticket = get_session_handoff_store().exchange(
            code=body.code,
            public_host=host,
        )
    except HandoffRateLimited:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many handoff requests. Try again later.",
        )
    except HandoffError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Pairing code is invalid, expired, or already used",
        )
    logger.info("Exchanged session handoff code for host=%s", host)
    return {"ok": True, "ticket": ticket}


@router.post("/session-handoff/complete")
async def complete_session_handoff(
    body: SessionHandoffCompleteRequest,
    request: Request,
    response: Response,
) -> dict:
    """Consume a ticket once and issue the normal HttpOnly session cookie."""

    _no_store(response)
    host = _request_frontend_host(request)
    if not host:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Handoff request has no valid frontend host",
        )
    try:
        ticket = get_session_handoff_store().consume_ticket(
            ticket=body.ticket,
            public_host=host,
        )
        claims = decrypt_ticket_payload(ticket)
        if str(claims.get("host")) != host:
            raise HandoffRejected("Invalid handoff ticket")
        try:
            expires_at = int(claims.get("exp"))
        except (TypeError, ValueError):
            raise HandoffRejected("Invalid handoff ticket") from None
        if expires_at <= int(time.time()):
            raise HandoffRejected("Invalid handoff ticket")

        mode = str(claims.get("mode"))
        if mode == "pocketbase":
            token = str(claims.get("token") or "")
            if not token or decode_token(token) is None:
                raise HandoffRejected("Invalid handoff ticket")
        elif mode == "builtin":
            token = create_token(
                str(claims.get("username") or ""),
                str(claims.get("role") or "user"),
                str(claims.get("user_id") or ""),
                device_credential_id=str(claims.get("device_credential_id") or ""),
                device_session_nonce=str(claims.get("device_session_nonce") or ""),
            )
        else:
            raise HandoffRejected("Invalid handoff ticket")
    except HandoffRateLimited:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many handoff requests. Try again later.",
        )
    except HandoffError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Handoff ticket is invalid, expired, or already used",
        )

    response.set_cookie(value=token, max_age=_COOKIE_MAX_AGE, **_cookie_attrs())
    logger.info("Completed session handoff for host=%s", host)
    return {"ok": True}


@router.post("/device/heartbeat")
async def device_heartbeat(
    response: Response,
    payload: TokenPayload | None = Depends(require_auth),
) -> dict:
    """Refresh a device lease and account bounded daily usage."""

    if not AUTH_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Device credentials require built-in authentication.",
        )
    if payload is None or not payload.device_credential_id or not payload.device_session_nonce:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This session does not use a device credential.",
        )
    try:
        device = heartbeat_device_credential(
            payload.device_credential_id,
            user_id=payload.user_id,
            session_nonce=payload.device_session_nonce,
        )
    except ValueError:
        response.delete_cookie(**_cookie_attrs())
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Device session is no longer active",
        ) from None
    return {"ok": not device.pop("limit_reached"), **device}


@router.post("/logout")
async def logout(response: Response) -> dict:
    """Clear the JWT cookie.

    Deletion attributes mirror ``login`` structurally via ``_cookie_attrs()``
    (see the rationale there and #623).
    """
    response.delete_cookie(**_cookie_attrs())
    return {"ok": True}


@router.post("/register/request-code", status_code=status.HTTP_202_ACCEPTED)
async def request_registration_code(
    body: EmailRegistrationRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict:
    """Request a one-time code without revealing whether an account exists."""
    if not AUTH_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Auth is disabled — registration is not available.",
        )
    auth_settings = load_auth_settings()
    if not bool(auth_settings.get("self_registration_enabled", True)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Self-registration is currently unavailable.",
        )
    if not bool(auth_settings.get("email_verification_required", True)):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Email verification is required for public registration.",
        )
    if POCKETBASE_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email registration is not available in PocketBase mode.",
        )
    # Private-host deployments gate credential issuance behind their private
    # origins exactly like /login; a public visitor must not be able to mint
    # registration codes (each one is a delivered email) for the deployment.
    _require_private_frontend(request)
    email = body.email
    client_ip = _client_ip(request)
    try:
        # Do the cheap persistent limiter check before bcrypt, which is
        # intentionally expensive and must not become an unauthenticated DoS
        # primitive.
        email_verification.check_registration_rate_limit(email, client_ip)
        password_hash = hash_password(body.password)
        challenge = email_verification.issue_challenge(email, password_hash, client_ip)
        deliverable = bool(
            email_verification.email_delivery_configured() and get_user_info(email) is None
        )
        # Existing and new accounts take the same bcrypt/ledger path and queue
        # the same background callable. SMTP latency or failure cannot turn
        # the endpoint into an account-existence oracle.
        background_tasks.add_task(
            email_verification.deliver_registration_challenge,
            challenge,
            deliverable=deliverable,
        )
    except (
        email_verification.VerificationRateLimited,
        email_verification.VerificationCooldown,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Please wait before requesting another verification code.",
            headers={"Retry-After": "60"},
        ) from exc
    except email_verification.EmailVerificationError as exc:
        logger.warning("Registration verification service unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Email verification is temporarily unavailable.",
        ) from exc

    return {
        "ok": True,
        "verification_required": True,
        "message": "If this address can register, a verification code has been sent.",
    }


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(body: VerifyRegistrationRequest, request: Request, response: Response) -> dict:
    """Consume a code and create the verified account exactly once."""
    if not AUTH_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Auth is disabled — registration is not available.",
        )
    auth_settings = load_auth_settings()
    if not bool(auth_settings.get("self_registration_enabled", True)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Self-registration is currently unavailable.",
        )
    if not bool(auth_settings.get("email_verification_required", True)):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Email verification is required for public registration.",
        )
    if POCKETBASE_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email registration is not available in PocketBase mode.",
        )
    # Mirror the request-code gate: with private login hosts configured, a
    # stolen one-time code must not complete registration from a public origin.
    _require_private_frontend(request)
    password_hash = email_verification.consume_challenge(body.email, body.code)
    if not password_hash:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification code.",
        )

    created, record = add_verified_user(body.email, password_hash)
    if not created:
        # The challenge is already consumed; never overwrite an existing
        # password when two verification requests race.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This email address is already registered.",
        )

    username = body.email
    role = str(record.get("role") or "user")
    user_id = str(record.get("id") or "")
    trial_pending = await _provision_commercial_trial(user_id, role, request)
    token = create_token(username, role, user_id)
    response.set_cookie(value=token, max_age=_COOKIE_MAX_AGE, **_cookie_attrs())
    logger.info(
        "Verified account created for email domain=%s role=%s",
        username.rsplit("@", 1)[-1],
        role,
    )
    return {
        "ok": True,
        "user_id": user_id,
        "username": username,
        "role": role,
        "is_first_user": role == "admin",
        "is_admin": role == "admin",
        "trial_pending": trial_pending,
    }


@router.get("/is_first_user")
async def check_is_first_user() -> dict:
    """Legacy compatibility endpoint without exposing deployment state."""
    # The registration UI no longer calls this endpoint. Keeping a constant
    # response avoids breaking older clients while preventing public account
    # enumeration of whether an admin has already registered.
    return {"is_first_user": False}


# ---------------------------------------------------------------------------
# Profile endpoints (any authenticated user, self-service)
# ---------------------------------------------------------------------------

_AVATAR_MAX_BYTES = 1 * 1024 * 1024
_AVATAR_MEDIA_TYPES = {"png": "image/png", "jpg": "image/jpeg", "webp": "image/webp"}


def _sniff_image(data: bytes) -> str | None:
    """Detect a supported raster image format from its magic bytes.

    The uploaded filename and Content-Type are attacker-controlled, so the
    stored extension (and the media type served back) is derived from the
    bytes alone. SVG is deliberately unsupported — serving user-supplied SVG
    is a stored-XSS vector.
    """
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def _require_profile_identity(payload: TokenPayload | None) -> TokenPayload:
    """Shared guard for the self-service profile endpoints."""
    if not AUTH_ENABLED or payload is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Auth is disabled — profiles are not available.",
        )
    return payload


@router.get("/profile", response_model=UserInfo)
async def get_profile(
    payload: TokenPayload | None = Depends(require_auth),
) -> UserInfo:
    """Return the current user's own account info."""
    current = _require_profile_identity(payload)
    info = get_user_info(current.username)
    if info is None:
        # PocketBase-backed identities have no local record; fall back to the
        # token claims so the profile page still renders.
        return UserInfo(
            id=current.user_id,
            username=current.username,
            role=current.role,
            created_at="",
        )
    return UserInfo(**info)


@router.get("/profile/export")
async def export_profile_data(
    payload: TokenPayload | None = Depends(require_auth),
) -> FileResponse:
    """Download a privacy-safe ZIP snapshot of the current local account."""
    current = _require_profile_identity(payload)
    if POCKETBASE_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Account export is managed by the configured identity provider.",
        )
    try:
        commercial_data = await _commercial_export_data(current.user_id)
        result = await asyncio.to_thread(
            account_lifecycle.export_account,
            current.username,
            actor_id=current.user_id,
            commercial_data=commercial_data,
        )
    except account_lifecycle.AccountNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found") from exc
    except (account_lifecycle.AccountLifecycleError, RuntimeError) as exc:
        logger.exception("Account export failed for user id %s", current.user_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Account export could not be created.",
        ) from exc
    return FileResponse(
        path=str(result.path),
        media_type="application/zip",
        filename="deeptutor-account-export.zip",
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        background=BackgroundTask(result.path.unlink, missing_ok=True),
    )


@router.delete("/profile")
async def delete_own_account(
    body: DeleteOwnAccountRequest,
    request: Request,
    response: Response,
    payload: TokenPayload | None = Depends(require_auth),
) -> dict:
    """Irreversibly erase the current regular account after password proof."""
    current = _require_profile_identity(payload)
    if POCKETBASE_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Account deletion is managed by the configured identity provider.",
        )
    if current.role == "admin":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Demote the administrator before deleting the account.",
        )

    client_ip = _client_ip(request)
    _check_login_rate_limit(current.username, client_ip)
    confirmed = authenticate(current.username, body.password)
    _record_login_result(current.username, client_ip, success=confirmed is not None)
    if confirmed is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password",
        )
    try:
        result = await _delete_account_saga(
            current.username,
            expected_user_id=current.user_id,
            actor_id=current.user_id,
            actor_username=current.username,
        )
    except account_lifecycle.AccountLifecycleForbidden as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except account_lifecycle.AccountLifecycleError as exc:
        logger.exception("Self-service account deletion failed for user id %s", current.user_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Account deletion could not be completed. The account is disabled.",
        ) from exc
    response.delete_cookie(**_cookie_attrs())
    return {"ok": result.deleted, "already_deleted": result.already_deleted}


@router.put("/profile")
async def update_profile(
    body: UpdateProfileRequest,
    payload: TokenPayload | None = Depends(require_auth),
) -> dict:
    """Update the current user's own avatar marker (icon choice or reset).

    Only the validated ``icon:<name>:<color>`` form (or empty string) is
    accepted here; ``img:`` markers are owned by the upload endpoint.
    """
    current = _require_profile_identity(payload)
    if not set_avatar(current.username, body.avatar):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    # The marker no longer references an uploaded image, so drop the file.
    from deeptutor.multi_user.identity import delete_avatar_file

    if current.user_id and _USER_ID_RE.match(current.user_id):
        delete_avatar_file(current.user_id)
    return {"ok": True, "avatar": body.avatar}


@router.put("/profile/avatar")
async def upload_avatar(
    file: UploadFile = File(...),
    payload: TokenPayload | None = Depends(require_auth),
) -> dict:
    """Upload an avatar image for the current user.

    The client is expected to crop/resize before uploading; the server only
    enforces a size cap and validates the format by magic bytes. Not available
    in PocketBase mode (those identities have no local user record).
    """
    current = _require_profile_identity(payload)
    if POCKETBASE_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Avatar upload is not available in PocketBase mode.",
        )
    if not current.user_id or not _USER_ID_RE.match(current.user_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot store an avatar for this account.",
        )
    info = get_user_info(current.username)
    if info is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    data = await file.read(_AVATAR_MAX_BYTES + 1)
    if len(data) > _AVATAR_MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Avatar image is too large (max 1 MB).",
        )
    ext = _sniff_image(data)
    if ext is None:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Avatar must be a PNG, JPEG or WebP image.",
        )

    from deeptutor.multi_user.identity import save_avatar_file

    # Bump the version embedded in the marker so clients cache-bust the URL.
    previous = str(info.get("avatar") or "")
    version = 1
    if previous.startswith("img:"):
        try:
            version = int(previous.split(":", 1)[1]) + 1
        except ValueError:
            version = 1
    marker = f"img:{version}"

    save_avatar_file(current.user_id, data, ext)
    if not set_avatar(current.username, marker):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    logger.info(f"User '{current.username}' uploaded a new avatar ({ext}, {len(data)} bytes)")
    return {"ok": True, "avatar": marker}


@router.delete("/profile/avatar")
async def remove_avatar(
    payload: TokenPayload | None = Depends(require_auth),
) -> dict:
    """Remove the current user's uploaded avatar image and reset the marker."""
    current = _require_profile_identity(payload)
    from deeptutor.multi_user.identity import delete_avatar_file

    if current.user_id and _USER_ID_RE.match(current.user_id):
        delete_avatar_file(current.user_id)
    set_avatar(current.username, "")
    return {"ok": True, "avatar": ""}


@router.get("/avatar/{user_id}")
async def get_avatar_image(
    user_id: str,
    _: TokenPayload | None = Depends(require_auth),
) -> FileResponse:
    """Serve a stored avatar image. Any authenticated user may view avatars
    (they appear in the admin table and next to the viewer's own profile)."""
    if not _USER_ID_RE.match(user_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Avatar not found")

    from deeptutor.multi_user.identity import get_avatar_file

    target = get_avatar_file(user_id)
    if target is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Avatar not found")

    media_type = _AVATAR_MEDIA_TYPES.get(target.suffix.lstrip("."), "application/octet-stream")
    headers = {
        # Private user content; the marker version in the URL handles busting.
        "Cache-Control": "private, max-age=86400",
        "X-Content-Type-Options": "nosniff",
        "Content-Disposition": "inline",
    }
    return FileResponse(path=str(target), media_type=media_type, headers=headers)


# ---------------------------------------------------------------------------
# Admin-only endpoints
# ---------------------------------------------------------------------------


@router.get("/devices")
async def list_devices(
    user_id: str | None = None,
    include_revoked: bool = False,
    _: TokenPayload = Depends(require_admin),
) -> dict:
    """List local device credential metadata without credential secrets."""

    _require_builtin_device_auth()
    credentials = list_device_credentials(user_id=user_id, include_revoked=include_revoked)
    users = {str(user.get("id") or ""): str(user.get("username") or "") for user in list_users()}
    return {
        "devices": [
            {**device, "username": users.get(device["user_id"], "")} for device in credentials
        ]
    }


@router.post("/devices", status_code=status.HTTP_201_CREATED)
async def issue_device(
    body: DeviceCredentialCreateRequest,
    current: TokenPayload = Depends(require_admin),
) -> dict:
    """Issue a revocable device credential for an ordinary local account."""

    _require_builtin_device_auth()
    if get_user_by_id(body.user_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    try:
        device, pairing_code, pin = issue_device_credential(
            user_id=body.user_id,
            device_name=body.device_name,
            expires_at=datetime.now(timezone.utc) + timedelta(days=body.expires_in_days),
            daily_limit_minutes=body.daily_limit_minutes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    log_admin_action(
        "device_credential_issue",
        target_user_id=body.user_id,
        summary={
            "device_credential_id": device["id"],
            "device_name": device["device_name"],
            "expires_at": device["expires_at"],
            "daily_limit_minutes": device["daily_limit_minutes"],
        },
    )
    logger.info(
        f"Admin '{current.username if current else 'local'}' issued device "
        f"credential {device['id']} for user id '{body.user_id}'"
    )
    return {
        "device": device,
        "pairing_code": pairing_code,
        "pin": pin,
    }


@router.delete("/devices/{device_credential_id}")
async def revoke_device(
    device_credential_id: str,
    current: TokenPayload = Depends(require_admin),
) -> dict:
    _require_builtin_device_auth()
    device = revoke_device_credential(
        device_credential_id,
        revoked_by=str(current.user_id if current else ""),
    )
    if device is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Device credential not found",
        )
    log_admin_action(
        "device_credential_revoke",
        target_user_id=device["user_id"],
        summary={"device_credential_id": device["id"]},
    )
    return {"device": device, "ok": True}


@router.get("/users", response_model=list[UserInfo])
async def get_users(_: TokenPayload = Depends(require_admin)) -> list[UserInfo]:
    """List all registered users. Requires admin role."""
    return [UserInfo(**u) for u in list_users()]


def _require_local_learner(current: TokenPayload) -> tuple[str, dict]:
    """Resolve a self-service profile request to its local learner account."""

    if current.role == "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Learner profile required"
        )
    account = get_user_by_id(current.user_id)
    if account is None or account[0] != current.username:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    if str(account[1].get("preset") or "standard") != "learner":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Learner profile required"
        )
    return account


@router.get("/profile/learner-profile")
async def get_current_learner_profile(current: TokenPayload = Depends(require_auth)) -> dict:
    """Return the authenticated learner's own profile."""
    _require_local_learner(current)
    profile = load_learner_profile(current.username)
    return {"learner_profile": profile}


@router.put("/profile/learner-profile")
async def put_current_learner_profile(
    body: LearnerProfileRequest,
    current: TokenPayload = Depends(require_auth),
) -> dict:
    """Update only the authenticated learner's own profile."""
    _require_local_learner(current)
    from deeptutor.multi_user.learner_profile import normalize_profile

    try:
        profile = normalize_profile(body.model_dump(exclude_none=True))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    updated = set_learner_profile(current.username, profile)
    log_usage(
        "learner_profile",
        current.user_id,
        "self_update",
        {"fields": sorted(profile or {})},
    )
    return {"learner_profile": updated}


@router.get("/users/{username}/learner-profile")
async def get_learner_profile(username: str, _: TokenPayload = Depends(require_admin)) -> dict:
    """Return the structured profile managed for an ordinary learner."""
    from deeptutor.multi_user.identity import get_user

    user = get_user(username)
    if (
        user is None
        or str(user.get("role") or "user") == "admin"
        or str(user.get("preset") or "standard") != "learner"
    ):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return {"learner_profile": user.get("learner_profile")}


@router.put("/users/{username}/learner-profile")
async def put_learner_profile(
    username: str,
    body: LearnerProfileRequest,
    current: TokenPayload = Depends(require_admin),
) -> dict:
    from deeptutor.multi_user.identity import get_user
    from deeptutor.multi_user.learner_profile import normalize_profile

    user = get_user(username)
    if (
        user is None
        or str(user.get("role") or "user") == "admin"
        or str(user.get("preset") or "standard") != "learner"
    ):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    try:
        profile = normalize_profile(body.model_dump(exclude_none=True))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    updated = set_learner_profile(username, profile)
    log_admin_action(
        "learner_profile_update",
        target_user_id=str(user.get("id") or ""),
        summary={"fields": sorted(profile or {})},
    )
    logger.info("Admin '%s' updated learner profile for '%s'", current.username, username)
    return {"learner_profile": updated}


def _validation_message(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error.get("loc", ()))
        parts.append(f"{location or 'row'}: {error.get('msg', 'Invalid value')}")
    return "; ".join(parts)


async def _create_admin_user(
    body: AdminCreateUserRequest,
    current: TokenPayload | None,
) -> dict:
    """Create one ordinary account through the shared admin provision path."""
    actor = current.username if current else "local"

    if POCKETBASE_ENABLED:
        if body.preset != "standard":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only the standard preset is available in PocketBase mode.",
            )
        result = register_pb(username=body.username, email=body.username, password=body.password)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Failed to create user — username may already be taken.",
            )
        logger.info("Admin '%s' created PocketBase user '%s'", actor, body.username)
        user_id = str(result.get("id") or "")
        trial_pending = await _provision_commercial_trial(user_id, "user")
        return {
            "ok": True,
            "user_id": user_id,
            "username": body.username,
            "role": "user",
            "is_admin": False,
            "trial_pending": trial_pending,
            "preset": "standard",
        }

    if any(item.get("username") == body.username for item in list_users()):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already taken")

    add_user(body.username, body.password, preset=body.preset)
    created = next(
        (item for item in list_users() if item.get("username") == body.username),
        None,
    )
    user_id = str(created.get("id") or "") if created else ""
    role = str(created.get("role") or "user") if created else "user"
    preset = str(created.get("preset") or "standard") if created else "standard"

    if role != "user":
        delete_user(body.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Only non-admin users can be provisioned by this endpoint.",
        )

    if preset == "learner":
        from deeptutor.multi_user.grants import learner_grant, save_grant

        try:
            save_grant(user_id, learner_grant(user_id))
        except Exception as exc:
            rolled_back = False
            try:
                rolled_back = delete_user(body.username)
            except Exception:
                logger.exception(
                    "Failed to roll back user '%s' after learner grant initialization failed",
                    body.username,
                )
            if not rolled_back:
                logger.error(
                    "Learner account '%s' may remain after grant initialization failed",
                    body.username,
                )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="The learner preset could not be initialized.",
            ) from exc

    logger.info(
        "Admin '%s' created user '%s' (role=%r, preset=%r)",
        actor,
        body.username,
        role,
        preset,
    )
    trial_pending = await _provision_commercial_trial(user_id, role)
    return {
        "ok": True,
        "user_id": user_id,
        "username": body.username,
        "role": role,
        "is_admin": role == "admin",
        "trial_pending": trial_pending,
        "preset": preset,
    }


@router.post("/users", status_code=status.HTTP_201_CREATED)
async def admin_create_user(
    body: AdminCreateUserRequest,
    current: TokenPayload = Depends(require_admin),
) -> dict:
    """Admin-only: create one ordinary user account."""
    if not AUTH_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Auth is disabled — user creation is not available.",
        )
    return await _create_admin_user(body, current)


@router.post("/users/import")
async def admin_import_users(
    file: UploadFile = File(...),
    current: TokenPayload = Depends(require_admin),
) -> dict:
    """Admin-only: provision ordinary users from a CSV file.

    The response is row-oriented so one invalid record does not erase the
    context of the valid records around it. Passwords are never returned.
    """
    if not AUTH_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Auth is disabled — user import is not available.",
        )

    content = await file.read(_USER_IMPORT_MAX_BYTES + 1)
    await file.close()
    if len(content) > _USER_IMPORT_MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="User import file exceeds 2 MB.",
        )
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User import must be a UTF-8 CSV file.",
        ) from exc

    reader = csv.DictReader(io.StringIO(text, newline=""))
    fieldnames = [str(name or "").strip().lower() for name in reader.fieldnames or []]
    expected = ["username", "password", "preset"]
    if fieldnames != expected:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="CSV header must be exactly: username,password,preset",
        )
    # DictReader keeps the original header spelling as dictionary keys.
    reader.fieldnames = expected

    parsed: list[tuple[int, AdminCreateUserRequest, str]] = []
    results: list[dict] = []
    rows = list(reader)
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="User import file contains no user rows.",
        )
    if len(rows) > _USER_IMPORT_MAX_ROWS:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"User import cannot exceed {_USER_IMPORT_MAX_ROWS} rows.",
        )
    for row_number, row in enumerate(rows, start=2):
        username = str(row.get("username") or "").strip()
        if row.get(None):
            results.append(
                {
                    "row": row_number,
                    "username": username[:100],
                    "ok": False,
                    "error": "Row has more values than the CSV header.",
                }
            )
            continue
        try:
            body = AdminCreateUserRequest(
                username=username,
                password=str(row.get("password") or ""),
                preset=str(row.get("preset") or "").strip() or "standard",
            )
        except ValidationError as exc:
            results.append(
                {
                    "row": row_number,
                    "username": username[:100],
                    "ok": False,
                    "error": _validation_message(exc),
                }
            )
            continue
        parsed.append((row_number, body, username[:100]))

    for row_number, body, username in parsed:
        try:
            created = await _create_admin_user(body, current)
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else "Failed to create user"
            results.append({"row": row_number, "username": username, "ok": False, "error": detail})
        except Exception:
            logger.exception("Admin user import failed for row %s", row_number)
            results.append(
                {
                    "row": row_number,
                    "username": username,
                    "ok": False,
                    "error": "Failed to create user.",
                }
            )
        else:
            results.append({"row": row_number, "username": username, "ok": True, "user": created})

    results.sort(key=lambda result: result["row"])
    created_count = sum(1 for result in results if result["ok"])
    failed_count = len(results) - created_count
    logger.info(
        "Admin '%s' imported users: %s created, %s failed",
        current.username if current else "local",
        created_count,
        failed_count,
    )
    return {
        "ok": failed_count == 0,
        "created_count": created_count,
        "failed_count": failed_count,
        "results": results,
    }


def _delete_admin_user(username: str, current: TokenPayload | None) -> tuple[int, str]:
    if current and username == current.username:
        return 400, "You cannot delete your own account"

    # Capture the id before the record disappears so the avatar file can go too.
    info = get_user_info(username)

    removed = delete_user(username)
    if not removed:
        return 404, "User not found"

    user_id = str(info.get("id") or "") if info else ""
    if user_id and _USER_ID_RE.match(user_id):
        from deeptutor.multi_user.identity import delete_avatar_file

        delete_avatar_file(user_id)

    actor = current.username if current else "local"
    logger.info("Admin '%s' deleted user '%s'", actor, username)
    return 200, ""


@router.delete("/users/{username}", status_code=status.HTTP_200_OK)
async def remove_user(
    username: str,
    current: TokenPayload = Depends(require_admin),
) -> dict:
    """Delete a user. Admins cannot delete their own own account."""
    canonical_username = username.strip().casefold() if "@" in username else username.strip()
    target_info = get_user_info(canonical_username)
    expected_user_id = str(target_info.get("id") or "") if target_info else None
    try:
        result = await _delete_account_saga(
            canonical_username,
            expected_user_id=expected_user_id,
            actor_id=current.user_id,
            actor_username=current.username,
        )
    except account_lifecycle.AccountNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found") from exc
    except account_lifecycle.AccountLifecycleForbidden as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except account_lifecycle.AccountLifecycleError as exc:
        logger.exception("Admin account deletion failed actor=%s", current.user_id or "unknown")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Account deletion could not be completed. The account is disabled.",
        ) from exc

    logger.info(
        "Admin '%s' deleted account (already_deleted=%s)",
        current.username,
        result.already_deleted,
    )
    return {"ok": result.deleted, "already_deleted": result.already_deleted}


@router.post("/users/batch-delete")
async def batch_remove_users(
    body: AdminBatchDeleteRequest,
    current: TokenPayload = Depends(require_admin),
) -> dict:
    """Delete selected users and report each outcome separately."""
    results: list[dict] = []
    for username in body.usernames:
        _status_code, detail = _delete_admin_user(username, current)
        results.append({"username": username, "ok": not detail, "error": detail or None})

    deleted_count = sum(1 for result in results if result["ok"])
    failed_count = len(results) - deleted_count
    logger.info(
        "Admin '%s' batch-deleted users: %s deleted, %s failed",
        current.username if current else "local",
        deleted_count,
        failed_count,
    )
    return {
        "ok": failed_count == 0,
        "deleted_count": deleted_count,
        "failed_count": failed_count,
        "results": results,
    }


@router.put("/users/{username}/role", status_code=status.HTTP_200_OK)
async def update_user_role(
    username: str,
    body: SetRoleRequest,
    current: TokenPayload = Depends(require_admin),
) -> dict:
    """Change a user's role. Admins cannot change their own role."""
    if current and username == current.username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot change your own role",
        )

    updated = set_role(username, body.role)
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    logger.info(
        f"Admin '{current.username if current else 'local'}' set '{username}' role to {body.role!r}"
    )
    return {"ok": True, "username": username, "role": body.role}
