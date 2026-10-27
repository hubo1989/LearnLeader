"""Framework-neutral request context and derived commercial grant projection."""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar, Token
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from deeptutor.multi_user.context import get_current_user, get_current_user_or_none

from .errors import ActiveSubscriptionExists, TrialAlreadyClaimed
from .models import ResolvedEntitlements, SubscriptionStatus, thaw_json
from .runtime import CommercialRuntime, get_commercial_runtime


class CommercialAccessDenied(PermissionError):
    """Stable, framework-neutral denial consumed by HTTP and WebSocket gates."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class CommercialAccess:
    """An active entitlement snapshot resolved for one authenticated owner."""

    owner_id: str
    resolved: ResolvedEntitlements

    @property
    def is_current(self) -> bool:
        valid_until = self.resolved.valid_until
        now = datetime.now(timezone.utc)
        return bool(self.resolved.active and valid_until is not None and valid_until > now)


_current_access: ContextVar[CommercialAccess | None] = ContextVar(
    "deeptutor_commercial_access",
    default=None,
)

#: Request-side trial-claim signals (already hashed) for the request in
#: flight.  The HTTP adapter installs it where the visitor address is known;
#: entry points without that context leave it unset and claim nothing.
_trial_claim_context: ContextVar[Mapping[str, Any] | None] = ContextVar(
    "deeptutor_trial_claim_context",
    default=None,
)


def set_trial_claim_context(
    context: Mapping[str, Any] | None,
) -> Token[Mapping[str, Any] | None]:
    return _trial_claim_context.set(context)


def reset_trial_claim_context(token: Token[Mapping[str, Any] | None]) -> None:
    _trial_claim_context.reset(token)


def get_trial_claim_context() -> Mapping[str, Any] | None:
    return _trial_claim_context.get()


def set_commercial_access(
    access: CommercialAccess | None,
) -> Token[CommercialAccess | None]:
    return _current_access.set(access)


def reset_commercial_access(token: Token[CommercialAccess | None]) -> None:
    _current_access.reset(token)


def get_commercial_access() -> CommercialAccess | None:
    return _current_access.get()


async def resolve_commercial_access_for_owner(
    owner_id: str,
    *,
    runtime: CommercialRuntime | None = None,
    claim_context: Mapping[str, Any] | None = None,
) -> CommercialAccess:
    """Ensure the one-time trial and resolve active or historical plan state."""

    current_runtime = runtime or get_commercial_runtime()
    effective_claim = claim_context if claim_context is not None else get_trial_claim_context()
    try:
        await current_runtime.ensure_trial_for_owner(owner_id, claim_context=effective_claim)
    except (ActiveSubscriptionExists, TrialAlreadyClaimed):
        # Paid access or a historical lifetime trial is resolved below; neither
        # condition is permission to mint another trial.
        pass
    resolved = await current_runtime.resolve_for_owner(owner_id)
    if resolved is None:
        now = datetime.now(timezone.utc)
        resolved = ResolvedEntitlements(
            customer_id="",
            subscription_id=None,
            plan_version_id=None,
            status=None,
            active=False,
            resolved_at=now,
            valid_until=None,
            values={},
        )
    if not resolved.active:
        resolved = await _with_historical_status(current_runtime, owner_id, resolved)
    return CommercialAccess(owner_id=owner_id, resolved=resolved)


async def install_commercial_access_for_current_user(
    *,
    runtime: CommercialRuntime | None = None,
) -> CommercialAccess | None:
    """Refresh and install access for HTTP or each inbound WebSocket message.

    Disabled mode and administrators clear the snapshot and return ``None``.
    Inactive users still receive a historical snapshot; only infrastructure
    failures escape to the caller.
    """

    set_commercial_access(None)
    current_runtime = runtime or get_commercial_runtime()
    if not current_runtime.settings.enabled:
        return None
    user = get_current_user()
    if user.is_admin:
        return None
    access = await resolve_commercial_access_for_owner(user.id, runtime=current_runtime)
    set_commercial_access(access)
    return access


async def _with_historical_status(
    runtime: CommercialRuntime,
    owner_id: str,
    resolved: ResolvedEntitlements,
) -> ResolvedEntitlements:
    """Enrich an inactive read model with the newest durable subscription row."""

    control_plane = getattr(runtime, "control_plane", None)
    if control_plane is None:
        return resolved
    customer = await control_plane.get_billing_customer_by_owner(owner_id)
    if customer is None:
        return resolved
    subscriptions = await control_plane.repository.list_subscriptions(customer.id)
    if not subscriptions:
        return resolved
    latest = max(subscriptions, key=lambda item: (item.updated_at, item.created_at, item.id))
    values = {
        item.key: thaw_json(item.value)
        for item in await control_plane.repository.list_entitlements(latest.plan_version_id)
    }
    valid_until = latest.current_period_end
    if latest.trial_ends_at is not None:
        valid_until = min(valid_until, latest.trial_ends_at)
    effective_status = latest.status
    if (
        effective_status in {SubscriptionStatus.TRIALING, SubscriptionStatus.ACTIVE}
        and valid_until <= resolved.resolved_at
    ):
        effective_status = SubscriptionStatus.EXPIRED
    return ResolvedEntitlements(
        customer_id=customer.id,
        subscription_id=latest.id,
        plan_version_id=latest.plan_version_id,
        status=effective_status,
        active=False,
        resolved_at=resolved.resolved_at,
        valid_until=valid_until,
        values=values,
    )


def require_active_commercial_access() -> CommercialAccess | None:
    """Require active access for a consuming operation; local/admin is a no-op."""

    runtime = get_commercial_runtime()
    if not runtime.settings.enabled:
        return None
    user = get_current_user_or_none()
    if user is not None and user.is_admin:
        return None
    access = get_commercial_access()
    if access is not None and access.is_current:
        return access
    status = access.resolved.status if access is not None else None
    if status is SubscriptionStatus.EXPIRED:
        raise CommercialAccessDenied("trial_expired", "The trial has expired.")
    raise CommercialAccessDenied(
        "subscription_required",
        "An active subscription or trial is required.",
    )


def require_capability(name: str) -> CommercialAccess | None:
    """Require ``capability.<name>`` for a consuming capability turn."""

    access = require_active_commercial_access()
    if access is None:
        return None
    key = f"capability.{name.strip()}"
    if access.resolved.values.get(key) is not True:
        raise CommercialAccessDenied(
            "capability_not_in_plan",
            f"Capability is not included in the current plan: {name}",
        )
    return access


def integer_limit(key: str) -> int | None:
    """Return a non-negative ``limits.*`` value; ``None`` means admin/local."""

    access = require_active_commercial_access()
    if access is None:
        return None
    entitlement_key = key if key.startswith("limits.") else f"limits.{key}"
    raw = access.resolved.values.get(entitlement_key)
    try:
        if isinstance(raw, bool):
            raise ValueError("boolean is not an integer limit")
        value = int(raw)
    except (TypeError, ValueError):
        return 0
    return max(0, value)


def project_grant_for_commercial(
    user_id: str,
    base_grant: dict[str, Any],
) -> dict[str, Any]:
    """Overlay the active plan onto a grant without persisting billing state.

    Called only by ``load_grant``. A snapshot that belongs to another user or
    has expired returns a deny-all projection, which prevents background work
    from continuing on a stale request context.
    """

    access = get_commercial_access()
    if access is None:
        runtime = get_commercial_runtime()
        if not runtime.settings.enabled:
            return base_grant
        current_user = get_current_user_or_none()
        if current_user is not None and current_user.is_admin:
            return base_grant
        return _deny_all_projection(deepcopy(base_grant))

    projected = deepcopy(base_grant)
    if access.owner_id != user_id or not access.is_current:
        return _deny_all_projection(projected)

    values = thaw_json(access.resolved.values)
    if not isinstance(values, Mapping):
        return _deny_all_projection(projected)

    model_binding = values.get("models.llm")
    if isinstance(model_binding, Mapping):
        profile_id = str(model_binding.get("profile_id") or "").strip()
        raw_model_ids = model_binding.get("model_ids")
        model_ids = (
            [str(item).strip() for item in raw_model_ids if str(item).strip()]
            if isinstance(raw_model_ids, list)
            else []
        )
    else:
        profile_id = ""
        model_ids = []
    projected["models"] = {
        "llm": (
            [{"profile_id": profile_id, "model_ids": model_ids}] if profile_id and model_ids else []
        )
    }

    projected["platform"] = {
        "embedding": {"enabled": _platform_service_enabled(values, "embedding")},
        "mineru": {"enabled": _platform_service_enabled(values, "mineru")},
    }
    projected["byok"] = {
        service: {"enabled": _byok_service_enabled(values, service)}
        for service in ("llm", "embedding", "mineru")
    }

    allowed_tools = sorted(
        key.removeprefix("tool.")
        for key, value in values.items()
        if key.startswith("tool.") and value is True
    )
    # ``sandbox`` is the commercial name for the code-execution surface.
    if "sandbox" in allowed_tools and "code_execution" not in allowed_tools:
        allowed_tools.append("code_execution")
    projected["enabled_tools"] = sorted(set(allowed_tools))
    projected["exec_enabled"] = bool(
        values.get("tool.exec") is True or values.get("tool.sandbox") is True
    )
    if values.get("feature.mcp") is not True:
        projected["mcp_tools"] = []
    if values.get("feature.partners") is not True:
        projected["partners"] = []

    projected["quota"] = _legacy_quota_projection(values)
    projected["token_quota"] = dict(projected["quota"]["llm"])
    return projected


def _platform_service_enabled(values: Mapping[str, Any], service: str) -> bool:
    if values.get(f"platform.{service}") is True:
        return True
    quota_key = "quota.embedding_tokens" if service == "embedding" else "quota.mineru_pages"
    return isinstance(values.get(quota_key), Mapping)


def _byok_service_enabled(values: Mapping[str, Any], service: str) -> bool:
    return bool(
        values.get("feature.byok") is True
        or values.get(f"feature.byok.{service}") is True
        or values.get(f"byok.{service}") is True
    )


def _quota_limit(values: Mapping[str, Any], key: str) -> tuple[int, str] | None:
    raw = values.get(key)
    if not isinstance(raw, Mapping):
        return None
    try:
        limit = int(raw.get("limit"))
    except (TypeError, ValueError):
        return None
    if limit < 0:
        return None
    window = str(raw.get("window") or "").strip()
    return limit, window


def _legacy_quota_projection(values: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    llm = {"daily_tokens": 0, "monthly_tokens": 0}
    embedding = {"daily_tokens": 0, "monthly_tokens": 0}
    mineru = {"daily_pages": 0, "monthly_pages": 0, "max_pages_per_file": 0}

    for key, target, daily_key, monthly_key in (
        ("quota.llm_tokens", llm, "daily_tokens", "monthly_tokens"),
        (
            "quota.embedding_tokens",
            embedding,
            "daily_tokens",
            "monthly_tokens",
        ),
        ("quota.mineru_pages", mineru, "daily_pages", "monthly_pages"),
    ):
        parsed = _quota_limit(values, key)
        if parsed is None:
            continue
        limit, window = parsed
        if window == "utc_day":
            target[daily_key] = limit
        elif window == "subscription":
            target[monthly_key] = limit

    try:
        max_pages = int(values.get("limits.mineru_max_pages_per_file", 0))
    except (TypeError, ValueError):
        max_pages = 0
    mineru["max_pages_per_file"] = max(0, max_pages)
    return {"llm": llm, "embedding": embedding, "mineru": mineru}


def _deny_all_projection(grant: dict[str, Any]) -> dict[str, Any]:
    grant["models"] = {"llm": []}
    grant["platform"] = {
        "embedding": {"enabled": False},
        "mineru": {"enabled": False},
    }
    grant["byok"] = {service: {"enabled": False} for service in ("llm", "embedding", "mineru")}
    grant["partners"] = []
    grant["enabled_tools"] = []
    grant["mcp_tools"] = []
    grant["exec_enabled"] = False
    grant["quota"] = _legacy_quota_projection({})
    grant["token_quota"] = dict(grant["quota"]["llm"])
    return grant


__all__ = [
    "CommercialAccess",
    "CommercialAccessDenied",
    "get_commercial_access",
    "get_trial_claim_context",
    "install_commercial_access_for_current_user",
    "integer_limit",
    "project_grant_for_commercial",
    "require_active_commercial_access",
    "require_capability",
    "reset_commercial_access",
    "reset_trial_claim_context",
    "resolve_commercial_access_for_owner",
    "set_commercial_access",
    "set_trial_claim_context",
]
