"""Authenticated commercial account read API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from deeptutor.multi_user.context import get_current_user
from deeptutor.services.auth import TokenPayload

from .access import get_commercial_access, require_commercial_access
from .models import thaw_json
from .runtime import get_commercial_runtime

router = APIRouter()


class CommercialMeterUsage(BaseModel):
    """One quota meter's consumption inside its current window."""

    limit: int | None
    window: str
    used: int
    reserved: int


class CommercialMeResponse(BaseModel):
    """Stable, sanitized read model consumed by the subscription UI."""

    enabled: bool
    is_admin: bool
    status: (
        Literal[
            "disabled",
            "admin",
            "trialing",
            "active",
            "past_due",
            "suspended",
            "canceled",
            "expired",
        ]
        | None
    )
    valid_until: datetime | None
    plan_version_id: str | None
    values: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, CommercialMeterUsage] = Field(default_factory=dict)


@router.get("/commercial/me", response_model=CommercialMeResponse)
async def get_my_commercial_access(
    _payload: TokenPayload | None = Depends(require_commercial_access),
):
    """Return the caller's derived plan state without provider credentials."""

    runtime = get_commercial_runtime()
    if not runtime.settings.enabled:
        return {
            "enabled": False,
            "is_admin": False,
            "status": "disabled",
            "valid_until": None,
            "plan_version_id": None,
            "values": {},
        }

    user = get_current_user()
    if user.is_admin:
        return {
            "enabled": True,
            "is_admin": True,
            "status": "admin",
            "valid_until": None,
            "plan_version_id": None,
            "values": {},
        }

    access = get_commercial_access()
    if access is None or access.owner_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Subscription state is temporarily unavailable.",
        )
    resolved = access.resolved
    usage: dict[str, Any] = {}
    if resolved.subscription_id is not None:
        control_plane = getattr(runtime, "control_plane", None)
        if control_plane is not None:
            usage = await control_plane.usage_summary(resolved.customer_id)
    return {
        "enabled": True,
        "is_admin": False,
        "status": resolved.status.value if resolved.status is not None else None,
        "valid_until": resolved.valid_until,
        "plan_version_id": resolved.plan_version_id,
        "values": thaw_json(resolved.values),
        "usage": usage,
    }


@router.get("/commercial/billing/preview")
async def get_my_billing_preview(
    _payload: TokenPayload | None = Depends(require_commercial_access),
) -> dict[str, Any]:
    """Estimate the caller's current-period bill; nothing is committed.

    Fixed charge from the subscription's plan snapshot plus overage
    estimated with the ``list_price`` catalog; catalog misses are flagged
    ``unpriced`` instead of guessed.
    """

    runtime = get_commercial_runtime()
    if not runtime.settings.enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Commercial billing is not enabled.",
        )
    user = get_current_user()
    if user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrators are not metered; nothing to preview.",
        )
    access = get_commercial_access()
    if access is None or access.owner_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Subscription state is temporarily unavailable.",
        )
    control_plane = runtime.control_plane
    if control_plane is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Subscription state is temporarily unavailable.",
        )
    from .errors import CommercialError, CommercialNotFound

    try:
        return await control_plane.billing_preview(access.resolved.customer_id)
    except CommercialNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except CommercialError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.get("/commercial/usage")
async def get_my_commercial_usage(
    _payload: TokenPayload | None = Depends(require_commercial_access),
    since: datetime | None = Query(default=None),
    until: datetime | None = Query(default=None),
    meter: str | None = Query(default=None),
) -> dict[str, Any]:
    """Return the caller's usage events with per-day/per-model rollups."""

    runtime = get_commercial_runtime()
    if not runtime.settings.enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Commercial usage accounting is not enabled.",
        )
    user = get_current_user()
    if user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrators are not metered; nothing to report.",
        )
    access = get_commercial_access()
    if access is None or access.owner_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Subscription state is temporarily unavailable.",
        )
    control_plane = runtime.control_plane
    if control_plane is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Subscription state is temporarily unavailable.",
        )
    if since is not None and since.tzinfo is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="since must carry a timezone offset",
        )
    if until is not None and until.tzinfo is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="until must carry a timezone offset",
        )
    return await control_plane.usage_details(
        access.resolved.customer_id,
        since=since,
        until=until,
        meter=meter,
    )


class CommercialPlanResponse(BaseModel):
    """One published plan version; entitlement values stay server-side."""

    id: str
    plan_code: str
    version: int
    name: str
    price_minor: int
    currency: str
    billing_interval: str
    trial_days: int


class CommercialOrderResponse(BaseModel):
    id: str
    order_type: str
    plan_version_id: str
    price_minor: int
    currency: str
    status: str
    period_start: datetime
    period_end: datetime
    paid_at: datetime | None = None
    fulfilled_at: datetime | None = None
    canceled_at: datetime | None = None
    subscription_id: str | None = None
    created_at: datetime


class CreateOrderRequest(BaseModel):
    plan_version_id: str
    order_type: str = "subscription_new"
    idempotency_key: str


def _order_response(order) -> CommercialOrderResponse:
    return CommercialOrderResponse(
        id=order.id,
        order_type=order.order_type.value,
        plan_version_id=order.plan_version_id,
        price_minor=order.price_minor,
        currency=order.currency,
        status=order.status.value,
        period_start=order.period_start,
        period_end=order.period_end,
        paid_at=order.paid_at,
        fulfilled_at=order.fulfilled_at,
        canceled_at=order.canceled_at,
        subscription_id=order.subscription_id,
        created_at=order.created_at,
    )


@router.get("/commercial/plans", response_model=list[CommercialPlanResponse])
async def list_commercial_plans(
    _payload: TokenPayload | None = Depends(require_commercial_access),
) -> list[CommercialPlanResponse]:
    """Published plan versions with prices; secrets never leave the server."""

    runtime = get_commercial_runtime()
    if not runtime.settings.enabled:
        return []
    control_plane = runtime.control_plane
    if control_plane is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Subscription state is temporarily unavailable.",
        )
    plans = await control_plane.list_plan_versions()
    return [
        CommercialPlanResponse(
            id=plan.id,
            plan_code=plan.plan_code,
            version=plan.version,
            name=plan.name,
            price_minor=plan.price_minor,
            currency=plan.currency,
            billing_interval=plan.billing_interval,
            trial_days=plan.trial_days,
        )
        for plan in plans
    ]


async def _require_customer_control_plane() -> tuple[Any, str]:
    runtime = get_commercial_runtime()
    if not runtime.settings.enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Commercial ordering is not enabled.",
        )
    user = get_current_user()
    if user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrators manage subscriptions directly, not through orders.",
        )
    access = get_commercial_access()
    if access is None or access.owner_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Subscription state is temporarily unavailable.",
        )
    control_plane = runtime.control_plane
    if control_plane is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Subscription state is temporarily unavailable.",
        )
    return control_plane, access.resolved.customer_id


@router.post("/commercial/orders", response_model=CommercialOrderResponse)
async def create_commercial_order(
    request: CreateOrderRequest,
    _payload: TokenPayload | None = Depends(require_commercial_access),
) -> CommercialOrderResponse:
    """Open an order awaiting offline payment; price and period are snapshotted."""

    from .errors import CommercialError, CommercialNotFound
    from .models import OrderType

    control_plane, customer_id = await _require_customer_control_plane()
    try:
        order_type = OrderType(request.order_type)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="order_type must be subscription_new or subscription_renew",
        ) from None
    try:
        order = await control_plane.create_order(
            customer_id,
            request.plan_version_id,
            order_type=order_type,
            idempotency_key=request.idempotency_key,
        )
    except CommercialNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except CommercialError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return _order_response(order)


@router.get("/commercial/orders", response_model=list[CommercialOrderResponse])
async def list_my_commercial_orders(
    _payload: TokenPayload | None = Depends(require_commercial_access),
    status_filter: str | None = Query(default=None, alias="status"),
) -> list[CommercialOrderResponse]:
    control_plane, customer_id = await _require_customer_control_plane()
    orders = await control_plane.list_orders(customer_id, status=status_filter)
    return [_order_response(order) for order in orders]


@router.post("/commercial/orders/{order_id}/cancel", response_model=CommercialOrderResponse)
async def cancel_commercial_order(
    order_id: str,
    _payload: TokenPayload | None = Depends(require_commercial_access),
) -> CommercialOrderResponse:
    from .errors import CommercialError, CommercialNotFound

    control_plane, customer_id = await _require_customer_control_plane()
    try:
        order = await control_plane.get_order(order_id)
        if order.customer_id != customer_id:
            raise CommercialNotFound(f"purchase order not found: {order_id}")
        canceled = await control_plane.cancel_order(order_id)
    except CommercialNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except CommercialError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return _order_response(canceled)


__all__ = ["CommercialMeResponse", "router"]
