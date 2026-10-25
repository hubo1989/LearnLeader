"""Administrator commercial API: plans, orders, and cross-user usage reads.

Order fulfillment is manual by design until a payment gateway exists: an
administrator confirms offline collection through ``mark-paid``, which
atomically lands payment and replaces effective access. Every write is
recorded in the admin audit log.

This module is domain/runtime code and must not import API adapters: the
router ships without an auth dependency, and the API adapter mounts it
behind ``require_admin`` (see ``deeptutor.api.main``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from deeptutor.multi_user.audit import log_admin_action
from deeptutor.multi_user.context import get_current_user

from .api import CommercialOrderResponse, CommercialPlanResponse, _order_response
from .errors import CommercialError, CommercialNotFound
from .runtime import get_commercial_runtime

router = APIRouter()


class PublishPlanRequest(BaseModel):
    plan_code: str
    version: int = Field(ge=1)
    name: str
    price_minor: int = Field(ge=0)
    currency: str
    billing_interval: Literal["month", "year"] = "month"
    entitlements: dict[str, Any]


class MarkPaidRequest(BaseModel):
    payment_reference: str | None = None


def _require_control_plane():
    runtime = get_commercial_runtime()
    if not runtime.settings.enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Commercial mode is not enabled.",
        )
    control_plane = runtime.control_plane
    if control_plane is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Commercial control plane is temporarily unavailable.",
        )
    return control_plane


@router.get("/admin/commercial/orders", response_model=list[CommercialOrderResponse])
async def list_orders(
    status_filter: str | None = Query(default=None, alias="status"),
    owner_id: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[CommercialOrderResponse]:
    """The order queue; filter to ``pending_payment`` for the worklist."""

    control_plane = _require_control_plane()
    if owner_id is not None:
        customer = await control_plane.get_billing_customer_by_owner(owner_id)
        if customer is None:
            return []
        orders = await control_plane.list_orders(customer.id, status=status_filter)
        orders = orders[-limit:]
    else:
        orders = await control_plane.repository.list_all_orders(
            status=status_filter,
            limit=limit,
        )
    return [_order_response(order) for order in orders]


@router.post(
    "/admin/commercial/orders/{order_id}/mark-paid",
    response_model=CommercialOrderResponse,
)
async def mark_order_paid(
    order_id: str,
    request: MarkPaidRequest,
) -> CommercialOrderResponse:
    """Confirm collected payment and activate the ordered subscription."""

    control_plane = _require_control_plane()
    operator = get_current_user().id
    try:
        order, _subscription = await control_plane.pay_and_fulfill_order(
            order_id,
            operator=operator,
            payment_reference=request.payment_reference,
        )
    except CommercialNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except CommercialError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    log_admin_action(
        "commercial.order.mark_paid",
        target_user_id=None,
        summary={
            "order_id": order.id,
            "customer_id": order.customer_id,
            "price_minor": order.price_minor,
            "currency": order.currency,
            "payment_reference": request.payment_reference,
        },
    )
    return _order_response(order)


@router.post("/admin/commercial/plans", response_model=CommercialPlanResponse)
async def publish_plan(
    request: PublishPlanRequest,
) -> CommercialPlanResponse:
    """Publish an immutable plan version; changes require a new version."""

    control_plane = _require_control_plane()
    try:
        plan = await control_plane.publish_plan_version(
            plan_code=request.plan_code,
            version=request.version,
            name=request.name,
            price_minor=request.price_minor,
            currency=request.currency,
            entitlements=request.entitlements,
            billing_interval=request.billing_interval,
        )
    except CommercialError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    log_admin_action(
        "commercial.plan.publish",
        summary={
            "plan_code": plan.plan_code,
            "version": plan.version,
            "price_minor": plan.price_minor,
        },
    )
    return CommercialPlanResponse(
        id=plan.id,
        plan_code=plan.plan_code,
        version=plan.version,
        name=plan.name,
        price_minor=plan.price_minor,
        currency=plan.currency,
        billing_interval=plan.billing_interval,
        trial_days=plan.trial_days,
    )


@router.get("/admin/commercial/usage/{owner_id}")
async def get_owner_usage(
    owner_id: str,
    since: datetime | None = Query(default=None),
    until: datetime | None = Query(default=None),
    meter: str | None = Query(default=None),
) -> dict[str, Any]:
    """Per-user token accounting for support and risk review."""

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
    control_plane = _require_control_plane()
    customer = await control_plane.get_billing_customer_by_owner(owner_id)
    if customer is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No billing customer for owner: {owner_id}",
        )
    return await control_plane.usage_details(
        customer.id,
        since=since,
        until=until,
        meter=meter,
    )


__all__ = ["router"]
