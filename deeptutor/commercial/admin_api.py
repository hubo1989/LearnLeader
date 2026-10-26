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
from .models import PriceEntryDraft
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


class PriceEntryRequest(BaseModel):
    """One catalog row: micros per single unit of ``meter`` for a tier."""

    meter: str
    model_tier: str = "default"
    unit_price_micros: int = Field(ge=0)
    currency: str


class PublishPriceRequest(BaseModel):
    price_version: str
    kind: Literal["platform_cost", "list_price"]
    effective_from: datetime
    entries: list[PriceEntryRequest] = Field(min_length=1)


class PriceEntryResponse(BaseModel):
    id: str
    price_version: str
    kind: str
    meter: str
    model_tier: str
    unit_price_micros: int
    currency: str
    effective_from: datetime
    created_at: datetime


def _price_response(entry) -> PriceEntryResponse:
    return PriceEntryResponse(
        id=entry.id,
        price_version=entry.price_version,
        kind=entry.kind,
        meter=entry.meter,
        model_tier=entry.model_tier,
        unit_price_micros=entry.unit_price_micros,
        currency=entry.currency,
        effective_from=entry.effective_from,
        created_at=entry.created_at,
    )


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
    offset: int = Query(default=0, ge=0),
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
            offset=offset,
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


@router.post("/admin/commercial/prices", response_model=list[PriceEntryResponse])
async def publish_price_version(
    request: PublishPriceRequest,
) -> list[PriceEntryResponse]:
    """Publish immutable price-catalog rows; identical replays are no-ops."""

    control_plane = _require_control_plane()
    try:
        entries = await control_plane.publish_price_version(
            price_version=request.price_version,
            kind=request.kind,
            effective_from=request.effective_from,
            entries=[
                PriceEntryDraft(
                    meter=item.meter,
                    model_tier=item.model_tier,
                    unit_price_micros=item.unit_price_micros,
                    currency=item.currency,
                )
                for item in request.entries
            ],
        )
    except CommercialError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    log_admin_action(
        "commercial.price.publish",
        summary={
            "price_version": request.price_version,
            "kind": request.kind,
            "effective_from": request.effective_from.isoformat(),
            "rows": len(entries),
        },
    )
    return [_price_response(entry) for entry in entries]


@router.get("/admin/commercial/prices", response_model=list[PriceEntryResponse])
async def list_price_entries(
    kind: str | None = Query(default=None),
) -> list[PriceEntryResponse]:
    """Every published catalog row, newest effective_from first."""

    control_plane = _require_control_plane()
    try:
        entries = await control_plane.list_price_entries(kind=kind)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    return [_price_response(entry) for entry in entries]


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
