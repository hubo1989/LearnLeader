"""Immutable domain records for DeepTutor's commercial source of truth.

Durable source-of-truth records are customers, immutable plan versions and
entitlements, subscriptions, the webhook inbox, usage events, and usage
reservations. ``ResolvedEntitlements`` and temporal activity are derived read
models; they are always rebuildable from those durable records.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from .keys import canonical_json


class SubscriptionStatus(str, Enum):
    TRIALING = "trialing"
    ACTIVE = "active"
    PAST_DUE = "past_due"
    SUSPENDED = "suspended"
    CANCELED = "canceled"
    EXPIRED = "expired"


class UsageReservationState(str, Enum):
    ACTIVE = "active"
    # Backward-friendly name for callers that think in reserve/finalize terms.
    RESERVED = "active"
    FINALIZED = "finalized"
    RELEASED = "released"


class OrderStatus(str, Enum):
    PENDING_PAYMENT = "pending_payment"
    PAID = "paid"
    FULFILLED = "fulfilled"
    CANCELED = "canceled"
    REFUNDED = "refunded"


class OrderType(str, Enum):
    SUBSCRIPTION_NEW = "subscription_new"
    SUBSCRIPTION_RENEW = "subscription_renew"


ALLOWED_ORDER_TRANSITIONS: Mapping[OrderStatus, frozenset[OrderStatus]] = {
    # ``fulfilled`` is reachable straight from pending_payment: manual
    # mark-paid fulfillment lands both transitions in one transaction.
    OrderStatus.PENDING_PAYMENT: frozenset(
        {OrderStatus.PAID, OrderStatus.FULFILLED, OrderStatus.CANCELED}
    ),
    OrderStatus.PAID: frozenset({OrderStatus.FULFILLED, OrderStatus.REFUNDED}),
    OrderStatus.FULFILLED: frozenset(),
    OrderStatus.CANCELED: frozenset(),
    OrderStatus.REFUNDED: frozenset(),
}


class QuotaWindow(str, Enum):
    UTC_DAY = "utc_day"
    SUBSCRIPTION = "subscription"


class PriceKind(str, Enum):
    """The two price ledgers sharing one catalog table."""

    PLATFORM_COST = "platform_cost"
    LIST_PRICE = "list_price"


PRICE_KINDS: frozenset[str] = frozenset(kind.value for kind in PriceKind)

#: Tier used when a model is absent or missing from ``pricing.model_tiers``.
DEFAULT_MODEL_TIER = "default"

#: Recorded on usage events when no catalog row matches; never a real price.
UNPRICED_PRICE_VERSION = "unpriced-v1"

#: One budget credit equals this many micros of platform cost.
PLATFORM_COST_CREDIT_MICROS = 1_000

#: Server-owned reservation metadata key holding the snapshotted platform
#: cost estimate of that reservation; excluded from replay fingerprints
#: because it is injected after the idempotency comparison, priced at
#: first-attempt time.
ESTIMATED_COST_METADATA_KEY = "estimated_cost_micros"


ALLOWED_SUBSCRIPTION_TRANSITIONS: Mapping[SubscriptionStatus, frozenset[SubscriptionStatus]] = {
    SubscriptionStatus.TRIALING: frozenset(
        {SubscriptionStatus.EXPIRED, SubscriptionStatus.CANCELED}
    ),
    SubscriptionStatus.ACTIVE: frozenset(
        {SubscriptionStatus.PAST_DUE, SubscriptionStatus.SUSPENDED, SubscriptionStatus.CANCELED}
    ),
    SubscriptionStatus.PAST_DUE: frozenset(
        {SubscriptionStatus.ACTIVE, SubscriptionStatus.CANCELED}
    ),
    # Frozen access still owns the live-subscription slot; only admin
    # reinstatement or a terminal cancel may move it.
    SubscriptionStatus.SUSPENDED: frozenset(
        {SubscriptionStatus.ACTIVE, SubscriptionStatus.CANCELED}
    ),
    SubscriptionStatus.CANCELED: frozenset(),
    SubscriptionStatus.EXPIRED: frozenset(),
}


def freeze_json(value: Any) -> Any:
    """Recursively freeze a JSON-like value for safe read-model sharing."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    return value


def reservation_metadata_fingerprint(metadata: Any) -> str:
    """Canonical replay fingerprint that ignores server-owned reserved keys.

    ``estimated_cost_micros`` is injected by the control plane after the
    idempotency comparison and is priced at first-attempt time, so a replay
    that has not re-resolved the catalog yet must still compare equal.
    """

    thawed = dict(thaw_json(metadata) or {})
    thawed.pop(ESTIMATED_COST_METADATA_KEY, None)
    return canonical_json(thawed)


def thaw_json(value: Any) -> Any:
    """Return regular dict/list containers suitable for JSON encoders."""
    if isinstance(value, Mapping):
        return {str(key): thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class BillingCustomer:
    id: str
    owner_id: str
    provider: str
    external_customer_id: str | None
    trial_claimed_at: datetime | None
    trial_subscription_id: str | None
    created_at: datetime
    updated_at: datetime
    #: sha256 hashes of request-side signals captured when the trial was
    #: claimed (``ip_hash``/``ua_hash``); never the raw values.  ``None`` for
    #: customers whose trial predates abuse signal collection.
    trial_claim_context: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "trial_claim_context", freeze_json(self.trial_claim_context))


@dataclass(frozen=True, slots=True)
class PlanVersion:
    id: str
    plan_code: str
    version: int
    name: str
    price_minor: int
    currency: str
    billing_interval: str
    trial_days: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Entitlement:
    id: str
    plan_version_id: str
    key: str
    value: Any
    created_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", freeze_json(self.value))


@dataclass(frozen=True, slots=True)
class Subscription:
    id: str
    customer_id: str
    plan_version_id: str
    status: SubscriptionStatus
    provider: str
    external_subscription_id: str | None
    current_period_start: datetime
    current_period_end: datetime
    trial_started_at: datetime | None
    trial_ends_at: datetime | None
    canceled_at: datetime | None
    version: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class WebhookEvent:
    id: str
    customer_id: str | None
    provider: str
    external_event_id: str
    dedupe_key: str
    event_type: str
    payload: Any
    payload_sha256: str
    occurred_at: datetime | None
    received_at: datetime
    processed_at: datetime | None
    processing_error: str | None
    processing_attempts: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", freeze_json(self.payload))


@dataclass(frozen=True, slots=True)
class PurchaseOrder:
    """A purchase intent with its price and period snapshotted at creation.

    Payment capture is deliberately decoupled: today an administrator moves a
    pending order to paid/fulfilled manually after offline collection; a
    future payment gateway does the same through the webhook inbox. Either
    way, fulfillment replaces effective access through the same atomic
    activate path a verified provider subscription uses.
    """

    id: str
    dedupe_key: str
    customer_id: str
    plan_version_id: str
    order_type: OrderType
    price_minor: int
    currency: str
    status: OrderStatus
    provider: str
    external_order_id: str | None
    subscription_id: str | None
    period_start: datetime
    period_end: datetime
    paid_at: datetime | None
    fulfilled_at: datetime | None
    canceled_at: datetime | None
    refunded_at: datetime | None
    metadata: Any
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", freeze_json(self.metadata))


@dataclass(frozen=True, slots=True)
class UsageReservation:
    id: str
    dedupe_key: str
    customer_id: str
    subscription_id: str
    meter: str
    quantity: int
    state: UsageReservationState
    metadata: Any
    period_start: datetime
    period_end: datetime
    expires_at: datetime
    created_at: datetime
    finalized_at: datetime | None
    released_at: datetime | None
    actual_quantity: int | None
    usage_event_id: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", freeze_json(self.metadata))


@dataclass(frozen=True, slots=True)
class UsageEvent:
    id: str
    dedupe_key: str
    reservation_id: str
    customer_id: str
    subscription_id: str
    meter: str
    quantity: int
    provider: str | None
    model: str | None
    usage_units: Any
    price_version: str
    cost_micros: int
    is_estimated: bool
    metadata: Any
    period_start: datetime
    period_end: datetime
    occurred_at: datetime
    created_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "usage_units", freeze_json(self.usage_units))
        object.__setattr__(self, "metadata", freeze_json(self.metadata))


@dataclass(frozen=True, slots=True)
class PriceEntryDraft:
    """Caller-supplied catalog row; identity fields are filled by the service."""

    meter: str
    model_tier: str
    unit_price_micros: int
    currency: str


@dataclass(frozen=True, slots=True)
class PriceEntry:
    """One immutable catalog row: unit price for a meter and model tier.

    ``unit_price_micros`` is the cost of a single unit of ``meter`` in
    micros of ``currency``.  Resolution picks the row with the latest
    ``effective_from`` at or before the evaluation instant; ties break by
    ``created_at`` then ``price_version`` (descending) so replays of the
    same query stay deterministic.
    """

    id: str
    price_version: str
    kind: str
    meter: str
    model_tier: str
    unit_price_micros: int
    currency: str
    effective_from: datetime
    created_at: datetime


@dataclass(frozen=True, slots=True)
class QuotaPolicy:
    limit: int | None
    window: QuotaWindow


@dataclass(frozen=True, slots=True)
class ResolvedEntitlements:
    """Derived, immutable access snapshot; never a billing source of truth."""

    customer_id: str
    subscription_id: str | None
    plan_version_id: str | None
    status: SubscriptionStatus | None
    active: bool
    resolved_at: datetime
    valid_until: datetime | None
    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", freeze_json(self.values))

    def allows(self, key: str) -> bool:
        return self.active and self.values.get(key) is True

    def quota(self, meter: str) -> QuotaPolicy:
        """Resolve a server-owned quota window; missing or malformed means deny."""
        value = self.values.get(f"quota.{meter}", _MISSING)
        if value is _MISSING:
            raise KeyError(meter)
        if not isinstance(value, Mapping):
            raise ValueError(f"quota.{meter} must define limit and window")
        limit = value.get("limit", _MISSING)
        if limit is not None and (
            limit is _MISSING or isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
        ):
            raise ValueError(f"quota.{meter}.limit must be a non-negative integer or null")
        try:
            window = QuotaWindow(value.get("window"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"quota.{meter}.window is invalid") from exc
        return QuotaPolicy(limit=None if limit is None else limit, window=window)


_MISSING = object()


@dataclass(frozen=True, slots=True)
class TurnLease:
    """One durable owner turn slot with heartbeat-based TTL expiry.

    ``owner_id`` plus the ``capability``/``turn_id`` identity makes each
    concurrent turn its own row, so ``CONCURRENT_TURN_LIMIT`` counts rows
    rather than trusting a single process's memory.  A crashed replica leaves
    its rows behind until ``expires_at`` passes; ``sweep`` removes them.
    """

    id: str
    owner_id: str
    capability: str | None
    turn_id: str | None
    acquired_at: datetime
    heartbeat_at: datetime
    expires_at: datetime


#: A third lifetime trial claimed from the same visitor address is flagged.
TRIAL_ABUSE_THRESHOLD = 3

#: Durable turn-lease time-to-live; expiry frees the slot after a crash.
TURN_LEASE_TTL_SECONDS = 900.0


@dataclass(frozen=True, slots=True)
class ReconciliationFinding:
    code: str
    entity_type: str
    entity_id: str
    detail: str
    repairable: bool


#: Reconcile health-signal thresholds for ``estimated_usage_share_high``:
#: a subscription period only counts once it has at least this many usage
#: events, and flags when the estimated share of events exceeds this ratio.
ESTIMATED_USAGE_SHARE_THRESHOLD = 0.5
ESTIMATED_USAGE_MIN_EVENTS = 5


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    checked_at: datetime
    findings: tuple[ReconciliationFinding, ...]
    repaired: int
