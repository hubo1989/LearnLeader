"""Asyncpg repository for the durable PostgreSQL commercial control plane.

``asyncpg`` is imported only by :meth:`PostgresCommercialRepository.connect`,
so the default non-commercial DeepTutor runtime has no PostgreSQL dependency.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
import json
from pathlib import Path
from typing import Any, AsyncIterator

from .errors import (
    ActiveSubscriptionExists,
    CommercialConflict,
    CommercialNotFound,
    EntitlementDenied,
    IdempotencyConflict,
    ImmutablePlanVersion,
    ImmutablePriceVersion,
    InvalidOrderTransition,
    InvalidSubscriptionTransition,
    MigrationDriftError,
    PendingOrderExists,
    TrialAlreadyClaimed,
    UsageAmountExceeded,
)
from .keys import canonical_json
from .migrations import discover_migrations
from .models import (
    ALLOWED_SUBSCRIPTION_TRANSITIONS,
    ESTIMATED_USAGE_MIN_EVENTS,
    ESTIMATED_USAGE_SHARE_THRESHOLD,
    BillingCustomer,
    Entitlement,
    OrderStatus,
    OrderType,
    PlanVersion,
    PriceEntry,
    PurchaseOrder,
    ReconciliationFinding,
    ReconciliationReport,
    Subscription,
    SubscriptionStatus,
    UsageEvent,
    UsageReservation,
    UsageReservationState,
    WebhookEvent,
    thaw_json,
)


class AsyncpgUnavailable(RuntimeError):
    """Raised only when commercial PostgreSQL is explicitly started without asyncpg."""


_MIGRATION_LOCK_KEY = 1_726_604_552_111_091_031
_MIGRATION_BOOTSTRAP = """
CREATE SCHEMA IF NOT EXISTS commercial;
CREATE TABLE IF NOT EXISTS commercial.schema_migration (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    checksum CHAR(64) NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""


class PostgresCommercialRepository:
    """Production repository with explicit transaction and replay semantics."""

    def __init__(self, pool: Any) -> None:
        self.pool = pool

    @classmethod
    async def connect(
        cls,
        dsn: str,
        *,
        min_size: int = 1,
        max_size: int = 10,
        command_timeout: float = 60.0,
        run_migrations: bool = True,
        migrations_dir: Path | None = None,
    ) -> "PostgresCommercialRepository":
        """Create the asyncpg pool and optionally migrate before accepting traffic."""
        try:
            import asyncpg  # type: ignore[import-not-found]
        except ImportError as exc:
            raise AsyncpgUnavailable(
                "commercial mode requires asyncpg; install the server commercial dependency"
            ) from exc
        pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=min_size,
            max_size=max_size,
            command_timeout=command_timeout,
        )
        repository = cls(pool)
        try:
            if run_migrations:
                await repository.migrate(migrations_dir)
        except BaseException:
            await pool.close()
            raise
        return repository

    async def close(self) -> None:
        await self.pool.close()

    async def migrate(self, migrations_dir: Path | None = None) -> None:
        migrations = discover_migrations(migrations_dir)
        async with self._connection() as connection:
            await connection.execute(_MIGRATION_BOOTSTRAP)
            for migration in migrations:
                async with connection.transaction():
                    await connection.fetchval(
                        "SELECT pg_advisory_xact_lock($1)",
                        _MIGRATION_LOCK_KEY,
                    )
                    existing = await connection.fetchrow(
                        """
                        SELECT version, name, checksum
                        FROM commercial.schema_migration
                        WHERE version = $1
                        """,
                        migration.version,
                    )
                    if existing is not None:
                        if existing["checksum"].strip() != migration.checksum:
                            raise MigrationDriftError(
                                f"commercial migration {migration.version:04d} checksum drift"
                            )
                        continue
                    await connection.execute(migration.sql)
                    await connection.execute(
                        """
                        INSERT INTO commercial.schema_migration (version, name, checksum)
                        VALUES ($1, $2, $3)
                        """,
                        migration.version,
                        migration.name,
                        migration.checksum,
                    )

    @asynccontextmanager
    async def _connection(self) -> AsyncIterator[Any]:
        async with self.pool.acquire() as connection:
            yield connection

    async def ensure_billing_customer(self, candidate: BillingCustomer) -> BillingCustomer:
        async with self._connection() as connection:
            try:
                row = await connection.fetchrow(
                    """
                    INSERT INTO commercial.billing_customer (
                        id, owner_id, provider, external_customer_id,
                        trial_claimed_at, trial_subscription_id, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, NULL, NULL, $5, $5)
                    ON CONFLICT (owner_id) DO NOTHING
                    RETURNING *
                    """,
                    candidate.id,
                    candidate.owner_id,
                    candidate.provider,
                    candidate.external_customer_id,
                    candidate.created_at,
                )
            except Exception as exc:
                raise CommercialConflict("billing customer identity already exists") from exc
            if row is None:
                row = await connection.fetchrow(
                    "SELECT * FROM commercial.billing_customer WHERE owner_id = $1",
                    candidate.owner_id,
                )
            existing = _billing_customer(row)
            if (
                existing.provider != candidate.provider
                or existing.external_customer_id != candidate.external_customer_id
            ):
                raise CommercialConflict(
                    "billing customer owner was reused with different provider identity"
                )
            return existing

    async def get_billing_customer(self, customer_id: str) -> BillingCustomer | None:
        async with self._connection() as connection:
            row = await connection.fetchrow(
                "SELECT * FROM commercial.billing_customer WHERE id = $1",
                customer_id,
            )
        return _billing_customer(row) if row is not None else None

    async def get_billing_customer_by_owner(self, owner_id: str) -> BillingCustomer | None:
        async with self._connection() as connection:
            row = await connection.fetchrow(
                "SELECT * FROM commercial.billing_customer WHERE owner_id = $1",
                owner_id,
            )
        return _billing_customer(row) if row is not None else None

    async def publish_plan_version(
        self,
        candidate: PlanVersion,
        entitlements: tuple[Entitlement, ...],
    ) -> PlanVersion:
        async with self._connection() as connection, connection.transaction():
            row = await connection.fetchrow(
                """
                INSERT INTO commercial.plan_version (
                    id, plan_code, version, name, price_minor, currency,
                    billing_interval, trial_days, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT DO NOTHING
                RETURNING *
                """,
                candidate.id,
                candidate.plan_code,
                candidate.version,
                candidate.name,
                candidate.price_minor,
                candidate.currency,
                candidate.billing_interval,
                candidate.trial_days,
                candidate.created_at,
            )
            inserted = row is not None
            if row is None:
                row = await connection.fetchrow(
                    """
                    SELECT * FROM commercial.plan_version
                    WHERE id = $1 OR (plan_code = $2 AND version = $3)
                    FOR SHARE
                    """,
                    candidate.id,
                    candidate.plan_code,
                    candidate.version,
                )
            if row is None:
                raise CommercialConflict("plan version insert conflicted without a visible row")
            existing = _plan_version(row)
            if inserted:
                await connection.executemany(
                    """
                    INSERT INTO commercial.entitlement (
                        id, plan_version_id, key, value_json, created_at
                    ) VALUES ($1, $2, $3, $4::jsonb, $5)
                    """,
                    [
                        (
                            item.id,
                            candidate.id,
                            item.key,
                            canonical_json(thaw_json(item.value)),
                            item.created_at,
                        )
                        for item in entitlements
                    ],
                )
                return existing
            stored_rows = await connection.fetch(
                """
                SELECT * FROM commercial.entitlement
                WHERE plan_version_id = $1 ORDER BY key
                """,
                existing.id,
            )
            if not _same_plan(existing, candidate) or not _same_entitlements(
                tuple(_entitlement(item) for item in stored_rows),
                entitlements,
            ):
                raise ImmutablePlanVersion(
                    f"plan version {candidate.plan_code}@{candidate.version} is immutable"
                )
            return existing

    async def get_plan_version(self, plan_version_id: str) -> PlanVersion | None:
        async with self._connection() as connection:
            row = await connection.fetchrow(
                "SELECT * FROM commercial.plan_version WHERE id = $1",
                plan_version_id,
            )
        return _plan_version(row) if row is not None else None

    async def list_plan_versions(self) -> tuple[PlanVersion, ...]:
        async with self._connection() as connection:
            rows = await connection.fetch(
                """
                SELECT * FROM commercial.plan_version
                ORDER BY plan_code, version
                """
            )
        return tuple(_plan_version(row) for row in rows)

    async def list_entitlements(self, plan_version_id: str) -> tuple[Entitlement, ...]:
        async with self._connection() as connection:
            rows = await connection.fetch(
                """
                SELECT * FROM commercial.entitlement
                WHERE plan_version_id = $1 ORDER BY key
                """,
                plan_version_id,
            )
        return tuple(_entitlement(row) for row in rows)

    async def publish_price_entries(
        self,
        entries: tuple[PriceEntry, ...],
    ) -> tuple[PriceEntry, ...]:
        async with self._connection() as connection, connection.transaction():
            published: list[PriceEntry] = []
            for candidate in entries:
                try:
                    row = await connection.fetchrow(
                        """
                        INSERT INTO commercial.price_version (
                            id, price_version, kind, meter, model_tier,
                            unit_price_micros, currency, effective_from, created_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        ON CONFLICT (price_version, kind, meter, model_tier) DO NOTHING
                        RETURNING *
                        """,
                        candidate.id,
                        candidate.price_version,
                        candidate.kind,
                        candidate.meter,
                        candidate.model_tier,
                        candidate.unit_price_micros,
                        candidate.currency,
                        candidate.effective_from,
                        candidate.created_at,
                    )
                except Exception as exc:
                    raise CommercialConflict(
                        f"price entry id already exists: {candidate.id}"
                    ) from exc
                if row is not None:
                    published.append(_price_entry(row))
                    continue
                existing_row = await connection.fetchrow(
                    """
                    SELECT * FROM commercial.price_version
                    WHERE price_version = $1 AND kind = $2
                      AND meter = $3 AND model_tier = $4
                    FOR SHARE
                    """,
                    candidate.price_version,
                    candidate.kind,
                    candidate.meter,
                    candidate.model_tier,
                )
                if existing_row is None:
                    raise CommercialConflict(
                        "price catalog insert conflicted without a visible row"
                    )
                existing = _price_entry(existing_row)
                if not _same_price_entry(existing, candidate):
                    raise ImmutablePriceVersion(
                        "price catalog row is immutable; publish a new price_version"
                    )
                published.append(existing)
            return tuple(published)

    async def resolve_price(
        self,
        kind: str,
        meter: str,
        model_tier: str,
        at: datetime,
    ) -> PriceEntry | None:
        async with self._connection() as connection:
            row = await connection.fetchrow(
                """
                SELECT * FROM commercial.price_version
                WHERE kind = $1 AND meter = $2 AND model_tier = $3
                  AND effective_from <= $4
                ORDER BY effective_from DESC, created_at DESC, price_version DESC
                LIMIT 1
                """,
                kind,
                meter,
                model_tier,
                at,
            )
        return _price_entry(row) if row is not None else None

    async def list_price_entries(self) -> tuple[PriceEntry, ...]:
        async with self._connection() as connection:
            rows = await connection.fetch(
                """
                SELECT * FROM commercial.price_version
                ORDER BY effective_from DESC, created_at DESC,
                         price_version DESC, kind, meter, model_tier
                """
            )
        return tuple(_price_entry(row) for row in rows)

    async def ensure_trial(
        self,
        candidate: Subscription,
        claimed_at: datetime,
    ) -> Subscription:
        async with self._connection() as connection, connection.transaction():
            customer_row = await connection.fetchrow(
                """
                SELECT * FROM commercial.billing_customer
                WHERE id = $1 FOR UPDATE
                """,
                candidate.customer_id,
            )
            if customer_row is None:
                raise CommercialNotFound(f"billing customer not found: {candidate.customer_id}")
            customer = _billing_customer(customer_row)
            if customer.trial_subscription_id is not None:
                previous_row = await connection.fetchrow(
                    "SELECT * FROM commercial.subscription WHERE id = $1",
                    customer.trial_subscription_id,
                )
                if previous_row is None:
                    raise CommercialConflict(
                        "billing customer references a missing trial subscription"
                    )
                previous = _subscription(previous_row)
                if previous.plan_version_id != candidate.plan_version_id:
                    raise TrialAlreadyClaimed("the lifetime trial was claimed for another plan")
                return previous
            live = await connection.fetchval(
                """
                SELECT id FROM commercial.subscription
                WHERE customer_id = $1 AND status IN ('trialing', 'active', 'past_due')
                LIMIT 1
                """,
                candidate.customer_id,
            )
            if live is not None:
                raise ActiveSubscriptionExists(
                    "customer already has a trialing or active subscription"
                )
            try:
                await self._insert_subscription(connection, candidate)
                await connection.execute(
                    """
                    UPDATE commercial.billing_customer
                    SET trial_claimed_at = $2, trial_subscription_id = $3, updated_at = $2
                    WHERE id = $1
                    """,
                    candidate.customer_id,
                    claimed_at,
                    candidate.id,
                )
            except Exception as exc:
                _raise_subscription_conflict(exc)
            return candidate

    async def create_subscription(self, candidate: Subscription) -> Subscription:
        async with self._connection() as connection, connection.transaction():
            customer = await connection.fetchval(
                "SELECT id FROM commercial.billing_customer WHERE id = $1 FOR UPDATE",
                candidate.customer_id,
            )
            if customer is None:
                raise CommercialNotFound(f"billing customer not found: {candidate.customer_id}")
            existing_row = None
            if candidate.external_subscription_id:
                existing_row = await connection.fetchrow(
                    """
                    SELECT * FROM commercial.subscription
                    WHERE provider = $1 AND external_subscription_id = $2
                    FOR UPDATE
                    """,
                    candidate.provider,
                    candidate.external_subscription_id,
                )
            if existing_row is None:
                existing_row = await connection.fetchrow(
                    "SELECT * FROM commercial.subscription WHERE id = $1 FOR UPDATE",
                    candidate.id,
                )
            if existing_row is not None:
                existing = _subscription(existing_row)
                if not _same_subscription_identity(existing, candidate):
                    raise IdempotencyConflict(
                        "subscription identity was reused with different semantics"
                    )
                return existing
            try:
                await self._insert_subscription(connection, candidate)
            except Exception as exc:
                _raise_subscription_conflict(exc)
            return candidate

    async def activate_paid_subscription(
        self,
        candidate: Subscription,
        changed_at: datetime,
    ) -> Subscription:
        """Cancel current effective rows and insert paid access in one transaction."""
        async with self._connection() as connection, connection.transaction():
            customer = await connection.fetchval(
                "SELECT id FROM commercial.billing_customer WHERE id = $1 FOR UPDATE",
                candidate.customer_id,
            )
            if customer is None:
                raise CommercialNotFound(f"billing customer not found: {candidate.customer_id}")
            existing_row = await connection.fetchrow(
                """
                SELECT * FROM commercial.subscription
                WHERE provider = $1 AND external_subscription_id = $2
                FOR UPDATE
                """,
                candidate.provider,
                candidate.external_subscription_id,
            )
            if existing_row is not None:
                existing = _subscription(existing_row)
                if not _same_subscription_identity(existing, candidate):
                    raise IdempotencyConflict(
                        "external subscription id was reused with different semantics"
                    )
                return existing
            await connection.execute(
                """
                UPDATE commercial.subscription
                SET status = 'canceled', canceled_at = $2,
                    version = version + 1, updated_at = $2
                WHERE customer_id = $1
                  AND status IN ('trialing', 'active', 'past_due')
                """,
                candidate.customer_id,
                changed_at,
            )
            try:
                await self._insert_subscription(connection, candidate)
            except Exception as exc:
                _raise_subscription_conflict(exc)
            return candidate

    @staticmethod
    async def _insert_subscription(connection: Any, candidate: Subscription) -> None:
        await connection.execute(
            """
            INSERT INTO commercial.subscription (
                id, customer_id, plan_version_id, status, provider,
                external_subscription_id, current_period_start, current_period_end,
                trial_started_at, trial_ends_at, canceled_at, version, created_at, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
            )
            """,
            candidate.id,
            candidate.customer_id,
            candidate.plan_version_id,
            candidate.status.value,
            candidate.provider,
            candidate.external_subscription_id,
            candidate.current_period_start,
            candidate.current_period_end,
            candidate.trial_started_at,
            candidate.trial_ends_at,
            candidate.canceled_at,
            candidate.version,
            candidate.created_at,
            candidate.updated_at,
        )

    async def get_subscription(self, subscription_id: str) -> Subscription | None:
        async with self._connection() as connection:
            row = await connection.fetchrow(
                "SELECT * FROM commercial.subscription WHERE id = $1",
                subscription_id,
            )
        return _subscription(row) if row is not None else None

    async def list_subscriptions(self, customer_id: str) -> tuple[Subscription, ...]:
        async with self._connection() as connection:
            rows = await connection.fetch(
                """
                SELECT * FROM commercial.subscription
                WHERE customer_id = $1 ORDER BY created_at, id
                """,
                customer_id,
            )
        return tuple(_subscription(row) for row in rows)

    async def transition_subscription(
        self,
        subscription_id: str,
        target: SubscriptionStatus,
        changed_at: datetime,
    ) -> Subscription:
        async with self._connection() as connection, connection.transaction():
            row = await connection.fetchrow(
                "SELECT * FROM commercial.subscription WHERE id = $1 FOR UPDATE",
                subscription_id,
            )
            if row is None:
                raise CommercialNotFound(f"subscription not found: {subscription_id}")
            existing = _subscription(row)
            if existing.status is target:
                return existing
            if target not in ALLOWED_SUBSCRIPTION_TRANSITIONS[existing.status]:
                raise InvalidSubscriptionTransition(
                    f"cannot transition subscription from {existing.status.value} to {target.value}"
                )
            try:
                updated = await connection.fetchrow(
                    """
                    UPDATE commercial.subscription
                    SET status = $2,
                        canceled_at = CASE
                            WHEN $2 = 'canceled' THEN $3
                            WHEN $2 = 'active' THEN NULL
                            ELSE canceled_at
                        END,
                        version = version + 1,
                        updated_at = $3
                    WHERE id = $1
                    RETURNING *
                    """,
                    subscription_id,
                    target.value,
                    changed_at,
                )
            except Exception as exc:
                _raise_subscription_conflict(exc)
            return _subscription(updated)

    async def get_entitled_subscription(
        self,
        customer_id: str,
        at: datetime,
    ) -> Subscription | None:
        async with self._connection() as connection:
            rows = await connection.fetch(
                """
                SELECT * FROM commercial.subscription
                WHERE customer_id = $1
                  AND status IN ('trialing', 'active')
                  AND current_period_start <= $2
                  AND current_period_end > $2
                  AND (status <> 'trialing' OR trial_ends_at > $2)
                ORDER BY created_at DESC
                LIMIT 2
                """,
                customer_id,
                at,
            )
        if len(rows) > 1:
            raise CommercialConflict("multiple entitled subscriptions violate the invariant")
        return _subscription(rows[0]) if rows else None

    async def record_webhook_event(self, candidate: WebhookEvent) -> WebhookEvent:
        async with self._connection() as connection:
            row = await connection.fetchrow(
                """
                INSERT INTO commercial.webhook_event (
                    id, customer_id, provider, external_event_id, dedupe_key, event_type,
                    payload_json, payload_sha256, occurred_at, received_at,
                    processed_at, processing_error, processing_attempts
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10, NULL, NULL, 0
                )
                ON CONFLICT (dedupe_key) DO NOTHING
                RETURNING *
                """,
                candidate.id,
                candidate.customer_id,
                candidate.provider,
                candidate.external_event_id,
                candidate.dedupe_key,
                candidate.event_type,
                canonical_json(thaw_json(candidate.payload)),
                candidate.payload_sha256,
                candidate.occurred_at,
                candidate.received_at,
            )
            if row is None:
                row = await connection.fetchrow(
                    "SELECT * FROM commercial.webhook_event WHERE dedupe_key = $1",
                    candidate.dedupe_key,
                )
            existing = _webhook(row)
            if (
                existing.customer_id != candidate.customer_id
                or existing.provider != candidate.provider
                or existing.external_event_id != candidate.external_event_id
                or existing.event_type != candidate.event_type
                or existing.payload_sha256 != candidate.payload_sha256
            ):
                raise IdempotencyConflict("webhook event id was replayed with different content")
            return existing

    async def mark_webhook_processed(
        self,
        event_id: str,
        processed_at: datetime,
        processing_error: str | None = None,
    ) -> WebhookEvent:
        async with self._connection() as connection, connection.transaction():
            row = await connection.fetchrow(
                "SELECT * FROM commercial.webhook_event WHERE id = $1 FOR UPDATE",
                event_id,
            )
            if row is None:
                raise CommercialNotFound(f"webhook event not found: {event_id}")
            existing = _webhook(row)
            if existing.processed_at is not None:
                if processing_error is None:
                    return existing
                raise IdempotencyConflict("processed webhook cannot be marked failed")
            row = await connection.fetchrow(
                """
                UPDATE commercial.webhook_event
                SET processed_at = CASE WHEN $2::text IS NULL THEN $3 ELSE NULL END,
                    processing_error = $2,
                    processing_attempts = processing_attempts + 1
                WHERE id = $1
                RETURNING *
                """,
                event_id,
                processing_error,
                processed_at,
            )
            return _webhook(row)

    async def get_usage_reservation_by_dedupe_key(
        self,
        dedupe_key: str,
    ) -> UsageReservation | None:
        async with self._connection() as connection:
            row = await connection.fetchrow(
                "SELECT * FROM commercial.usage_reservation WHERE dedupe_key = $1",
                dedupe_key,
            )
        return _reservation(row) if row is not None else None

    async def reserve_usage(
        self,
        candidate: UsageReservation,
        quota_limit: int | None,
        *,
        budget_limit_micros: int | None = None,
        estimated_cost_micros: int = 0,
    ) -> UsageReservation:
        async with self._connection() as connection, connection.transaction():
            existing_row = await connection.fetchrow(
                """
                SELECT * FROM commercial.usage_reservation
                WHERE dedupe_key = $1 FOR UPDATE
                """,
                candidate.dedupe_key,
            )
            if existing_row is not None:
                existing = _reservation(existing_row)
                _assert_same_reservation(existing, candidate)
                return existing
            # Locking the customer serializes quota accounting across app instances.
            customer_exists = await connection.fetchval(
                "SELECT id FROM commercial.billing_customer WHERE id = $1 FOR UPDATE",
                candidate.customer_id,
            )
            if customer_exists is None:
                raise CommercialNotFound(f"billing customer not found: {candidate.customer_id}")
            # A concurrent retry may have committed while this transaction waited
            # for the customer lock, so re-check the durable idempotency key.
            existing_row = await connection.fetchrow(
                """
                SELECT * FROM commercial.usage_reservation
                WHERE dedupe_key = $1 FOR UPDATE
                """,
                candidate.dedupe_key,
            )
            if existing_row is not None:
                existing = _reservation(existing_row)
                _assert_same_reservation(existing, candidate)
                return existing
            subscription_row = await connection.fetchrow(
                """
                SELECT * FROM commercial.subscription
                WHERE id = $1 FOR SHARE
                """,
                candidate.subscription_id,
            )
            if subscription_row is None:
                raise CommercialNotFound(f"subscription not found: {candidate.subscription_id}")
            subscription = _subscription(subscription_row)
            if (
                subscription.customer_id != candidate.customer_id
                or subscription.status
                not in {SubscriptionStatus.TRIALING, SubscriptionStatus.ACTIVE}
                or not (
                    subscription.current_period_start
                    <= candidate.created_at
                    < subscription.current_period_end
                )
                or (
                    subscription.status is SubscriptionStatus.TRIALING
                    and (
                        subscription.trial_ends_at is None
                        or candidate.created_at >= subscription.trial_ends_at
                    )
                )
            ):
                raise EntitlementDenied(
                    "subscription became ineligible before usage reservation committed"
                )
            # Reclaim abandoned reservations in the same serialized customer
            # transaction before evaluating quota.  Merely waiting for the
            # startup reconciler would let a long-lived process permanently
            # strand capacity.  Marking them released also prevents a late
            # finalizer from overspending quota after replacement capacity has
            # already been granted.
            await connection.execute(
                """
                UPDATE commercial.usage_reservation
                SET state = 'released', released_at = $6
                WHERE customer_id = $1 AND subscription_id = $2 AND meter = $3
                  AND period_start = $4 AND period_end = $5
                  AND state = 'active' AND expires_at <= $6
                """,
                candidate.customer_id,
                candidate.subscription_id,
                candidate.meter,
                candidate.period_start,
                candidate.period_end,
                candidate.created_at,
            )
            consumed = await connection.fetchval(
                """
                SELECT COALESCE(SUM(quantity), 0)
                FROM commercial.usage_event
                WHERE customer_id = $1 AND subscription_id = $2 AND meter = $3
                  AND period_start = $4 AND period_end = $5
                """,
                candidate.customer_id,
                candidate.subscription_id,
                candidate.meter,
                candidate.period_start,
                candidate.period_end,
            )
            reserved = await connection.fetchval(
                """
                SELECT COALESCE(SUM(quantity), 0)
                FROM commercial.usage_reservation
                WHERE customer_id = $1 AND subscription_id = $2 AND meter = $3
                  AND period_start = $4 AND period_end = $5 AND state = 'active'
                  AND expires_at > $6
                """,
                candidate.customer_id,
                candidate.subscription_id,
                candidate.meter,
                candidate.period_start,
                candidate.period_end,
                candidate.created_at,
            )
            if quota_limit is not None and consumed + reserved + candidate.quantity > quota_limit:
                raise EntitlementDenied(
                    f"quota.{candidate.meter} exceeded: limit={quota_limit}, "
                    f"used={consumed}, reserved={reserved}, requested={candidate.quantity}"
                )
            if budget_limit_micros is not None:
                # Platform-cost budget: finalized events only (their rows
                # already carry attributed cost).  Held but unfinalized
                # reservations have no attributed cost yet; their meter
                # quota already bounds the missing amounts.
                spent = await connection.fetchval(
                    """
                    SELECT COALESCE(SUM(cost_micros), 0)
                    FROM commercial.usage_event
                    WHERE customer_id = $1 AND subscription_id = $2
                    """,
                    candidate.customer_id,
                    candidate.subscription_id,
                )
                if spent + max(0, estimated_cost_micros) > budget_limit_micros:
                    raise EntitlementDenied(
                        "budget.platform_cost exceeded: "
                        f"limit_micros={budget_limit_micros}, "
                        f"spent_micros={spent}, "
                        f"estimated_micros={max(0, estimated_cost_micros)}"
                    )
            try:
                row = await connection.fetchrow(
                    """
                    INSERT INTO commercial.usage_reservation (
                        id, dedupe_key, customer_id, subscription_id, meter, quantity,
                        state, metadata_json, period_start, period_end, expires_at,
                        created_at, finalized_at, released_at, actual_quantity, usage_event_id
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, 'active', $7::jsonb, $8, $9, $10,
                        $11, NULL, NULL, NULL, NULL
                    ) RETURNING *
                    """,
                    candidate.id,
                    candidate.dedupe_key,
                    candidate.customer_id,
                    candidate.subscription_id,
                    candidate.meter,
                    candidate.quantity,
                    canonical_json(thaw_json(candidate.metadata)),
                    candidate.period_start,
                    candidate.period_end,
                    candidate.expires_at,
                    candidate.created_at,
                )
            except Exception as exc:
                raise IdempotencyConflict("usage reservation insert conflicted") from exc
            return _reservation(row)

    async def get_usage_reservation(
        self,
        reservation_id: str,
    ) -> UsageReservation | None:
        async with self._connection() as connection:
            row = await connection.fetchrow(
                "SELECT * FROM commercial.usage_reservation WHERE id = $1",
                reservation_id,
            )
        return _reservation(row) if row is not None else None

    async def list_usage_reservations(
        self,
        customer_id: str,
    ) -> tuple[UsageReservation, ...]:
        async with self._connection() as connection:
            rows = await connection.fetch(
                """
                SELECT * FROM commercial.usage_reservation
                WHERE customer_id = $1 ORDER BY created_at, id
                """,
                customer_id,
            )
        return tuple(_reservation(row) for row in rows)

    async def list_usage_events(
        self,
        customer_id: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        meter: str | None = None,
        limit: int | None = None,
    ) -> tuple[UsageEvent, ...]:
        # Filters are pushed down so a long-lived ledger never fully loads.
        async with self._connection() as connection:
            rows = await connection.fetch(
                """
                SELECT * FROM commercial.usage_event
                WHERE customer_id = $1
                  AND ($2::timestamptz IS NULL OR occurred_at >= $2)
                  AND ($3::timestamptz IS NULL OR occurred_at < $3)
                  AND ($4::text IS NULL OR meter = $4)
                ORDER BY occurred_at DESC, id DESC
                LIMIT $5
                """,
                customer_id,
                since,
                until,
                meter,
                limit if limit is not None else 2**31 - 1,
            )
        return tuple(_usage_event(row) for row in rows)

    async def finalize_usage(
        self,
        reservation_id: str,
        actual_quantity: int,
        event_candidate: UsageEvent,
        finalized_at: datetime,
    ) -> UsageEvent:
        async with self._connection() as connection, connection.transaction():
            reservation_row = await connection.fetchrow(
                """
                SELECT * FROM commercial.usage_reservation
                WHERE id = $1 FOR UPDATE
                """,
                reservation_id,
            )
            if reservation_row is None:
                raise CommercialNotFound(f"usage reservation not found: {reservation_id}")
            reservation = _reservation(reservation_row)
            if reservation.state is UsageReservationState.FINALIZED:
                if reservation.actual_quantity != actual_quantity:
                    raise IdempotencyConflict("finalize retry changed the actual usage quantity")
                row = await connection.fetchrow(
                    "SELECT * FROM commercial.usage_event WHERE reservation_id = $1",
                    reservation_id,
                )
                existing_event = _usage_event(row)
                if not _same_usage_event_accounting(existing_event, event_candidate):
                    raise IdempotencyConflict(
                        "finalize retry changed immutable usage cost attribution"
                    )
                return existing_event
            if reservation.state is UsageReservationState.RELEASED:
                raise IdempotencyConflict("released usage reservation cannot be finalized")
            if actual_quantity > reservation.quantity:
                raise UsageAmountExceeded(
                    f"actual usage {actual_quantity} exceeds reserved {reservation.quantity}"
                )
            row = await connection.fetchrow(
                """
                INSERT INTO commercial.usage_event (
                    id, dedupe_key, reservation_id, customer_id, subscription_id,
                    meter, quantity, provider, model, usage_units_json, price_version,
                    cost_micros, is_estimated, metadata_json, period_start, period_end,
                    occurred_at, created_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11,
                    $12, $13, $14::jsonb, $15, $16, $17, $18
                )
                RETURNING *
                """,
                event_candidate.id,
                event_candidate.dedupe_key,
                event_candidate.reservation_id,
                event_candidate.customer_id,
                event_candidate.subscription_id,
                event_candidate.meter,
                event_candidate.quantity,
                event_candidate.provider,
                event_candidate.model,
                canonical_json(thaw_json(event_candidate.usage_units)),
                event_candidate.price_version,
                event_candidate.cost_micros,
                event_candidate.is_estimated,
                canonical_json(thaw_json(event_candidate.metadata)),
                event_candidate.period_start,
                event_candidate.period_end,
                event_candidate.occurred_at,
                event_candidate.created_at,
            )
            await connection.execute(
                """
                UPDATE commercial.usage_reservation
                SET state = 'finalized', finalized_at = $2, actual_quantity = $3,
                    usage_event_id = $4
                WHERE id = $1
                """,
                reservation_id,
                finalized_at,
                actual_quantity,
                event_candidate.id,
            )
            return _usage_event(row)

    async def release_usage(
        self,
        reservation_id: str,
        released_at: datetime,
    ) -> UsageReservation:
        async with self._connection() as connection, connection.transaction():
            row = await connection.fetchrow(
                """
                SELECT * FROM commercial.usage_reservation
                WHERE id = $1 FOR UPDATE
                """,
                reservation_id,
            )
            if row is None:
                raise CommercialNotFound(f"usage reservation not found: {reservation_id}")
            existing = _reservation(row)
            if existing.state is UsageReservationState.RELEASED:
                return existing
            if existing.state is UsageReservationState.FINALIZED:
                raise IdempotencyConflict("finalized usage reservation cannot be released")
            row = await connection.fetchrow(
                """
                UPDATE commercial.usage_reservation
                SET state = 'released', released_at = $2
                WHERE id = $1 RETURNING *
                """,
                reservation_id,
                released_at,
            )
            return _reservation(row)

    async def reconcile(self, at: datetime, *, repair: bool) -> ReconciliationReport:
        async with self._connection() as connection, connection.transaction():
            lock_clause = " FOR UPDATE" if repair else ""
            trial_rows = await connection.fetch(
                """
                SELECT id FROM commercial.subscription
                WHERE status = 'trialing' AND trial_ends_at <= $1
                """
                + lock_clause,
                at,
            )
            reservation_rows = await connection.fetch(
                """
                SELECT id FROM commercial.usage_reservation
                WHERE state = 'active' AND expires_at <= $1
                """
                + lock_clause,
                at,
            )
            findings = [
                ReconciliationFinding(
                    code="expired_trial_not_materialized",
                    entity_type="subscription",
                    entity_id=row["id"],
                    detail="trial end timestamp passed while status remained trialing",
                    repairable=True,
                )
                for row in trial_rows
            ]
            findings.extend(
                ReconciliationFinding(
                    code="stale_usage_reservation",
                    entity_type="usage_reservation",
                    entity_id=row["id"],
                    detail="reservation lease expired without finalize or release",
                    repairable=True,
                )
                for row in reservation_rows
            )
            estimated_rows = await connection.fetch(
                """
                SELECT customer_id, subscription_id,
                       COUNT(*) AS total,
                       COUNT(*) FILTER (WHERE is_estimated) AS estimated
                FROM commercial.usage_event
                GROUP BY customer_id, subscription_id
                HAVING COUNT(*) >= $1
                   AND COUNT(*) FILTER (WHERE is_estimated)::float / COUNT(*) > $2
                ORDER BY customer_id, subscription_id
                """,
                ESTIMATED_USAGE_MIN_EVENTS,
                ESTIMATED_USAGE_SHARE_THRESHOLD,
            )
            findings.extend(
                ReconciliationFinding(
                    code="estimated_usage_share_high",
                    entity_type="subscription",
                    entity_id=row["subscription_id"],
                    detail=(
                        f"customer {row['customer_id']}: {row['estimated']}/{row['total']} "
                        f"usage events are estimated "
                        f"({row['estimated'] / row['total']:.0%}); provider usage "
                        "reporting looks unhealthy"
                    ),
                    repairable=False,
                )
                for row in estimated_rows
            )
            repaired = 0
            if repair and trial_rows:
                result = await connection.execute(
                    """
                    UPDATE commercial.subscription
                    SET status = 'expired', version = version + 1, updated_at = $1
                    WHERE status = 'trialing' AND trial_ends_at <= $1
                    """,
                    at,
                )
                repaired += _affected_rows(result)
            if repair and reservation_rows:
                result = await connection.execute(
                    """
                    UPDATE commercial.usage_reservation
                    SET state = 'released', released_at = $1
                    WHERE state = 'active' AND expires_at <= $1
                    """,
                    at,
                )
                repaired += _affected_rows(result)
            return ReconciliationReport(
                checked_at=at,
                findings=tuple(findings),
                repaired=repaired,
            )

    async def create_order(self, candidate: PurchaseOrder) -> PurchaseOrder:
        async with self._connection() as connection, connection.transaction():
            existing_row = await connection.fetchrow(
                "SELECT * FROM commercial.purchase_order WHERE dedupe_key = $1",
                candidate.dedupe_key,
            )
            if existing_row is not None:
                existing = _order(existing_row)
                if not _same_order_identity(existing, candidate):
                    raise IdempotencyConflict(
                        "order idempotency key was replayed with different semantics"
                    )
                return existing
            pending = await connection.fetchval(
                """
                SELECT id FROM commercial.purchase_order
                WHERE customer_id = $1 AND status = 'pending_payment'
                """,
                candidate.customer_id,
            )
            if pending is not None:
                raise PendingOrderExists("cancel or fulfill the open order before creating another")
            await connection.execute(
                """
                INSERT INTO commercial.purchase_order (
                    id, dedupe_key, customer_id, plan_version_id, order_type,
                    price_minor, currency, status, provider, external_order_id,
                    subscription_id, period_start, period_end, paid_at,
                    fulfilled_at, canceled_at, refunded_at, metadata_json,
                    created_at, updated_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                    NULL, $11, $12, NULL, NULL, NULL, NULL, $13, $14, $15
                )
                """,
                candidate.id,
                candidate.dedupe_key,
                candidate.customer_id,
                candidate.plan_version_id,
                candidate.order_type.value,
                candidate.price_minor,
                candidate.currency,
                candidate.status.value,
                candidate.provider,
                candidate.external_order_id,
                candidate.period_start,
                candidate.period_end,
                json.dumps(thaw_json(candidate.metadata)),
                candidate.created_at,
                candidate.updated_at,
            )
            return candidate

    async def get_order(self, order_id: str) -> PurchaseOrder | None:
        async with self._connection() as connection:
            row = await connection.fetchrow(
                "SELECT * FROM commercial.purchase_order WHERE id = $1",
                order_id,
            )
        return _order(row) if row is not None else None

    async def list_orders(
        self,
        customer_id: str,
        *,
        status: str | None = None,
    ) -> tuple[PurchaseOrder, ...]:
        async with self._connection() as connection:
            rows = await connection.fetch(
                """
                SELECT * FROM commercial.purchase_order
                WHERE customer_id = $1
                  AND ($2::text IS NULL OR status = $2)
                ORDER BY created_at, id
                """,
                customer_id,
                status,
            )
        return tuple(_order(row) for row in rows)

    async def list_all_orders(
        self,
        *,
        status: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> tuple[PurchaseOrder, ...]:
        async with self._connection() as connection:
            rows = await connection.fetch(
                """
                SELECT * FROM commercial.purchase_order
                WHERE ($1::text IS NULL OR status = $1)
                ORDER BY created_at DESC, id DESC
                LIMIT $2 OFFSET $3
                """,
                status,
                max(1, limit),
                max(0, offset),
            )
        return tuple(_order(row) for row in rows)

    async def pay_and_fulfill_order(
        self,
        order_id: str,
        subscription_candidate: Subscription | None,
        changed_at: datetime,
        *,
        operator: str,
        payment_reference: str | None = None,
        extend_subscription_id: str | None = None,
    ) -> tuple[PurchaseOrder, Subscription]:
        async with self._connection() as connection, connection.transaction():
            order_row = await connection.fetchrow(
                "SELECT * FROM commercial.purchase_order WHERE id = $1 FOR UPDATE",
                order_id,
            )
            if order_row is None:
                raise CommercialNotFound(f"purchase order not found: {order_id}")
            order = _order(order_row)
            if order.status is OrderStatus.FULFILLED:
                subscription_row = await connection.fetchrow(
                    "SELECT * FROM commercial.subscription WHERE id = $1",
                    order.subscription_id,
                )
                if subscription_row is None:
                    raise CommercialConflict("fulfilled order lost its subscription")
                return order, _subscription(subscription_row)
            if order.status not in {OrderStatus.PENDING_PAYMENT, OrderStatus.PAID}:
                raise InvalidOrderTransition(
                    f"order in status {order.status.value} cannot be fulfilled"
                )
            if extend_subscription_id is not None:
                # Early renewal: extend the live subscription instead of
                # replacing it, so paid time is never interrupted.
                extended_row = await connection.fetchrow(
                    """
                    UPDATE commercial.subscription
                    SET current_period_end = $2, version = version + 1,
                        updated_at = $3
                    WHERE id = $1
                      AND status IN ('trialing', 'active', 'past_due')
                      AND current_period_end < $2
                    RETURNING *
                    """,
                    extend_subscription_id,
                    order.period_end,
                    changed_at,
                )
                if extended_row is None:
                    raise CommercialConflict(
                        "renewal target subscription is not live or not extended"
                    )
                subscription = _subscription(extended_row)
            else:
                assert subscription_candidate is not None
                # Same replacement semantics as activate_paid_subscription,
                # keyed to the order id so a retried fulfillment converges.
                existing_row = await connection.fetchrow(
                    """
                    SELECT * FROM commercial.subscription
                    WHERE provider = $1 AND external_subscription_id = $2
                    """,
                    subscription_candidate.provider,
                    subscription_candidate.external_subscription_id,
                )
                if existing_row is not None:
                    existing = _subscription(existing_row)
                    if not _same_subscription_identity(existing, subscription_candidate):
                        raise IdempotencyConflict(
                            "external subscription id was reused with different semantics"
                        )
                    subscription = existing
                else:
                    await connection.execute(
                        """
                        UPDATE commercial.subscription
                        SET status = 'canceled', canceled_at = $2,
                            version = version + 1, updated_at = $2
                        WHERE customer_id = $1
                          AND status IN ('trialing', 'active', 'past_due')
                        """,
                        order.customer_id,
                        changed_at,
                    )
                    try:
                        await self._insert_subscription(connection, subscription_candidate)
                    except Exception as exc:
                        _raise_subscription_conflict(exc)
                    subscription = subscription_candidate
            metadata = dict(thaw_json(order.metadata) or {})
            metadata["fulfilled_by"] = operator
            if payment_reference:
                metadata["payment_reference"] = payment_reference
            updated_row = await connection.fetchrow(
                """
                UPDATE commercial.purchase_order
                SET status = 'fulfilled', paid_at = COALESCE(paid_at, $2),
                    fulfilled_at = $2, subscription_id = $3,
                    metadata_json = $4, updated_at = $2
                WHERE id = $1
                RETURNING *
                """,
                order.id,
                changed_at,
                subscription.id,
                json.dumps(metadata),
            )
            return _order(updated_row), subscription

    async def cancel_order(
        self,
        order_id: str,
        canceled_at: datetime,
    ) -> PurchaseOrder:
        async with self._connection() as connection, connection.transaction():
            order_row = await connection.fetchrow(
                "SELECT * FROM commercial.purchase_order WHERE id = $1 FOR UPDATE",
                order_id,
            )
            if order_row is None:
                raise CommercialNotFound(f"purchase order not found: {order_id}")
            order = _order(order_row)
            if order.status is OrderStatus.CANCELED:
                return order
            if order.status is not OrderStatus.PENDING_PAYMENT:
                raise InvalidOrderTransition(
                    f"order in status {order.status.value} cannot be canceled"
                )
            updated_row = await connection.fetchrow(
                """
                UPDATE commercial.purchase_order
                SET status = 'canceled', canceled_at = $2, updated_at = $2
                WHERE id = $1
                RETURNING *
                """,
                order_id,
                canceled_at,
            )
            return _order(updated_row)

    async def erase_customer(self, owner_id: str) -> bool:
        async with self._connection() as connection, connection.transaction():
            customer_id = await connection.fetchval(
                """
                SELECT id FROM commercial.billing_customer
                WHERE owner_id = $1 FOR UPDATE
                """,
                owner_id,
            )
            if customer_id is None:
                return False
            # The customer/trial-subscription cycle is intentionally deferrable.
            await connection.execute("SET CONSTRAINTS ALL DEFERRED")
            await connection.fetchval(
                "SELECT set_config('deeptutor.commercial_erasure', 'on', true)"
            )
            await connection.execute(
                "DELETE FROM commercial.webhook_event WHERE customer_id = $1",
                customer_id,
            )
            await connection.execute(
                "DELETE FROM commercial.usage_event WHERE customer_id = $1",
                customer_id,
            )
            await connection.execute(
                "DELETE FROM commercial.usage_reservation WHERE customer_id = $1",
                customer_id,
            )
            # The order→subscription FK is deferrable: deleting orders
            # before subscriptions satisfies it at commit, and nulling
            # subscription_id instead would violate the fulfilled-shape
            # check (checks are NOT deferrable).
            await connection.execute(
                "DELETE FROM commercial.purchase_order WHERE customer_id = $1",
                customer_id,
            )
            await connection.execute(
                "DELETE FROM commercial.subscription WHERE customer_id = $1",
                customer_id,
            )
            await connection.execute(
                "DELETE FROM commercial.billing_customer WHERE id = $1",
                customer_id,
            )
            return True


def _billing_customer(row: Any) -> BillingCustomer:
    return BillingCustomer(
        id=row["id"],
        owner_id=row["owner_id"],
        provider=row["provider"],
        external_customer_id=row["external_customer_id"],
        trial_claimed_at=row["trial_claimed_at"],
        trial_subscription_id=row["trial_subscription_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _plan_version(row: Any) -> PlanVersion:
    return PlanVersion(
        id=row["id"],
        plan_code=row["plan_code"],
        version=row["version"],
        name=row["name"],
        price_minor=row["price_minor"],
        currency=row["currency"].strip(),
        billing_interval=row["billing_interval"],
        trial_days=row["trial_days"],
        created_at=row["created_at"],
    )


def _entitlement(row: Any) -> Entitlement:
    return Entitlement(
        id=row["id"],
        plan_version_id=row["plan_version_id"],
        key=row["key"],
        value=_decode_json(row["value_json"]),
        created_at=row["created_at"],
    )


def _price_entry(row: Any) -> PriceEntry:
    return PriceEntry(
        id=row["id"],
        price_version=row["price_version"],
        kind=row["kind"],
        meter=row["meter"],
        model_tier=row["model_tier"],
        unit_price_micros=row["unit_price_micros"],
        currency=row["currency"].strip(),
        effective_from=row["effective_from"],
        created_at=row["created_at"],
    )


def _same_price_entry(existing: PriceEntry, candidate: PriceEntry) -> bool:
    """Semantic identity for replayed catalog rows; ids and timestamps differ."""

    return (
        existing.price_version,
        existing.kind,
        existing.meter,
        existing.model_tier,
        existing.unit_price_micros,
        existing.currency,
        existing.effective_from,
    ) == (
        candidate.price_version,
        candidate.kind,
        candidate.meter,
        candidate.model_tier,
        candidate.unit_price_micros,
        candidate.currency,
        candidate.effective_from,
    )


def _subscription(row: Any) -> Subscription:
    return Subscription(
        id=row["id"],
        customer_id=row["customer_id"],
        plan_version_id=row["plan_version_id"],
        status=SubscriptionStatus(row["status"]),
        provider=row["provider"],
        external_subscription_id=row["external_subscription_id"],
        current_period_start=row["current_period_start"],
        current_period_end=row["current_period_end"],
        trial_started_at=row["trial_started_at"],
        trial_ends_at=row["trial_ends_at"],
        canceled_at=row["canceled_at"],
        version=row["version"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _order(row: Any) -> PurchaseOrder:
    return PurchaseOrder(
        id=row["id"],
        dedupe_key=row["dedupe_key"].strip(),
        customer_id=row["customer_id"],
        plan_version_id=row["plan_version_id"],
        order_type=OrderType(row["order_type"]),
        price_minor=row["price_minor"],
        currency=row["currency"],
        status=OrderStatus(row["status"]),
        provider=row["provider"],
        external_order_id=row["external_order_id"],
        subscription_id=row["subscription_id"],
        period_start=row["period_start"],
        period_end=row["period_end"],
        paid_at=row["paid_at"],
        fulfilled_at=row["fulfilled_at"],
        canceled_at=row["canceled_at"],
        refunded_at=row["refunded_at"],
        metadata=_decode_json(row["metadata_json"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _same_order_identity(existing: PurchaseOrder, candidate: PurchaseOrder) -> bool:
    return (
        existing.customer_id,
        existing.plan_version_id,
        existing.order_type,
        existing.price_minor,
        existing.currency,
        existing.provider,
    ) == (
        candidate.customer_id,
        candidate.plan_version_id,
        candidate.order_type,
        candidate.price_minor,
        candidate.currency,
        candidate.provider,
    )


def _webhook(row: Any) -> WebhookEvent:
    return WebhookEvent(
        id=row["id"],
        customer_id=row["customer_id"],
        provider=row["provider"],
        external_event_id=row["external_event_id"],
        dedupe_key=row["dedupe_key"].strip(),
        event_type=row["event_type"],
        payload=_decode_json(row["payload_json"]),
        payload_sha256=row["payload_sha256"].strip(),
        occurred_at=row["occurred_at"],
        received_at=row["received_at"],
        processed_at=row["processed_at"],
        processing_error=row["processing_error"],
        processing_attempts=row["processing_attempts"],
    )


def _reservation(row: Any) -> UsageReservation:
    return UsageReservation(
        id=row["id"],
        dedupe_key=row["dedupe_key"].strip(),
        customer_id=row["customer_id"],
        subscription_id=row["subscription_id"],
        meter=row["meter"],
        quantity=row["quantity"],
        state=UsageReservationState(row["state"]),
        metadata=_decode_json(row["metadata_json"]),
        period_start=row["period_start"],
        period_end=row["period_end"],
        expires_at=row["expires_at"],
        created_at=row["created_at"],
        finalized_at=row["finalized_at"],
        released_at=row["released_at"],
        actual_quantity=row["actual_quantity"],
        usage_event_id=row["usage_event_id"],
    )


def _usage_event(row: Any) -> UsageEvent:
    return UsageEvent(
        id=row["id"],
        dedupe_key=row["dedupe_key"].strip(),
        reservation_id=row["reservation_id"],
        customer_id=row["customer_id"],
        subscription_id=row["subscription_id"],
        meter=row["meter"],
        quantity=row["quantity"],
        provider=row["provider"],
        model=row["model"],
        usage_units=_decode_json(row["usage_units_json"]),
        price_version=row["price_version"],
        cost_micros=row["cost_micros"],
        is_estimated=row["is_estimated"],
        metadata=_decode_json(row["metadata_json"]),
        period_start=row["period_start"],
        period_end=row["period_end"],
        occurred_at=row["occurred_at"],
        created_at=row["created_at"],
    )


def _decode_json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _same_plan(existing: PlanVersion, candidate: PlanVersion) -> bool:
    return (
        existing.plan_code,
        existing.version,
        existing.name,
        existing.price_minor,
        existing.currency,
        existing.billing_interval,
        existing.trial_days,
    ) == (
        candidate.plan_code,
        candidate.version,
        candidate.name,
        candidate.price_minor,
        candidate.currency,
        candidate.billing_interval,
        candidate.trial_days,
    )


def _same_entitlements(
    existing: tuple[Entitlement, ...],
    candidate: tuple[Entitlement, ...],
) -> bool:
    def normalized(values: tuple[Entitlement, ...]) -> dict[str, str]:
        return {item.key: canonical_json(thaw_json(item.value)) for item in values}

    return normalized(existing) == normalized(candidate)


def _same_subscription_identity(existing: Subscription, candidate: Subscription) -> bool:
    return (
        existing.customer_id,
        existing.plan_version_id,
        existing.provider,
        existing.external_subscription_id,
        existing.current_period_start,
        existing.current_period_end,
    ) == (
        candidate.customer_id,
        candidate.plan_version_id,
        candidate.provider,
        candidate.external_subscription_id,
        candidate.current_period_start,
        candidate.current_period_end,
    )


def _assert_same_reservation(
    existing: UsageReservation,
    candidate: UsageReservation,
) -> None:
    if (
        existing.customer_id,
        existing.subscription_id,
        existing.meter,
        existing.quantity,
        canonical_json(thaw_json(existing.metadata)),
    ) != (
        candidate.customer_id,
        candidate.subscription_id,
        candidate.meter,
        candidate.quantity,
        canonical_json(thaw_json(candidate.metadata)),
    ):
        raise IdempotencyConflict(
            "usage request id was replayed with different reservation semantics"
        )


def _same_usage_event_accounting(existing: UsageEvent, candidate: UsageEvent) -> bool:
    return (
        existing.quantity,
        existing.provider,
        existing.model,
        canonical_json(thaw_json(existing.usage_units)),
        existing.price_version,
        existing.cost_micros,
        existing.is_estimated,
    ) == (
        candidate.quantity,
        candidate.provider,
        candidate.model,
        canonical_json(thaw_json(candidate.usage_units)),
        candidate.price_version,
        candidate.cost_micros,
        candidate.is_estimated,
    )


def _raise_subscription_conflict(exc: Exception) -> None:
    constraint = getattr(exc, "constraint_name", "") or ""
    if constraint == "subscription_one_live_per_customer_uidx":
        raise ActiveSubscriptionExists(
            "customer already has a trialing, active, or past-due subscription"
        ) from exc
    raise CommercialConflict("subscription write violated a durable constraint") from exc


def _affected_rows(command_status: str) -> int:
    try:
        return int(command_status.rsplit(" ", 1)[-1])
    except (TypeError, ValueError):
        return 0
