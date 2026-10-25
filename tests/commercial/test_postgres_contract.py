from __future__ import annotations

import builtins
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from deeptutor.commercial.migrations import discover_migrations
from deeptutor.commercial.models import UsageReservation, UsageReservationState
from deeptutor.commercial.postgres import AsyncpgUnavailable, PostgresCommercialRepository


def test_initial_postgres_migration_defines_all_control_plane_tables_and_guards() -> None:
    migrations = discover_migrations()
    assert [migration.version for migration in migrations] == [1, 2, 3]
    sql = migrations[0].sql.lower()

    for table in (
        "schema_migration",
        "billing_customer",
        "plan_version",
        "subscription",
        "entitlement",
        "webhook_event",
        "usage_event",
        "usage_reservation",
    ):
        assert f"table if not exists commercial.{table}" in sql

    assert "where status in ('trialing', 'active', 'past_due')" in sql
    assert "prevent_plan_version_mutation" in sql
    assert "validate_subscription_transition" in sql
    assert "trial_subscription_id" in sql
    assert "cost_micros" in sql
    assert "price_version" in sql
    assert "usage_units_json" in sql
    assert sql.count("on delete no action") >= 3
    assert "deeptutor.commercial_erasure" in sql
    assert migrations[0].checksum == migrations[0].checksum
    assert len(migrations[0].checksum) == 64
    assert "deeptutor/commercial/sql" in str(migrations[0].path)


def test_migration_discovery_rejects_duplicate_versions(tmp_path: Path) -> None:
    (tmp_path / "0001_first.sql").write_text("select 1", encoding="utf-8")
    (tmp_path / "0001_second.sql").write_text("select 2", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate migration version"):
        discover_migrations(tmp_path)


def test_orders_migration_defines_purchase_order_state_machine() -> None:
    migrations = discover_migrations()
    sql = migrations[1].sql.lower()

    assert "table if not exists commercial.purchase_order" in sql
    assert "purchase_order_one_pending_uidx" in sql
    assert "purchase_order_provider_external_uidx" in sql
    assert "validate_purchase_order_transition" in sql
    # pending_payment → paid | fulfilled | canceled; paid → fulfilled | refunded.
    assert "new.status in ('paid', 'fulfilled', 'canceled')" in sql
    assert "old.status = 'paid' and new.status in ('fulfilled', 'refunded')" in sql
    assert "price_minor" in sql and "period_end > period_start" in sql


def test_pricing_migration_defines_immutable_price_catalog() -> None:
    migrations = discover_migrations()
    sql = migrations[2].sql.lower()

    assert "table if not exists commercial.price_version" in sql
    # Two ledgers in one table, split by kind.
    assert "kind in ('platform_cost', 'list_price')" in sql
    assert "unit_price_micros bigdecimal" not in sql
    assert "unit_price_micros" in sql and "unit_price_micros >= 0" in sql
    assert "effective_from timestamptz" in sql
    assert "currency char(3)" in sql
    assert "unique (price_version, kind, meter, model_tier)" in sql
    assert "price_version_lookup_idx" in sql
    # Catalog rows are immutable like plan versions.
    assert "price_version_immutable" in sql
    assert "prevent_plan_version_mutation" in sql


@pytest.mark.asyncio
async def test_asyncpg_is_imported_only_when_connecting(monkeypatch) -> None:
    repository = PostgresCommercialRepository(pool=object())
    assert repository is not None

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "asyncpg":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(AsyncpgUnavailable):
        await PostgresCommercialRepository.connect("postgresql://example.invalid/db")


@pytest.mark.asyncio
async def test_reserve_releases_expired_active_rows_before_counting_quota(monkeypatch) -> None:
    now = datetime(2026, 8, 5, 9, 0, tzinfo=timezone.utc)
    candidate = UsageReservation(
        id="res-new",
        dedupe_key="req-new",
        customer_id="cus-1",
        subscription_id="sub-1",
        meter="llm_tokens",
        quantity=20,
        state=UsageReservationState.ACTIVE,
        metadata={},
        period_start=now - timedelta(hours=1),
        period_end=now + timedelta(days=1),
        expires_at=now + timedelta(minutes=15),
        created_at=now,
        finalized_at=None,
        released_at=None,
        actual_quantity=None,
        usage_event_id=None,
    )
    subscription_row = {
        "id": "sub-1",
        "customer_id": "cus-1",
        "plan_version_id": "plan-1",
        "status": "active",
        "provider": "internal",
        "external_subscription_id": None,
        "current_period_start": candidate.period_start,
        "current_period_end": candidate.period_end,
        "trial_started_at": None,
        "trial_ends_at": None,
        "canceled_at": None,
        "version": 1,
        "created_at": candidate.period_start,
        "updated_at": candidate.period_start,
    }
    inserted_row = {
        "id": candidate.id,
        "dedupe_key": candidate.dedupe_key,
        "customer_id": candidate.customer_id,
        "subscription_id": candidate.subscription_id,
        "meter": candidate.meter,
        "quantity": candidate.quantity,
        "state": "active",
        "metadata_json": {},
        "period_start": candidate.period_start,
        "period_end": candidate.period_end,
        "expires_at": candidate.expires_at,
        "created_at": candidate.created_at,
        "finalized_at": None,
        "released_at": None,
        "actual_quantity": None,
        "usage_event_id": None,
    }

    class Transaction:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    class Connection:
        def __init__(self) -> None:
            self.reclaimed = False
            self.reserved_query = ""
            self.reserved_args: tuple[object, ...] = ()

        def transaction(self):
            return Transaction()

        async def fetchrow(self, sql, *_args):
            if "WHERE dedupe_key" in sql:
                return None
            if "FROM commercial.subscription" in sql:
                return subscription_row
            if "INSERT INTO commercial.usage_reservation" in sql:
                return inserted_row
            raise AssertionError(f"unexpected fetchrow SQL: {sql}")

        async def fetchval(self, sql, *_args):
            if "FROM commercial.billing_customer" in sql:
                return "cus-1"
            if "FROM commercial.usage_event" in sql:
                return 0
            if "FROM commercial.usage_reservation" in sql:
                assert self.reclaimed is True
                self.reserved_query = sql
                self.reserved_args = _args
                return 0
            raise AssertionError(f"unexpected fetchval SQL: {sql}")

        async def execute(self, sql, *_args):
            assert "expires_at <= $6" in sql
            assert _args[-1] == now
            self.reclaimed = True
            return "UPDATE 1"

    connection = Connection()

    @asynccontextmanager
    async def connection_scope():
        yield connection

    repository = PostgresCommercialRepository(pool=object())
    monkeypatch.setattr(repository, "_connection", connection_scope)

    reserved = await repository.reserve_usage(candidate, quota_limit=20)

    assert reserved.id == candidate.id
    assert connection.reclaimed is True
    assert "expires_at > $6" in connection.reserved_query
    assert connection.reserved_args[-1] == now


@pytest.mark.asyncio
async def test_reserve_usage_enforces_platform_cost_budget(monkeypatch) -> None:
    """The budget check runs after quota and aggregates attributed event cost."""

    from deeptutor.commercial.errors import EntitlementDenied

    now = datetime(2026, 10, 25, 9, 0, tzinfo=timezone.utc)
    candidate = UsageReservation(
        id="res-budget",
        dedupe_key="req-budget",
        customer_id="cus-1",
        subscription_id="sub-1",
        meter="llm_tokens",
        quantity=20,
        state=UsageReservationState.ACTIVE,
        metadata={},
        period_start=now - timedelta(hours=1),
        period_end=now + timedelta(days=1),
        expires_at=now + timedelta(minutes=15),
        created_at=now,
        finalized_at=None,
        released_at=None,
        actual_quantity=None,
        usage_event_id=None,
    )
    budget_query: dict[str, object] = {}

    class Transaction:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    class Connection:
        def transaction(self):
            return Transaction()

        async def fetchrow(self, sql, *_args):
            if "WHERE dedupe_key" in sql:
                return None
            if "FROM commercial.subscription" in sql:
                return {
                    "id": "sub-1",
                    "customer_id": "cus-1",
                    "plan_version_id": "plan-1",
                    "status": "active",
                    "provider": "internal",
                    "external_subscription_id": None,
                    "current_period_start": candidate.period_start,
                    "current_period_end": candidate.period_end,
                    "trial_started_at": None,
                    "trial_ends_at": None,
                    "canceled_at": None,
                    "version": 1,
                    "created_at": candidate.period_start,
                    "updated_at": candidate.period_start,
                }
            raise AssertionError(f"unexpected fetchrow SQL: {sql}")

        async def fetchval(self, sql, *_args):
            if "FROM commercial.billing_customer" in sql:
                return "cus-1"
            if "SUM(cost_micros)" in sql:
                budget_query["sql"] = sql
                budget_query["args"] = _args
                return 900
            if "FROM commercial.usage_event" in sql or "FROM commercial.usage_reservation" in sql:
                return 0
            raise AssertionError(f"unexpected fetchval SQL: {sql}")

        async def execute(self, sql, *_args):
            assert "expires_at <= $6" in sql
            return "UPDATE 0"

    connection = Connection()

    @asynccontextmanager
    async def connection_scope():
        yield connection

    repository = PostgresCommercialRepository(pool=object())
    monkeypatch.setattr(repository, "_connection", connection_scope)

    # 900 spent + 1500 estimated > 1000 limit: the gate denies.
    with pytest.raises(EntitlementDenied, match="budget.platform_cost exceeded"):
        await repository.reserve_usage(
            candidate,
            quota_limit=None,
            budget_limit_micros=1_000,
            estimated_cost_micros=1_500,
        )
    assert "customer_id = $1 AND subscription_id = $2" in budget_query["sql"]


@pytest.mark.asyncio
async def test_reconcile_queries_estimated_usage_share_per_subscription(monkeypatch) -> None:
    """The estimated-share finding groups events per customer subscription."""

    now = datetime(2026, 10, 25, 9, 0, tzinfo=timezone.utc)

    class Transaction:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    class Connection:
        def __init__(self) -> None:
            self.estimated_query = ""

        def transaction(self):
            return Transaction()

        async def fetch(self, sql, *_args):
            if "FROM commercial.subscription" in sql:
                return []
            if "FROM commercial.usage_reservation" in sql:
                return []
            if "GROUP BY customer_id, subscription_id" in sql:
                self.estimated_query = sql
                return [
                    {
                        "customer_id": "cus-1",
                        "subscription_id": "sub-1",
                        "total": 8,
                        "estimated": 6,
                    }
                ]
            raise AssertionError(f"unexpected fetch SQL: {sql}")

        async def execute(self, *_args):
            return "UPDATE 0"

    connection = Connection()

    @asynccontextmanager
    async def connection_scope():
        yield connection

    repository = PostgresCommercialRepository(pool=object())
    monkeypatch.setattr(repository, "_connection", connection_scope)

    report = await repository.reconcile(now, repair=True)

    assert "FILTER (WHERE is_estimated)" in connection.estimated_query
    assert len(report.findings) == 1
    finding = report.findings[0]
    assert finding.code == "estimated_usage_share_high"
    assert finding.entity_id == "sub-1"
    assert finding.repairable is False
    assert "6/8" in finding.detail
