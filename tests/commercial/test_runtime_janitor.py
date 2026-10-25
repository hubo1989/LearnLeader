"""The commercial runtime's periodic reconcile janitor."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pytest

from deeptutor.commercial.runtime import (
    DEFAULT_RECONCILE_INTERVAL_SECONDS,
    CommercialRuntime,
    CommercialSettings,
)
from deeptutor.commercial.seed import TrialV1ModelBinding

ENABLED_ENV = {
    "DEEPTUTOR_COMMERCIAL_ENABLED": "true",
    "DEEPTUTOR_COMMERCIAL_DATABASE_URL": "postgresql://localhost/commercial",
    "DEEPTUTOR_COMMERCIAL_TRIAL_LLM_MODELS": ('{"profile_id": "trial", "model_ids": ["m-1"]}'),
}


def test_reconcile_interval_defaults_and_disables_via_env():
    default = CommercialSettings.from_env(ENABLED_ENV)
    assert default.reconcile_interval_seconds == DEFAULT_RECONCILE_INTERVAL_SECONDS

    disabled = CommercialSettings.from_env(
        {**ENABLED_ENV, "DEEPTUTOR_COMMERCIAL_RECONCILE_INTERVAL_SECONDS": "0"}
    )
    assert disabled.reconcile_interval_seconds == 0.0

    custom = CommercialSettings.from_env(
        {**ENABLED_ENV, "DEEPTUTOR_COMMERCIAL_RECONCILE_INTERVAL_SECONDS": "45.5"}
    )
    assert custom.reconcile_interval_seconds == 45.5


def test_reconcile_interval_rejects_garbage_and_negative():
    with pytest.raises(Exception, match="RECONCILE_INTERVAL"):
        CommercialSettings.from_env(
            {**ENABLED_ENV, "DEEPTUTOR_COMMERCIAL_RECONCILE_INTERVAL_SECONDS": "soon"}
        )
    with pytest.raises(Exception, match="negative"):
        CommercialSettings.from_env(
            {**ENABLED_ENV, "DEEPTUTOR_COMMERCIAL_RECONCILE_INTERVAL_SECONDS": "-1"}
        )


@dataclass
class _RecordingControlPlane:
    calls: list[bool] = field(default_factory=list)
    error: Exception | None = None

    async def reconcile(self, *, repair: bool = False):
        # Mirrors CommercialControlPlane.reconcile's keyword-only signature:
        # a stub accepting (at, repair) would hide caller signature drift.
        self.calls.append(repair)
        if self.error is not None:
            raise self.error
        from deeptutor.commercial.models import ReconciliationReport

        return ReconciliationReport(checked_at=datetime.now(timezone.utc), findings=(), repaired=0)


def _runtime_with(interval: float) -> CommercialRuntime:
    runtime = CommercialRuntime(
        CommercialSettings(
            enabled=True,
            database_url="postgresql://localhost/commercial",
            trial_model_binding=TrialV1ModelBinding(profile_id="trial", model_ids=("m-1",)),
            reconcile_interval_seconds=interval,
        )
    )
    return runtime


@pytest.mark.asyncio
async def test_sweep_applies_repair_with_utc_clock():
    runtime = _runtime_with(300.0)
    control_plane = _RecordingControlPlane()
    runtime.control_plane = control_plane  # type: ignore[assignment]

    await runtime.run_reconcile_sweep()

    assert control_plane.calls == [True]


@pytest.mark.asyncio
async def test_sweep_repairs_stale_state_through_the_real_control_plane():
    """The sweep must call the control plane's actual signature.

    The recording-stub test above cannot catch a signature mismatch — the
    stub mirrors whatever the caller passes. This one exercises a real
    CommercialControlPlane: a leaked reservation and an expired trial must
    both be repaired by one sweep.
    """

    from deeptutor.commercial import CommercialControlPlane, InMemoryCommercialRepository
    from deeptutor.commercial.models import (
        SubscriptionStatus,
        UsageReservationState,
    )

    repository = InMemoryCommercialRepository()
    service = CommercialControlPlane(repository)
    runtime = _runtime_with(300.0)
    runtime.control_plane = service  # type: ignore[assignment]

    from dataclasses import replace
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    customer = await service.ensure_billing_customer(owner_id="u_janitor")
    plan = await service.publish_plan_version(
        plan_code="pro",
        version=1,
        name="Pro",
        price_minor=100,
        currency="CNY",
        entitlements={"quota.llm_tokens": {"limit": 100, "window": "subscription"}},
    )
    trial = await service.ensure_trial(customer.id, plan.id)
    # Reserve while the trial is still entitled, then age both past expiry.
    reservation = await service.reserve_usage(customer.id, "llm_tokens", 10, "janitor-stale-1")
    async with repository._lock:
        repository._subscriptions[trial.id] = replace(
            trial,
            trial_ends_at=now - timedelta(days=1),
            current_period_end=now - timedelta(days=1),
        )
        repository._reservations[reservation.id] = replace(
            reservation, expires_at=now - timedelta(minutes=1)
        )

    await runtime.run_reconcile_sweep()

    refreshed_trial = await repository.get_subscription(trial.id)
    assert refreshed_trial is not None
    assert refreshed_trial.status is SubscriptionStatus.EXPIRED
    refreshed_reservation = await repository.get_usage_reservation(reservation.id)
    assert refreshed_reservation is not None
    assert refreshed_reservation.state is UsageReservationState.RELEASED


@pytest.mark.asyncio
async def test_sweep_swallows_failures_to_survive_the_next_tick():
    runtime = _runtime_with(300.0)
    control_plane = _RecordingControlPlane()
    control_plane.error = RuntimeError("ledger unavailable")
    runtime.control_plane = control_plane  # type: ignore[assignment]

    await runtime.run_reconcile_sweep()  # must not raise

    assert len(control_plane.calls) == 1
