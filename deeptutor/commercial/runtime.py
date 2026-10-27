"""Fail-closed environment settings and lifespan hooks for commercial mode."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
from typing import Mapping

from .errors import CommercialConfigurationError
from .models import PlanVersion, ResolvedEntitlements, Subscription
from .postgres import PostgresCommercialRepository
from .seed import TrialV1ModelBinding, seed_trial_v1
from .service import DEFAULT_TRIAL_DAYS, CommercialControlPlane

logger = logging.getLogger(__name__)

COMMERCIAL_ENABLED_ENV = "DEEPTUTOR_COMMERCIAL_ENABLED"
COMMERCIAL_DATABASE_URL_ENV = "DEEPTUTOR_COMMERCIAL_DATABASE_URL"
COMMERCIAL_DATABASE_URL_FILE_ENV = "DEEPTUTOR_COMMERCIAL_DATABASE_URL_FILE"
COMMERCIAL_TRIAL_DAYS_ENV = "DEEPTUTOR_COMMERCIAL_TRIAL_DAYS"
COMMERCIAL_TRIAL_LLM_MODELS_ENV = "DEEPTUTOR_COMMERCIAL_TRIAL_LLM_MODELS"
COMMERCIAL_RECONCILE_INTERVAL_ENV = "DEEPTUTOR_COMMERCIAL_RECONCILE_INTERVAL_SECONDS"
DEFAULT_RECONCILE_INTERVAL_SECONDS = 300.0


@dataclass(frozen=True, slots=True)
class CommercialSettings:
    enabled: bool = False
    database_url: str | None = None
    trial_days: int = DEFAULT_TRIAL_DAYS
    trial_model_binding: TrialV1ModelBinding | None = None
    reconcile_interval_seconds: float = DEFAULT_RECONCILE_INTERVAL_SECONDS

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "CommercialSettings":
        values = os.environ if environ is None else environ
        enabled = commercial_mode_requested(values)
        if not enabled:
            # Disabled commercial mode must remain inert even with stale ancillary env.
            return cls(enabled=False)
        inline_url = values.get(COMMERCIAL_DATABASE_URL_ENV, "").strip()
        url_file = values.get(COMMERCIAL_DATABASE_URL_FILE_ENV, "").strip()
        if inline_url and url_file:
            raise CommercialConfigurationError(
                "configure only one of DEEPTUTOR_COMMERCIAL_DATABASE_URL and "
                "DEEPTUTOR_COMMERCIAL_DATABASE_URL_FILE"
            )
        database_url = inline_url or None
        if url_file:
            try:
                database_url = Path(url_file).read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise CommercialConfigurationError(
                    f"cannot read commercial database URL file: {url_file}"
                ) from exc
            if not database_url:
                raise CommercialConfigurationError("commercial database URL file is empty")
        raw_trial_days = values.get(COMMERCIAL_TRIAL_DAYS_ENV, str(DEFAULT_TRIAL_DAYS)).strip()
        try:
            trial_days = int(raw_trial_days)
        except ValueError as exc:
            raise CommercialConfigurationError(
                "DEEPTUTOR_COMMERCIAL_TRIAL_DAYS must be an integer"
            ) from exc
        if trial_days != DEFAULT_TRIAL_DAYS:
            raise CommercialConfigurationError(
                "Commercial Foundation Trial v1 requires "
                f"DEEPTUTOR_COMMERCIAL_TRIAL_DAYS={DEFAULT_TRIAL_DAYS}; publish v2 to change it"
            )
        raw_models = values.get(COMMERCIAL_TRIAL_LLM_MODELS_ENV, "").strip()
        model_binding = TrialV1ModelBinding.from_json(raw_models) if raw_models else None
        raw_interval = values.get(COMMERCIAL_RECONCILE_INTERVAL_ENV, "").strip()
        reconcile_interval_seconds = DEFAULT_RECONCILE_INTERVAL_SECONDS
        if raw_interval:
            try:
                reconcile_interval_seconds = float(raw_interval)
            except ValueError as exc:
                raise CommercialConfigurationError(
                    f"{COMMERCIAL_RECONCILE_INTERVAL_ENV} must be a number of seconds"
                ) from exc
            if reconcile_interval_seconds < 0:
                raise CommercialConfigurationError(
                    f"{COMMERCIAL_RECONCILE_INTERVAL_ENV} must not be negative; use 0 to disable"
                )
        if enabled and not database_url:
            raise CommercialConfigurationError(
                "commercial mode is enabled but DEEPTUTOR_COMMERCIAL_DATABASE_URL "
                "or DEEPTUTOR_COMMERCIAL_DATABASE_URL_FILE is missing"
            )
        if model_binding is None:
            raise CommercialConfigurationError(
                "commercial mode requires DEEPTUTOR_COMMERCIAL_TRIAL_LLM_MODELS "
                "so Trial v1 cannot drift with runtime defaults"
            )
        return cls(
            enabled=enabled,
            database_url=database_url,
            trial_days=trial_days,
            trial_model_binding=model_binding,
            reconcile_interval_seconds=reconcile_interval_seconds,
        )


class CommercialRuntime:
    """Idempotent startup/shutdown adapter intended for FastAPI lifespan."""

    def __init__(self, settings: CommercialSettings) -> None:
        if settings.enabled and not settings.database_url:
            raise CommercialConfigurationError(
                "commercial mode is enabled but its PostgreSQL URL is missing"
            )
        if settings.enabled and settings.trial_model_binding is None:
            raise CommercialConfigurationError(
                "commercial mode requires an explicit Trial v1 model binding"
            )
        self.settings = settings
        self.repository: PostgresCommercialRepository | None = None
        self.control_plane: CommercialControlPlane | None = None
        self.trial_plan: PlanVersion | None = None
        self._lock = asyncio.Lock()
        self._janitor_task: asyncio.Task[None] | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "CommercialRuntime":
        return cls(CommercialSettings.from_env(environ))

    async def startup(self) -> CommercialControlPlane | None:
        if not self.settings.enabled:
            return None
        async with self._lock:
            if self.control_plane is not None:
                return self.control_plane
            repository = await PostgresCommercialRepository.connect(
                self.settings.database_url or "",
                run_migrations=True,
            )
            control_plane = CommercialControlPlane(
                repository,
                trial_days=self.settings.trial_days,
            )
            try:
                self.trial_plan = await seed_trial_v1(
                    control_plane,
                    self.settings.trial_model_binding,
                )
            except BaseException:
                await repository.close()
                raise
            self.repository = repository
            self.control_plane = control_plane
            if self.settings.reconcile_interval_seconds > 0:
                self._janitor_task = asyncio.create_task(
                    self._reconcile_janitor_loop(),
                    name="commercial-reconcile-janitor",
                )
            return control_plane

    async def run_reconcile_sweep(self) -> None:
        """Apply deterministic repair to expired trials and leaked reservations.

        Expiry (`expires_at`) already protects quota accounting, but nothing
        materializes that protection: without this sweep, stale reservations
        keep inflating ``usage_summary``'s held totals and trialing
        subscriptions linger past their end. Failures are logged, not raised,
        so one bad sweep never takes the process down.
        """

        control_plane = self.control_plane
        if control_plane is None:
            return
        try:
            report = await control_plane.reconcile(repair=True)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("commercial reconcile sweep failed")
            report = None
        if report is not None and report.findings:
            logger.info(
                "commercial reconcile sweep: %d findings, %d repaired",
                len(report.findings),
                report.repaired,
            )
        # Turn leases already expire lazily on acquire; sweeping here also
        # reclaims rows from owners who stopped making turns entirely.
        repository = self.repository
        if repository is None:
            return
        try:
            await repository.sweep_turn_leases(datetime.now(timezone.utc))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("commercial turn-lease sweep failed")

    async def _reconcile_janitor_loop(self) -> None:
        interval = self.settings.reconcile_interval_seconds
        while True:
            await asyncio.sleep(interval)
            await self.run_reconcile_sweep()

    async def shutdown(self) -> None:
        async with self._lock:
            janitor = self._janitor_task
            self._janitor_task = None
            repository = self.repository
            self.repository = None
            self.control_plane = None
            self.trial_plan = None
        if janitor is not None:
            janitor.cancel()
            with suppress(asyncio.CancelledError):
                await janitor
        if repository is not None:
            await repository.close()

    @property
    def trial_plan_version_id(self) -> str | None:
        return self.trial_plan.id if self.trial_plan is not None else None

    async def ensure_trial_for_owner(
        self,
        owner_id: str,
        *,
        claim_context: Mapping[str, str] | None = None,
    ) -> Subscription:
        control_plane, trial_plan = self._require_started()
        customer = await control_plane.ensure_billing_customer(owner_id=owner_id)
        return await control_plane.ensure_trial(
            customer.id,
            trial_plan.id,
            claim_context=claim_context,
        )

    async def resolve_for_owner(self, owner_id: str) -> ResolvedEntitlements | None:
        control_plane, _trial_plan = self._require_started()
        customer = await control_plane.get_billing_customer_by_owner(owner_id)
        if customer is None:
            return None
        return await control_plane.resolve_entitlements(customer.id)

    def _require_started(self) -> tuple[CommercialControlPlane, PlanVersion]:
        if self.control_plane is None or self.trial_plan is None:
            raise RuntimeError("commercial runtime has not completed startup")
        return self.control_plane, self.trial_plan


_runtime_singleton: CommercialRuntime | None = None


def get_commercial_runtime(
    environ: Mapping[str, str] | None = None,
) -> CommercialRuntime:
    """Return the process-wide pool/control-plane owner used by every entry point."""
    global _runtime_singleton
    if _runtime_singleton is None:
        _runtime_singleton = CommercialRuntime.from_env(environ)
    return _runtime_singleton


async def reset_commercial_runtime_for_tests() -> None:
    """Close and clear the singleton; intended for isolated app/test lifespans."""
    global _runtime_singleton
    runtime = _runtime_singleton
    _runtime_singleton = None
    if runtime is not None:
        await runtime.shutdown()


def _parse_bool(raw: str) -> bool:
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise CommercialConfigurationError(
        f"{COMMERCIAL_ENABLED_ENV} must be true/false, 1/0, yes/no, or on/off"
    )


def commercial_mode_requested(environ: Mapping[str, str] | None = None) -> bool:
    """Return the deployment's explicit commercial-mode switch.

    Security-sensitive modules import before FastAPI lifespan validation, so
    they need a lightweight way to choose fail-closed cookie/secret behavior
    without constructing the PostgreSQL runtime first.
    """

    values = os.environ if environ is None else environ
    return _parse_bool(values.get(COMMERCIAL_ENABLED_ENV, "false"))
