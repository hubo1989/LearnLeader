"""Domain errors for the commercial control plane."""

from __future__ import annotations


class CommercialError(RuntimeError):
    """Base class for expected commercial control-plane failures."""


class CommercialConfigurationError(ValueError):
    """Commercial mode is configured in a way that cannot fail safely."""


class CommercialNotFound(CommercialError):
    """A referenced commercial entity does not exist."""


class CommercialConflict(CommercialError):
    """A durable commercial invariant would be violated."""


class ImmutablePlanVersion(CommercialConflict):
    """An existing plan version was republished with different content."""


class ImmutablePriceVersion(CommercialConflict):
    """An existing catalog row was republished with different content."""


class ActiveSubscriptionExists(CommercialConflict):
    """A customer already owns a trialing or active subscription."""


class TrialAlreadyClaimed(CommercialConflict):
    """The customer's one lifetime trial has already been claimed."""


class InvalidSubscriptionTransition(CommercialConflict):
    """A subscription status transition is not part of the state machine."""


class InvalidOrderTransition(CommercialConflict):
    """A purchase-order status transition is not part of the state machine."""


class PendingOrderExists(CommercialConflict):
    """The customer already has an open order awaiting payment."""


class IdempotencyConflict(CommercialConflict):
    """A deduplication key was reused for different semantics."""


class EntitlementDenied(CommercialConflict):
    """The customer is not entitled to the requested feature or quantity."""


class UsageAmountExceeded(CommercialConflict):
    """Final usage exceeds the amount reserved before work began."""


class MigrationDriftError(CommercialConflict):
    """An already-applied migration no longer has its recorded checksum."""
