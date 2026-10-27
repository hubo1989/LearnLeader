-- DeepTutor commercial risk & periods v4: turn leases, trial abuse signals,
-- and the SUSPENDED subscription state.  Replayable like 0001-0003; the
-- runner records and verifies its checksum.

-- 1. Durable concurrent-turn leases (owner-scoped, TTL expiry).
CREATE TABLE IF NOT EXISTS commercial.turn_lease (
    id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    capability TEXT,
    turn_id TEXT,
    acquired_at TIMESTAMPTZ NOT NULL,
    heartbeat_at TIMESTAMPTZ NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT turn_lease_expiry_order CHECK (expires_at > acquired_at)
);

CREATE INDEX IF NOT EXISTS turn_lease_owner_idx
    ON commercial.turn_lease (owner_id, expires_at);

CREATE INDEX IF NOT EXISTS turn_lease_expiry_idx
    ON commercial.turn_lease (expires_at);

-- 2. Trial claim context (sha256 hashes only, never raw values).
ALTER TABLE commercial.billing_customer
    ADD COLUMN IF NOT EXISTS trial_claim_context JSONB;

-- 3. SUSPENDED subscription status: widen the status check constraint and
--    the live-per-customer unique index, then teach the transition trigger.
ALTER TABLE commercial.subscription DROP CONSTRAINT IF EXISTS subscription_status_check;

ALTER TABLE commercial.subscription
    ADD CONSTRAINT subscription_status_check CHECK (
        status IN ('trialing', 'active', 'past_due', 'suspended', 'canceled', 'expired')
    );

DROP INDEX IF EXISTS subscription_one_live_per_customer_uidx;

CREATE UNIQUE INDEX IF NOT EXISTS subscription_one_live_per_customer_uidx
    ON commercial.subscription (customer_id)
    WHERE status IN ('trialing', 'active', 'past_due', 'suspended');

CREATE OR REPLACE FUNCTION commercial.validate_subscription_transition()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.status = NEW.status THEN
        RETURN NEW;
    END IF;
    IF NOT (
        (OLD.status = 'trialing' AND NEW.status IN ('expired', 'canceled'))
        OR (OLD.status = 'active'
            AND NEW.status IN ('past_due', 'suspended', 'canceled'))
        OR (OLD.status = 'past_due' AND NEW.status IN ('active', 'canceled'))
        OR (OLD.status = 'suspended' AND NEW.status IN ('active', 'canceled'))
    ) THEN
        RAISE EXCEPTION 'invalid subscription transition: % -> %', OLD.status, NEW.status
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NEW;
END;
$$;

-- 4. Keep the trial claim immutable after first write; the new context
--    column is part of the protected claim.
CREATE OR REPLACE FUNCTION commercial.protect_trial_claim()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.trial_subscription_id IS NOT NULL
       AND (
           NEW.trial_subscription_id IS DISTINCT FROM OLD.trial_subscription_id
           OR NEW.trial_claimed_at IS DISTINCT FROM OLD.trial_claimed_at
           OR NEW.trial_claim_context IS DISTINCT FROM OLD.trial_claim_context
       ) THEN
        RAISE EXCEPTION 'lifetime trial claim is immutable'
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NEW;
END;
$$;

-- 5. Billing-period resolution and budget gating look up fulfilled orders by
--    subscription and period; without this index each budget-gated reserve
--    and each reconcile LATERAL query scans the growing order table.
CREATE INDEX IF NOT EXISTS purchase_order_fulfilled_period_idx
    ON commercial.purchase_order (subscription_id, period_start, period_end)
    WHERE status = 'fulfilled';
