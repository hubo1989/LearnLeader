-- DeepTutor commercial orders v2: purchase orders with manual fulfillment.
-- Replayable like 0001; the runner records and verifies its checksum.

CREATE TABLE IF NOT EXISTS commercial.purchase_order (
    id TEXT PRIMARY KEY,
    dedupe_key CHAR(71) NOT NULL UNIQUE,
    customer_id TEXT NOT NULL
        REFERENCES commercial.billing_customer(id) ON DELETE RESTRICT,
    plan_version_id TEXT NOT NULL
        REFERENCES commercial.plan_version(id) ON DELETE RESTRICT,
    order_type TEXT NOT NULL CHECK (
        order_type IN ('subscription_new', 'subscription_renew')
    ),
    -- Price and period are snapshotted at creation so republished plan
    -- versions cannot drift an order that is already awaiting payment.
    price_minor BIGINT NOT NULL CHECK (price_minor >= 0),
    currency CHAR(3) NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('pending_payment', 'paid', 'fulfilled', 'canceled', 'refunded')
    ),
    provider TEXT NOT NULL,
    external_order_id TEXT,
    subscription_id TEXT,
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    paid_at TIMESTAMPTZ,
    fulfilled_at TIMESTAMPTZ,
    canceled_at TIMESTAMPTZ,
    refunded_at TIMESTAMPTZ,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT purchase_order_period_order CHECK (period_end > period_start),
    CONSTRAINT purchase_order_paid_shape CHECK (
        (status IN ('pending_payment', 'canceled') AND paid_at IS NULL)
        OR (status NOT IN ('pending_payment', 'canceled') AND paid_at IS NOT NULL)
    ),
    CONSTRAINT purchase_order_fulfilled_shape CHECK (
        (status = 'fulfilled' AND fulfilled_at IS NOT NULL AND subscription_id IS NOT NULL)
        OR (status <> 'fulfilled' AND fulfilled_at IS NULL AND subscription_id IS NULL)
    ),
    CONSTRAINT purchase_order_canceled_shape CHECK (
        (status = 'canceled' AND canceled_at IS NOT NULL) OR canceled_at IS NULL
    ),
    CONSTRAINT purchase_order_refunded_shape CHECK (
        (status = 'refunded' AND refunded_at IS NOT NULL) OR refunded_at IS NULL
    ),
    CONSTRAINT purchase_order_provider_shape CHECK (
        (provider = 'manual' AND external_order_id IS NULL)
        OR (provider <> 'manual')
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS purchase_order_provider_external_uidx
    ON commercial.purchase_order (provider, external_order_id)
    WHERE external_order_id IS NOT NULL;

-- One open order per customer: repeated checkout clicks reuse the same
-- pending row instead of piling up payment instructions.
CREATE UNIQUE INDEX IF NOT EXISTS purchase_order_one_pending_uidx
    ON commercial.purchase_order (customer_id)
    WHERE status = 'pending_payment';

CREATE INDEX IF NOT EXISTS purchase_order_customer_idx
    ON commercial.purchase_order (customer_id, created_at, id);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'purchase_order_subscription_fk'
          AND conrelid = 'commercial.purchase_order'::regclass
    ) THEN
        ALTER TABLE commercial.purchase_order
            ADD CONSTRAINT purchase_order_subscription_fk
            FOREIGN KEY (subscription_id)
            REFERENCES commercial.subscription(id)
            ON DELETE NO ACTION
            DEFERRABLE INITIALLY DEFERRED;
    END IF;
END
$$;

CREATE OR REPLACE FUNCTION commercial.validate_purchase_order_transition()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.status = NEW.status THEN
        RETURN NEW;
    END IF;
    IF NOT (
        (OLD.status = 'pending_payment'
            AND NEW.status IN ('paid', 'fulfilled', 'canceled'))
        OR (OLD.status = 'paid' AND NEW.status IN ('fulfilled', 'refunded'))
    ) THEN
        RAISE EXCEPTION 'invalid purchase order transition: % -> %', OLD.status, NEW.status
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NEW;
END;
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgname = 'purchase_order_state_machine'
          AND tgrelid = 'commercial.purchase_order'::regclass
    ) THEN
        CREATE TRIGGER purchase_order_state_machine
            BEFORE UPDATE OF status ON commercial.purchase_order
            FOR EACH ROW EXECUTE FUNCTION commercial.validate_purchase_order_transition();
    END IF;
END
$$;
