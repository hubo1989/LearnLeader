-- DeepTutor commercial pricing v3: versioned, immutable price catalog.
-- Replayable like 0001/0002; the runner records and verifies its checksum.
--
-- Two ledgers live in one table, separated by ``kind``:
--   platform_cost — what a metered unit costs the platform (budget gate,
--                   finalize-time cost attribution on usage_event).
--   list_price    — what a metered unit is sold for (billing preview).
-- Rows are immutable after publication: price changes require a new
-- ``price_version`` or a later ``effective_from`` row, never an UPDATE.

CREATE TABLE IF NOT EXISTS commercial.price_version (
    id TEXT PRIMARY KEY,
    price_version TEXT NOT NULL CHECK (length(btrim(price_version)) > 0),
    kind TEXT NOT NULL CHECK (kind IN ('platform_cost', 'list_price')),
    meter TEXT NOT NULL CHECK (length(btrim(meter)) > 0),
    model_tier TEXT NOT NULL CHECK (length(btrim(model_tier)) > 0),
    -- Micros of ``currency`` charged per one unit of ``meter`` for the
    -- given tier.  BIGINT keeps token-level catalogs exact.
    unit_price_micros BIGINT NOT NULL CHECK (unit_price_micros >= 0),
    currency CHAR(3) NOT NULL,
    effective_from TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT price_version_unique_row UNIQUE (price_version, kind, meter, model_tier)
);

-- Resolution scans (kind, meter, model_tier) backwards by effective_from.
CREATE INDEX IF NOT EXISTS price_version_lookup_idx
    ON commercial.price_version (kind, meter, model_tier, effective_from DESC);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgname = 'price_version_immutable'
          AND tgrelid = 'commercial.price_version'::regclass
    ) THEN
        CREATE TRIGGER price_version_immutable
            BEFORE UPDATE OR DELETE ON commercial.price_version
            FOR EACH ROW EXECUTE FUNCTION commercial.prevent_plan_version_mutation();
    END IF;
END
$$;

-- Budget accounting sums raw micros across the platform_cost ledger, so a
-- second currency in that ledger would compare unrelated amounts.  The
-- application pre-checks with a friendly error; this trigger closes the
-- concurrent-publication window (updates are already blocked as immutable).
CREATE OR REPLACE FUNCTION commercial.enforce_price_ledger_currency()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM commercial.price_version
        WHERE kind = NEW.kind
          AND currency <> NEW.currency
    ) THEN
        RAISE EXCEPTION 'price ledger % must stay single-currency', NEW.kind
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NEW;
END;
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgname = 'price_version_ledger_currency'
          AND tgrelid = 'commercial.price_version'::regclass
    ) THEN
        CREATE TRIGGER price_version_ledger_currency
            BEFORE INSERT ON commercial.price_version
            FOR EACH ROW EXECUTE FUNCTION commercial.enforce_price_ledger_currency();
    END IF;
END
$$;
