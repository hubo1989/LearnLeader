import { apiFetch, apiUrl } from "@/lib/api";

import { parseCommercialAccessSnapshot } from "./access";
import type { CommercialAccessSnapshot, CommercialOrder, CommercialPlan } from "./types";

export async function fetchCommercialAccess(): Promise<CommercialAccessSnapshot> {
  const response = await apiFetch(apiUrl("/api/v1/commercial/me"), {
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(`Commercial access probe failed (${response.status})`);
  }

  const parsed = parseCommercialAccessSnapshot(await response.json());
  if (!parsed) throw new Error("Commercial access probe returned invalid data");
  return parsed;
}

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `request failed (${response.status})`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* keep status-derived message */
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

function mapPlan(raw: Record<string, unknown>): CommercialPlan {
  return {
    id: String(raw.id),
    planCode: String(raw.plan_code),
    version: Number(raw.version),
    name: String(raw.name),
    priceMinor: Number(raw.price_minor),
    currency: String(raw.currency),
    billingInterval: String(raw.billing_interval),
    trialDays: Number(raw.trial_days),
  };
}

function mapOrder(raw: Record<string, unknown>): CommercialOrder {
  return {
    id: String(raw.id),
    orderType: (raw.order_type ?? "subscription_new") as CommercialOrder["orderType"],
    planVersionId: String(raw.plan_version_id),
    priceMinor: Number(raw.price_minor),
    currency: String(raw.currency),
    status: raw.status as CommercialOrder["status"],
    periodStart: String(raw.period_start),
    periodEnd: String(raw.period_end),
    paidAt: (raw.paid_at as string | null) ?? null,
    fulfilledAt: (raw.fulfilled_at as string | null) ?? null,
    canceledAt: (raw.canceled_at as string | null) ?? null,
    subscriptionId: (raw.subscription_id as string | null) ?? null,
    createdAt: String(raw.created_at),
  };
}

export async function fetchCommercialPlans(): Promise<CommercialPlan[]> {
  const raw = await readJson<Record<string, unknown>[]>(
    await apiFetch(apiUrl("/api/v1/commercial/plans"), { cache: "no-store" }),
  );
  return raw.map(mapPlan);
}

export async function fetchCommercialOrders(
  status?: string,
): Promise<CommercialOrder[]> {
  const query = status ? `?status=${encodeURIComponent(status)}` : "";
  const raw = await readJson<Record<string, unknown>[]>(
    await apiFetch(apiUrl(`/api/v1/commercial/orders${query}`), { cache: "no-store" }),
  );
  return raw.map(mapOrder);
}

export async function createCommercialOrder(input: {
  planVersionId: string;
  orderType?: string;
  idempotencyKey: string;
}): Promise<CommercialOrder> {
  const raw = await readJson<Record<string, unknown>>(
    await apiFetch(apiUrl("/api/v1/commercial/orders"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        plan_version_id: input.planVersionId,
        order_type: input.orderType ?? "subscription_new",
        idempotency_key: input.idempotencyKey,
      }),
    }),
  );
  return mapOrder(raw);
}

export async function cancelCommercialOrder(orderId: string): Promise<CommercialOrder> {
  const raw = await readJson<Record<string, unknown>>(
    await apiFetch(apiUrl(`/api/v1/commercial/orders/${orderId}/cancel`), {
      method: "POST",
    }),
  );
  return mapOrder(raw);
}

export async function markOrderPaid(
  orderId: string,
  paymentReference?: string,
): Promise<CommercialOrder> {
  const raw = await readJson<Record<string, unknown>>(
    await apiFetch(apiUrl(`/api/v1/admin/commercial/orders/${orderId}/mark-paid`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ payment_reference: paymentReference ?? null }),
    }),
  );
  return mapOrder(raw);
}

export async function fetchAdminOrders(
  status?: string,
  limit = 100,
  offset = 0,
): Promise<CommercialOrder[]> {
  const params = new URLSearchParams();
  if (status) params.set("status", status);
  params.set("limit", String(limit));
  params.set("offset", String(offset));
  const raw = await readJson<Record<string, unknown>[]>(
    await apiFetch(
      apiUrl(`/api/v1/admin/commercial/orders?${params.toString()}`),
      { cache: "no-store" },
    ),
  );
  return raw.map(mapOrder);
}
