import { describe, expect, it, vi, beforeEach } from "vitest";

const rawOrder = {
  id: "ord_1",
  order_type: "subscription_new",
  plan_version_id: "planv_1",
  price_minor: 4900,
  currency: "CNY",
  status: "pending_payment",
  period_start: "2026-08-05T09:00:00Z",
  period_end: "2026-09-05T09:00:00Z",
  paid_at: null,
  fulfilled_at: null,
  canceled_at: null,
  subscription_id: null,
  created_at: "2026-08-05T09:00:00Z",
};

const ok = (body: unknown) =>
  Promise.resolve({
    ok: true,
    status: 200,
    json: () => Promise.resolve(body),
  }) as unknown as Promise<Response>;

describe("commercial order api client", () => {
  beforeEach(() => {
    vi.resetModules();
    const fetchMock = vi.fn().mockImplementation((_url: string, init?: RequestInit) => {
      if (init?.method === "POST") return ok(rawOrder);
      return ok([rawOrder]);
    });
    vi.stubGlobal("fetch", fetchMock);
  });

  it("maps snake_case order payloads to camelCase types", async () => {
    const { fetchCommercialOrders } = await import("@/features/commercial/api");
    const orders = await fetchCommercialOrders();
    expect(orders).toHaveLength(1);
    expect(orders[0]).toMatchObject({
      id: "ord_1",
      orderType: "subscription_new",
      planVersionId: "planv_1",
      priceMinor: 4900,
      status: "pending_payment",
      subscriptionId: null,
    });
  });

  it("surfaces the server's detail message on failure", async () => {
    const { createCommercialOrder } = await import("@/features/commercial/api");
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        Promise.resolve({
          ok: false,
          status: 409,
          json: () => Promise.resolve({ detail: "cancel or fulfill the open order" }),
        }) as unknown as Promise<Response>,
      ),
    );
    await expect(
      createCommercialOrder({ planVersionId: "planv_1", idempotencyKey: "k" }),
    ).rejects.toThrow("cancel or fulfill the open order");
  });
});
