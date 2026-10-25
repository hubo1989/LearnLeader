"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import {
  cancelCommercialOrder,
  createCommercialOrder,
  fetchCommercialOrders,
  fetchCommercialPlans,
} from "./api";
import type { CommercialOrder, CommercialPlan } from "./types";
import { formatMinorAmount } from "@/lib/currency";

const formatPrice = formatMinorAmount;

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString();
}

export default function OrdersSection() {
  const { t } = useTranslation();
  const [plans, setPlans] = useState<CommercialPlan[]>([]);
  const [orders, setOrders] = useState<CommercialOrder[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const [planList, orderList] = await Promise.all([
        fetchCommercialPlans(),
        fetchCommercialOrders(),
      ]);
      setPlans(planList.filter((plan) => plan.priceMinor > 0));
      setOrders(orderList);
      setError(null);
      // Drop idempotency keys whose order is no longer open: the server
      // treats a stable key as "retry this order", so a fresh purchase
      // after cancel/fulfill must mint a fresh key.
      const openPlanIds = new Set(
        orderList
          .filter((order) => order.status === "pending_payment")
          .map((order) => order.planVersionId),
      );
      for (const key of orderKeys.current.keys()) {
        if (!openPlanIds.has(key)) orderKeys.current.delete(key);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // One idempotency key per plan until that order leaves pending: a
  // network retry reuses it and lands on the same server-side order.
  const orderKeys = useRef<Map<string, string>>(new Map());

  const pendingOrder = useMemo(
    () => orders.find((order) => order.status === "pending_payment") ?? null,
    [orders],
  );

  async function handleOrder(plan: CommercialPlan) {
    setBusy(true);
    try {
      let key = orderKeys.current.get(plan.id);
      if (!key) {
        key =
          typeof crypto !== "undefined" && "randomUUID" in crypto
            ? `web-${plan.id}-${crypto.randomUUID()}`
            : `web-${plan.id}-${Math.random().toString(36).slice(2)}`;
        orderKeys.current.set(plan.id, key);
      }
      await createCommercialOrder({ planVersionId: plan.id, idempotencyKey: key });
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function handleCancel(orderId: string) {
    setBusy(true);
    try {
      await cancelCommercialOrder(orderId);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold">{t("Plans and orders")}</h2>
        <p className="mt-1 text-sm text-[var(--muted-foreground)]">
          {t(
            "Orders are paid offline. Create an order, then note its id with your transfer so it can be confirmed.",
          )}
        </p>
      </div>

      {error ? (
        <div className="rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-700 dark:border-red-900/60 dark:bg-red-950/30 dark:text-red-400">
          {error}
        </div>
      ) : null}

      <div className="grid gap-3 sm:grid-cols-2">
        {plans.map((plan) => (
          <div
            key={plan.id}
            className="rounded-lg border border-[var(--border)] bg-[var(--card)] p-4"
          >
            <div className="flex items-baseline justify-between">
              <span className="font-medium">{plan.name}</span>
              <span className="text-sm">{formatPrice(plan.priceMinor, plan.currency)}</span>
            </div>
            <p className="mt-1 text-xs text-[var(--muted-foreground)]">
              {t("per {{interval}}", { interval: plan.billingInterval })}
            </p>
            <button
              type="button"
              disabled={busy || pendingOrder !== null}
              onClick={() => void handleOrder(plan)}
              className="mt-3 rounded-md border border-[var(--border)] px-3 py-1.5 text-sm disabled:opacity-50"
            >
              {pendingOrder
                ? t("An order is already awaiting payment")
                : t("Create order")}
            </button>
          </div>
        ))}
        {plans.length === 0 ? (
          <p className="text-sm text-[var(--muted-foreground)]">
            {t("No purchasable plans yet.")}
          </p>
        ) : null}
      </div>

      {pendingOrder ? (
        <div className="rounded-lg border border-amber-300 bg-amber-50 p-4 text-sm dark:border-amber-900/60 dark:bg-amber-950/30">
          <p className="font-medium">{t("Awaiting payment confirmation")}</p>
          <p className="mt-1 text-xs text-[var(--muted-foreground)]">
            {t("Order id")}{" "}
            <code className="rounded bg-[var(--muted)] px-1">{pendingOrder.id}</code>
            {" · "}
            {formatPrice(pendingOrder.priceMinor, pendingOrder.currency)}
            {" · "}
            {formatDate(pendingOrder.periodStart)} → {formatDate(pendingOrder.periodEnd)}
          </p>
          <button
            type="button"
            disabled={busy}
            onClick={() => void handleCancel(pendingOrder.id)}
            className="mt-2 rounded-md border border-[var(--border)] px-3 py-1 text-xs disabled:opacity-50"
          >
            {t("Cancel order")}
          </button>
        </div>
      ) : null}

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-[var(--border)] text-left text-xs text-[var(--muted-foreground)]">
              <th className="py-2 pr-4">{t("Order")}</th>
              <th className="py-2 pr-4">{t("Status")}</th>
              <th className="py-2 pr-4">{t("Period")}</th>
              <th className="py-2 pr-4">{t("Created")}</th>
            </tr>
          </thead>
          <tbody>
            {orders.map((order) => (
              <tr key={order.id} className="border-b border-[var(--border)]/60">
                <td className="py-2 pr-4 font-mono text-xs">{order.id}</td>
                <td className="py-2 pr-4">{order.status}</td>
                <td className="py-2 pr-4">
                  {formatDate(order.periodStart)} → {formatDate(order.periodEnd)}
                </td>
                <td className="py-2 pr-4">{formatDate(order.createdAt)}</td>
              </tr>
            ))}
            {orders.length === 0 ? (
              <tr>
                <td colSpan={4} className="py-3 text-[var(--muted-foreground)]">
                  {t("No orders yet.")}
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </div>
  );
}
