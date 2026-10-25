"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import { fetchAdminOrders, markOrderPaid } from "@/features/commercial/api";
import {
  COMMERCIAL_ORDER_STATUSES,
  type CommercialOrder,
  type CommercialOrderStatus,
} from "@/features/commercial/types";
import { formatMinorAmount } from "@/lib/currency";

export default function AdminCommercialPage() {
  const { t } = useTranslation();
  const [orders, setOrders] = useState<CommercialOrder[]>([]);
  const [statusFilter, setStatusFilter] = useState<CommercialOrderStatus | "">( "pending_payment" );
  const [references, setReferences] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [hasMore, setHasMore] = useState(false);

  // A slow response for an old filter must not overwrite the current one.
  const requestSeq = useRef(0);

  const refresh = useCallback(async () => {
    const seq = ++requestSeq.current;
    try {
      const fetched = await fetchAdminOrders(statusFilter || undefined, 100, 0);
      if (seq !== requestSeq.current) return;
      setOrders(fetched);
      setHasMore(fetched.length >= 100);
      setError(null);
    } catch (err) {
      if (seq !== requestSeq.current) return;
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [statusFilter]);

  const loadMore = useCallback(async () => {
    const seq = ++requestSeq.current;
    try {
      const fetched = await fetchAdminOrders(
        statusFilter || undefined,
        100,
        orders.length,
      );
      if (seq !== requestSeq.current) return;
      setOrders((current) => [...current, ...fetched]);
      setHasMore(fetched.length >= 100);
    } catch (err) {
      if (seq !== requestSeq.current) return;
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [statusFilter, orders.length]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function handleMarkPaid(orderId: string) {
    setBusy(true);
    try {
      await markOrderPaid(orderId, references[orderId] || undefined);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto max-w-4xl space-y-4 p-6">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-semibold">{t("Orders")}</h1>
        <select
          value={statusFilter}
          onChange={(event) =>
            setStatusFilter(event.target.value as CommercialOrderStatus | "")
          }
          className="rounded-md border border-[var(--border)] bg-[var(--background)] px-2 py-1 text-sm"
        >
          <option value="">{t("All statuses")}</option>
          {COMMERCIAL_ORDER_STATUSES.map((status) => (
            <option key={status} value={status}>
              {t(status)}
            </option>
          ))}
        </select>
      </div>

      {error ? (
        <div className="rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-700 dark:border-red-900/60 dark:bg-red-950/30 dark:text-red-400">
          {error}
        </div>
      ) : null}

      <div className="overflow-x-auto rounded-lg border border-[var(--border)]">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-[var(--border)] text-left text-xs text-[var(--muted-foreground)]">
              <th className="p-3">{t("Order")}</th>
              <th className="p-3">{t("Status")}</th>
              <th className="p-3">{t("Amount")}</th>
              <th className="p-3">{t("Created")}</th>
              <th className="p-3">{t("Actions")}</th>
            </tr>
          </thead>
          <tbody>
            {orders.map((order) => (
              <tr key={order.id} className="border-b border-[var(--border)]/60">
                <td className="p-3 font-mono text-xs">{order.id}</td>
                <td className="p-3">{order.status}</td>
                <td className="p-3">
                  {formatMinorAmount(order.priceMinor, order.currency)}
                </td>
                <td className="p-3">{new Date(order.createdAt).toLocaleString()}</td>
                <td className="p-3">
                  {order.status === "pending_payment" ? (
                    <div className="flex items-center gap-2">
                      <input
                        value={references[order.id] ?? ""}
                        onChange={(event) =>
                          setReferences((current) => ({
                            ...current,
                            [order.id]: event.target.value,
                          }))
                        }
                        placeholder={t("Payment reference")}
                        className="w-36 rounded-md border border-[var(--border)] bg-[var(--background)] px-2 py-1 text-xs"
                      />
                      <button
                        type="button"
                        disabled={busy}
                        onClick={() => void handleMarkPaid(order.id)}
                        className="rounded-md border border-[var(--border)] px-2 py-1 text-xs disabled:opacity-50"
                      >
                        {t("Confirm payment")}
                      </button>
                    </div>
                  ) : (
                    <span className="text-xs text-[var(--muted-foreground)]">—</span>
                  )}
                </td>
              </tr>
            ))}
            {hasMore ? (
              <tr>
                <td colSpan={5} className="p-3 text-center">
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => void loadMore()}
                    className="rounded-md border border-[var(--border)] px-3 py-1 text-xs disabled:opacity-50"
                  >
                    {t("Load more")}
                  </button>
                </td>
              </tr>
            ) : null}
            {orders.length === 0 ? (
              <tr>
                <td colSpan={5} className="p-3 text-[var(--muted-foreground)]">
                  {t("No orders in this view.")}
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </div>
  );
}
