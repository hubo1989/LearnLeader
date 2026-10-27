"use client";

import { ClockAlert } from "lucide-react";
import Link from "next/link";
import { useTranslation } from "react-i18next";

import type { CommercialStatus } from "./types";

function inactiveCopyPrefix(status: CommercialStatus): string {
  if (status === "past_due") return "commercial.inactive.pastDue";
  if (status === "suspended") return "commercial.inactive.suspended";
  if (status === "canceled") return "commercial.inactive.canceled";
  if (status === "trialing" || status === "expired") {
    return "commercial.inactive.trialExpired";
  }
  return "commercial.inactive.subscriptionRequired";
}

export function TrialExpiredNotice({
  status,
}: {
  status: CommercialStatus;
}) {
  const { t } = useTranslation();
  const copyPrefix = inactiveCopyPrefix(status);

  return (
    <div className="flex h-full w-full items-center justify-center overflow-y-auto p-6">
      <div className="flex max-w-lg flex-col items-center gap-4 rounded-2xl border border-[var(--border)] bg-[var(--secondary)]/40 px-8 py-10 text-center">
        <div className="flex h-12 w-12 items-center justify-center rounded-full bg-amber-500/10 text-amber-700 dark:text-amber-300">
          <ClockAlert aria-hidden="true" size={21} strokeWidth={1.8} />
        </div>
        <h2 className="text-base font-semibold text-[var(--foreground)]">
          {t(`${copyPrefix}.title`)}
        </h2>
        <p className="text-sm leading-relaxed text-[var(--muted-foreground)]">
          {t(`${copyPrefix}.description`)}
        </p>
        <p className="text-xs leading-relaxed text-[var(--muted-foreground)]">
          {t("commercial.inactive.dataSafe")}
        </p>
        <div className="mt-1 flex flex-wrap justify-center gap-2">
          <Link
            href="/space/chat-history"
            className="rounded-lg border border-[var(--border)] bg-[var(--card)] px-3 py-2 text-xs font-medium text-[var(--foreground)] transition-colors hover:bg-[var(--muted)]"
          >
            {t("commercial.inactive.openHistory")}
          </Link>
          <Link
            href="/profile#account-data"
            className="rounded-lg border border-[var(--border)] bg-[var(--card)] px-3 py-2 text-xs font-medium text-[var(--foreground)] transition-colors hover:bg-[var(--muted)]"
          >
            {t("commercial.inactive.openAccountData")}
          </Link>
        </div>
      </div>
    </div>
  );
}
