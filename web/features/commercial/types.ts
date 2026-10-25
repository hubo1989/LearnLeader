export const COMMERCIAL_STATUSES = [
  'disabled',
  'admin',
  'trialing',
  'active',
  'past_due',
  'canceled',
  'expired',
] as const

export type CommercialStatus = (typeof COMMERCIAL_STATUSES)[number] | null

/** One quota meter's consumption inside its current window. */
export type CommercialMeterUsage = {
  limit: number | null
  window: 'utc_day' | 'subscription'
  used: number
  reserved: number
}

/** Public, credential-free commercial state returned for the signed-in user. */
export type CommercialAccessSnapshot = {
  enabled: boolean
  isAdmin: boolean
  status: CommercialStatus
  validUntil: string | null
  planVersionId: string | null
  values: Record<string, unknown>
  /** Per-meter usage; absent on older payloads. */
  usage: Record<string, CommercialMeterUsage>
}

/** One published plan version with its price. */
export type CommercialPlan = {
  id: string
  planCode: string
  version: number
  name: string
  priceMinor: number
  currency: string
  billingInterval: string
  trialDays: number
}

export const COMMERCIAL_ORDER_STATUSES = [
  'pending_payment',
  'paid',
  'fulfilled',
  'canceled',
  'refunded',
] as const

export type CommercialOrderStatus = (typeof COMMERCIAL_ORDER_STATUSES)[number]

/** A purchase order snapshotted at creation (price and period included). */
export type CommercialOrder = {
  id: string
  orderType: 'subscription_new' | 'subscription_renew'
  planVersionId: string
  priceMinor: number
  currency: string
  status: CommercialOrderStatus
  periodStart: string
  periodEnd: string
  paidAt: string | null
  fulfilledAt: string | null
  canceledAt: string | null
  subscriptionId: string | null
  createdAt: string
}
