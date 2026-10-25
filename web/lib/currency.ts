/** Currencies whose minor unit is not 1/100 (ISO 4217 exponents). */
const ZERO_DECIMAL = new Set(['JPY', 'KRW', 'VND', 'CLP', 'ISK', 'UGX', 'TZS'])
const THREE_DECIMAL = new Set(['BHD', 'IQD', 'JOD', 'KWD', 'OMR', 'TND'])

export function currencyExponent(currency: string): number {
  const code = currency.trim().toUpperCase()
  if (THREE_DECIMAL.has(code)) return 3
  if (ZERO_DECIMAL.has(code)) return 0
  return 2
}

/**
 * Format a minor-unit amount for display (and offline transfers): the
 * returned string is the exact payable amount, so JPY must not gain two
 * phantom decimal places.
 */
export function formatMinorAmount(priceMinor: number, currency: string): string {
  const exponent = currencyExponent(currency)
  const units = priceMinor / 10 ** exponent
  return `${units.toLocaleString(undefined, {
    minimumFractionDigits: exponent === 0 ? 0 : exponent === 3 ? 3 : 2,
    maximumFractionDigits: exponent,
  })} ${currency.trim().toUpperCase()}`
}
