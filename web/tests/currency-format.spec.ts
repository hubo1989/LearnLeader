import { describe, expect, it } from 'vitest'

import { currencyExponent, formatMinorAmount } from '@/lib/currency'

describe('formatMinorAmount', () => {
  it('keeps two decimals for common hundredth-unit currencies', () => {
    expect(formatMinorAmount(4900, 'CNY')).toBe('49.00 CNY')
    expect(formatMinorAmount(1599, 'usd')).toBe('15.99 USD')
  })

  it('does not invent decimals for zero-decimal currencies', () => {
    expect(formatMinorAmount(4900, 'JPY')).toBe('4,900 JPY')
    expect(currencyExponent('KRW')).toBe(0)
  })

  it('renders three decimals where the minor unit is a thousandth', () => {
    expect(formatMinorAmount(4900, 'KWD')).toBe('4.900 KWD')
  })
})
