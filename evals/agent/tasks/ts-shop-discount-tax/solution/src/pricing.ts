export interface Coupon {
  code: string;
  percentOff?: number; // e.g. 10 for 10 %
  amountOffCents?: number;
  minSubtotalCents?: number; // the coupon only applies from this subtotal on
}

export interface PricedLine {
  quantity: number;
  unitPriceCents: number;
}

export interface Totals {
  subtotalCents: number;
  discountCents: number;
  taxCents: number;
  totalCents: number;
}

export const TAX_RATE = 0.08;

export const COUPONS: Record<string, Coupon> = {
  SAVE10: { code: "SAVE10", percentOff: 10 },
  FIVEOFF: { code: "FIVEOFF", amountOffCents: 500, minSubtotalCents: 2000 },
};

/** The discount (in cents) a coupon gives on a subtotal: 0 below its minimum, never more than the subtotal. */
export function applyCoupon(subtotalCents: number, coupon?: Coupon): number {
  if (!coupon) return 0;
  if (coupon.minSubtotalCents !== undefined && subtotalCents < coupon.minSubtotalCents) return 0;
  let discount = 0;
  if (coupon.percentOff !== undefined) discount = Math.round((subtotalCents * coupon.percentOff) / 100);
  if (coupon.amountOffCents !== undefined) discount += coupon.amountOffCents;
  return Math.min(discount, subtotalCents);
}

/** Subtotal, coupon discount, tax and total. Tax is charged on the discounted subtotal. */
export function cartTotal(lines: PricedLine[], couponCode?: string, taxRate: number = TAX_RATE): Totals {
  const subtotalCents = lines.reduce((sum, l) => sum + l.quantity * l.unitPriceCents, 0);
  const coupon = couponCode ? COUPONS[couponCode.toUpperCase()] : undefined;
  const discountCents = applyCoupon(subtotalCents, coupon);
  const taxCents = Math.round((subtotalCents - discountCents) * taxRate);
  return { subtotalCents, discountCents, taxCents, totalCents: subtotalCents - discountCents + taxCents };
}
