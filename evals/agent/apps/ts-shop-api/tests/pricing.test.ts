import assert from "node:assert/strict";
import { test } from "node:test";
import { cartTotal } from "../src/pricing.ts";

test("totals without a coupon", () => {
  const t = cartTotal([{ quantity: 3, unitPriceCents: 1000 }]);
  assert.deepEqual(t, { subtotalCents: 3000, discountCents: 0, taxCents: 240, totalCents: 3240 });
});

test("empty cart", () => {
  assert.equal(cartTotal([]).totalCents, 0);
});
