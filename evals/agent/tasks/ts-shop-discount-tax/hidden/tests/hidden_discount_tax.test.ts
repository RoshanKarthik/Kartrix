import assert from "node:assert/strict";
import { test } from "node:test";
import { cartTotal } from "../src/pricing.ts";

test("tax is charged after the percentage discount", () => {
  const t = cartTotal([{ quantity: 1, unitPriceCents: 5000 }], "SAVE10");
  assert.deepEqual(t, { subtotalCents: 5000, discountCents: 500, taxCents: 360, totalCents: 4860 });
});

test("tax is charged after a fixed discount", () => {
  const t = cartTotal([{ quantity: 2, unitPriceCents: 1500 }], "fiveoff");
  assert.deepEqual(t, { subtotalCents: 3000, discountCents: 500, taxCents: 200, totalCents: 2700 });
});

test("no coupon unchanged", () => {
  assert.equal(cartTotal([{ quantity: 1, unitPriceCents: 1000 }]).totalCents, 1080);
});
