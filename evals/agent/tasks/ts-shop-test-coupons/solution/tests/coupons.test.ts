import assert from "node:assert/strict";
import { test } from "node:test";
import { applyCoupon } from "../src/pricing.ts";

test("percentage coupon", () => {
  assert.equal(applyCoupon(5000, { code: "P", percentOff: 10 }), 500);
});

test("fixed coupon", () => {
  assert.equal(applyCoupon(3000, { code: "F", amountOffCents: 500 }), 500);
});

test("minimum subtotal", () => {
  const coupon = { code: "M", amountOffCents: 500, minSubtotalCents: 2000 };
  assert.equal(applyCoupon(1999, coupon), 0);
  assert.equal(applyCoupon(2000, coupon), 500);
});

test("discount never exceeds the subtotal", () => {
  assert.equal(applyCoupon(300, { code: "F", amountOffCents: 500 }), 300);
});

test("no coupon", () => {
  assert.equal(applyCoupon(1000), 0);
});
