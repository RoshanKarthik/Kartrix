import assert from "node:assert/strict";
import { test } from "node:test";
import { loadCatalog } from "../src/catalog.ts";
import { createApp } from "../src/router.ts";

const app = () => createApp(loadCatalog());

test("lists products", () => {
  const res = app()({ method: "GET", path: "/products" });
  assert.equal(res.status, 200);
  assert.equal((res.body as unknown[]).length, 4);
});

test("gets one product", () => {
  const res = app()({ method: "GET", path: "/products/p1" });
  assert.deepEqual(res.body, { id: "p1", name: "Ceramic mug", priceCents: 1250 });
});

test("adds to the cart and totals it", () => {
  const handle = app();
  assert.equal(handle({ method: "POST", path: "/cart/items", body: { productId: "p1", quantity: 2 } }).status, 201);
  const res = handle({ method: "GET", path: "/cart" });
  const body = res.body as { subtotalCents: number; taxCents: number; totalCents: number };
  assert.equal(body.subtotalCents, 2500);
  assert.equal(body.taxCents, 200);
  assert.equal(body.totalCents, 2700);
});

test("unknown product cannot be added", () => {
  const res = app()({ method: "POST", path: "/cart/items", body: { productId: "nope" } });
  assert.equal(res.status, 404);
});

test("gets product p3", () => {
  const res = app()({ method: "GET", path: "/products/p3" });
  assert.deepEqual(res.body, { id: "p3", name: "Coffee beans", priceCents: 1599 });
});
