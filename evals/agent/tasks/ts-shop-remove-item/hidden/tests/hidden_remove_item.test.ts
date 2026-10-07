import assert from "node:assert/strict";
import { test } from "node:test";
import { loadCatalog } from "../src/catalog.ts";
import { createApp } from "../src/router.ts";

test("removes a line", () => {
  const app = createApp(loadCatalog());
  app({ method: "POST", path: "/cart/items", body: { productId: "p1", quantity: 2 } });
  app({ method: "POST", path: "/cart/items", body: { productId: "p4", quantity: 1 } });
  const res = app({ method: "DELETE", path: "/cart/items/p1" });
  assert.equal(res.status, 204);
  assert.ok(res.body === undefined || res.body === null || res.body === "");
  const cart = app({ method: "GET", path: "/cart" }).body as { lines: { productId: string }[]; subtotalCents: number };
  assert.deepEqual(cart.lines.map((l) => l.productId), ["p4"]);
  assert.equal(cart.subtotalCents, 450);
});

test("not in cart is 404", () => {
  const res = createApp(loadCatalog())({ method: "DELETE", path: "/cart/items/p2" });
  assert.equal(res.status, 404);
  assert.deepEqual(res.body, { error: "not in cart" });
});
