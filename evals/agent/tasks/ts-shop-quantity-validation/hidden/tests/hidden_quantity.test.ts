import assert from "node:assert/strict";
import { test } from "node:test";
import { loadCatalog } from "../src/catalog.ts";
import { createApp } from "../src/router.ts";

const error = { error: "quantity must be a whole number from 1 to 99" };

for (const quantity of [0, -3, 1.5, "2", 100, null, true]) {
  test(`rejects quantity ${JSON.stringify(quantity)}`, () => {
    const res = createApp(loadCatalog())({ method: "POST", path: "/cart/items", body: { productId: "p1", quantity } });
    assert.equal(res.status, 400);
    assert.deepEqual(res.body, error);
  });
}

test("accepts 1, 99 and a missing quantity", () => {
  const app = createApp(loadCatalog());
  assert.equal(app({ method: "POST", path: "/cart/items", body: { productId: "p1", quantity: 1 } }).status, 201);
  assert.equal(app({ method: "POST", path: "/cart/items", body: { productId: "p2", quantity: 99 } }).status, 201);
  assert.equal(app({ method: "POST", path: "/cart/items", body: { productId: "p3" } }).status, 201);
});
