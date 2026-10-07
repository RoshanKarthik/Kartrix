import assert from "node:assert/strict";
import { test } from "node:test";
import { loadCatalog } from "../src/catalog.ts";
import { createApp } from "../src/router.ts";

test("unknown product is 404", () => {
  const res = createApp(loadCatalog())({ method: "GET", path: "/products/p999" });
  assert.equal(res.status, 404);
  assert.deepEqual(res.body, { error: "product not found" });
});

test("known product still works", () => {
  const res = createApp(loadCatalog())({ method: "GET", path: "/products/p2" });
  assert.equal(res.status, 200);
  assert.deepEqual(res.body, { id: "p2", name: "Pour-over kettle", priceCents: 4900 });
});
