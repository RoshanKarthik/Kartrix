import assert from "node:assert/strict";
import { test } from "node:test";
import { paginate, parseDuration, slugify } from "../src/index.ts";

test("slugify basic titles", () => {
  assert.equal(slugify("Hello World"), "hello-world");
  assert.equal(slugify("  Release v2.0  "), "release-v2-0");
});

test("parseDuration", () => {
  assert.equal(parseDuration("1h30m"), 5400);
  assert.equal(parseDuration("2d"), 172800);
  assert.equal(parseDuration("1500ms"), 2);
  assert.throws(() => parseDuration("soon"));
});

test("paginate", () => {
  const items = [1, 2, 3, 4, 5];
  assert.deepEqual(paginate(items, 2, 2), { items: [3, 4], page: 2, pageCount: 3 });
  assert.deepEqual(paginate(items, 4, 2).items, []);
  assert.throws(() => paginate(items, 0, 2));
});
