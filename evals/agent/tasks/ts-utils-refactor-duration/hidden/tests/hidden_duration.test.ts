import assert from "node:assert/strict";
import { test } from "node:test";
import { parseDuration, tokenizeDuration } from "../src/duration.ts";

test("tokenizeDuration", () => {
  assert.deepEqual(tokenizeDuration("1h30m"), [[1, "h"], [30, "m"]]);
  assert.deepEqual(tokenizeDuration(" 2d 1.5s "), [[2, "d"], [1.5, "s"]]);
  assert.deepEqual(tokenizeDuration("250ms"), [[250, "ms"]]);
  assert.throws(() => tokenizeDuration("3 weeks"));
  assert.throws(() => tokenizeDuration(""));
});

test("parseDuration unchanged", () => {
  assert.equal(parseDuration("1h30m"), 5400);
  assert.equal(parseDuration("1d 2h 3m 4s"), 93784);
  assert.equal(parseDuration("1500ms"), 2);
  assert.equal(parseDuration("2H"), 7200);
  assert.throws(() => parseDuration("10 apples"));
});

test("parseDuration uses tokenizeDuration", async () => {
  const { readFileSync } = await import("node:fs");
  const src = readFileSync(new URL("../src/duration.ts", import.meta.url), "utf8");
  const body = src.slice(src.indexOf("export function parseDuration"));
  assert.match(body, /tokenizeDuration\(/);
});
