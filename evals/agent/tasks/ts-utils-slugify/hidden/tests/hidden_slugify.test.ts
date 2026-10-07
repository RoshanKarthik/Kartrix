import assert from "node:assert/strict";
import { test } from "node:test";
import { slugify } from "../src/slugify.ts";

test("accents removed, single dashes", () => {
  assert.equal(slugify("Crème brûlée, à la carte!"), "creme-brulee-a-la-carte");
  assert.equal(slugify("Ça va?  Très bien"), "ca-va-tres-bien");
  assert.equal(slugify("--Hello---World--"), "hello-world");
});

test("plain titles unchanged", () => {
  assert.equal(slugify("Hello World"), "hello-world");
  assert.equal(slugify("Release v2.0"), "release-v2-0");
});
