import assert from "node:assert/strict";
import { test } from "node:test";

import { SseParser } from "../src/sse";

test("messages split across chunks, ids, live-only items and pings", () => {
  const parser = new SseParser();
  assert.deepEqual(parser.push('id: 1\nevent: event\ndata: {"type":"notice",'), []);
  assert.deepEqual(parser.push('"text":"hi"}\n\n: ping\n\nevent: event\ndata: {"type":"assistant_delta"}\n\n'), [
    { id: 1, kind: "event", data: { type: "notice", text: "hi" } },
    { id: null, kind: "event", data: { type: "assistant_delta" } },
  ]);
});

test("CRLF line endings and multi-line data", () => {
  const parser = new SseParser();
  assert.deepEqual(parser.push('id: 7\r\nevent: run_finished\r\ndata: {"status":\r\ndata: "completed"}\r\n\r\n'), [
    { id: 7, kind: "run_finished", data: { status: "completed" } },
  ]);
});

test("malformed or non-object data is skipped", () => {
  const parser = new SseParser();
  assert.deepEqual(parser.push("event: event\ndata: {nope\n\nevent: event\ndata: [1,2]\n\nevent: x\n\n"), []);
});
