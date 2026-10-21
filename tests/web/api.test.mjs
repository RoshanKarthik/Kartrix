// The web UI's client logic (kartrix/web/static/api.js), run by tests/unit/test_web_ui.py with `node --test`.

import assert from "node:assert/strict";
import { test } from "node:test";

import { Api, ApiError, SseParser, parseInput } from "../../kartrix/web/static/api.js";

test("SSE blocks split across chunks, with ids, comments and bad data", () => {
  const parser = new SseParser();
  assert.deepEqual(parser.push('id: 1\nevent: event\ndata: {"type":"run_st'), []);
  const out = parser.push('arted"}\r\n\r\n: ping\n\nevent: event\ndata: {"type":"assistant_delta","text":"hi"}\n\ndata: [1]\n\ndata: nope\n\n');
  assert.deepEqual(out, [
    { id: 1, kind: "event", data: { type: "run_started" } },
    { id: null, kind: "event", data: { type: "assistant_delta", text: "hi" } },
  ]);
});

test("input: plain text asks; /plan and other commands get a hint, not a run", () => {
  assert.deepEqual(parseInput("  what does app.py do?\n"), { kind: "ask", input: "what does app.py do?" });
  assert.deepEqual(parseInput("/ask why?"), { kind: "ask", input: "why?" });
  assert.match(parseInput("").error, /Type a request/);
  assert.match(parseInput("/ask").error, /Usage/);
  assert.match(parseInput("/plan a shop").error, /plan review/);
  assert.match(parseInput("/undo").error, /isn't available in the browser/);
  assert.deepEqual(parseInput("/path/to/file is broken?"), { kind: "ask", input: "/path/to/file is broken?" });
});

function stream(text) {
  return new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(text));
      controller.close();
    },
  });
}

test("calls send the token as a bearer header and turn errors into ApiError", async () => {
  const seen = [];
  const api = new Api("kx_secret", async (url, init) => {
    seen.push({ url, init });
    if (url === "/v1/me") return new Response(JSON.stringify({ email: "a@b.co" }), { status: 200 });
    return new Response(JSON.stringify({ detail: "a run is already in progress in this workspace" }), { status: 409 });
  });
  assert.deepEqual(await api.me(), { email: "a@b.co" });
  assert.equal(seen[0].init.headers.Authorization, "Bearer kx_secret");
  await assert.rejects(api.startRun("w1", "ask", "hi"), (e) => e instanceof ApiError && e.status === 409 && /in progress/.test(e.message));
  assert.deepEqual(JSON.parse(seen[1].init.body), { kind: "ask", input: "hi", new_session: false, model: null });
  assert.equal(seen[1].url, "/v1/workspaces/w1/runs");
  const down = new Api(null, async () => {
    throw new TypeError("Failed to fetch");
  });
  await assert.rejects(down.health(), (e) => e instanceof ApiError && e.status === 0 && /kartrix serve/.test(e.message));
});

test("the feed resumes after the last stored item and ends at run_finished", async () => {
  const urls = [];
  const bodies = [
    'id: 1\nevent: event\ndata: {"type":"run_started"}\n\nevent: event\ndata: {"type":"assistant_delta","text":"4"}\n\n',
    'id: 2\nevent: event\ndata: {"type":"assistant_message","text":"4"}\n\nid: 3\nevent: run_finished\ndata: {"status":"completed"}\n\nid: 4\nevent: event\ndata: {}\n\n',
  ];
  const api = new Api("kx_t", async (url, init) => {
    urls.push(url);
    assert.equal(init.headers.Authorization, "Bearer kx_t");
    return new Response(stream(bodies.shift()), { status: 200, headers: { "Content-Type": "text/event-stream" } });
  });
  const got = [];
  for await (const message of api.events("r1")) got.push([message.id, message.kind, message.data.type ?? message.data.status]);
  assert.deepEqual(urls, ["/v1/runs/r1/events?after=0", "/v1/runs/r1/events?after=1"]);
  assert.deepEqual(got, [
    [1, "event", "run_started"],
    [null, "event", "assistant_delta"],
    [2, "event", "assistant_message"],
    [3, "run_finished", "completed"],
  ]);
});

test("a feed that can't be read fails with the server's reason", async () => {
  const api = new Api("kx_t", async () => new Response(JSON.stringify({ detail: "run not found" }), { status: 404 }));
  await assert.rejects(
    (async () => {
      for await (const _ of api.events("nope"));
    })(),
    (e) => e instanceof ApiError && e.status === 404 && e.message === "run not found",
  );
});
