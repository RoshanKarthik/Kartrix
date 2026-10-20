import assert from "node:assert/strict";
import { test } from "node:test";

import { ApiError, KartrixApi, findWorkspace } from "../src/api";
import { parseInput } from "../src/messages";
import { isLocal } from "../src/server";

type Call = { url: string; init?: RequestInit };

function fakeFetch(handler: (url: string, init?: RequestInit) => Response): { fetch: typeof fetch; calls: Call[] } {
  const calls: Call[] = [];
  const impl = async (input: string | URL | Request, init?: RequestInit) => {
    const url = String(input);
    calls.push({ url, init });
    return handler(url, init);
  };
  return { fetch: impl as typeof fetch, calls };
}

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

test("requests carry the token and JSON; errors carry the server's detail", async () => {
  const { fetch, calls } = fakeFetch((url) =>
    url.endsWith("/runs") ? json({ detail: "a run is already in progress in this workspace" }, 409) : json({ id: "r1" }),
  );
  const api = new KartrixApi("http://127.0.0.1:8765/", "tok", fetch);
  await api.answerPlanReview("r1", "q1", false, "smaller steps");
  assert.equal(calls[0].url, "http://127.0.0.1:8765/v1/runs/r1/plan-reviews/q1");
  assert.equal((calls[0].init?.headers as Record<string, string>).Authorization, "Bearer tok");
  assert.deepEqual(JSON.parse(String(calls[0].init?.body)), { approved: false, plan: null, feedback: "smaller steps" });
  await assert.rejects(api.startRun("w1", "ask", "hi"), (e: unknown) => e instanceof ApiError && e.status === 409 && /in progress/.test(e.detail));
});

test("an unreachable server is an ApiError with status 0", async () => {
  const api = new KartrixApi("http://127.0.0.1:1", "t", (async () => {
    throw new TypeError("fetch failed");
  }) as typeof fetch);
  await assert.rejects(api.me(), (e: unknown) => e instanceof ApiError && e.status === 0 && /can't reach/.test(e.detail));
});

test("the feed resumes after the last stored item when the stream drops", async () => {
  const first = 'id: 1\nevent: event\ndata: {"type":"notice","text":"one"}\n\nevent: event\ndata: {"type":"assistant_delta","text":"x"}\n\n';
  const second = 'id: 2\nevent: run_finished\ndata: {"status":"completed"}\n\n';
  const { fetch, calls } = fakeFetch((url) => new Response(url.endsWith("after=0") ? first : second));
  const api = new KartrixApi("http://s", "t", fetch);
  const kinds: string[] = [];
  for await (const message of api.events("r1")) kinds.push(`${message.kind}:${message.id}`);
  assert.deepEqual(kinds, ["event:1", "event:null", "run_finished:2"]);
  assert.deepEqual(
    calls.map((c) => c.url),
    ["http://s/v1/runs/r1/events?after=0", "http://s/v1/runs/r1/events?after=1"],
  );
});

test("the open folder's workspace is found by path (case-insensitive on Windows) or created", async () => {
  const workspaces = [{ id: "w1", name: "repo", root_path: "C:\\Work\\Repo" }];
  const { fetch, calls } = fakeFetch((_url, init) =>
    init?.method === "POST" ? json({ id: "w2", name: "new", root_path: "C:\\New" }, 201) : json(workspaces),
  );
  const api = new KartrixApi("http://s", "t", fetch);
  assert.equal((await findWorkspace(api, "c:\\work\\repo\\", "repo", "win32")).id, "w1");
  assert.equal((await findWorkspace(api, "C:\\New", "new", "win32")).id, "w2");
  assert.deepEqual(JSON.parse(String(calls.at(-1)?.init?.body)), { name: "new", path: "C:\\New" });
});

test("a hosted server is refused for local folders", async () => {
  const { fetch } = fakeFetch((_url, init) => (init?.method === "POST" ? json({ detail: "no" }, 403) : json([])));
  await assert.rejects(findWorkspace(new KartrixApi("http://s", "t", fetch), "/repo", "repo", "linux"), /local `kartrix serve`/);
});

test("input: plain text and /ask ask, /plan plans, /undo /redo /checkpoints run here, others are pointed to the terminal", () => {
  assert.deepEqual(parseInput("  what does slugify do?  "), { kind: "ask", input: "what does slugify do?" });
  assert.deepEqual(parseInput("/ask where is it tested?"), { kind: "ask", input: "where is it tested?" });
  assert.deepEqual(parseInput("/plan a todo API\nwith tests"), { kind: "plan", input: "a todo API\nwith tests" });
  assert.ok("error" in parseInput("/plan"));
  assert.deepEqual(parseInput("/undo"), { command: "undo" });
  assert.deepEqual(parseInput(" /checkpoints "), { command: "checkpoints" });
  assert.ok("error" in parseInput("/undo force"));
  assert.ok("error" in parseInput("/mode auto"));
  assert.ok("error" in parseInput("   "));
});

test("local server addresses", () => {
  assert.ok(isLocal("http://127.0.0.1:8765"));
  assert.ok(isLocal("http://localhost:9000"));
  assert.ok(!isLocal("https://kartrix.example.com"));
  assert.ok(!isLocal("not a url"));
});
