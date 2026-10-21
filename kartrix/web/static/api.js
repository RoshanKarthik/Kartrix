/** The Kartrix server's HTTP API and a run's SSE feed (kartrix/server/app.py), called the way the VS Code extension
 * (vscode/src/api.ts) and the CLI (kartrix/client/api.py) call them. The page and the API share one origin. */

export class ApiError extends Error {
  constructor(status, detail) {
    super(detail);
    this.status = status;
    this.detail = detail;
  }
}

export class Api {
  constructor(token = null, fetchImpl = (...args) => fetch(...args)) {
    this.token = token;
    this.fetch = fetchImpl;
  }

  async call(method, path, body, signal) {
    const headers = {};
    if (this.token) headers.Authorization = `Bearer ${this.token}`;
    if (body !== undefined) headers["Content-Type"] = "application/json";
    let response;
    try {
      response = await this.fetch(`/v1${path}`, {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
        cache: "no-store",
        signal,
      });
    } catch (e) {
      if (signal?.aborted) throw e;
      throw new ApiError(0, "Can't reach the Kartrix server — is `kartrix serve` still running?");
    }
    if (!response.ok) throw new ApiError(response.status, await detail(response));
    const text = await response.text();
    return text ? JSON.parse(text) : undefined;
  }

  health() {
    return this.call("GET", "/health");
  }

  login(email, password) {
    return this.call("POST", "/auth/login", { email, password, token_name: "web" });
  }

  signup(email, password) {
    return this.call("POST", "/auth/signup", { email, password, token_name: "web" });
  }

  logout() {
    return this.call("POST", "/auth/logout");
  }

  me() {
    return this.call("GET", "/me");
  }

  workspaces() {
    return this.call("GET", "/workspaces");
  }

  createWorkspace(name, path) {
    return this.call("POST", "/workspaces", { name, path: path || null });
  }

  /** The workspace's latest runs, newest first. */
  runs(workspaceId, limit = 20) {
    return this.call("GET", `/workspaces/${encodeURIComponent(workspaceId)}/runs?limit=${limit}`);
  }

  startRun(workspaceId, kind, input, { newSession = false, model = null } = {}) {
    return this.call("POST", `/workspaces/${encodeURIComponent(workspaceId)}/runs`, {
      kind,
      input,
      new_session: newSession,
      model,
    });
  }

  run(runId) {
    return this.call("GET", `/runs/${encodeURIComponent(runId)}`);
  }

  stop(runId) {
    return this.call("POST", `/runs/${encodeURIComponent(runId)}/stop`);
  }

  /** The run's feed until it ends ({id, kind, data} messages); a dropped connection resumes after the last stored
   * item (`?after=`). A fetch rather than EventSource, so the token goes in a header, never in the URL. */
  async *events(runId, signal) {
    let after = 0;
    for (let attempt = 0; attempt < 5; attempt++) {
      let response;
      try {
        response = await this.fetch(`/v1/runs/${encodeURIComponent(runId)}/events?after=${after}`, {
          headers: { Authorization: `Bearer ${this.token}`, Accept: "text/event-stream" },
          cache: "no-store",
          signal,
        });
      } catch {
        if (signal?.aborted) return;
        await sleep(500 * (attempt + 1));
        continue;
      }
      if (!response.ok || !response.body) throw new ApiError(response.status, await detail(response));
      const parser = new SseParser();
      const decoder = new TextDecoder();
      const reader = response.body.getReader();
      try {
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          for (const message of parser.push(decoder.decode(value, { stream: true }))) {
            if (message.id !== null) after = message.id;
            yield message;
            if (message.kind === "run_finished") return;
          }
        }
      } catch {
        if (signal?.aborted) return;
      } finally {
        reader.releaseLock();
      }
      if (signal?.aborted) return;
      // The stream ended without run_finished (a restart, a proxy timeout): resume after the last stored item.
    }
    throw new ApiError(0, "Lost the connection to the run's feed.");
  }
}

/** Server-Sent Events parsed from text chunks; comments (`: ping`) and malformed data are skipped. */
export class SseParser {
  constructor() {
    this.buffer = "";
  }

  push(chunk) {
    this.buffer += chunk.replace(/\r\n?/g, "\n");
    const messages = [];
    let end;
    while ((end = this.buffer.indexOf("\n\n")) !== -1) {
      const message = parseBlock(this.buffer.slice(0, end));
      this.buffer = this.buffer.slice(end + 2);
      if (message) messages.push(message);
    }
    return messages;
  }
}

function parseBlock(block) {
  let id = null;
  let kind = "message";
  const data = [];
  for (const line of block.split("\n")) {
    if (!line || line.startsWith(":")) continue;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    const value = colon === -1 ? "" : line.slice(colon + 1).replace(/^ /, "");
    if (field === "id" && /^\d+$/.test(value)) id = Number(value);
    else if (field === "event") kind = value;
    else if (field === "data") data.push(value);
  }
  if (data.length === 0) return null;
  try {
    const parsed = JSON.parse(data.join("\n"));
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) return null;
    return { id, kind, data: parsed };
  } catch {
    return null;
  }
}

async function detail(response) {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((d) => d?.msg ?? String(d)).join("; ");
  } catch {
    // not JSON
  }
  return response.statusText || `HTTP ${response.status}`;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** What the composer sends: plain text asks. `/plan` needs plan review and other slash commands run elsewhere for
 * now, so they get a hint instead of a run. */
export function parseInput(text) {
  const trimmed = text.trim();
  if (!trimmed) return { error: "Type a request." };
  const match = /^\/(\w+)(?:\s+([\s\S]*))?$/.exec(trimmed);
  if (!match) return { kind: "ask", input: trimmed };
  const [, command, rest = ""] = match;
  if (command === "ask") return rest.trim() ? { kind: "ask", input: rest.trim() } : { error: "Usage: /ask <question>" };
  if (command === "plan") {
    return { error: "/plan needs plan review, which the browser can't do yet — run it in the terminal or VS Code." };
  }
  return { error: `/${command} isn't available in the browser yet — run it in the terminal (kartrix) or VS Code.` };
}
