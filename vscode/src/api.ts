/** The Kartrix server's HTTP API (kartrix/server/app.py), as the CLI's client uses it (kartrix/client/api.py). */

import { SseMessage, SseParser } from "./sse";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
  ) {
    super(detail);
  }
}

export interface Workspace {
  id: string;
  name: string;
  root_path: string;
}

export interface Run {
  id: string;
  status: string;
  detail: string | null;
  answer: string | null;
}

export interface ModelInfo {
  id: string;
  name: string;
  provider: string;
  tier: string;
  description: string;
  default: boolean;
  available: boolean;
  unavailable_reason: string | null;
  trains_on_inputs: boolean;
}

export interface Catalog {
  default: string;
  models: ModelInfo[];
}

export interface Decision {
  type: "approve" | "approve_session" | "edit" | "reject";
  command?: string | null;
  message?: string | null;
}

type Fetch = typeof fetch;

export class KartrixApi {
  readonly server: string;

  constructor(
    server: string,
    private readonly token: string,
    private readonly fetchImpl: Fetch = fetch,
  ) {
    this.server = server.replace(/\/+$/, "");
  }

  private async call<T>(method: string, path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
    let response: Response;
    try {
      response = await this.fetchImpl(`${this.server}/v1${path}`, {
        method,
        headers: { Authorization: `Bearer ${this.token}`, ...(body === undefined ? {} : { "Content-Type": "application/json" }) },
        body: body === undefined ? undefined : JSON.stringify(body),
        signal,
      });
    } catch (e) {
      throw new ApiError(0, `can't reach the Kartrix server at ${this.server} (${(e as Error).message})`);
    }
    if (!response.ok) throw new ApiError(response.status, await detail(response));
    const text = await response.text();
    return (text ? JSON.parse(text) : undefined) as T;
  }

  health(): Promise<{ name: string; allow_local_paths: boolean }> {
    return this.call("GET", "/health", undefined, AbortSignal.timeout(3000));
  }

  me(): Promise<{ id: string; email: string }> {
    return this.call("GET", "/me");
  }

  models(): Promise<Catalog> {
    return this.call("GET", "/models");
  }

  workspaces(): Promise<Workspace[]> {
    return this.call("GET", "/workspaces");
  }

  createWorkspace(name: string, path: string | null): Promise<Workspace> {
    return this.call("POST", "/workspaces", { name, path });
  }

  startRun(
    workspaceId: string,
    kind: "ask" | "plan",
    input: string,
    options: { newSession?: boolean; model?: string | null } = {},
  ): Promise<Run> {
    return this.call("POST", `/workspaces/${workspaceId}/runs`, {
      kind,
      input,
      new_session: options.newSession ?? false,
      model: options.model ?? null,
    });
  }

  run(runId: string): Promise<Run> {
    return this.call("GET", `/runs/${runId}`);
  }

  answerApproval(runId: string, requestId: string, decisions: Decision[]): Promise<void> {
    return this.call("POST", `/runs/${runId}/approvals/${requestId}`, { decisions });
  }

  answerPlanReview(
    runId: string,
    requestId: string,
    approved: boolean,
    feedback = "",
    plan: Record<string, unknown> | null = null,
  ): Promise<void> {
    return this.call("POST", `/runs/${runId}/plan-reviews/${requestId}`, { approved, plan, feedback });
  }

  stop(runId: string): Promise<void> {
    return this.call("POST", `/runs/${runId}/stop`);
  }

  command(workspaceId: string, name: string, args: Record<string, unknown>): Promise<{ name: string; result: Record<string, unknown> }> {
    return this.call("POST", `/workspaces/${workspaceId}/commands`, { name, args });
  }

  /** The run's feed until it ends; a dropped connection resumes after the last stored item (`?after=`). */
  async *events(runId: string, signal?: AbortSignal): AsyncGenerator<SseMessage> {
    let after = 0;
    for (let attempt = 0; attempt < 5; attempt++) {
      let response: Response;
      try {
        response = await this.fetchImpl(`${this.server}/v1/runs/${runId}/events?after=${after}`, {
          headers: { Authorization: `Bearer ${this.token}`, Accept: "text/event-stream" },
          signal,
        });
      } catch (e) {
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
      } catch (e) {
        if (signal?.aborted) return;
      } finally {
        reader.releaseLock();
      }
      if (signal?.aborted) return;
      // The stream ended without run_finished (server restart, proxy timeout): resume after the last item.
    }
    throw new ApiError(0, "lost the connection to the run's feed");
  }
}

async function detail(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((d) => (d as { msg?: string }).msg ?? String(d)).join("; ");
  } catch {
    // not JSON
  }
  return response.statusText || `HTTP ${response.status}`;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function samePath(a: string, b: string, platform: string): boolean {
  const norm = (p: string) => {
    let out = p.replace(/[\\/]+/g, "/").replace(/\/$/, "");
    if (platform === "win32") out = out.toLowerCase();
    return out;
  };
  return norm(a) === norm(b);
}

/** The server workspace for `folder`, registered on first use (like the CLI's find_workspace). A hosted server
 * (no local folders) is refused: the extension works on the files open in the editor. */
export async function findWorkspace(api: KartrixApi, folder: string, name: string, platform: string = process.platform): Promise<Workspace> {
  const existing = await api.workspaces();
  const found = existing.find((ws) => samePath(ws.root_path, folder, platform));
  if (found) return found;
  try {
    return await api.createWorkspace(name, folder);
  } catch (e) {
    if (e instanceof ApiError && e.status === 403) {
      throw new ApiError(403, "this server doesn't work on folders of your machine — sign in to a local `kartrix serve`");
    }
    throw e;
  }
}
