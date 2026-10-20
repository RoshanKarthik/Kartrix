/** The chat view: connects to the Kartrix server, runs requests in the open folder's workspace and shows their
 * live feed — agent steps, tool calls, the streamed answer — with approvals and plan reviews answered in the view. */

import * as path from "node:path";
import * as vscode from "vscode";

import { ApiError, KartrixApi, Workspace, findWorkspace } from "./api";
import { Changes, ChangesHost, Connection } from "./checkpoints";
import { loadCredentials } from "./credentials";
import { ApprovalRequestView, FromView, ModelOption, Status, ToView, parseInput } from "./messages";
import { DEFAULT_SERVER, LocalServer, isLocal, reachable } from "./server";

export class ChatController implements vscode.WebviewViewProvider, ChangesHost {
  readonly changes = new Changes(this);
  private view: vscode.WebviewView | null = null;
  private api: KartrixApi | null = null;
  private workspace: Workspace | null = null;
  private status: Status = { state: "connecting", text: "Connecting…" };
  private models: ModelOption[] = [];
  private model: string | null = null;
  private newSession = false;
  private runId: string | null = null;
  private following: AbortController | null = null;
  private connecting: Promise<void> | null = null;

  constructor(
    private readonly context: vscode.ExtensionContext,
    private readonly server: LocalServer,
    private readonly output: vscode.OutputChannel,
  ) {
    this.model = context.workspaceState.get<string>("kartrix.model") ?? null;
  }

  resolveWebviewView(view: vscode.WebviewView): void {
    this.view = view;
    const media = vscode.Uri.joinPath(this.context.extensionUri, "dist");
    view.webview.options = { enableScripts: true, localResourceRoots: [media, vscode.Uri.joinPath(this.context.extensionUri, "media")] };
    view.webview.html = this.html(view.webview);
    view.webview.onDidReceiveMessage((message: FromView) => this.onMessage(message).catch((e) => this.fail(e)));
    view.onDidDispose(() => (this.view = null));
  }

  post(message: ToView): void {
    void this.view?.webview.postMessage(message);
  }

  get viewVisible(): boolean {
    return this.view?.visible ?? false;
  }

  private setStatus(status: Status): void {
    this.status = status;
    this.post({ type: "status", status });
  }

  private fail(e: unknown): void {
    const text = e instanceof Error ? e.message : String(e);
    this.output.appendLine(`error: ${text}`);
    this.post({ type: "error", text });
  }

  // ── connection ──────────────────────────────────────────────────────

  connect(): Promise<void> {
    this.connecting ??= this.doConnect().finally(() => (this.connecting = null));
    return this.connecting;
  }

  private async doConnect(): Promise<void> {
    const config = vscode.workspace.getConfiguration("kartrix");
    const creds = await loadCredentials();
    const server = (config.get<string>("serverUrl") || creds?.server || DEFAULT_SERVER).replace(/\/+$/, "");
    this.api = null;
    this.workspace = null;
    this.setStatus({ state: "connecting", text: `Connecting to ${server}…` });
    if (!(await reachable(server))) {
      if (!isLocal(server) || !config.get<boolean>("autoStartServer", true)) {
        this.setStatus({ state: "error", text: `Can't reach the Kartrix server at ${server}.` });
        return;
      }
      this.setStatus({ state: "connecting", text: "Starting the local Kartrix server (the first start takes a minute)…" });
      try {
        await this.server.start(config.get<string>("cliPath") || "kartrix", server);
      } catch (e) {
        this.setStatus({ state: "error", text: `${(e as Error).message}` });
        return;
      }
    }
    if (!creds || creds.server !== server) {
      this.setStatus({ state: "signed_out", server });
      return;
    }
    const folder = vscode.workspace.workspaceFolders?.[0];
    if (!folder || folder.uri.scheme !== "file") {
      this.setStatus({ state: "error", text: "Open a folder to work on it with Kartrix." });
      return;
    }
    const api = new KartrixApi(server, creds.token);
    try {
      const me = await api.me();
      this.workspace = await findWorkspace(api, folder.uri.fsPath, folder.name || path.basename(folder.uri.fsPath));
      this.api = api;
      this.setStatus({ state: "ready", server, email: me.email, workspace: this.workspace.name });
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) {
        this.setStatus({ state: "signed_out", server, text: "Your login has expired." });
      } else {
        this.setStatus({ state: "error", text: (e as Error).message });
      }
      return;
    }
    await this.loadModels();
  }

  private async loadModels(): Promise<void> {
    if (!this.api) return;
    try {
      const catalog = await this.api.models();
      this.models = catalog.models.map((m) => ({ id: m.id, name: m.name, available: m.available, reason: m.unavailable_reason }));
      if (!this.models.some((m) => m.id === this.model && m.available)) this.model = catalog.default;
      this.post({ type: "models", models: this.models, current: this.model });
    } catch (e) {
      this.fail(e);
    }
  }

  /** The server and workspace for the open folder; connects first if needed (a diff editor restored at startup). */
  async connection(): Promise<Connection | null> {
    if (this.connecting) await this.connecting;
    else if (!this.api) await this.connect();
    const folder = vscode.workspace.workspaceFolders?.[0];
    return this.api && this.workspace && folder ? { api: this.api, workspace: this.workspace, folder: folder.uri } : null;
  }

  // ── messages from the view ─────────────────────────────────────────

  private async onMessage(message: FromView): Promise<void> {
    switch (message.type) {
      case "ready":
        this.post({ type: "status", status: this.status });
        if (this.models.length) this.post({ type: "models", models: this.models, current: this.model });
        this.post({ type: "busy", busy: this.runId !== null });
        if (this.status.state === "connecting" && !this.connecting) await this.connect();
        return;
      case "send":
        return this.send(message.text);
      case "approval":
        await this.api?.answerApproval(
          message.runId,
          message.requestId,
          message.decisions.map((d) => ({ type: d.type, command: d.command ?? null, message: d.message ?? null })),
        );
        return;
      case "plan_review":
        await this.api?.answerPlanReview(message.runId, message.requestId, message.approved, message.feedback);
        return;
      case "stop":
        return this.stop();
      case "new":
        return this.newConversation();
      case "model":
        return this.pickModel(message.id);
      case "signIn":
        return this.signIn(message.signup);
      case "reconnect":
        return this.connect();
      case "open":
        return this.changes.open(message.path);
      case "diff":
        return this.changes.diff(message.entry, message.path, message.label);
      case "undo":
        return this.changes.step("undo", message.entry);
      case "redo":
        return this.changes.step("redo");
    }
  }

  async send(text: string): Promise<void> {
    const parsed = parseInput(text);
    if ("error" in parsed) {
      this.post({ type: "info", text: parsed.error });
      return;
    }
    if ("command" in parsed) {
      if (parsed.command === "checkpoints") return this.changes.showCheckpoints();
      return this.changes.step(parsed.command);
    }
    if (!this.api || !this.workspace) {
      this.post({ type: "error", text: "Not connected to a Kartrix server yet." });
      return;
    }
    if (this.runId) {
      this.post({ type: "info", text: "A request is still running — wait for it or stop it." });
      return;
    }
    this.post({ type: "user", text: parsed.input, kind: parsed.kind });
    const run = await this.api.startRun(this.workspace.id, parsed.kind, parsed.input, {
      newSession: this.newSession,
      model: this.model,
    });
    this.newSession = false;
    await this.follow(run.id);
  }

  private async follow(runId: string): Promise<void> {
    const api = this.api;
    if (!api) return;
    this.runId = runId;
    this.following = new AbortController();
    this.post({ type: "busy", busy: true });
    let reported = false;
    try {
      for await (const message of api.events(runId, this.following.signal)) {
        if (message.kind === "event") {
          reported ||= message.data.type === "run_finished";
          this.post({ type: "event", event: message.data });
        } else if (message.kind === "approval_needed") {
          this.post({
            type: "approval",
            runId,
            requestId: String(message.data.request_id),
            requests: (message.data.requests as ApprovalRequestView[]) ?? [],
          });
        } else if (message.kind === "plan_review_needed") {
          this.post({
            type: "plan_review",
            runId,
            requestId: String(message.data.request_id),
            plan: (message.data.plan as Record<string, unknown>) ?? {},
          });
        } else if (message.kind === "run_finished") {
          const status = String(message.data.status ?? "error");
          const detail = typeof message.data.detail === "string" ? message.data.detail : null;
          this.post({ type: "finished", status, detail, reported });
        }
      }
    } catch (e) {
      this.fail(e);
    } finally {
      this.runId = null;
      this.following = null;
      this.post({ type: "busy", busy: false });
    }
  }

  async stop(): Promise<void> {
    if (!this.api || !this.runId) return;
    try {
      await this.api.stop(this.runId);
      this.post({ type: "info", text: "Stopping…" });
    } catch (e) {
      if (!(e instanceof ApiError && e.status === 409)) throw e; // 409: it finished meanwhile
    }
  }

  newConversation(): void {
    this.newSession = true;
    this.post({ type: "cleared" });
  }

  async pickModel(id?: string): Promise<void> {
    if (!this.models.length) await this.loadModels();
    if (id === undefined) {
      const picked = await vscode.window.showQuickPick(
        this.models.map((m) => ({
          label: m.name,
          description: m.id === this.model ? "current" : m.available ? m.id : `unavailable: ${m.reason ?? ""}`,
          id: m.id,
          available: m.available,
        })),
        { placeHolder: "Model for the next requests" },
      );
      if (!picked) return;
      if (!picked.available) {
        void vscode.window.showWarningMessage(`${picked.label} isn't available here.`);
        return;
      }
      id = picked.id;
    }
    const model = this.models.find((m) => m.id === id);
    if (!model?.available) {
      this.post({ type: "error", text: `${id}: ${model?.reason ?? "unknown model"}` });
      return;
    }
    this.model = model.id;
    await this.context.workspaceState.update("kartrix.model", model.id);
    this.post({ type: "models", models: this.models, current: this.model });
  }

  /** `kartrix login` asks for the password in a terminal (never in the editor); the view reconnects after. */
  signIn(signup = false): void {
    const config = vscode.workspace.getConfiguration("kartrix");
    const cli = config.get<string>("cliPath") || "kartrix";
    const server = this.status.state === "signed_out" ? this.status.server : config.get<string>("serverUrl") || DEFAULT_SERVER;
    const terminal = vscode.window.createTerminal({ name: "Kartrix sign-in" });
    terminal.show();
    terminal.sendText(`${quote(cli)} login --server ${server}${signup ? " --signup" : ""}`);
    this.post({ type: "info", text: "Sign in in the terminal, then press Reconnect." });
  }

  dispose(): void {
    this.following?.abort();
  }

  private html(webview: vscode.Webview): string {
    const nonce = [...crypto.getRandomValues(new Uint8Array(16))].map((b) => b.toString(16).padStart(2, "0")).join("");
    const script = webview.asWebviewUri(vscode.Uri.joinPath(this.context.extensionUri, "dist", "webview.js"));
    const style = webview.asWebviewUri(vscode.Uri.joinPath(this.context.extensionUri, "media", "chat.css"));
    const csp = [
      "default-src 'none'",
      `style-src ${webview.cspSource}`,
      `script-src 'nonce-${nonce}'`,
      `img-src ${webview.cspSource} data:`,
    ].join("; ");
    return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="${csp}">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="${style}">
<title>Kartrix</title>
</head>
<body>
<div id="app"></div>
<script nonce="${nonce}" src="${script}"></script>
</body>
</html>`;
  }
}

function quote(value: string): string {
  return /\s/.test(value) ? `"${value}"` : value;
}
