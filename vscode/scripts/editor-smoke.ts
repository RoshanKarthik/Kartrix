/** Runs inside a real VS Code extension host (`--extensionTestsPath`, see run-editor-smoke.mjs): activates Kartrix,
 * opens its side bar and drives the chat controller through the same messages the view sends — a question, an
 * approval round trip and a plan review — against a live `kartrix serve`. Needs a login (`kartrix login`) and a model
 * key; writes what it saw to KARTRIX_SMOKE_REPORT. */

import { writeFileSync } from "node:fs";
import * as vscode from "vscode";

import type { ChatController } from "../src/chat";
import type { FromView, ToView } from "../src/messages";

/** The controller as this test drives it (`post` and `onMessage` are private to the extension). */
interface Driven {
  post(message: ToView): void;
  onMessage(message: FromView): Promise<void>;
  send(text: string): Promise<void>;
  stop(): Promise<void>;
  newConversation(): void;
  api: { command(workspaceId: string, name: string, args: Record<string, unknown>): Promise<{ result: Record<string, unknown> }> };
  workspace: { id: string; root_path: string };
}

let approveEverything = false; // the change step: approve what the agent asks, like a click on Approve

const report: { step: string; ok: boolean; detail: string }[] = [];
const seen: ToView[] = [];

function note(step: string, ok: boolean, detail = ""): void {
  report.push({ step, ok, detail });
  console.log(`${ok ? "PASS" : "FAIL"} ${step}${detail ? ` — ${detail}` : ""}`);
}

async function until<T>(what: string, find: () => T | undefined, timeoutMs: number): Promise<T> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const found = find();
    if (found !== undefined) return found;
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error(`timed out waiting for ${what}`);
}

function since(start: number): ToView[] {
  return seen.slice(start);
}

export async function run(): Promise<void> {
  try {
    await smoke();
  } catch (e) {
    note("smoke", false, (e as Error).message);
  } finally {
    const out = process.env.KARTRIX_SMOKE_REPORT;
    if (out) writeFileSync(out, JSON.stringify({ report, seen: seen.map(summarize) }, null, 2));
  }
  if (report.some((r) => !r.ok)) throw new Error("editor smoke test failed");
}

function summarize(m: ToView): string {
  if (m.type === "event") return `event:${String(m.event.type)}`;
  if (m.type === "status") return `status:${m.status.state}`;
  if (m.type === "error" || m.type === "info") return `${m.type}:${m.text}`;
  if (m.type === "finished") return `finished:${m.status}`;
  return m.type;
}

async function smoke(): Promise<void> {
  const extension = vscode.extensions.getExtension<ChatController>("roshankarthik.kartrix");
  if (!extension) throw new Error("the Kartrix extension isn't loaded");
  const chat = (await extension.activate()) as unknown as Driven;
  note("activates and returns its controller", typeof chat.send === "function");

  const post = chat.post.bind(chat);
  chat.post = (message: ToView) => {
    seen.push(message);
    post(message);
    if (approveEverything && message.type === "approval") {
      void chat.onMessage({
        type: "approval",
        runId: message.runId,
        requestId: message.requestId,
        decisions: message.requests.map(() => ({ type: "approve" as const })),
      });
    }
  };
  const commands = await vscode.commands.getCommands(true);
  const ours = ["kartrix.newConversation", "kartrix.stop", "kartrix.pickModel", "kartrix.signIn", "kartrix.reconnect"];
  note("registers its commands", ours.every((c) => commands.includes(c)), ours.filter((c) => !commands.includes(c)).join(", "));

  // Opening the side bar resolves the webview; its script says "ready", which connects.
  await vscode.commands.executeCommand("workbench.view.extension.kartrix");
  await vscode.commands.executeCommand("kartrix.chat.focus"); // the view itself, even if its section is collapsed
  const status = await until(
    "the view to connect",
    () => seen.find((m) => m.type === "status" && m.status.state !== "connecting"),
    180_000,
  );
  const state = status.type === "status" ? status.status : null;
  note("the view loads and connects", state?.state === "ready", JSON.stringify(state));
  if (state?.state !== "ready") return;
  const models = await until("the model list", () => seen.find((m) => m.type === "models"), 30_000);
  note("loads the model catalog", models.type === "models" && models.models.some((m) => m.available));

  // 1. A question: live feed to the answer.
  let start = seen.length;
  chat.newConversation();
  await chat.send("In one sentence: what does slugify in app.py do?");
  const kinds = since(start).map(summarize);
  note(
    "a question streams to an answer",
    kinds.includes("event:assistant_message") && kinds.includes("finished:completed"),
    [...new Set(kinds)].join(" "),
  );

  // 2. A change: the agent edits app.py (what it asks is approved), the file changes on disk, undo restores it.
  const file = vscode.Uri.joinPath(vscode.Uri.file(chat.workspace.root_path), "app.py");
  const before = new TextDecoder().decode(await vscode.workspace.fs.readFile(file));
  start = seen.length;
  approveEverything = true;
  await chat.send('In app.py add a function shout(text) that returns text.upper() + "!". Only edit the file; do not run any commands.');
  approveEverything = false;
  const after = new TextDecoder().decode(await vscode.workspace.fs.readFile(file));
  const changed = since(start).find((m) => m.type === "event" && m.event.type === "files_changed");
  note(
    "a change request edits the file",
    after !== before && /def shout/.test(after) && changed !== undefined,
    `${since(start).filter((m) => m.type === "approval").length} approval(s); files_changed: ${changed?.type === "event" ? JSON.stringify(changed.event.files) : "none"}`,
  );
  const undo = await chat.api.command(chat.workspace.id, "undo", { force: false });
  const restored = new TextDecoder().decode(await vscode.workspace.fs.readFile(file));
  note("undo restores it", restored === before, JSON.stringify((undo.result.result as { restored?: string[] } | null)?.restored ?? undo.result));

  // 3. An approval: the agent wants to run a command; the view's Reject goes back to the server.
  start = seen.length;
  const asking = chat.send(
    "Run this exact command in the repository and tell me what it printed: python -c \"print(6 * 7)\"",
  );
  const approval = await until("an approval card", () => since(start).find((m) => m.type === "approval"), 300_000).catch(() => undefined);
  if (approval?.type === "approval") {
    await chat.onMessage({
      type: "approval",
      runId: approval.runId,
      requestId: approval.requestId,
      decisions: approval.requests.map(() => ({ type: "reject" as const, message: "editor smoke test: not now" })),
    });
  }
  await asking;
  const resolved = since(start).find((m) => m.type === "event" && m.event.type === "approval_resolved");
  note(
    "an approval card is answered from the editor",
    approval !== undefined && resolved !== undefined,
    approval?.type === "approval" ? approval.requests.map((r) => r.command).join(" | ") : "no approval was asked",
  );

  // 4. /plan: the plan card, sent back with feedback, then stopped.
  start = seen.length;
  const planning = chat.send("/plan add a shout(text) function to app.py that returns text.upper() + '!'");
  const review = await until("a plan card", () => since(start).find((m) => m.type === "plan_review"), 400_000).catch(() => undefined);
  if (review?.type === "plan_review") {
    await chat.onMessage({ type: "plan_review", runId: review.runId, requestId: review.requestId, approved: false, feedback: "Keep it to one task." });
    await until("the re-plan to start", () => since(start).find((m) => m.type === "event" && m.event.type === "plan_reviewed"), 60_000).catch(() => undefined);
    await chat.stop();
  }
  await planning;
  const reviewed = since(start).find((m) => m.type === "event" && m.event.type === "plan_reviewed");
  note(
    "a plan card is reviewed from the editor",
    review !== undefined && reviewed !== undefined,
    review?.type === "plan_review" ? `${(review.plan.tasks as unknown[] | undefined)?.length ?? 0} tasks` : "no plan review",
  );
  note("the run ends after Stop", since(start).some((m) => m.type === "busy" && !m.busy));
}
