/** The chat view (runs in the webview). Everything from the agent is rendered with textContent — never as HTML. */

import type { ApprovalRequestView, DecisionView, FromView, ModelOption, Status, ToView } from "../src/messages";

declare function acquireVsCodeApi(): { postMessage(message: FromView): void };

const vscode = acquireVsCodeApi();
const send = (message: FromView) => vscode.postMessage(message);

// ── tiny DOM helpers ──────────────────────────────────────────────────

type Child = Node | string | null | undefined | false;

function h<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Record<string, string> = {}, ...children: Child[]): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") el.className = value;
    else el.setAttribute(key, value);
  }
  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    el.append(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return el;
}

function button(label: string, onClick: () => void, cls = ""): HTMLButtonElement {
  const b = h("button", { class: cls, type: "button" }, label);
  b.addEventListener("click", onClick);
  return b;
}

/** A small, safe subset of Markdown: fenced code, headings, lists, paragraphs, `code` and **bold**. */
function markdown(text: string): HTMLElement {
  const root = h("div", { class: "md" });
  const parts = text.split(/^```[^\n]*\n?/m);
  parts.forEach((part, i) => {
    if (i % 2 === 1) {
      root.append(h("pre", {}, h("code", {}, part.replace(/\n$/, ""))));
      return;
    }
    let list: HTMLElement | null = null;
    let paragraph: string[] = [];
    const flush = () => {
      if (paragraph.length) root.append(h("p", {}, ...inline(paragraph.join(" "))));
      paragraph = [];
    };
    for (const line of part.split("\n")) {
      const heading = /^(#{1,4})\s+(.*)$/.exec(line);
      const item = /^\s*(?:[-*]|\d+\.)\s+(.*)$/.exec(line);
      if (heading) {
        flush();
        list = null;
        root.append(h("h4", {}, ...inline(heading[2])));
      } else if (item) {
        flush();
        if (!list) {
          list = h("ul");
          root.append(list);
        }
        list.append(h("li", {}, ...inline(item[1])));
      } else if (!line.trim()) {
        flush();
        list = null;
      } else {
        list = null;
        paragraph.push(line.trim());
      }
    }
    flush();
  });
  return root;
}

function inline(text: string): Node[] {
  const nodes: Node[] = [];
  const pattern = /(`[^`]+`|\*\*[^*]+\*\*)/g;
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    if (match.index > last) nodes.push(document.createTextNode(text.slice(last, match.index)));
    const token = match[0];
    nodes.push(token.startsWith("`") ? h("code", {}, token.slice(1, -1)) : h("strong", {}, token.slice(2, -2)));
    last = match.index + token.length;
  }
  if (last < text.length) nodes.push(document.createTextNode(text.slice(last)));
  return nodes;
}

// ── layout ─────────────────────────────────────────────────────────────

const app = document.getElementById("app")!;
const statusLine = h("div", { class: "status" });
const modelSelect = h("select", { class: "model", title: "Model for the next requests" });
const header = h("div", { class: "header" }, statusLine, modelSelect);
const log = h("div", { class: "log", role: "log", "aria-live": "polite" });
const input = h("textarea", {
  class: "input",
  rows: "3",
  placeholder: "Ask about the code or request a change (Enter sends, Shift+Enter: new line). /plan <goal> plans a project; /undo reverts the last change.",
});
const sendButton = button("Send", () => submit(), "primary");
const stopButton = button("Stop", () => send({ type: "stop" }), "danger");
stopButton.hidden = true;
const composer = h("div", { class: "composer" }, input, h("div", { class: "actions" }, stopButton, sendButton));
app.append(header, log, composer);

modelSelect.addEventListener("change", () => send({ type: "model", id: modelSelect.value }));
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    submit();
  }
});

let busy = false;
let answer: { el: HTMLElement; text: string } | null = null; // the streamed answer of the current run
const tools = new Map<string, HTMLElement>(); // tool call id → its line
type Lock = (note: string) => void;
const openRequests = new Map<string, Lock>(); // approval tool call id → lock its card (answered elsewhere, run ended)
const openPlans = new Set<Lock>(); // the same for open plan cards

function submit(): void {
  const text = input.value.trim();
  if (!text || busy) return;
  input.value = "";
  send({ type: "send", text });
}

function append(el: HTMLElement): HTMLElement {
  const stick = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
  log.append(el);
  if (stick) log.scrollTop = log.scrollHeight;
  return el;
}

function line(text: string, cls = "dim"): HTMLElement {
  return append(h("div", { class: `line ${cls}` }, text));
}

// ── status and models ─────────────────────────────────────────────────

function showStatus(status: Status): void {
  statusLine.replaceChildren();
  const ready = status.state === "ready";
  input.disabled = !ready;
  sendButton.disabled = !ready || busy;
  if (status.state === "ready") {
    statusLine.append(h("span", { class: "ok" }, "●"), ` ${status.workspace} · ${status.email}`);
    statusLine.title = status.server;
  } else if (status.state === "connecting") {
    statusLine.append(h("span", { class: "dim" }, status.text));
  } else if (status.state === "signed_out") {
    statusLine.append(
      h("span", {}, status.text ? `${status.text} ` : "Sign in to the Kartrix server "),
      button("Sign in", () => send({ type: "signIn", signup: false })),
      button("Create account", () => send({ type: "signIn", signup: true })),
      button("Reconnect", () => send({ type: "reconnect" })),
    );
  } else {
    statusLine.append(h("span", { class: "err" }, status.text), " ", button("Retry", () => send({ type: "reconnect" })));
  }
}

function showModels(models: ModelOption[], current: string | null): void {
  modelSelect.replaceChildren(
    ...models.map((m) => {
      const option = h("option", { value: m.id }, m.available ? m.name : `${m.name} (unavailable)`);
      option.disabled = !m.available;
      if (m.reason) option.title = m.reason;
      option.selected = m.id === current;
      return option;
    }),
  );
}

// ── the feed ──────────────────────────────────────────────────────────

const AGENT_LABEL: Record<string, string> = { router: "router", explorer: "explorer", coder: "coder", reviewer: "reviewer" };

function onEvent(event: Record<string, unknown>): void {
  const type = String(event.type);
  switch (type) {
    case "agent_step": {
      const agent = AGENT_LABEL[String(event.agent)] ?? String(event.agent);
      if (event.status === "started") line(`◆ ${agent} working…`, "step");
      else if (event.summary) line(`◆ ${agent}: ${String(event.summary)}`, "step");
      return;
    }
    case "tool_call_started": {
      const target = event.target ? ` ${String(event.target)}` : "";
      const el = line(`→ ${String(event.tool)}${target}`, "tool");
      if (event.call_id) tools.set(String(event.call_id), el);
      return;
    }
    case "tool_call_finished": {
      const el = event.call_id ? tools.get(String(event.call_id)) : undefined;
      if (el && event.outcome !== "ok") {
        el.classList.add("failed");
        el.append(h("span", { class: "outcome" }, ` (${String(event.outcome)})`));
      }
      return;
    }
    case "assistant_delta": {
      if (!answer) answer = { el: append(h("div", { class: "answer streaming" })), text: "" };
      answer.text += String(event.text ?? "");
      answer.el.textContent = answer.text;
      log.scrollTop = log.scrollHeight;
      return;
    }
    case "assistant_message": {
      const el = answer?.el ?? append(h("div", { class: "answer" }));
      el.className = "answer";
      el.replaceChildren(markdown(String(event.text ?? "")));
      answer = null;
      return;
    }
    case "notice": {
      const level = String(event.level ?? "info");
      line(String(event.text ?? ""), level === "error" ? "err" : level === "warning" ? "warn" : level === "success" ? "ok" : "dim");
      return;
    }
    case "approval_resolved": {
      openRequests.get(String(event.tool_call_id))?.("answered elsewhere");
      const decision = String(event.decision);
      const by = event.by === "policy" ? " by policy" : "";
      const label: Record<string, string> = {
        approve: "✓ approved",
        approve_session: "✓ allowed for the session",
        edit: "✓ edited and approved",
        reject: "✗ rejected",
      };
      line(`${label[decision] ?? decision}${by}`, decision === "reject" ? "warn" : "ok");
      return;
    }
    case "plan_reviewed":
      for (const lock of [...openPlans]) lock("answered elsewhere");
      line(event.approved ? "✓ plan approved" : "✗ plan sent back for changes", event.approved ? "ok" : "warn");
      return;
    case "task_started":
      line(`▶ ${String(event.title)}`, "step");
      return;
    case "task_finished": {
      const ok = event.status === "completed";
      line(`${ok ? "✓" : "✗"} ${String(event.title)}${event.detail ? ` — ${String(event.detail)}` : ""}`, ok ? "ok" : "warn");
      return;
    }
    case "progress":
      line(`${String(event.completed)}/${String(event.total)} tasks done`, "dim");
      return;
    case "files_changed":
      filesBlock(event);
      return;
    case "run_finished": {
      if (event.status !== "completed") {
        const detail = event.detail ? `: ${String(event.detail)}` : "";
        line(`Run ${String(event.status)}${detail}`, event.status === "stopped" ? "warn" : "err");
      }
      const usage = event.usage as Record<string, number> | undefined;
      if (usage) {
        const tokens = (usage.input_tokens ?? 0) + (usage.output_tokens ?? 0);
        line(`${tokens.toLocaleString()} tokens · ${usage.tool_calls ?? 0} tool calls · ${Math.round(usage.seconds ?? 0)} s`, "dim usage");
      }
      return;
    }
    default:
      return; // run_started, model_call, context_assembled, approval_requested, plan_proposed … (shown elsewhere)
  }
}

// ── changed files and undo ────────────────────────────────────────────

/** A file name that opens the file; with a checkpoint entry, also a "diff" that opens before ↔ after. */
function fileItem(path: string, diff?: { entry: string; label: string }): HTMLElement {
  const open = button(path, () => send({ type: "open", path }), "link");
  open.title = `Open ${path}`;
  const item = h("li", {}, open);
  if (diff) {
    const show = button("diff", () => send({ type: "diff", entry: diff.entry, path, label: diff.label }), "small");
    show.title = "Show this change in a diff editor";
    item.append(" ", show);
  }
  return item;
}

function filesBlock(event: Record<string, unknown>): void {
  const files = (event.files as string[]) ?? [];
  const count = Number(event.count ?? files.length);
  const entry = typeof event.entry === "string" ? event.entry : undefined;
  const label = String(event.label ?? "");
  const list = h("ul", {}, ...files.map((f) => fileItem(f, entry ? { entry, label } : undefined)));
  if (count > files.length) list.append(h("li", { class: "dim" }, `… and ${count - files.length} more`));
  const details = h("details", { open: "" }, h("summary", {}, `Changed ${count} file(s)`), list);
  const undo = button("Undo", () => send({ type: "undo", entry }), "small");
  undo.title = "Revert the files of this change (Kartrix: Undo Last Change)";
  append(h("div", { class: "files" }, details, h("div", { class: "buttons" }, undo)));
}

function stepBlock(message: Extract<ToView, { type: "step" }>): void {
  const block = h("div", { class: "files" }, h("div", { class: "line ok" }, message.text));
  if (message.restored.length) {
    block.append(h("details", {}, h("summary", {}, `Restored ${message.restored.length}`), h("ul", {}, ...message.restored.map((f) => fileItem(f)))));
  }
  if (message.deleted.length) {
    block.append(h("details", {}, h("summary", {}, `Deleted ${message.deleted.length}`), h("ul", {}, ...message.deleted.map((f) => h("li", {}, f)))));
  }
  for (const error of message.errors) block.append(h("div", { class: "line err" }, `Could not delete ${error}`));
  const again = message.verb === "undo" ? button("Redo", () => send({ type: "redo" }), "small") : button("Undo", () => send({ type: "undo" }), "small");
  block.append(h("div", { class: "buttons" }, again));
  append(block);
}

// ── approvals ─────────────────────────────────────────────────────────

function approvalCard(runId: string, requestId: string, requests: ApprovalRequestView[]): void {
  const decisions: (DecisionView | null)[] = requests.map(() => null);
  const card = append(h("div", { class: "card approval" }, h("div", { class: "card-title" }, "Approval needed")));
  const notes: HTMLElement[] = [];
  const lock: Lock = (note) => {
    card.classList.add("answered");
    card.querySelectorAll("button, textarea, input").forEach((el) => ((el as HTMLButtonElement).disabled = true));
    notes.forEach((el, i) => {
      if (decisions[i] === null && note) el.textContent = note;
    });
    requests.forEach((r) => openRequests.delete(r.tool_call_id));
  };
  const finish = () => {
    if (decisions.some((d) => d === null)) return;
    lock("");
    send({ type: "approval", runId, requestId, decisions: decisions as DecisionView[] });
  };
  requests.forEach((request, i) => {
    const command = h("pre", { class: "command" }, request.command);
    const editor = h("textarea", { class: "edit", rows: "3" });
    editor.value = request.command;
    editor.hidden = true;
    const reason = h("input", { class: "reason", type: "text", placeholder: "Why not? (optional, tells the agent)" });
    const decided = h("div", { class: "decided" });
    notes.push(decided);
    openRequests.set(request.tool_call_id, lock);
    const decide = (decision: DecisionView, label: string) => {
      decisions[i] = decision;
      row.querySelectorAll("button, textarea, input").forEach((el) => ((el as HTMLButtonElement).disabled = true));
      decided.textContent = label;
      finish();
    };
    const row = h(
      "div",
      { class: "request" },
      h("div", { class: "meta" }, `${request.tool} · ${request.category} · in ${request.directory}`),
      command,
      editor,
      h("div", { class: "why" }, request.reason),
      h(
        "div",
        { class: "buttons" },
        button("Approve", () => decide({ type: "approve" }, "✓ approved"), "primary"),
        request.allow_session ? button("Allow for this session", () => decide({ type: "approve_session" }, "✓ allowed for this session")) : null,
        request.editable
          ? button("Edit", () => {
              if (editor.hidden) {
                editor.hidden = false;
                command.hidden = true;
                editor.focus();
              } else if (editor.value.trim()) {
                decide({ type: "edit", command: editor.value.trim() }, "✓ edited and approved");
              }
            })
          : null,
        button("Reject", () => decide({ type: "reject", message: reason.value.trim() || undefined }, "✗ rejected"), "danger"),
      ),
      reason,
      decided,
    );
    card.append(row);
  });
}

// ── plan review ───────────────────────────────────────────────────────

function planCard(runId: string, requestId: string, plan: Record<string, unknown>): void {
  const tasks = (plan.tasks as Record<string, unknown>[]) ?? [];
  const list = (items: unknown) => h("ul", {}, ...((items as string[]) ?? []).map((x) => h("li", {}, String(x))));
  const feedback = h("textarea", { class: "feedback", rows: "2", placeholder: "What should change? (for Re-plan)" });
  const card = append(
    h(
      "div",
      { class: "card plan" },
      h("div", { class: "card-title" }, `Plan: ${String(plan.project_name ?? "")}`),
      h("p", {}, String(plan.goal_summary ?? "")),
      (plan.tech_stack as string[] | undefined)?.length ? h("div", { class: "meta" }, `Stack: ${(plan.tech_stack as string[]).join(", ")}`) : null,
      h(
        "ol",
        { class: "tasks" },
        ...tasks.map((t) =>
          h(
            "li",
            {},
            h("strong", {}, String(t.title ?? t.id)),
            h("div", { class: "dim" }, String(t.description ?? "")),
            (t.output_files as string[] | undefined)?.length ? h("div", { class: "meta" }, (t.output_files as string[]).join(", ")) : null,
          ),
        ),
      ),
      (plan.risks as string[] | undefined)?.length ? h("details", {}, h("summary", {}, "Risks"), list(plan.risks)) : null,
      (plan.assumptions as string[] | undefined)?.length ? h("details", {}, h("summary", {}, "Assumptions"), list(plan.assumptions)) : null,
      feedback,
    ),
  );
  const note = h("div", { class: "decided" });
  const lock: Lock = (text) => {
    card.classList.add("answered");
    card.querySelectorAll("button, textarea").forEach((el) => ((el as HTMLButtonElement).disabled = true));
    if (text) note.textContent = text;
    openPlans.delete(lock);
  };
  openPlans.add(lock);
  const answer = (approved: boolean) => {
    lock("");
    send({ type: "plan_review", runId, requestId, approved, feedback: approved ? "" : feedback.value.trim() });
  };
  card.append(h("div", { class: "buttons" }, button("Approve and run", () => answer(true), "primary"), button("Re-plan", () => answer(false))), note);
}

// ── messages from the extension ───────────────────────────────────────

window.addEventListener("message", (e: MessageEvent<ToView>) => {
  const message = e.data;
  switch (message.type) {
    case "status":
      showStatus(message.status);
      return;
    case "models":
      showModels(message.models, message.current);
      return;
    case "busy":
      busy = message.busy;
      stopButton.hidden = !busy;
      sendButton.disabled = busy || input.disabled;
      if (!busy) {
        for (const lock of [...openRequests.values(), ...openPlans]) lock("the run ended");
        answer = null;
        tools.clear();
        input.focus();
      }
      return;
    case "user":
      append(h("div", { class: "user" }, message.kind === "plan" ? `/plan ${message.text}` : message.text));
      return;
    case "event":
      onEvent(message.event);
      return;
    case "approval":
      approvalCard(message.runId, message.requestId, message.requests);
      return;
    case "plan_review":
      planCard(message.runId, message.requestId, message.plan);
      return;
    case "finished":
      if (!message.reported && message.status !== "completed") {
        line(`Run ${message.status}${message.detail ? `: ${message.detail}` : ""}`, message.status === "stopped" ? "warn" : "err");
      }
      return;
    case "error":
      line(message.text, "err");
      return;
    case "info":
      line(message.text, message.level ?? "dim");
      return;
    case "step":
      stepBlock(message);
      return;
    case "cleared":
      log.replaceChildren();
      line("New conversation — the next request starts fresh.", "dim");
      return;
  }
});

send({ type: "ready" });
