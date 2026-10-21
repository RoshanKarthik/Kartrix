/** The chat log: a run's live feed — agent steps, tool calls, the streamed answer — and earlier runs. Everything from
 * the agent or the server is put in the page as text (textContent), never as HTML. */

export function h(tag, attrs = {}, ...children) {
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

/** A small, safe subset of Markdown: fenced code, headings, lists, paragraphs, `code` and **bold**. */
export function markdown(text) {
  const root = h("div", { class: "md" });
  text.split(/^```[^\n]*\n?/m).forEach((part, i) => {
    if (i % 2 === 1) {
      root.append(h("pre", {}, h("code", {}, part.replace(/\n$/, ""))));
      return;
    }
    let list = null;
    let paragraph = [];
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

function inline(text) {
  const nodes = [];
  let last = 0;
  for (const match of text.matchAll(/(`[^`]+`|\*\*[^*]+\*\*)/g)) {
    if (match.index > last) nodes.push(document.createTextNode(text.slice(last, match.index)));
    const token = match[0];
    nodes.push(token.startsWith("`") ? h("code", {}, token.slice(1, -1)) : h("strong", {}, token.slice(2, -2)));
    last = match.index + token.length;
  }
  if (last < text.length) nodes.push(document.createTextNode(text.slice(last)));
  return nodes;
}

const LEVEL = { error: "err", warning: "warn", success: "ok" };

export class Feed {
  constructor(log) {
    this.log = log;
    this.answer = null; // the streamed answer of the current run: {el, text}
    this.tools = new Map(); // tool call id → its line
    this.placeholder = null;
  }

  append(el) {
    const stick = this.log.scrollHeight - this.log.scrollTop - this.log.clientHeight < 60;
    this.log.append(el);
    if (stick) this.log.scrollTop = this.log.scrollHeight;
    return el;
  }

  line(text, cls = "dim") {
    return this.append(h("div", { class: `line ${cls}` }, text));
  }

  clear() {
    this.log.replaceChildren();
    this.reset();
  }

  /** The run ended: an answer cut off mid-stream (stopped, failed) keeps what arrived, formatted. */
  reset() {
    if (this.answer) {
      this.answer.el.className = "answer";
      this.answer.el.replaceChildren(markdown(this.answer.text));
    }
    this.answer = null;
    this.tools.clear();
  }

  /** A line shown only until the first request. */
  hint(text) {
    this.placeholder = this.line(text);
  }

  user(text, kind = "ask") {
    this.placeholder?.remove();
    this.placeholder = null;
    this.append(h("div", { class: "user" }, kind === "plan" ? `/plan ${text}` : text));
  }

  /** A finished run from the workspace's history: the request, then its answer or how it ended. */
  past(run) {
    this.user(run.input, run.kind);
    if (run.answer) this.append(h("div", { class: "answer" }, markdown(run.answer)));
    if (run.status !== "completed") this.ended(run.status, run.detail);
  }

  ended(status, detail) {
    if (status === "completed") return;
    this.line(`Run ${status}${detail ? `: ${detail}` : ""}`, status === "stopped" ? "warn" : "err");
  }

  /** One message of the SSE feed. */
  message(message) {
    if (message.kind === "event") this.event(message.data);
    else if (message.kind === "approval_needed") this.approvalNeeded(message.data);
    else if (message.kind === "plan_review_needed") this.planReviewNeeded(message.data);
  }

  event(event) {
    switch (String(event.type)) {
      case "agent_step": {
        if (event.status === "started") this.line(`◆ ${String(event.agent)} working…`, "step");
        else if (event.summary) this.line(`◆ ${String(event.agent)}: ${String(event.summary)}`, "step");
        return;
      }
      case "tool_call_started": {
        const el = this.line(`→ ${String(event.tool)}${event.target ? ` ${String(event.target)}` : ""}`, "tool");
        if (event.call_id) this.tools.set(String(event.call_id), el);
        return;
      }
      case "tool_call_finished": {
        const el = event.call_id ? this.tools.get(String(event.call_id)) : undefined;
        if (el && event.outcome !== "ok") {
          el.classList.add("failed");
          el.append(h("span", { class: "outcome" }, ` (${String(event.outcome)})`));
        }
        return;
      }
      case "assistant_delta": {
        if (!this.answer) this.answer = { el: this.append(h("div", { class: "answer streaming" })), text: "" };
        this.answer.text += String(event.text ?? "");
        this.answer.el.textContent = this.answer.text;
        this.log.scrollTop = this.log.scrollHeight;
        return;
      }
      case "assistant_message": {
        const el = this.answer?.el ?? this.append(h("div", { class: "answer" }));
        el.className = "answer";
        el.replaceChildren(markdown(String(event.text ?? "")));
        this.answer = null;
        return;
      }
      case "notice":
        this.line(String(event.text ?? ""), LEVEL[String(event.level)] ?? "dim");
        return;
      case "approval_resolved": {
        const labels = { approve: "✓ approved", approve_session: "✓ allowed for the session", edit: "✓ edited and approved", reject: "✗ rejected" };
        const decision = String(event.decision);
        this.line(`${labels[decision] ?? decision}${event.by === "policy" ? " by policy" : ""}`, decision === "reject" ? "warn" : "ok");
        return;
      }
      case "plan_reviewed":
        this.line(event.approved ? "✓ plan approved" : "✗ plan sent back for changes", event.approved ? "ok" : "warn");
        return;
      case "task_started":
        this.line(`▶ ${String(event.title)}`, "step");
        return;
      case "task_finished": {
        const ok = event.status === "completed";
        this.line(`${ok ? "✓" : "✗"} ${String(event.title)}${event.detail ? ` — ${String(event.detail)}` : ""}`, ok ? "ok" : "warn");
        return;
      }
      case "progress":
        this.line(`${String(event.completed)}/${String(event.total)} tasks done`);
        return;
      case "files_changed":
        this.filesChanged(event);
        return;
      case "run_finished": {
        this.ended(String(event.status), event.detail ? String(event.detail) : null);
        const usage = event.usage;
        if (usage && typeof usage === "object") {
          const tokens = (usage.input_tokens ?? 0) + (usage.output_tokens ?? 0);
          this.line(`${tokens.toLocaleString()} tokens · ${usage.tool_calls ?? 0} tool calls · ${Math.round(usage.seconds ?? 0)} s`, "dim usage");
        }
        return;
      }
      default:
        return; // run_started, model_call, context_assembled, approval_requested, plan_proposed …
    }
  }

  filesChanged(event) {
    const files = Array.isArray(event.files) ? event.files.map(String) : [];
    const count = Number(event.count ?? files.length);
    const list = h("ul", {}, ...files.map((f) => h("li", {}, h("code", {}, f))));
    if (count > files.length) list.append(h("li", { class: "dim" }, `… and ${count - files.length} more`));
    this.append(h("details", { class: "files", open: "" }, h("summary", {}, `Changed ${count} file(s)`), list));
  }

  // Approvals and plan review in the browser are the next step (roadmap 6.2); until then the run says what it waits
  // for, and Stop ends it.

  approvalNeeded(data) {
    const requests = Array.isArray(data.requests) ? data.requests : [];
    const card = this.append(h("div", { class: "card waiting" }, h("div", { class: "card-title" }, "Waiting for your approval")));
    for (const request of requests) {
      card.append(
        h("div", { class: "meta" }, `${String(request.tool ?? "")} · ${String(request.category ?? "")} · in ${String(request.directory ?? "")}`),
        h("pre", { class: "command" }, String(request.command ?? "")),
        request.reason ? h("div", { class: "dim" }, String(request.reason)) : null,
      );
    }
    card.append(h("p", {}, "The browser can't approve commands yet. Press Stop to end this run, and send requests that need approval from the terminal or VS Code for now."));
  }

  planReviewNeeded(data) {
    const plan = data.plan && typeof data.plan === "object" ? data.plan : {};
    const tasks = Array.isArray(plan.tasks) ? plan.tasks : [];
    this.append(
      h(
        "div",
        { class: "card waiting" },
        h("div", { class: "card-title" }, `Plan waiting for review: ${String(plan.project_name ?? "")}`),
        plan.goal_summary ? h("p", {}, String(plan.goal_summary)) : null,
        h("ol", {}, ...tasks.map((t) => h("li", {}, String(t.title ?? t.id ?? "")))),
        h("p", {}, "The browser can't review plans yet. Press Stop to end this run, and use the terminal or VS Code for /plan for now."),
      ),
    );
  }
}
