/** Messages between the extension and the chat view (webview/main.ts). */

export interface ModelOption {
  id: string;
  name: string;
  available: boolean;
  reason: string | null;
}

export type Status =
  | { state: "connecting"; text: string }
  | { state: "signed_out"; server: string; text?: string }
  | { state: "error"; text: string }
  | { state: "ready"; server: string; email: string; workspace: string };

export interface ApprovalRequestView {
  tool_call_id: string;
  tool: string;
  command: string;
  directory: string;
  category: string;
  reason: string;
  allow_session: boolean;
  editable: boolean;
}

export type ToView =
  | { type: "status"; status: Status }
  | { type: "models"; models: ModelOption[]; current: string | null }
  | { type: "busy"; busy: boolean }
  | { type: "user"; text: string; kind: "ask" | "plan" }
  | { type: "event"; event: Record<string, unknown> }
  | { type: "approval"; runId: string; requestId: string; requests: ApprovalRequestView[] }
  | { type: "plan_review"; runId: string; requestId: string; plan: Record<string, unknown> }
  | { type: "finished"; status: string; detail: string | null; reported: boolean }
  | { type: "error"; text: string }
  | { type: "info"; text: string; level?: "ok" | "warn" | "dim" }
  | { type: "step"; verb: "undo" | "redo"; text: string; restored: string[]; deleted: string[]; errors: string[] }
  | { type: "cleared" };

export interface DecisionView {
  type: "approve" | "approve_session" | "edit" | "reject";
  command?: string;
  message?: string;
}

export type FromView =
  | { type: "ready" }
  | { type: "send"; text: string }
  | { type: "approval"; runId: string; requestId: string; decisions: DecisionView[] }
  | { type: "plan_review"; runId: string; requestId: string; approved: boolean; feedback: string }
  | { type: "stop" }
  | { type: "new" }
  | { type: "model"; id: string }
  | { type: "signIn"; signup: boolean }
  | { type: "reconnect" }
  | { type: "open"; path: string }
  | { type: "diff"; entry: string; path: string; label: string }
  | { type: "undo"; entry?: string }
  | { type: "redo" };

export type Input = { kind: "ask" | "plan"; input: string } | { command: "undo" | "redo" | "checkpoints" } | { error: string };

/** `/plan <goal>` plans and runs a project; `/ask <q>` and plain text ask; `/undo`, `/redo` and `/checkpoints` run
 * in the editor (a conflict is asked about there, so no `force`). Other slash commands aren't sent. */
export function parseInput(text: string): Input {
  const trimmed = text.trim();
  if (!trimmed) return { error: "Type a request." };
  const match = /^\/(\w+)(?:\s+([\s\S]*))?$/.exec(trimmed);
  if (!match) return { kind: "ask", input: trimmed };
  const [, command, rest = ""] = match;
  if (command === "ask" || command === "plan") {
    return rest.trim() ? { kind: command, input: rest.trim() } : { error: `Usage: /${command} <text>` };
  }
  if (command === "undo" || command === "redo" || command === "checkpoints") {
    return rest.trim() ? { error: `Usage: /${command} (on a conflict you're asked whether to overwrite)` } : { command };
  }
  return { error: `/${command} isn't available in the editor yet — run it in the terminal (kartrix).` };
}
