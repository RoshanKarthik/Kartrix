/** Changed files in the editor: open a file, diff a change (before ↔ after, both read from the server's checkpoint
 * store as `kartrix:` documents), undo / redo with the terminal's confirmation flow, and a checkpoints picker. */

import * as vscode from "vscode";

import { ApiError, KartrixApi, Workspace } from "./api";
import {
  CheckpointEntry,
  CheckpointList,
  FileVersion,
  SCHEME,
  Side,
  StepResult,
  Verb,
  cleanPath,
  conflictPrompt,
  isBlocked,
  parseVersionPath,
  shorten,
  stepSummary,
  undoTarget,
  versionPath,
  versionText,
} from "./changes";
import { ToView } from "./messages";

export interface Connection {
  api: KartrixApi;
  workspace: Workspace;
  folder: vscode.Uri;
}

/** What this needs from the chat view's controller. */
export interface ChangesHost {
  connection(): Promise<Connection | null>;
  post(message: ToView): void;
  readonly viewVisible: boolean;
}

export class Changes implements vscode.TextDocumentContentProvider {
  constructor(private readonly host: ChangesHost) {}

  private async connection(): Promise<Connection> {
    const connection = await this.host.connection();
    if (!connection) throw new Error("Not connected to a Kartrix server yet.");
    return connection;
  }

  private async command<T>(name: string, args: Record<string, unknown> = {}): Promise<T> {
    const { api, workspace } = await this.connection();
    return (await api.command(workspace.id, name, args)).result as T;
  }

  /** A message in the chat view; a notification as well when the view isn't showing. */
  private tell(text: string, level: "ok" | "warn" | "dim" | "err" = "dim"): void {
    this.host.post(level === "err" ? { type: "error", text } : { type: "info", text, level });
    if (this.host.viewVisible) return;
    if (level === "err") void vscode.window.showErrorMessage(text);
    else if (level === "warn") void vscode.window.showWarningMessage(text);
    else void vscode.window.showInformationMessage(text);
  }

  // ── the `kartrix:` documents ────────────────────────────────────────

  async provideTextDocumentContent(uri: vscode.Uri): Promise<string> {
    const target = parseVersionPath(uri.path);
    if (!target) throw new Error(`Not a Kartrix checkpoint file: ${uri.toString()}`);
    return versionText(await this.command<FileVersion>("checkpoints.show", target));
  }

  // ── open and diff ───────────────────────────────────────────────────

  async open(path: string): Promise<void> {
    const rel = cleanPath(path);
    if (rel === null) return this.tell(`Not a file in the workspace: ${path}`, "err");
    const { folder } = await this.connection();
    const uri = vscode.Uri.joinPath(folder, ...rel.split("/"));
    try {
      await vscode.workspace.fs.stat(uri);
    } catch {
      return this.tell(`${rel} doesn't exist any more.`, "warn");
    }
    await vscode.window.showTextDocument(uri, { preview: true });
  }

  async diff(entry: string, path: string, label: string): Promise<void> {
    const rel = cleanPath(path);
    if (rel === null) return this.tell(`Not a file in the workspace: ${path}`, "err");
    const side = (s: Side) => vscode.Uri.from({ scheme: SCHEME, path: versionPath(entry, s, rel) });
    const name = rel.split("/").at(-1);
    await vscode.commands.executeCommand("vscode.diff", side("before"), side("after"), `${name} (before ↔ after “${shorten(label, 40)}”)`, {
      preview: true,
    });
  }

  // ── undo / redo ─────────────────────────────────────────────────────

  /** Undo (or redo) the latest change. With `entry` (the view's Undo button), first check that it is that change. */
  async step(verb: Verb, entry?: string): Promise<void> {
    try {
      if (entry && verb === "undo" && !(await this.confirmTarget(entry))) return;
      let result = await this.command<StepResult>(verb, { force: false });
      if (isBlocked(result)) {
        const prompt = conflictPrompt(result);
        const choice = await vscode.window.showWarningMessage(prompt.message, { modal: true, detail: prompt.detail }, prompt.action);
        if (choice !== prompt.action) return this.tell("Nothing was changed.");
        result = await this.command<StepResult>(verb, { force: true });
      }
      this.report(result);
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) return this.tell(`A request is running in this workspace — ${verb} when it ends.`, "warn");
      this.tell(e instanceof Error ? e.message : String(e), "err");
    }
  }

  private async confirmTarget(entry: string): Promise<boolean> {
    const list = await this.command<CheckpointList>("checkpoints");
    if (!list.available) {
      this.tell(`Checkpoints are off: ${list.reason ?? "unknown reason"}`, "warn");
      return false;
    }
    const target = undoTarget(list, entry);
    if (target.ok) return true;
    if (target.reason === "undone") this.tell("That change is already undone.");
    if (target.reason === "gone") this.tell("That change is no longer kept, so it can't be undone.", "warn");
    if (target.reason !== "not_latest") return false;
    const action = "Undo the Latest";
    const choice = await vscode.window.showWarningMessage(
      `Undo goes back one change at a time, newest first. The latest change is “${shorten(target.latest?.label ?? "")}”. Undo that one?`,
      { modal: true },
      action,
    );
    return choice === action;
  }

  private report(result: StepResult): void {
    const summary = stepSummary(result);
    const outcome = result.result;
    if (outcome?.applied) {
      this.host.post({ type: "step", verb: result.verb, text: summary.text, restored: outcome.restored, deleted: outcome.deleted, errors: outcome.errors });
      if (!this.host.viewVisible) void vscode.window.showInformationMessage(summary.text);
      return;
    }
    this.tell(summary.text, summary.level);
  }

  // ── checkpoints ─────────────────────────────────────────────────────

  async showCheckpoints(): Promise<void> {
    let list: CheckpointList;
    try {
      list = await this.command<CheckpointList>("checkpoints");
    } catch (e) {
      return this.tell(e instanceof Error ? e.message : String(e), "err");
    }
    if (!list.available) return this.tell(`Checkpoints are off: ${list.reason ?? "unknown reason"}`, "warn");
    type Item = vscode.QuickPickItem & { entry?: CheckpointEntry; action?: Verb };
    const item = (entry: CheckpointEntry, note: string): Item => ({
      label: entry.label,
      description: `${new Date(entry.created * 1000).toLocaleString()} · ${entry.file_count} file(s)`,
      detail: note || undefined,
      entry,
    });
    const items: Item[] = [];
    if (list.undo.length) items.push({ label: "$(discard) Undo the latest change", action: "undo" });
    if (list.redo.length) items.push({ label: "$(redo) Redo", action: "redo" });
    if (items.length) items.push({ label: "Changes (newest first)", kind: vscode.QuickPickItemKind.Separator });
    items.push(...[...list.redo].reverse().map((e) => item(e, "undone")));
    items.push(...[...list.undo].reverse().map((e, i) => item(e, i === 0 ? "latest — Undo reverts it" : "")));
    if (!list.undo.length && !list.redo.length) {
      return this.tell("No checkpoints yet — every request and /plan task that changes files gets one.");
    }
    const picked = await vscode.window.showQuickPick(items, { placeHolder: "Kartrix checkpoints — pick a change to see its files" });
    if (!picked) return;
    if (picked.action) return this.step(picked.action);
    if (picked.entry) return this.pickFile(picked.entry);
  }

  private async pickFile(entry: CheckpointEntry): Promise<void> {
    const files = entry.files ?? [];
    const items: vscode.QuickPickItem[] = files.map((f) => ({ label: f }));
    if (entry.file_count > files.length) items.push({ label: `… and ${entry.file_count - files.length} more (not listed)`, kind: vscode.QuickPickItemKind.Separator });
    const picked = await vscode.window.showQuickPick(items, { placeHolder: `Diff a file of “${shorten(entry.label)}”` });
    if (picked && files.includes(picked.label)) await this.diff(entry.id, picked.label, entry.label);
  }
}
