/** Pure helpers for changed files, diffs and undo (no `vscode` import, so node:test can load them). The data shapes
 * are what the server's workspace commands return (kartrix/core/commands.py: checkpoints, checkpoints.show, undo,
 * redo). */

export const SCHEME = "kartrix";

export type Side = "before" | "after";
export type Verb = "undo" | "redo";

export interface CheckpointEntry {
  id: string;
  label: string;
  created: number;
  file_count: number;
  files?: string[];
}

export interface CheckpointList {
  available: boolean;
  reason?: string | null;
  undo: CheckpointEntry[];
  redo: CheckpointEntry[];
}

export interface FileVersion {
  available: boolean;
  reason?: string | null;
  entry: string;
  label: string;
  side: Side;
  path: string;
  exists: boolean;
  kind: "file" | "symlink" | "other";
  size: number;
  binary: boolean;
  too_large: boolean;
  content: string | null;
}

export interface StepOutcome {
  label: string;
  applied: boolean;
  restored: string[];
  deleted: string[];
  conflicts: string[];
  errors: string[];
}

export interface StepResult {
  available: boolean;
  reason?: string | null;
  verb: Verb;
  result: StepOutcome | null;
}

/** A workspace-relative path in git's form (forward slashes), or null when it isn't one: absolute, a drive, `..`.
 * The same rule as the server's `clean_path` (kartrix/security/checkpoints.py). */
export function cleanPath(rel: string): string | null {
  const path = rel.replace(/\\/g, "/").trim();
  const parts = path.split("/").filter((p) => p !== "" && p !== ".");
  if (!parts.length || path.startsWith("/") || /^[A-Za-z]:/.test(path) || parts.includes("..") || path.includes("\0")) return null;
  return parts.join("/");
}

/** The path of a file version's virtual document: `/<entry>/<side>/<file>` — ending in the file's own name, so the
 * editor picks its language. */
export function versionPath(entry: string, side: Side, path: string): string {
  const clean = cleanPath(path);
  if (!/^[0-9a-f]{4,40}$/.test(entry) || clean === null) throw new Error(`not a checkpoint file: ${entry} ${path}`);
  return `/${entry}/${side}/${clean}`;
}

export function parseVersionPath(uriPath: string): { entry: string; side: Side; path: string } | null {
  const match = /^\/([0-9a-f]{4,40})\/(before|after)\/(.+)$/.exec(uriPath);
  if (!match) return null;
  const path = cleanPath(match[3]);
  return path === null ? null : { entry: match[1], side: match[2] as Side, path };
}

/** What the virtual document shows: the text, or a note when there is none to show. */
export function versionText(version: FileVersion): string {
  if (!version.available) return `(checkpoints are off: ${version.reason ?? "unknown reason"})`;
  if (!version.exists) return "";
  if (version.kind === "other") return "(a folder or nested repository, not a file)";
  if (version.binary) return `(binary file, ${version.size.toLocaleString("en-US")} bytes — not shown)`;
  if (version.too_large || version.content === null) return `(${version.size.toLocaleString("en-US")} bytes — too large to show)`;
  return version.content;
}

/** Undo/redo stopped because files changed since: the user decides whether to overwrite them (then `force`). */
export function isBlocked(step: StepResult): boolean {
  return step.available && step.result !== null && !step.result.applied;
}

export function conflictPrompt(step: StepResult, limit = 15): { message: string; detail: string; action: string } {
  const outcome = step.result!;
  const files = outcome.conflicts.slice(0, limit);
  if (outcome.conflicts.length > limit) files.push(`… and ${outcome.conflicts.length - limit} more`);
  const verb = step.verb === "undo" ? "Undo" : "Redo";
  return {
    message: `These files changed after “${shorten(outcome.label)}”. ${verb} anyway and overwrite those changes?`,
    detail: files.join("\n"),
    action: `${verb} Anyway`,
  };
}

export function stepSummary(step: StepResult): { text: string; level: "ok" | "warn" | "dim" } {
  if (!step.available) return { text: `Checkpoints are off: ${step.reason ?? "unknown reason"}`, level: "warn" };
  const outcome = step.result;
  if (outcome === null) return { text: `Nothing to ${step.verb}.`, level: "dim" };
  if (!outcome.applied) return { text: "Nothing was changed.", level: "dim" };
  const done = step.verb === "undo" ? "Undid" : "Redid";
  const kept = outcome.conflicts.length ? " The overwritten versions are kept in the checkpoint store." : "";
  return {
    text: `✓ ${done} “${shorten(outcome.label)}” (${outcome.restored.length} restored, ${outcome.deleted.length} deleted).${kept}`,
    level: "ok",
  };
}

/** Is `entry` the change an undo would revert now? Undo goes back one change at a time, newest first. */
export function undoTarget(list: CheckpointList, entry: string): { ok: true } | { ok: false; reason: "undone" | "gone" | "not_latest"; latest?: CheckpointEntry } {
  const latest = list.undo.at(-1);
  if (latest?.id === entry) return { ok: true };
  if (list.redo.some((e) => e.id === entry)) return { ok: false, reason: "undone" };
  if (!list.undo.some((e) => e.id === entry)) return { ok: false, reason: "gone" };
  return { ok: false, reason: "not_latest", latest };
}

export function shorten(text: string, max = 80): string {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}
