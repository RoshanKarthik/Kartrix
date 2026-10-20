import assert from "node:assert/strict";
import { test } from "node:test";

import {
  CheckpointList,
  FileVersion,
  StepResult,
  cleanPath,
  conflictPrompt,
  isBlocked,
  parseVersionPath,
  stepSummary,
  undoTarget,
  versionPath,
  versionText,
} from "../src/changes";

test("paths stay inside the workspace (same rule as the server)", () => {
  assert.equal(cleanPath("src/app.py"), "src/app.py");
  assert.equal(cleanPath("./src\\deep//app.py"), "src/deep/app.py");
  assert.equal(cleanPath("notes/a:b.txt"), "notes/a:b.txt");
  for (const bad of ["", ".", "../x", "a/../../x", "/etc/passwd", "C:/Windows/x", "c:x", "a/\0"]) {
    assert.equal(cleanPath(bad), null, bad);
  }
});

test("a file version's document path round-trips and refuses anything else", () => {
  const path = versionPath("0123456789ab", "before", "src/my app.py");
  assert.equal(path, "/0123456789ab/before/src/my app.py");
  assert.deepEqual(parseVersionPath(path), { entry: "0123456789ab", side: "before", path: "src/my app.py" });
  assert.deepEqual(parseVersionPath("/0123456789ab/after/a.py"), { entry: "0123456789ab", side: "after", path: "a.py" });
  assert.equal(parseVersionPath("/0123456789ab/now/a.py"), null);
  assert.equal(parseVersionPath("/0123456789ab/before/../../secret"), null);
  assert.equal(parseVersionPath("/not-an-id/before/a.py"), null);
  assert.throws(() => versionPath("0123456789ab", "after", "../x"));
  assert.throws(() => versionPath("../x", "after", "a.py"));
});

test("a version's text, or a note when there is none to show", () => {
  const base: FileVersion = {
    available: true,
    entry: "e1",
    label: "l",
    side: "before",
    path: "a.py",
    exists: true,
    kind: "file",
    size: 12,
    binary: false,
    too_large: false,
    content: "print('v1')\n",
  };
  assert.equal(versionText(base), "print('v1')\n");
  assert.equal(versionText({ ...base, exists: false, content: null }), "");
  assert.match(versionText({ ...base, binary: true, size: 1234, content: null }), /binary file, 1,234 bytes/);
  assert.match(versionText({ ...base, too_large: true, size: 2_000_000, content: null }), /too large/);
  assert.match(versionText({ ...base, kind: "other", content: null }), /folder/);
  assert.match(versionText({ ...base, available: false, reason: "git was not found" }), /checkpoints are off: git was not found/);
});

const blocked: StepResult = {
  available: true,
  verb: "undo",
  result: { label: "/ask add a --json flag", applied: false, restored: [], deleted: [], conflicts: ["a.py", "b.py", "c.py"], errors: [] },
};

test("undo blocked by later changes asks before overwriting them", () => {
  assert.ok(isBlocked(blocked));
  const prompt = conflictPrompt(blocked, 2);
  assert.match(prompt.message, /changed after “\/ask add a --json flag”\. Undo anyway/);
  assert.equal(prompt.detail, "a.py\nb.py\n… and 1 more");
  assert.equal(prompt.action, "Undo Anyway");
  assert.equal(conflictPrompt({ ...blocked, verb: "redo" }).action, "Redo Anyway");
});

test("undo / redo summaries", () => {
  const applied: StepResult = {
    available: true,
    verb: "undo",
    result: { label: "turn", applied: true, restored: ["a.py"], deleted: ["new.py", "x.py"], conflicts: ["a.py"], errors: [] },
  };
  assert.ok(!isBlocked(applied));
  assert.deepEqual(stepSummary(applied), {
    text: "✓ Undid “turn” (1 restored, 2 deleted). The overwritten versions are kept in the checkpoint store.",
    level: "ok",
  });
  assert.equal(stepSummary({ ...applied, verb: "redo", result: { ...applied.result!, conflicts: [] } }).text, "✓ Redid “turn” (1 restored, 2 deleted).");
  assert.deepEqual(stepSummary({ available: true, verb: "redo", result: null }), { text: "Nothing to redo.", level: "dim" });
  assert.ok(!isBlocked({ available: true, verb: "redo", result: null }));
  assert.equal(stepSummary({ available: false, reason: "turned off", verb: "undo", result: null }).level, "warn");
});

test("the view's Undo button only undoes its own change when that change is the latest", () => {
  const entry = (id: string) => ({ id, label: `change ${id}`, created: 0, file_count: 1 });
  const list: CheckpointList = { available: true, undo: [entry("aaaa"), entry("bbbb")], redo: [entry("cccc")] };
  assert.deepEqual(undoTarget(list, "bbbb"), { ok: true });
  assert.deepEqual(undoTarget(list, "aaaa"), { ok: false, reason: "not_latest", latest: entry("bbbb") });
  assert.deepEqual(undoTarget(list, "cccc"), { ok: false, reason: "undone" });
  assert.deepEqual(undoTarget(list, "dddd"), { ok: false, reason: "gone" });
});
