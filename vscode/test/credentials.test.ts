import assert from "node:assert/strict";
import { mkdtemp, mkdir, writeFile } from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { test } from "node:test";

import { credentialsFile, loadCredentials, userDataDir } from "../src/credentials";

test("the data folder matches kartrix/paths.py on every OS", () => {
  assert.equal(userDataDir({ LOCALAPPDATA: "C:\\Users\\a\\AppData\\Local" }, "win32", "C:\\Users\\a"), "C:\\Users\\a\\AppData\\Local\\kartrix");
  assert.equal(userDataDir({}, "darwin", "/Users/a"), "/Users/a/Library/Application Support/kartrix");
  assert.equal(userDataDir({}, "linux", "/home/a"), "/home/a/.local/share/kartrix");
  assert.equal(userDataDir({ XDG_DATA_HOME: "/data" }, "linux", "/home/a"), "/data/kartrix");
  assert.equal(userDataDir({ KARTRIX_HOME: "/k" }, "linux", "/home/a"), "/k/data");
  assert.equal(credentialsFile({ KARTRIX_HOME: "D:\\k" }, "win32"), "D:\\k\\data\\credentials.json");
});

test("credentials: environment first, then the CLI's file; null when signed out", async () => {
  const home = await mkdtemp(path.join(os.tmpdir(), "kartrix-ext-"));
  const env = { KARTRIX_HOME: home };
  assert.equal(await loadCredentials(env), null);
  await mkdir(path.join(home, "data"), { recursive: true });
  await writeFile(path.join(home, "data", "credentials.json"), JSON.stringify({ server: "http://127.0.0.1:8765/", token: "t", email: "a@b.c" }));
  assert.deepEqual(await loadCredentials(env), { server: "http://127.0.0.1:8765", token: "t", email: "a@b.c" });
  assert.deepEqual(await loadCredentials({ ...env, KARTRIX_SERVER_URL: "http://h:1/", KARTRIX_API_TOKEN: "x" }), {
    server: "http://h:1",
    token: "x",
    email: "",
  });
  await writeFile(path.join(home, "data", "credentials.json"), "{not json");
  assert.equal(await loadCredentials(env), null);
});
