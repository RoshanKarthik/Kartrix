// Launches VS Code with this extension and scripts/editor-smoke.ts in its extension host — a separate profile and
// extensions folder (never your own), on the folder given (default: the current one). Usage:
//   npm run build && node scripts/run-editor-smoke.mjs [folder]
// VSCODE_EXE: the VS Code executable (default: found next to the `code` command); KARTRIX_SMOKE_DIR: where the
// profile, extensions folder and report go (default: <tmp>/kartrix-editor-smoke).
import { execFileSync, spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, rmSync } from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";

const here = path.dirname(fileURLToPath(import.meta.url));
const extension = path.resolve(here, "..");
const folder = path.resolve(process.argv[2] ?? process.cwd());
const work = process.env.KARTRIX_SMOKE_DIR ?? path.join(os.tmpdir(), "kartrix-editor-smoke");
mkdirSync(work, { recursive: true });

function vscodeExe() {
  if (process.env.VSCODE_EXE) return process.env.VSCODE_EXE;
  const finder = process.platform === "win32" ? "where" : "which";
  const cli = execFileSync(finder, ["code"], { encoding: "utf8" }).split(/\r?\n/)[0].trim();
  if (process.platform === "win32") return path.resolve(path.dirname(cli), "..", "Code.exe"); // <install>/bin/code.cmd
  if (process.platform === "darwin") return path.resolve(path.dirname(cli), "..", "..", "..", "MacOS", "Electron");
  return cli;
}

await build({
  entryPoints: [path.join(here, "editor-smoke.ts")],
  outfile: path.join(work, "tests", "index.js"),
  bundle: true,
  platform: "node",
  format: "cjs",
  external: ["vscode"],
  logLevel: "warning",
});

const exe = vscodeExe();
if (!existsSync(exe)) throw new Error(`VS Code not found at ${exe} — set VSCODE_EXE`);
const reportFile = path.join(work, "report.json");
rmSync(reportFile, { force: true });
// A fresh profile every run: a layout left by an earlier run (a closed or moved view) would keep the chat hidden.
rmSync(path.join(work, "user"), { recursive: true, force: true });
const args = [
  folder,
  `--extensionDevelopmentPath=${extension}`,
  `--extensionTestsPath=${path.join(work, "tests", "index.js")}`,
  `--user-data-dir=${path.join(work, "user")}`,
  `--extensions-dir=${path.join(work, "extensions")}`,
  "--disable-workspace-trust",
  "--skip-welcome",
  "--skip-release-notes",
  "--disable-telemetry",
  "--new-window",
];
const env = { ...process.env, KARTRIX_SMOKE_REPORT: reportFile };
delete env.ELECTRON_RUN_AS_NODE;
console.log(`${exe} ${args.join(" ")}`);
const code = await new Promise((resolve) => spawn(exe, args, { env, stdio: "inherit" }).on("exit", resolve));
if (existsSync(reportFile)) console.log(readFileSync(reportFile, "utf8"));
process.exit(code ?? 1);
