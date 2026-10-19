/** Starts `kartrix serve` in the background when the local server isn't running (and stops it with VS Code). */

import { ChildProcess, spawn } from "node:child_process";
import * as os from "node:os";

export const DEFAULT_SERVER = "http://127.0.0.1:8765";

export function isLocal(server: string): boolean {
  try {
    return ["127.0.0.1", "localhost", "[::1]"].includes(new URL(server).hostname);
  } catch {
    return false;
  }
}

export async function reachable(server: string, fetchImpl: typeof fetch = fetch): Promise<boolean> {
  try {
    const response = await fetchImpl(`${server.replace(/\/+$/, "")}/v1/health`, { signal: AbortSignal.timeout(2000) });
    return response.ok;
  } catch {
    return false;
  }
}

export class LocalServer {
  private child: ChildProcess | null = null;

  constructor(private readonly log: (line: string) => void) {}

  get started(): boolean {
    return this.child !== null && this.child.exitCode === null;
  }

  /** Start `<cli> serve` on the server's port and wait until it answers (the first start creates the built-in
   * database, which takes a while). */
  async start(cli: string, server: string, timeoutMs = 120_000): Promise<void> {
    if (await reachable(server)) return;
    if (!this.started) {
      const port = new URL(server).port || "8765";
      this.log(`Starting ${cli} serve (port ${port})…`);
      const env = { ...process.env, KARTRIX_SERVER__PORT: port };
      const child = spawn(cli, ["serve"], { cwd: os.homedir(), env, windowsHide: true, stdio: ["ignore", "pipe", "pipe"], shell: false });
      child.stdout?.on("data", (chunk: Buffer) => this.log(chunk.toString().trimEnd()));
      child.stderr?.on("data", (chunk: Buffer) => this.log(chunk.toString().trimEnd()));
      child.on("exit", (code) => {
        this.log(`kartrix serve exited (code ${code})`);
        if (this.child === child) this.child = null;
      });
      const failed = new Promise<never>((_, reject) => {
        child.once("error", (e) => reject(new Error(`couldn't start "${cli} serve": ${e.message} — set kartrix.cliPath`)));
        child.once("exit", (code) => reject(new Error(`"${cli} serve" exited (code ${code}) — see the Kartrix output`)));
      });
      failed.catch(() => undefined);
      this.child = child;
      const deadline = Date.now() + timeoutMs;
      while (Date.now() < deadline) {
        await Promise.race([sleep(1000), failed]);
        if (await reachable(server)) return;
      }
      throw new Error(`the Kartrix server didn't answer within ${Math.round(timeoutMs / 1000)} s`);
    }
  }

  /** Stop the server this extension started; its workers end when their pipe closes. */
  stop(): void {
    if (this.child && this.child.exitCode === null) this.child.kill();
    this.child = null;
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
