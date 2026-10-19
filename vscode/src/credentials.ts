/** The login the CLI saved with `kartrix login` (kartrix/client/api.py), so the extension uses the same account.
 * The folder mirrors kartrix/paths.py: `KARTRIX_HOME/data`, else platformdirs' per-user data folder. */

import { readFile } from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";

export interface Credentials {
  server: string;
  token: string;
  email: string;
}

type Env = Record<string, string | undefined>;

export function userDataDir(env: Env = process.env, platform: string = process.platform, home = os.homedir()): string {
  const pathFor = platform === "win32" ? path.win32 : path.posix;
  if (env.KARTRIX_HOME) return pathFor.join(env.KARTRIX_HOME, "data");
  if (platform === "win32") return pathFor.join(env.LOCALAPPDATA || pathFor.join(home, "AppData", "Local"), "kartrix");
  if (platform === "darwin") return pathFor.join(home, "Library", "Application Support", "kartrix");
  return pathFor.join(env.XDG_DATA_HOME || pathFor.join(home, ".local", "share"), "kartrix");
}

export function credentialsFile(env: Env = process.env, platform: string = process.platform): string {
  return (platform === "win32" ? path.win32 : path.posix).join(userDataDir(env, platform), "credentials.json");
}

/** `KARTRIX_SERVER_URL` + `KARTRIX_API_TOKEN` win (as in the CLI), else the saved file; null when signed out. */
export async function loadCredentials(env: Env = process.env, platform: string = process.platform): Promise<Credentials | null> {
  if (env.KARTRIX_SERVER_URL && env.KARTRIX_API_TOKEN) {
    return { server: env.KARTRIX_SERVER_URL.replace(/\/+$/, ""), token: env.KARTRIX_API_TOKEN, email: "" };
  }
  try {
    const data: unknown = JSON.parse(await readFile(credentialsFile(env, platform), "utf8"));
    if (typeof data !== "object" || data === null) return null;
    const { server, token, email } = data as Record<string, unknown>;
    if (typeof server !== "string" || typeof token !== "string" || !server || !token) return null;
    return { server: server.replace(/\/+$/, ""), token, email: typeof email === "string" ? email : "" };
  } catch {
    return null;
  }
}
