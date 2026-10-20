/** Kartrix for VS Code: a thin client over a local `kartrix serve` — all agent and safety logic stays there. */

import * as vscode from "vscode";

import { SCHEME } from "./changes";
import { ChatController } from "./chat";
import { LocalServer } from "./server";

let server: LocalServer | null = null;

/** Returns the chat controller (the extension's exports), so an extension-host test can drive it. */
export function activate(context: vscode.ExtensionContext): ChatController {
  const output = vscode.window.createOutputChannel("Kartrix");
  server = new LocalServer((line) => output.appendLine(line));
  const chat = new ChatController(context, server, output);
  context.subscriptions.push(
    output,
    chat,
    vscode.window.registerWebviewViewProvider("kartrix.chat", chat, { webviewOptions: { retainContextWhenHidden: true } }),
    vscode.commands.registerCommand("kartrix.newConversation", () => chat.newConversation()),
    vscode.commands.registerCommand("kartrix.stop", () => chat.stop()),
    vscode.commands.registerCommand("kartrix.pickModel", () => chat.pickModel()),
    vscode.commands.registerCommand("kartrix.signIn", () => chat.signIn()),
    vscode.commands.registerCommand("kartrix.reconnect", () => chat.connect()),
    vscode.workspace.registerTextDocumentContentProvider(SCHEME, chat.changes),
    vscode.commands.registerCommand("kartrix.undo", () => chat.changes.step("undo")),
    vscode.commands.registerCommand("kartrix.redo", () => chat.changes.step("redo")),
    vscode.commands.registerCommand("kartrix.showCheckpoints", () => chat.changes.showCheckpoints()),
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration("kartrix.serverUrl")) void chat.connect();
    }),
  );
  return chat;
}

export function deactivate(): void {
  server?.stop();
  server = null;
}
